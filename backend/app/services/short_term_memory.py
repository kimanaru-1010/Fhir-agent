"""Short-term conversation memory built from PostgreSQL messages."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.db.session import AsyncSessionFactory
from app.db.models import Conversation, Message
from app.services.short_term_summarizer import (
    ConversationSummarizer,
    LLMConversationSummarizer,
)
from app.services.token_counter import (
    ApproximateTokenCounter,
    ConversationMessage,
    TokenCounter,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShortTermContext:
    summary: str
    recent_messages: list[ConversationMessage]
    estimated_tokens: int


@dataclass(frozen=True)
class CompactionResult:
    compacted: bool
    summary: str
    summary_through_message_id: UUID | None
    tokens_before: int
    tokens_after: int
    messages_compacted: int


@dataclass(frozen=True)
class _MemoryState:
    conversation: Conversation
    messages_before_current: list[ConversationMessage]
    unsummarized_messages: list[ConversationMessage]
    version: int
    cutoff_id: UUID | None


def split_messages_for_compaction(
    messages: list[ConversationMessage],
    *,
    recent_token_budget: int,
    token_counter: TokenCounter,
    protected_message_ids: set[UUID] | None = None,
) -> tuple[list[ConversationMessage], list[ConversationMessage]]:
    protected = protected_message_ids or set()
    split_index = len(messages)
    recent_tokens = 0

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        message_tokens = token_counter.count_messages([message])
        if message.id in protected:
            split_index = index
            recent_tokens += message_tokens
            continue

        if recent_tokens + message_tokens > recent_token_budget:
            break

        recent_tokens += message_tokens
        split_index = index

    return messages[:split_index], messages[split_index:]


def build_conversation_context(
    *,
    summary: str,
    recent_messages: list[ConversationMessage],
) -> str:
    sections: list[str] = []
    if summary.strip():
        sections.append(
            "SHORT-TERM CONVERSATION SUMMARY\n"
            f"{summary.strip()}"
        )
    if recent_messages:
        lines = [
            f"{message.role.upper()}: {message.content}"
            for message in recent_messages
        ]
        sections.append(
            "RECENT CONVERSATION\n"
            + "\n\n".join(lines)
        )
    return "\n\n".join(sections)


async def load_short_term_context(
    *,
    session: AsyncSession,
    conversation_id: UUID,
    user_id: UUID,
    current_user_message_id: UUID,
) -> ShortTermContext:
    service = ShortTermMemoryService(
        session_factory=None,
        token_counter=ApproximateTokenCounter(),
        summarizer=None,
    )
    state = await service._load_state(
        session=session,
        conversation_id=conversation_id,
        user_id=user_id,
        current_user_message_id=current_user_message_id,
    )
    summary = state.conversation.summary or ""
    tokens = service._estimate_context_tokens(
        summary=summary,
        messages=state.unsummarized_messages,
        current_content="",
    )
    return ShortTermContext(
        summary=summary,
        recent_messages=state.unsummarized_messages,
        estimated_tokens=tokens,
    )


class ShortTermMemoryService:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None = AsyncSessionFactory,
        token_counter: TokenCounter | None = None,
        summarizer: ConversationSummarizer | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.token_counter = token_counter or ApproximateTokenCounter()
        self.summarizer = summarizer or LLMConversationSummarizer(
            token_counter=self.token_counter,
        )

    async def prepare_context(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        current_user_message_id: UUID,
        current_content: str,
    ) -> ShortTermContext:
        if not settings.short_term_enabled:
            return ShortTermContext(summary="", recent_messages=[], estimated_tokens=0)
        if self.session_factory is None:
            raise RuntimeError("ShortTermMemoryService requires a session factory")

        async with self.session_factory() as session:
            state = await self._load_state(
                session=session,
                conversation_id=conversation_id,
                user_id=user_id,
                current_user_message_id=current_user_message_id,
            )

        return await self._prepare_from_state(
            state,
            conversation_id=conversation_id,
            user_id=user_id,
            current_user_message_id=current_user_message_id,
            current_content=current_content,
            retry_on_version_mismatch=True,
        )

    async def _prepare_from_state(
        self,
        state: _MemoryState,
        *,
        conversation_id: UUID,
        user_id: UUID,
        current_user_message_id: UUID,
        current_content: str,
        retry_on_version_mismatch: bool,
    ) -> ShortTermContext:
        summary = state.conversation.summary or ""
        tokens_before = self._estimate_context_tokens(
            summary=summary,
            messages=state.unsummarized_messages,
            current_content=current_content,
        )
        threshold_tokens = int(
            settings.short_term_max_tokens
            * settings.short_term_compaction_threshold
        )

        if tokens_before < threshold_tokens:
            _, recent_messages = split_messages_for_compaction(
                state.unsummarized_messages,
                recent_token_budget=settings.short_term_recent_tokens,
                token_counter=self.token_counter,
            )
            return ShortTermContext(
                summary=summary,
                recent_messages=recent_messages,
                estimated_tokens=self._estimate_context_tokens(
                    summary=summary,
                    messages=recent_messages,
                    current_content=current_content,
                ),
            )

        old_messages, recent_messages = split_messages_for_compaction(
            state.unsummarized_messages,
            recent_token_budget=settings.short_term_recent_tokens,
            token_counter=self.token_counter,
        )
        if not old_messages:
            return self._bounded_fallback_context(
                summary=summary,
                messages=state.unsummarized_messages,
                current_content=current_content,
            )

        start = time.monotonic()
        try:
            new_summary = await self.summarizer.summarize(
                previous_summary=summary,
                messages=old_messages,
            )
        except Exception as exc:
            logger.warning(
                "SHORT_TERM_COMPACTION_FAILED | conversation_id=%s error_type=%s",
                conversation_id,
                type(exc).__name__,
            )
            return self._bounded_fallback_context(
                summary=summary,
                messages=state.unsummarized_messages,
                current_content=current_content,
            )

        cutoff_id = old_messages[-1].id
        persisted = await self._persist_summary_if_current(
            conversation_id=conversation_id,
            user_id=user_id,
            expected_version=state.version,
            expected_cutoff_id=state.cutoff_id,
            summary=new_summary,
            cutoff_id=cutoff_id,
        )
        if not persisted:
            if retry_on_version_mismatch and self.session_factory is not None:
                async with self.session_factory() as session:
                    refreshed = await self._load_state(
                        session=session,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        current_user_message_id=current_user_message_id,
                    )
                return await self._prepare_from_state(
                    refreshed,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    current_user_message_id=current_user_message_id,
                    current_content=current_content,
                    retry_on_version_mismatch=False,
                )

            logger.warning(
                "SHORT_TERM_COMPACTION_FAILED | conversation_id=%s error_type=%s",
                conversation_id,
                "SummaryPersistConflict",
            )
            return self._bounded_fallback_context(
                summary=summary,
                messages=state.unsummarized_messages,
                current_content=current_content,
            )

        tokens_after = self._estimate_context_tokens(
            summary=new_summary,
            messages=recent_messages,
            current_content=current_content,
        )
        logger.info(
            "SHORT_TERM_COMPACTION | conversation_id=%s tokens_before=%s "
            "tokens_after=%s messages_compacted=%s summary_tokens=%s duration_ms=%s",
            conversation_id,
            tokens_before,
            tokens_after,
            len(old_messages),
            self.token_counter.count_text(new_summary),
            int((time.monotonic() - start) * 1000),
        )
        return ShortTermContext(
            summary=new_summary,
            recent_messages=recent_messages,
            estimated_tokens=tokens_after,
        )

    async def _load_state(
        self,
        *,
        session: AsyncSession,
        conversation_id: UUID,
        user_id: UUID,
        current_user_message_id: UUID,
    ) -> _MemoryState:
        conversation_result = await session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        conversation = conversation_result.scalar_one_or_none()
        if conversation is None:
            raise RuntimeError("Conversation not found")

        message_result = await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
        ordered_messages = [
            ConversationMessage(
                id=message.id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            )
            for message in message_result.scalars().all()
        ]
        current_index = next(
            (
                index
                for index, message in enumerate(ordered_messages)
                if message.id == current_user_message_id
            ),
            None,
        )
        if current_index is None:
            raise RuntimeError("Current user message not found")

        messages_before_current = ordered_messages[:current_index]
        cutoff_id = conversation.summary_through_message_id
        if cutoff_id is None:
            unsummarized_messages = messages_before_current
        else:
            cutoff_index = next(
                (
                    index
                    for index, message in enumerate(messages_before_current)
                    if message.id == cutoff_id
                ),
                None,
            )
            unsummarized_messages = (
                messages_before_current
                if cutoff_index is None
                else messages_before_current[cutoff_index + 1 :]
            )

        return _MemoryState(
            conversation=conversation,
            messages_before_current=messages_before_current,
            unsummarized_messages=unsummarized_messages,
            version=conversation.memory_version or 0,
            cutoff_id=cutoff_id,
        )

    async def _persist_summary_if_current(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        expected_version: int,
        expected_cutoff_id: UUID | None,
        summary: str,
        cutoff_id: UUID,
    ) -> bool:
        if self.session_factory is None:
            return False
        async with self.session_factory() as session:
            result = await session.execute(
                select(Conversation)
                .where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                )
                .with_for_update()
            )
            conversation = result.scalar_one_or_none()
            if conversation is None:
                await session.rollback()
                raise RuntimeError("Conversation not found")
            if (
                (conversation.memory_version or 0) != expected_version
                or conversation.summary_through_message_id != expected_cutoff_id
            ):
                await session.rollback()
                return False

            conversation.summary = summary.strip()
            conversation.summary_through_message_id = cutoff_id
            conversation.summary_updated_at = datetime.now(timezone.utc)
            conversation.memory_version = (conversation.memory_version or 0) + 1
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                logger.warning(
                    "SHORT_TERM_COMPACTION_FAILED | conversation_id=%s error_type=%s",
                    conversation_id,
                    "SummaryPersistError",
                )
                return False
            return True

    def _bounded_fallback_context(
        self,
        *,
        summary: str,
        messages: list[ConversationMessage],
        current_content: str,
    ) -> ShortTermContext:
        _, recent_messages = split_messages_for_compaction(
            messages,
            recent_token_budget=settings.short_term_recent_tokens,
            token_counter=self.token_counter,
        )
        return ShortTermContext(
            summary=summary,
            recent_messages=recent_messages,
            estimated_tokens=self._estimate_context_tokens(
                summary=summary,
                messages=recent_messages,
                current_content=current_content,
            ),
        )

    def _estimate_context_tokens(
        self,
        *,
        summary: str,
        messages: list[ConversationMessage],
        current_content: str,
    ) -> int:
        return (
            self.token_counter.count_text(summary)
            + self.token_counter.count_messages(messages)
            + self.token_counter.count_text(current_content)
        )
