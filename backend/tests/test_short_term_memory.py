"""Tests for short-term memory splitting and context building."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import anyio

from app.services.short_term_memory import (
    ShortTermMemoryService,
    _MemoryState,
    build_conversation_context,
    split_messages_for_compaction,
)
from app.services.token_counter import ConversationMessage, TokenCounter


class FixedTokenCounter:
    def count_text(self, text: str) -> int:
        return len(text)

    def count_messages(self, messages: list[ConversationMessage]) -> int:
        return sum(len(message.content) for message in messages)


def _message(
    content: str,
    *,
    id: UUID | None = None,
    offset: int = 0,
) -> ConversationMessage:
    return ConversationMessage(
        id=id or uuid4(),
        role="user",
        content=content,
        created_at=datetime(2026, 7, 29, tzinfo=timezone.utc) + timedelta(seconds=offset),
    )


def test_split_below_budget_keeps_all_messages_recent():
    messages = [_message("aa", offset=1), _message("bb", offset=2)]

    old, recent = split_messages_for_compaction(
        messages,
        recent_token_budget=10,
        token_counter=FixedTokenCounter(),
    )

    assert old == []
    assert recent == messages


def test_split_over_budget_moves_old_messages_to_summarize_and_preserves_order():
    messages = [
        _message("aaaa", offset=1),
        _message("bbbb", offset=2),
        _message("cccc", offset=3),
    ]

    old, recent = split_messages_for_compaction(
        messages,
        recent_token_budget=8,
        token_counter=FixedTokenCounter(),
    )

    assert old == messages[:1]
    assert recent == messages[1:]
    assert old + recent == messages


def test_recent_messages_are_contiguous_suffix():
    messages = [
        _message("aaa", offset=1),
        _message("bbbbb", offset=2),
        _message("ccccccc", offset=3),
    ]

    old, recent = split_messages_for_compaction(
        messages,
        recent_token_budget=10,
        token_counter=FixedTokenCounter(),
    )

    assert old == messages[:2]
    assert recent == messages[2:]
    assert old + recent == messages
    assert recent == messages[len(old):]


def test_split_does_not_duplicate_messages_between_old_and_recent():
    messages = [_message("aaaa", offset=1), _message("bbbb", offset=2)]

    old, recent = split_messages_for_compaction(
        messages,
        recent_token_budget=4,
        token_counter=FixedTokenCounter(),
    )

    assert {message.id for message in old}.isdisjoint(
        {message.id for message in recent}
    )
    assert old + recent == messages
    assert recent == messages[len(old):]


def test_split_keeps_protected_message_even_when_over_budget():
    protected = _message("x" * 20, offset=2)
    messages = [_message("aaaa", offset=1), protected]

    old, recent = split_messages_for_compaction(
        messages,
        recent_token_budget=4,
        token_counter=FixedTokenCounter(),
        protected_message_ids={protected.id},
    )

    assert protected in recent
    assert old == [messages[0]]
    assert old + recent == messages


def test_build_conversation_context_omits_current_request_and_empty_sections():
    recent = [
        ConversationMessage(
            id=uuid4(),
            role="assistant",
            content="Patient/123 was found.",
            created_at=datetime.now(timezone.utc),
        )
    ]

    context = build_conversation_context(
        summary="The user is checking Patient/123.",
        recent_messages=recent,
    )

    assert "SHORT-TERM CONVERSATION SUMMARY" in context
    assert "RECENT CONVERSATION" in context
    assert "ASSISTANT: Patient/123 was found." in context
    assert "CURRENT USER REQUEST" not in context


def test_token_counter_protocol_is_satisfied_by_fixed_counter():
    counter: TokenCounter = FixedTokenCounter()
    assert counter.count_text("abc") == 3


def test_prepare_from_state_does_not_use_new_summary_when_persist_fails(caplog, monkeypatch):
    async def run_test():
        messages = [
            _message("aaaa", offset=1),
            _message("bbbb", offset=2),
            _message("cccc", offset=3),
        ]
        conversation = MagicMock()
        conversation.summary = "old summary"
        conversation.memory_version = 1
        conversation.summary_through_message_id = None

        state = _MemoryState(
            conversation=conversation,
            messages_before_current=messages,
            unsummarized_messages=messages,
            version=1,
            cutoff_id=None,
        )
        summarizer = AsyncMock()
        summarizer.summarize.return_value = "new summary"
        service = ShortTermMemoryService(
            session_factory=None,
            token_counter=FixedTokenCounter(),
            summarizer=summarizer,
        )

        monkeypatch.setattr("app.services.short_term_memory.settings.short_term_max_tokens", 10)
        monkeypatch.setattr("app.services.short_term_memory.settings.short_term_recent_tokens", 4)
        monkeypatch.setattr("app.services.short_term_memory.settings.short_term_compaction_threshold", 0.5)

        with (
            patch.object(service, "_persist_summary_if_current", AsyncMock(return_value=False)),
            caplog.at_level(logging.INFO, logger="app.services.short_term_memory"),
        ):
            result = await service._prepare_from_state(
                state,
                conversation_id=uuid4(),
                user_id=uuid4(),
                current_user_message_id=uuid4(),
                current_content="dddd",
                retry_on_version_mismatch=False,
            )

        assert result.summary == "old summary"
        assert "new summary" not in result.summary
        assert result.recent_messages == messages[2:]
        assert messages == state.unsummarized_messages
        assert "SHORT_TERM_COMPACTION_FAILED" in caplog.text
        assert "SummaryPersistConflict" in caplog.text
        assert "SHORT_TERM_COMPACTION |" not in caplog.text

    anyio.run(run_test)
