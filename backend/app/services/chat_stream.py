"""Shared SSE streaming for persisted chat exchanges."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.graph.client import get_collector
from app.db.session import AsyncSessionFactory
from app.db.models import Conversation, Message
from app.schemas.conversation import ConversationResponse
from app.schemas.message import MessageResponse
from app.services.chat import generate_assistant_response, persist_chat_memory

logger = logging.getLogger(__name__)


def format_sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, default=str, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def serialize_message(message: Message) -> dict[str, Any]:
    return MessageResponse.model_validate(message).model_dump(mode="json")


def serialize_conversation(conversation: Conversation) -> dict[str, Any]:
    return ConversationResponse.model_validate(conversation).model_dump(mode="json")


async def persist_assistant_message(
    *,
    conversation_id: UUID,
    user_id: UUID,
    content: str,
) -> tuple[Conversation, Message]:
    async with AsyncSessionFactory() as session:
        result = await session.execute(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        conversation = result.scalar_one_or_none()
        if conversation is None:
            raise RuntimeError("Conversation not found")

        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=content,
        )
        try:
            session.add(assistant_message)
            conversation.updated_at = datetime.now(timezone.utc)
            await session.commit()
        except Exception:
            await session.rollback()
            raise

        await session.refresh(conversation)
        await session.refresh(assistant_message)
        return conversation, assistant_message


async def stream_persisted_exchange(
    *,
    conversation_id: UUID,
    user_id: UUID,
    user_message_id: UUID,
    content: str,
    start_event: str,
    start_payload: dict[str, Any],
    overall_timeout: float = 900.0,
) -> AsyncIterator[str]:
    event_queue: asyncio.Queue = asyncio.Queue()
    collector = get_collector()
    agent_task: asyncio.Task | None = None
    queue_token = None
    loop = asyncio.get_event_loop()
    start_time = loop.time()

    try:
        yield format_sse(start_event, start_payload)

        collector.drain()
        collector.drain_tool_calls()
        queue_token = collector.set_event_queue(event_queue)

        agent_task = asyncio.create_task(
            generate_assistant_response(
                content=content,
                user_id=str(user_id),
                conversation_id=str(conversation_id),
                current_user_message_id=user_message_id,
            )
        )

        while True:
            if loop.time() - start_time > overall_timeout:
                agent_task.cancel()
                raise TimeoutError("Request exceeded maximum duration")

            if agent_task.done() and event_queue.empty():
                break

            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            event_name = event.get("event")
            if event_name in {"tool_start", "tool_end"}:
                yield format_sse(event_name, event.get("data") or {})

        assistant_content = await agent_task
        conversation, assistant_message = await persist_assistant_message(
            conversation_id=conversation_id,
            user_id=user_id,
            content=assistant_content,
        )

        conversation_payload = serialize_conversation(conversation)
        assistant_payload = serialize_message(assistant_message)

        yield format_sse(
            "text_delta",
            {
                "text": assistant_content,
                "delta": assistant_content,
            },
        )

        try:
            await persist_chat_memory(
                user_id=str(user_id),
                conversation_id=str(conversation_id),
                user_message=content,
                assistant_message=assistant_content,
            )
        except Exception:
            logger.exception(
                "Failed to persist memory for conversation_id=%s",
                conversation_id,
            )

        yield format_sse(
            "done",
            {
                "conversation": conversation_payload,
                "user_message": start_payload["user_message"],
                "assistant_message": assistant_payload,
                "response": assistant_content,
            },
        )
    except asyncio.CancelledError:
        if agent_task is not None and not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except (asyncio.CancelledError, Exception):
                pass
        raise
    except Exception:
        logger.exception(
            "Failed to stream persisted exchange for conversation_id=%s user_message_id=%s",
            conversation_id,
            user_message_id,
        )
        yield format_sse("error", {"detail": "Unable to process message"})
    finally:
        if agent_task is not None and not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except (asyncio.CancelledError, Exception):
                pass
        collector.clear_event_queue(queue_token)
