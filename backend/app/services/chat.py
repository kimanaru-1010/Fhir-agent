"""Shared chat service for conversation and message APIs."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from app.agents.fhir import generate_agent_response
from app.services.long_term_memory import save_conversation_memory
from app.services.short_term_memory import (
    ShortTermMemoryService,
    build_conversation_context,
)


async def generate_assistant_response(
    *,
    content: str,
    user_id: str,
    conversation_id: str,
    current_user_message_id: UUID | None = None,
) -> str:
    short_term_context = ""
    if current_user_message_id is not None:
        context = await ShortTermMemoryService().prepare_context(
            conversation_id=UUID(conversation_id),
            user_id=UUID(user_id),
            current_user_message_id=current_user_message_id,
            current_content=content,
        )
        short_term_context = build_conversation_context(
            summary=context.summary,
            recent_messages=context.recent_messages,
        )

    result = await generate_agent_response(
        content,
        session_id=conversation_id,
        user_id=user_id,
        short_term_context=short_term_context,
    )
    assistant_content = extract_agent_text(result)
    if not assistant_content.strip():
        raise RuntimeError("Agent returned an empty response")
    return assistant_content


def extract_agent_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("response", "content", "message", "text", "output"):
            value = result.get(key)
            if value is not None:
                return str(value)
        return str(result)
    for attr in ("response", "content", "message", "text", "output"):
        value = getattr(result, attr, None)
        if value is not None:
            return str(value)
    return str(result)


async def persist_chat_memory(
    *,
    user_id: str,
    conversation_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    await save_conversation_memory(
        user_id=user_id,
        session_id=conversation_id,
        user_message=user_message,
        assistant_message=assistant_message,
    )
