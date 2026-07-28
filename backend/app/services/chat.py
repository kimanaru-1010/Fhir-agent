"""Shared chat service for conversation and message APIs."""

from __future__ import annotations

from typing import Any

from app.agent import generate_agent_response
from app.memory import save_conversation_memory


async def generate_assistant_response(
    *,
    content: str,
    user_id: str,
    conversation_id: str,
) -> str:
    result = await generate_agent_response(
        content,
        session_id=conversation_id,
        user_id=user_id,
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
