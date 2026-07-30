"""Tests for agent response generation and memory persistence boundaries."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import anyio

from app.agent import (
    SYSTEM_PROMPT,
    _format_memory_context,
    _LEGACY_SYSTEM_PROMPT,
    _LEGACY_SYSTEM_PROMPT_V2,
    generate_agent_response,
    handle_message,
)


def test_system_prompt_is_bounded_and_preserves_legacy_prompt():
    assert "TARGETED GRAPH EXPLORATION" in _LEGACY_SYSTEM_PROMPT
    assert "BOUNDED TOOL USE" in _LEGACY_SYSTEM_PROMPT_V2
    assert len(SYSTEM_PROMPT) < len(_LEGACY_SYSTEM_PROMPT_V2)
    assert len(SYSTEM_PROMPT) < len(_LEGACY_SYSTEM_PROMPT)
    assert "provide every required argument" in SYSTEM_PROMPT
    assert "Do not repeat the same call" in SYSTEM_PROMPT
    assert "Use read-only Cypher only" in SYSTEM_PROMPT
    assert "complete matching set" in SYSTEM_PROMPT
    assert "Answer directly in the user's language" in SYSTEM_PROMPT


def test_format_memory_context_includes_timestamps_and_conflict_rule():
    memories = [
        {
            "memory": "User prefers responses in Vietnamese.",
            "created_at": "2026-07-01T08:00:00+00:00",
        },
        {
            "memory": "User prefers responses in English.",
            "created_at": "2026-07-29T08:00:00+00:00",
        },
    ]

    context = _format_memory_context(memories)

    assert "ordered by relevance, not by time" in context
    assert "prefer the memory with the latest created_at timestamp" in context
    assert "The current request always overrides all memories" in context
    assert (
        "[created_at=2026-07-01T08:00:00+00:00] "
        "User prefers responses in Vietnamese."
    ) in context
    assert (
        "[created_at=2026-07-29T08:00:00+00:00] "
        "User prefers responses in English."
    ) in context


def test_generate_agent_response_does_not_save_memory():
    async def run_test():
        model_result = MagicMock()
        model_result.output = "Assistant response"
        model_result.usage.return_value = {"total_tokens": 10}

        with (
            patch(
                "app.agent._prepare_run",
                AsyncMock(return_value=("conversation-1", [], "No memories")),
            ),
            patch("app.agent.agent.run", AsyncMock(return_value=model_result)),
            patch("app.agent.save_conversation_memory", AsyncMock()) as save_memory,
        ):
            result = await generate_agent_response(
                "Hello",
                session_id="conversation-1",
                user_id="user-1",
            )

        assert result["response"] == "Assistant response"
        assert result["session_id"] == "conversation-1"
        assert "diagnostic_run_id" in result
        save_memory.assert_not_awaited()

    anyio.run(run_test)


def test_generate_agent_response_includes_short_term_context_without_message_history():
    async def run_test():
        model_result = MagicMock()
        model_result.output = "Assistant response"
        model_result.usage.return_value = {"total_tokens": 10}
        runner = AsyncMock(return_value=model_result)

        with (
            patch(
                "app.agent._prepare_run",
                AsyncMock(return_value=("conversation-1", [], "No memories")),
            ),
            patch("app.agent.agent.run", runner),
        ):
            await generate_agent_response(
                "Nguoi nay co thuoc active nao?",
                session_id="conversation-1",
                user_id="user-1",
                short_term_context=(
                    "SHORT-TERM CONVERSATION SUMMARY\n"
                    "Patient/123 was identified earlier."
                ),
            )

        effective_message = runner.await_args.args[0]
        assert "<long_term_memory>" in effective_message
        assert "</long_term_memory>" in effective_message
        assert "<conversation_history>" in effective_message
        assert "</conversation_history>" in effective_message
        assert "Patient/123 was identified earlier." in effective_message
        assert "<current_request>" in effective_message
        assert "</current_request>" in effective_message
        assert effective_message.count("Nguoi nay co thuoc active nao?") == 1
        assert runner.await_args.kwargs["message_history"] == []

    anyio.run(run_test)


def test_handle_message_wraps_generation_and_saves_memory():
    async def run_test():
        generated = {
            "response": "Assistant response",
            "session_id": "conversation-1",
            "graph_data": None,
            "diagnostic_run_id": "run-1",
        }

        with (
            patch("app.agent.generate_agent_response", AsyncMock(return_value=generated)) as generate,
            patch("app.agent.save_conversation_memory", AsyncMock()) as save_memory,
        ):
            result = await handle_message(
                "Hello",
                session_id="conversation-1",
                user_id="user-1",
            )

        assert result == generated
        generate.assert_awaited_once_with(
            "Hello",
            session_id="conversation-1",
            user_id="user-1",
        )
        save_memory.assert_awaited_once_with(
            user_id="user-1",
            session_id="conversation-1",
            user_message="Hello",
            assistant_message="Assistant response",
        )

    anyio.run(run_test)
