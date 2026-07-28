"""Tests for agent response generation and memory persistence boundaries."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import anyio

from app.agent import generate_agent_response, handle_message


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
