"""Tests for chat service short-term memory integration."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import anyio

from app.services.short_term_memory import ShortTermContext
from app.services.token_counter import ConversationMessage


def test_generate_assistant_response_prepares_short_term_context_and_runs_agent_once():
    async def run_test():
        from app.services.chat import generate_assistant_response

        current_message_id = uuid4()
        recent = ConversationMessage(
            id=uuid4(),
            role="assistant",
            content="Patient/123 was found.",
            created_at=datetime.now(timezone.utc),
        )
        service = AsyncMock()
        service.prepare_context.return_value = ShortTermContext(
            summary="The user is asking about Patient/123.",
            recent_messages=[recent],
            estimated_tokens=20,
        )

        with (
            patch("app.services.chat.ShortTermMemoryService", return_value=service),
            patch(
                "app.services.chat.generate_agent_response",
                AsyncMock(return_value={"response": "Answer"}),
            ) as agent,
        ):
            result = await generate_assistant_response(
                content="What medications are active?",
                user_id=str(uuid4()),
                conversation_id=str(uuid4()),
                current_user_message_id=current_message_id,
            )

        assert result == "Answer"
        service.prepare_context.assert_awaited_once()
        agent.assert_awaited_once()
        short_term_context = agent.await_args.kwargs["short_term_context"]
        assert "SHORT-TERM CONVERSATION SUMMARY" in short_term_context
        assert "Patient/123 was found." in short_term_context

    anyio.run(run_test)
