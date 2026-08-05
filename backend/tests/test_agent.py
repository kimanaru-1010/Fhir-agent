"""Tests for agent response generation and memory persistence boundaries."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import anyio

from app.agents.fhir import (
    AgentDeps,
    SYSTEM_PROMPT,
    _BATCH_RESOURCE_LIMIT,
    _CURRENT_IMAGE_ATTACHMENTS,
    _TOOL_RESULT_LIMIT,
    _batch_size_error,
    _format_memory_context,
    _normalize_optional_exact_filter,
    find_patient_skin_images,
    generate_agent_response,
    handle_message,
    start_skin_diagnostic,
)
from app.schemas.message import ChatImageAttachment


def test_system_prompt_is_concise_and_preserves_core_rules():
    assert len(SYSTEM_PROMPT) < 7_000
    assert "The current user request is the only task to solve" in SYSTEM_PROMPT
    assert "follow tool schemas exactly" in SYSTEM_PROMPT
    assert "prefer batch operations" in SYSTEM_PROMPT
    assert "repeat a tool call with the same or equivalent arguments" in SYSTEM_PROMPT
    assert "Never interpret unknown codes" in SYSTEM_PROMPT
    assert "Answer in the user's language" in SYSTEM_PROMPT
    assert "Prefer clinical or business" in SYSTEM_PROMPT
    assert "meaning over technical identifiers" in SYSTEM_PROMPT
    assert "do not summarize" in SYSTEM_PROMPT
    assert "lists of IDs" in SYSTEM_PROMPT


def test_system_prompt_forbids_duplicate_tool_calls():
    assert "repeat a tool call with the same or equivalent arguments" in SYSTEM_PROMPT
    assert "reuse the existing result" in SYSTEM_PROMPT


def test_system_prompt_routes_image_diagnosis_from_attachment_metadata():
    assert "IMAGE_ATTACHMENT" in SYSTEM_PROMPT
    assert "Use start_skin_diagnostic only when" in SYSTEM_PROMPT
    assert "Use the doctor's current message exactly as initial_complaint" in SYSTEM_PROMPT
    assert "Do not call start_skin_diagnostic" in SYSTEM_PROMPT


def test_batch_size_error_requires_smaller_batches():
    assert _batch_size_error(["1"] * _BATCH_RESOURCE_LIMIT) is None

    payload = json.loads(
        _batch_size_error(
            [str(index) for index in range(_BATCH_RESOURCE_LIMIT + 1)]
        )
    )

    assert payload["status"] == "error"
    assert "split the ids into smaller batches" in payload["message"]


def test_optional_exact_filter_normalizes_wildcard_to_no_filter():
    assert _normalize_optional_exact_filter("*") == ""
    assert _normalize_optional_exact_filter(" * ") == ""
    assert _normalize_optional_exact_filter("") == ""
    assert _normalize_optional_exact_filter("Condition") == "Condition"


def test_find_patient_skin_images_returns_metadata_only_and_collects_attachment():
    async def run_test():
        ctx = MagicMock()
        ctx.deps = AgentDeps(session_id="session-1", user_id="doctor-1")
        attachments: list[ChatImageAttachment] = []
        token = _CURRENT_IMAGE_ATTACHMENTS.set(attachments)

        try:
            with patch(
                "app.agents.fhir.search_patient_skin_images",
                AsyncMock(
                    return_value=[
                        {
                            "patient_id": "12261",
                            "diagnostic_report_id": "report-1",
                            "media_id": "media-1",
                            "binary_id": "17761",
                            "created_at": "2026-08-04T03:00:00Z",
                            "conclusion": "Skin image conclusion",
                            "content_type": "image/jpeg",
                            "data": "must-not-leak",
                        }
                    ]
                ),
            ) as search:
                result = await find_patient_skin_images(
                    ctx,
                    patient_id="12261",
                    count=1,
                    sort="desc",
                )
        finally:
            _CURRENT_IMAGE_ATTACHMENTS.reset(token)

        payload = json.loads(result)
        row = payload["data"][0]
        filters = search.await_args.args[0]
        assert filters.patient_id == "12261"
        assert filters.count == 1
        assert filters.sort == "desc"
        assert row == {
            "patient_id": "12261",
            "diagnostic_report_id": "report-1",
            "media_id": "media-1",
            "binary_id": "17761",
            "created_at": "2026-08-04T03:00:00Z",
            "content_type": "image/jpeg",
            "url": "/api/skin-images/files/17761",
        }
        assert "data" not in row
        assert "conclusion" not in row
        assert "must-not-leak" not in json.dumps(payload)
        assert attachments[0].url == "/api/skin-images/files/17761"
        assert attachments[0].description is None

    anyio.run(run_test)


def test_find_patient_skin_images_requires_patient_context():
    async def run_test():
        ctx = MagicMock()
        ctx.deps = AgentDeps(session_id="session-1", user_id="doctor-1")

        result = await find_patient_skin_images(ctx, patient_id=None)

        payload = json.loads(result)
        assert payload["status"] == "patient_required"
        assert payload["data"] == []

    anyio.run(run_test)


def test_start_skin_diagnostic_tool_uses_binary_id_and_exact_complaint():
    async def run_test():
        ctx = MagicMock()
        ctx.deps = AgentDeps(session_id="session-1", user_id="doctor-1")
        run = MagicMock()
        run.id = "run-1"

        with patch(
            "app.agents.fhir.start_skin_diagnostic_from_binary",
            AsyncMock(return_value=run),
        ) as starter:
            result = await start_skin_diagnostic(
                ctx,
                patient_id="12261",
                binary_id="binary-123",
                initial_complaint="Với ảnh trên tôi cảm thấy ngứa khó chịu",
            )

        starter.assert_awaited_once_with(
            user_id="doctor-1",
            conversation_id="session-1",
            patient_id="12261",
            binary_id="binary-123",
            initial_complaint="Với ảnh trên tôi cảm thấy ngứa khó chịu",
        )
        payload = json.loads(result)
        assert payload["status"] == "ok"
        assert payload["data"]["run_id"] == "run-1"
        assert payload["data"]["current_step"] == "visual_extract"
        assert payload["data"]["binary_id"] == "binary-123"

    anyio.run(run_test)


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
                "app.agents.fhir._prepare_run",
                AsyncMock(return_value=("conversation-1", [], "No memories")),
            ),
            patch("app.agents.fhir.agent.run", AsyncMock(return_value=model_result)),
            patch("app.agents.fhir.save_conversation_memory", AsyncMock()) as save_memory,
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
                "app.agents.fhir._prepare_run",
                AsyncMock(return_value=("conversation-1", [], "No memories")),
            ),
            patch("app.agents.fhir.agent.run", runner),
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
            patch("app.agents.fhir.generate_agent_response", AsyncMock(return_value=generated)) as generate,
            patch("app.agents.fhir.save_conversation_memory", AsyncMock()) as save_memory,
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
