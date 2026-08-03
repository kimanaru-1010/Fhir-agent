"""Tests for conversation-aware SSE creation."""

from __future__ import annotations

import json
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from app.db import session as _db_module
from app.db.models import Conversation, User
from app.dependencies import auth as _auth_dep
from tests.test_messages import (
    _SessionFactory,
    _build_mock_session,
    _make_user,
    _result,
)


def _make_test_app(
    *,
    current_user: User | None = None,
    use_real_auth: bool = False,
) -> tuple[FastAPI, MagicMock, MagicMock]:
    from app.api.conversations import router as conversations_router

    app = FastAPI()
    app.include_router(conversations_router, prefix="/api")

    mock_session, mock_result = _build_mock_session()

    async def _override_get_db():
        yield mock_session

    app.dependency_overrides[_db_module.get_db] = _override_get_db

    if not use_real_auth:
        resolved_user = current_user or _make_user()

        async def _override_current_user():
            return resolved_user

        app.dependency_overrides[_auth_dep.get_current_user] = _override_current_user

    return app, mock_session, mock_result


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in text.strip().split("\n\n"):
        event_name = ""
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event_name and data is not None:
            events.append((event_name, data))
    return events


def _created_conversation(session: MagicMock) -> Conversation:
    for call in session.add.call_args_list:
        obj = call.args[0]
        if isinstance(obj, Conversation):
            return obj
    raise AssertionError("Conversation was not added")


def test_create_conversation_stream_success_starts_conversation_and_persists_exchange():
    from app.graph.client import get_collector

    user = _make_user()
    app, user_session, _ = _make_test_app(current_user=user)
    assistant_session, _ = _build_mock_session()

    async def _lookup_created_conversation(*args, **kwargs):
        return _result(scalar_one_or_none=_created_conversation(user_session))

    assistant_session.execute = AsyncMock(side_effect=_lookup_created_conversation)

    async def _agent(
        *,
        content: str,
        user_id: str,
        conversation_id: str,
        current_user_message_id=None,
    ) -> str:
        collector = get_collector()
        collector.emit_tool_start("search_patient", {"query": "Nguyen Van A"})
        collector.collect([{"patient": "A"}])
        collector.collect_tool_call(
            "search_patient",
            {"query": "Nguyen Van A"},
            "1 patient found",
        )
        return "Assistant streamed answer"

    memory = AsyncMock()
    with (
        patch("app.services.chat_stream.generate_assistant_response", AsyncMock(side_effect=_agent)) as agent,
        patch("app.services.chat_stream.persist_chat_memory", memory),
        patch("app.services.chat_stream.AsyncSessionFactory", _SessionFactory(assistant_session)),
    ):
        resp = TestClient(app).post(
            "/api/conversations/stream",
            json={"first_message": "  First question  "},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)
    assert [name for name, _ in events] == [
        "conversation_started",
        "tool_start",
        "tool_end",
        "text_delta",
        "done",
    ]

    started = events[0][1]
    assert started["conversation"]["id"]
    assert started["conversation"]["title"] == "First question"
    assert started["user_message"]["id"]
    assert started["user_message"]["conversation_id"] == started["conversation"]["id"]
    assert started["user_message"]["role"] == "user"
    assert started["user_message"]["content"] == "First question"

    assert events[1][1] == {
        "name": "search_patient",
        "inputs": {"query": "Nguyen Van A"},
    }
    assert events[2][1]["output_preview"] == "1 patient found"
    assert events[2][1]["graph_data"] == {"results": [{"patient": "A"}]}
    assert events[3][1] == {
        "text": "Assistant streamed answer",
        "delta": "Assistant streamed answer",
    }

    done = events[4][1]
    assert done["conversation"]["id"] == started["conversation"]["id"]
    assert done["user_message"] == started["user_message"]
    assert done["assistant_message"]["role"] == "assistant"
    assert done["assistant_message"]["conversation_id"] == started["conversation"]["id"]
    assert done["assistant_message"]["content"] == "Assistant streamed answer"
    assert done["response"] == "Assistant streamed answer"

    agent.assert_awaited_once_with(
        content="First question",
        user_id=str(user.id),
        conversation_id=started["conversation"]["id"],
        current_user_message_id=ANY,
    )
    memory.assert_awaited_once_with(
        user_id=str(user.id),
        conversation_id=started["conversation"]["id"],
        user_message="First question",
        assistant_message="Assistant streamed answer",
    )
    user_session.commit.assert_awaited_once()
    assistant_session.commit.assert_awaited_once()
    assert [call.args[0].__class__.__name__ for call in user_session.add.call_args_list] == [
        "Conversation",
        "Message",
    ]
    assert [call.args[0].role for call in assistant_session.add.call_args_list] == ["assistant"]


def test_create_conversation_stream_no_token_and_extra_field_validation():
    app, _, _ = _make_test_app(use_real_auth=True)
    client = TestClient(app)
    assert client.post(
        "/api/conversations/stream",
        json={"first_message": "Hello"},
    ).status_code == 401

    app, _, _ = _make_test_app()
    assert TestClient(app).post(
        "/api/conversations/stream",
        json={"first_message": "Hello", "title": "Client title"},
    ).status_code == 422


def test_create_conversation_stream_user_commit_error_returns_500_before_agent():
    user = _make_user()
    app, user_session, _ = _make_test_app(current_user=user)
    user_session.commit = AsyncMock(
        side_effect=IntegrityError(
            statement=None,
            params=None,
            orig=Exception("user commit"),
        )
    )
    agent = AsyncMock(return_value="Answer")
    memory = AsyncMock()

    with (
        patch("app.services.chat_stream.generate_assistant_response", agent),
        patch("app.services.chat_stream.persist_chat_memory", memory),
    ):
        resp = TestClient(app).post(
            "/api/conversations/stream",
            json={"first_message": "Hello"},
        )

    assert resp.status_code == 500
    user_session.rollback.assert_awaited_once()
    agent.assert_not_awaited()
    memory.assert_not_awaited()


def test_create_conversation_stream_agent_error_keeps_user_message_and_emits_error():
    user = _make_user()
    app, user_session, _ = _make_test_app(current_user=user)
    memory = AsyncMock()

    with (
        patch(
            "app.services.chat_stream.generate_assistant_response",
            AsyncMock(side_effect=RuntimeError("agent failed")),
        ),
        patch("app.services.chat_stream.persist_chat_memory", memory),
    ):
        resp = TestClient(app).post(
            "/api/conversations/stream",
            json={"first_message": "Hello"},
        )

    events = _parse_sse(resp.text)
    assert [name for name, _ in events] == ["conversation_started", "error"]
    assert events[-1][1] == {"detail": "Unable to process message"}
    assert user_session.commit.await_count == 1
    memory.assert_not_awaited()


def test_create_conversation_stream_assistant_commit_error_emits_error_without_memory():
    user = _make_user()
    app, user_session, _ = _make_test_app(current_user=user)
    assistant_session, _ = _build_mock_session()

    async def _lookup_created_conversation(*args, **kwargs):
        return _result(scalar_one_or_none=_created_conversation(user_session))

    assistant_session.execute = AsyncMock(side_effect=_lookup_created_conversation)
    assistant_session.commit = AsyncMock(
        side_effect=IntegrityError(
            statement=None,
            params=None,
            orig=Exception("assistant commit"),
        )
    )
    memory = AsyncMock()

    with (
        patch("app.services.chat_stream.generate_assistant_response", AsyncMock(return_value="Answer")),
        patch("app.services.chat_stream.persist_chat_memory", memory),
        patch("app.services.chat_stream.AsyncSessionFactory", _SessionFactory(assistant_session)),
    ):
        resp = TestClient(app).post(
            "/api/conversations/stream",
            json={"first_message": "Hello"},
        )

    events = _parse_sse(resp.text)
    assert [name for name, _ in events] == ["conversation_started", "error"]
    assistant_session.rollback.assert_awaited_once()
    memory.assert_not_awaited()


def test_create_conversation_stream_memory_error_still_emits_done():
    user = _make_user()
    app, user_session, _ = _make_test_app(current_user=user)
    assistant_session, _ = _build_mock_session()

    async def _lookup_created_conversation(*args, **kwargs):
        return _result(scalar_one_or_none=_created_conversation(user_session))

    assistant_session.execute = AsyncMock(side_effect=_lookup_created_conversation)

    with (
        patch("app.services.chat_stream.generate_assistant_response", AsyncMock(return_value="Answer")),
        patch(
            "app.services.chat_stream.persist_chat_memory",
            AsyncMock(side_effect=RuntimeError("mem0 failed")),
        ),
        patch("app.services.chat_stream.AsyncSessionFactory", _SessionFactory(assistant_session)),
    ):
        resp = TestClient(app).post(
            "/api/conversations/stream",
            json={"first_message": "Hello"},
        )

    events = _parse_sse(resp.text)
    assert events[-2] == ("text_delta", {"text": "Answer", "delta": "Answer"})
    assert events[-1][0] == "done"
    assert "error" not in [name for name, _ in events]
    user_session.rollback.assert_not_awaited()
    assistant_session.rollback.assert_not_awaited()


def test_conversation_stream_route_registered_without_hiding_static_path():
    from app.main import app

    routes = {
        (path, method)
        for path, methods in app.openapi()["paths"].items()
        if path in {"/api/conversations/stream", "/api/conversations/{conversation_id}"}
        for method in methods
    }

    assert ("/api/conversations/stream", "post") in routes
    assert ("/api/conversations/{conversation_id}", "get") in routes
    assert ("/api/conversations/{conversation_id}", "delete") in routes
