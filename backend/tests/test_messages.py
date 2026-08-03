"""Tests for conversation message API."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import anyio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from app.db import session as _db_module
from app.dependencies import auth as _auth_dep
from app.db.models import Conversation, Message, User


def _build_mock_session() -> tuple[MagicMock, MagicMock]:
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()

    async def _flush():
        for call in mock_session.add.call_args_list:
            obj = call.args[0]
            if getattr(obj, "id", None) is None:
                obj.id = uuid4()

    async def _refresh(obj):
        if getattr(obj, "id", None) is None:
            obj.id = uuid4()
        if getattr(obj, "created_at", None) is None:
            obj.created_at = datetime.now(timezone.utc)
        if hasattr(obj, "updated_at") and getattr(obj, "updated_at", None) is None:
            obj.updated_at = datetime.now(timezone.utc)

    mock_session.flush = AsyncMock(side_effect=_flush)
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.refresh = AsyncMock(side_effect=_refresh)
    mock_session.delete = AsyncMock()
    return mock_session, mock_result


def _make_test_app(
    *,
    current_user: User | None = None,
    use_real_auth: bool = False,
) -> tuple[FastAPI, MagicMock, MagicMock]:
    from app.api.messages import router as messages_router

    app = FastAPI()
    app.include_router(messages_router, prefix="/api")

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


def _make_user(id: UUID | None = None, username: str = "test_user") -> User:
    return User(
        id=id or uuid4(),
        username=username,
        password_hash="hashed",
        external_id="EXT-001",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _make_conversation(
    id: UUID | None = None,
    user_id: UUID | None = None,
    title: str = "Test conversation",
) -> Conversation:
    return Conversation(
        id=id or uuid4(),
        user_id=user_id or uuid4(),
        title=title,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _make_message(
    *,
    id: UUID | None = None,
    conversation_id: UUID | None = None,
    role: str = "user",
    content: str = "Hello",
    created_at: datetime | None = None,
) -> Message:
    return Message(
        id=id or uuid4(),
        conversation_id=conversation_id or uuid4(),
        role=role,
        content=content,
        created_at=created_at or datetime.now(timezone.utc),
    )


def _result(
    *,
    scalar_one_or_none=None,
    scalar_one=None,
    all_items: list | None = None,
) -> MagicMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar_one_or_none
    result.scalar_one.return_value = scalar_one
    result.scalars.return_value.all.return_value = all_items or []
    return result


def _configure_execute_results(mock_session: MagicMock, results: list[MagicMock]) -> None:
    mock_session.execute = AsyncMock(side_effect=results)


class _SessionFactory:
    def __init__(self, session: MagicMock):
        self.session = session

    def __call__(self):
        return self

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return None


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in text.strip().split("\n\n"):
        event_name = ""
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event_name and data is not None:
            events.append((event_name, data))
    return events


def test_list_messages_success():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    first = _make_message(conversation_id=conv.id, role="user", content="First")
    second = _make_message(conversation_id=conv.id, role="assistant", content="Second")
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(
        mock_session,
        [
            _result(scalar_one_or_none=conv),
            _result(scalar_one=2),
            _result(all_items=[first, second]),
        ],
    )

    resp = TestClient(app).get(f"/api/conversations/{conv.id}/messages")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert [item["content"] for item in data["items"]] == ["First", "Second"]


def test_list_messages_orders_by_created_at_then_id():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(
        mock_session,
        [
            _result(scalar_one_or_none=conv),
            _result(scalar_one=0),
            _result(all_items=[]),
        ],
    )

    resp = TestClient(app).get(f"/api/conversations/{conv.id}/messages")

    assert resp.status_code == 200
    list_stmt = mock_session.execute.call_args_list[2].args[0]
    rendered = str(list_stmt)
    assert "ORDER BY messages.created_at ASC, messages.id ASC" in rendered


def test_list_messages_empty_total():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(
        mock_session,
        [
            _result(scalar_one_or_none=conv),
            _result(scalar_one=0),
            _result(all_items=[]),
        ],
    )

    resp = TestClient(app).get(f"/api/conversations/{conv.id}/messages")

    assert resp.status_code == 200
    assert resp.json() == {"items": [], "total": 0}


def test_list_messages_pagination_validation():
    app, _, _ = _make_test_app()
    client = TestClient(app)
    conv_id = uuid4()

    assert client.get(f"/api/conversations/{conv_id}/messages?skip=-1").status_code == 422
    assert client.get(f"/api/conversations/{conv_id}/messages?limit=0").status_code == 422
    assert client.get(f"/api/conversations/{conv_id}/messages?limit=201").status_code == 422


def test_list_messages_conversation_not_found():
    user = _make_user()
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=None)])

    resp = TestClient(app).get(f"/api/conversations/{uuid4()}/messages")

    assert resp.status_code == 404


def test_list_messages_other_user_returns_404():
    user = _make_user()
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=None)])

    resp = TestClient(app).get(f"/api/conversations/{uuid4()}/messages")

    assert resp.status_code == 404


def test_list_messages_no_token():
    app, _, _ = _make_test_app(use_real_auth=True)

    resp = TestClient(app).get(f"/api/conversations/{uuid4()}/messages")

    assert resp.status_code == 401


def test_list_messages_invalid_uuid():
    app, _, _ = _make_test_app()

    resp = TestClient(app).get("/api/conversations/not-a-uuid/messages")

    assert resp.status_code == 422


def test_create_message_success():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=conv)])
    agent = AsyncMock(return_value="Assistant answer")
    memory = AsyncMock()

    with (
        patch("app.api.messages.generate_assistant_response", agent),
        patch("app.api.messages.persist_chat_memory", memory),
    ):
        resp = TestClient(app).post(
            f"/api/conversations/{conv.id}/messages",
            json={"content": "  User question  "},
        )

    assert resp.status_code == 201
    data = resp.json()
    assert data["conversation_id"] == str(conv.id)
    assert data["user_message"]["role"] == "user"
    assert data["user_message"]["content"] == "User question"
    assert data["assistant_message"]["role"] == "assistant"
    assert data["assistant_message"]["content"] == "Assistant answer"
    assert data["user_message"]["conversation_id"] == str(conv.id)
    assert data["assistant_message"]["conversation_id"] == str(conv.id)
    agent.assert_awaited_once_with(
        content="User question",
        user_id=str(user.id),
        conversation_id=str(conv.id),
        current_user_message_id=ANY,
    )
    assert mock_session.commit.await_count == 2
    memory.assert_awaited_once_with(
        user_id=str(user.id),
        conversation_id=str(conv.id),
        user_message="User question",
        assistant_message="Assistant answer",
    )
    assert conv.updated_at is not None

    added = [call.args[0] for call in mock_session.add.call_args_list]
    assert len(added) == 2
    user_message, assistant_message = added
    assert isinstance(user_message, Message)
    assert isinstance(assistant_message, Message)
    assert user_message.role == "user"
    assert assistant_message.role == "assistant"
    assert user_message.conversation_id == conv.id
    assert assistant_message.conversation_id == conv.id


def test_create_message_validation_and_forbidden_fields():
    app, _, _ = _make_test_app()
    client = TestClient(app)
    conv_id = uuid4()

    assert client.post(f"/api/conversations/{conv_id}/messages", json={"content": ""}).status_code == 422
    assert client.post(f"/api/conversations/{conv_id}/messages", json={"content": "   "}).status_code == 422
    assert client.post(f"/api/conversations/{conv_id}/messages", json={"content": "a" * 10_001}).status_code == 422
    assert client.post(f"/api/conversations/{conv_id}/messages", json={"content": "Hi", "role": "assistant"}).status_code == 422
    assert client.post(f"/api/conversations/{conv_id}/messages", json={"content": "Hi", "conversation_id": str(uuid4())}).status_code == 422
    assert client.post(f"/api/conversations/{conv_id}/messages", json={"content": "Hi", "user_id": str(uuid4())}).status_code == 422


def test_create_message_other_user_returns_404_and_does_not_call_agent():
    user = _make_user()
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=None)])
    agent = AsyncMock(return_value="Assistant answer")
    memory = AsyncMock()

    with (
        patch("app.api.messages.generate_assistant_response", agent),
        patch("app.api.messages.persist_chat_memory", memory),
    ):
        resp = TestClient(app).post(
            f"/api/conversations/{uuid4()}/messages",
            json={"content": "Hello"},
        )

    assert resp.status_code == 404
    agent.assert_not_awaited()
    memory.assert_not_awaited()


def test_create_message_agent_error_rolls_back():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=conv)])
    agent = AsyncMock(side_effect=RuntimeError("agent exploded"))
    memory = AsyncMock()

    with (
        patch("app.api.messages.generate_assistant_response", agent),
        patch("app.api.messages.persist_chat_memory", memory),
    ):
        resp = TestClient(app).post(
            f"/api/conversations/{conv.id}/messages",
            json={"content": "Hello"},
        )

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Unable to process message"
    mock_session.rollback.assert_awaited_once()
    assert mock_session.commit.await_count == 1
    memory.assert_not_awaited()
    added = [call.args[0] for call in mock_session.add.call_args_list]
    assert len(added) == 1
    assert added[0].role == "user"


def test_create_message_empty_agent_response_rolls_back():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=conv)])

    memory = AsyncMock()
    with (
        patch("app.api.messages.generate_assistant_response", AsyncMock(return_value="")),
        patch("app.api.messages.persist_chat_memory", memory),
    ):
        resp = TestClient(app).post(
            f"/api/conversations/{conv.id}/messages",
            json={"content": "Hello"},
        )

    assert resp.status_code == 500
    mock_session.rollback.assert_awaited_once()
    assert mock_session.commit.await_count == 1
    memory.assert_not_awaited()


def test_create_message_none_agent_response_rolls_back():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=conv)])

    memory = AsyncMock()
    with (
        patch("app.api.messages.generate_assistant_response", AsyncMock(return_value=None)),
        patch("app.api.messages.persist_chat_memory", memory),
    ):
        resp = TestClient(app).post(
            f"/api/conversations/{conv.id}/messages",
            json={"content": "Hello"},
        )

    assert resp.status_code == 500
    mock_session.rollback.assert_awaited_once()
    assert mock_session.commit.await_count == 1
    memory.assert_not_awaited()


def test_create_message_user_commit_error_rolls_back_before_agent():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=conv)])
    mock_session.commit = AsyncMock(
        side_effect=IntegrityError(
            statement=None,
            params=None,
            orig=Exception("constraint"),
        )
    )

    memory = AsyncMock()
    with (
        patch("app.api.messages.generate_assistant_response", AsyncMock(return_value="Answer")),
        patch("app.api.messages.persist_chat_memory", memory),
    ):
        resp = TestClient(app).post(
            f"/api/conversations/{conv.id}/messages",
            json={"content": "Hello"},
        )

    assert resp.status_code == 500
    mock_session.rollback.assert_awaited_once()
    memory.assert_not_awaited()


def test_create_message_commit_error_rolls_back():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=conv)])
    mock_session.commit = AsyncMock(
        side_effect=IntegrityError(
            statement=None,
            params=None,
            orig=Exception("commit"),
        )
    )

    memory = AsyncMock()
    with (
        patch("app.api.messages.generate_assistant_response", AsyncMock(return_value="Answer")),
        patch("app.api.messages.persist_chat_memory", memory),
    ):
        resp = TestClient(app).post(
            f"/api/conversations/{conv.id}/messages",
            json={"content": "Hello"},
        )

    assert resp.status_code == 500
    assert mock_session.rollback.await_count >= 1
    memory.assert_not_awaited()


def test_create_message_memory_error_after_commit_still_returns_201():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=conv)])
    memory = AsyncMock(side_effect=RuntimeError("mem0 failed"))

    with (
        patch("app.api.messages.generate_assistant_response", AsyncMock(return_value="Answer")),
        patch("app.api.messages.persist_chat_memory", memory),
    ):
        resp = TestClient(app).post(
            f"/api/conversations/{conv.id}/messages",
            json={"content": "Hello"},
        )

    assert resp.status_code == 201
    assert resp.json()["assistant_message"]["content"] == "Answer"
    mock_session.rollback.assert_not_awaited()
    assert mock_session.commit.await_count == 2
    assert "mem0 failed" not in resp.text


def test_stream_message_success_forwards_tool_events_and_persists_messages():
    from app.graph.client import get_collector

    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, user_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(user_session, [_result(scalar_one_or_none=conv)])
    assistant_session, _ = _build_mock_session()
    _configure_execute_results(assistant_session, [_result(scalar_one_or_none=conv)])

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
            f"/api/conversations/{conv.id}/messages/stream",
            json={"content": "  User question  "},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)
    assert [name for name, _ in events] == [
        "message_started",
        "tool_start",
        "tool_end",
        "text_delta",
        "done",
    ]

    started = events[0][1]
    assert started["conversation_id"] == str(conv.id)
    assert started["user_message"]["id"]
    assert started["user_message"]["role"] == "user"
    assert started["user_message"]["content"] == "User question"

    assert events[1][1] == {
        "name": "search_patient",
        "inputs": {"query": "Nguyen Van A"},
    }
    tool_end = events[2][1]
    assert tool_end["name"] == "search_patient"
    assert tool_end["output_preview"] == "1 patient found"
    assert tool_end["graph_data"] == {"results": [{"patient": "A"}]}

    assert events[3][1]["text"] == "Assistant streamed answer"
    done = events[4][1]
    assert done["conversation"]["id"] == str(conv.id)
    assert done["user_message"] == started["user_message"]
    assert done["assistant_message"]["role"] == "assistant"
    assert done["assistant_message"]["content"] == "Assistant streamed answer"
    assert done["response"] == done["assistant_message"]["content"]

    agent.assert_awaited_once_with(
        content="User question",
        user_id=str(user.id),
        conversation_id=str(conv.id),
        current_user_message_id=ANY,
    )
    memory.assert_awaited_once_with(
        user_id=str(user.id),
        conversation_id=str(conv.id),
        user_message="User question",
        assistant_message="Assistant streamed answer",
    )
    user_session.commit.assert_awaited_once()
    assistant_session.commit.assert_awaited_once()
    assert [call.args[0].role for call in user_session.add.call_args_list] == ["user"]
    assert [call.args[0].role for call in assistant_session.add.call_args_list] == ["assistant"]


def test_stream_message_ignores_collector_text_delta_and_done_duplicates():
    from app.graph.client import get_collector

    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, user_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(user_session, [_result(scalar_one_or_none=conv)])
    assistant_session, _ = _build_mock_session()
    _configure_execute_results(assistant_session, [_result(scalar_one_or_none=conv)])

    async def _agent(
        *,
        content: str,
        user_id: str,
        conversation_id: str,
        current_user_message_id=None,
    ) -> str:
        collector = get_collector()
        collector.emit_tool_start("search_patient", {"query": "A"})
        collector.collect_tool_call("search_patient", {"query": "A"}, "done")
        collector.emit_text_delta("old delta")
        collector.emit_done("old done", conversation_id)
        return "Final answer"

    with (
        patch("app.services.chat_stream.generate_assistant_response", AsyncMock(side_effect=_agent)),
        patch("app.services.chat_stream.persist_chat_memory", AsyncMock()),
        patch("app.services.chat_stream.AsyncSessionFactory", _SessionFactory(assistant_session)),
    ):
        resp = TestClient(app).post(
            f"/api/conversations/{conv.id}/messages/stream",
            json={"content": "Hello"},
        )

    events = _parse_sse(resp.text)
    assert [name for name, _ in events].count("text_delta") == 1
    assert [name for name, _ in events].count("done") == 1
    assert events[-2] == ("text_delta", {"text": "Final answer", "delta": "Final answer"})
    assert events[-1][0] == "done"
    assert events[-1][1]["response"] == "Final answer"


def test_stream_message_ownership_failure_does_not_call_agent():
    user = _make_user()
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=None)])
    agent = AsyncMock(return_value="Answer")

    with patch("app.services.chat_stream.generate_assistant_response", agent):
        resp = TestClient(app).post(
            f"/api/conversations/{uuid4()}/messages/stream",
            json={"content": "Hello"},
        )

    assert resp.status_code == 404
    agent.assert_not_awaited()


def test_stream_message_no_token_and_invalid_uuid():
    app, _, _ = _make_test_app(use_real_auth=True)
    client = TestClient(app)
    assert client.post(
        f"/api/conversations/{uuid4()}/messages/stream",
        json={"content": "Hello"},
    ).status_code == 401

    app, _, _ = _make_test_app()
    assert TestClient(app).post(
        "/api/conversations/not-a-uuid/messages/stream",
        json={"content": "Hello"},
    ).status_code == 422


def test_stream_message_user_commit_error_returns_500_before_agent():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=conv)])
    mock_session.commit = AsyncMock(
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
            f"/api/conversations/{conv.id}/messages/stream",
            json={"content": "Hello"},
        )

    assert resp.status_code == 500
    mock_session.rollback.assert_awaited_once()
    agent.assert_not_awaited()
    memory.assert_not_awaited()


def test_stream_message_agent_error_keeps_user_message_and_emits_error():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, user_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(user_session, [_result(scalar_one_or_none=conv)])
    memory = AsyncMock()

    with (
        patch(
            "app.services.chat_stream.generate_assistant_response",
            AsyncMock(side_effect=RuntimeError("agent failed")),
        ),
        patch("app.services.chat_stream.persist_chat_memory", memory),
    ):
        resp = TestClient(app).post(
            f"/api/conversations/{conv.id}/messages/stream",
            json={"content": "Hello"},
        )

    events = _parse_sse(resp.text)
    assert [name for name, _ in events] == ["message_started", "error"]
    assert events[-1][1] == {"detail": "Unable to process message"}
    assert "done" not in [name for name, _ in events]
    assert user_session.commit.await_count == 1
    memory.assert_not_awaited()


def test_stream_message_assistant_commit_error_emits_error_without_memory():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, user_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(user_session, [_result(scalar_one_or_none=conv)])
    assistant_session, _ = _build_mock_session()
    _configure_execute_results(assistant_session, [_result(scalar_one_or_none=conv)])
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
            f"/api/conversations/{conv.id}/messages/stream",
            json={"content": "Hello"},
        )

    events = _parse_sse(resp.text)
    assert [name for name, _ in events] == ["message_started", "error"]
    assert "done" not in [name for name, _ in events]
    assistant_session.rollback.assert_awaited_once()
    memory.assert_not_awaited()


def test_stream_message_memory_error_still_emits_done():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, user_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(user_session, [_result(scalar_one_or_none=conv)])
    assistant_session, _ = _build_mock_session()
    _configure_execute_results(assistant_session, [_result(scalar_one_or_none=conv)])

    with (
        patch("app.services.chat_stream.generate_assistant_response", AsyncMock(return_value="Answer")),
        patch(
            "app.services.chat_stream.persist_chat_memory",
            AsyncMock(side_effect=RuntimeError("mem0 failed")),
        ),
        patch("app.services.chat_stream.AsyncSessionFactory", _SessionFactory(assistant_session)),
    ):
        resp = TestClient(app).post(
            f"/api/conversations/{conv.id}/messages/stream",
            json={"content": "Hello"},
        )

    events = _parse_sse(resp.text)
    assert events[-2][0] == "text_delta"
    assert events[-1][0] == "done"
    assert "error" not in [name for name, _ in events]
    user_session.rollback.assert_not_awaited()
    assistant_session.rollback.assert_not_awaited()


def test_collector_event_queue_is_context_local_for_concurrent_streams():
    from app.graph.client import get_collector

    async def run_test():
        collector = get_collector()

        async def run_one(label: str):
            event_queue: asyncio.Queue = asyncio.Queue()
            token = collector.set_event_queue(event_queue)
            try:
                collector.emit_tool_start("tool", {"stream": label})
                return await event_queue.get()
            finally:
                collector.clear_event_queue(token)

        async with anyio.create_task_group() as tg:
            results: dict[str, dict] = {}

            async def capture(label: str):
                results[label] = await run_one(label)

            tg.start_soon(capture, "A")
            tg.start_soon(capture, "B")

        assert results["A"]["data"]["inputs"] == {"stream": "A"}
        assert results["B"]["data"]["inputs"] == {"stream": "B"}

    anyio.run(run_test)


def test_message_routes_registered_without_patch_or_delete():
    from app.main import app

    message_routes = {
        (path, method)
        for path, methods in app.openapi()["paths"].items()
        if path in {
            "/api/conversations/{conversation_id}/messages",
            "/api/conversations/{conversation_id}/messages/stream",
        }
        for method in methods
    }

    assert ("/api/conversations/{conversation_id}/messages", "get") in message_routes
    assert ("/api/conversations/{conversation_id}/messages", "post") in message_routes
    assert ("/api/conversations/{conversation_id}/messages/stream", "post") in message_routes
    assert ("/api/conversations/{conversation_id}/messages", "patch") not in message_routes
    assert ("/api/conversations/{conversation_id}/messages", "delete") not in message_routes
    assert ("/api/conversations/{conversation_id}/messages/stream", "patch") not in message_routes
    assert ("/api/conversations/{conversation_id}/messages/stream", "delete") not in message_routes
