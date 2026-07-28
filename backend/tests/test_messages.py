"""Tests for conversation message API."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

from app import database as _db_module
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

    with patch("app.api.messages.run_agent_for_message", agent):
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
    )
    mock_session.flush.assert_awaited_once()
    mock_session.commit.assert_awaited_once()
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

    with patch("app.api.messages.run_agent_for_message", agent):
        resp = TestClient(app).post(
            f"/api/conversations/{uuid4()}/messages",
            json={"content": "Hello"},
        )

    assert resp.status_code == 404
    agent.assert_not_awaited()


def test_create_message_agent_error_rolls_back():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=conv)])
    agent = AsyncMock(side_effect=RuntimeError("agent exploded"))

    with patch("app.api.messages.run_agent_for_message", agent):
        resp = TestClient(app).post(
            f"/api/conversations/{conv.id}/messages",
            json={"content": "Hello"},
        )

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Unable to process message"
    mock_session.rollback.assert_awaited_once()
    mock_session.commit.assert_not_awaited()
    added = [call.args[0] for call in mock_session.add.call_args_list]
    assert len(added) == 1
    assert added[0].role == "user"


def test_create_message_empty_agent_response_rolls_back():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=conv)])

    with patch("app.api.messages.run_agent_for_message", AsyncMock(return_value="")):
        resp = TestClient(app).post(
            f"/api/conversations/{conv.id}/messages",
            json={"content": "Hello"},
        )

    assert resp.status_code == 500
    mock_session.rollback.assert_awaited_once()
    mock_session.commit.assert_not_awaited()


def test_create_message_none_agent_response_rolls_back():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=conv)])

    with patch("app.api.messages.run_agent_for_message", AsyncMock(return_value=None)):
        resp = TestClient(app).post(
            f"/api/conversations/{conv.id}/messages",
            json={"content": "Hello"},
        )

    assert resp.status_code == 500
    mock_session.rollback.assert_awaited_once()
    mock_session.commit.assert_not_awaited()


def test_create_message_flush_error_rolls_back():
    user = _make_user()
    conv = _make_conversation(user_id=user.id)
    app, mock_session, _ = _make_test_app(current_user=user)
    _configure_execute_results(mock_session, [_result(scalar_one_or_none=conv)])
    mock_session.flush = AsyncMock(
        side_effect=IntegrityError(
            statement=None,
            params=None,
            orig=Exception("constraint"),
        )
    )

    with patch("app.api.messages.run_agent_for_message", AsyncMock(return_value="Answer")):
        resp = TestClient(app).post(
            f"/api/conversations/{conv.id}/messages",
            json={"content": "Hello"},
        )

    assert resp.status_code == 500
    mock_session.rollback.assert_awaited_once()


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

    with patch("app.api.messages.run_agent_for_message", AsyncMock(return_value="Answer")):
        resp = TestClient(app).post(
            f"/api/conversations/{conv.id}/messages",
            json={"content": "Hello"},
        )

    assert resp.status_code == 500
    mock_session.rollback.assert_awaited_once()


def test_message_routes_registered_without_patch_or_delete():
    from app.main import app

    message_routes = {
        (path, method)
        for path, methods in app.openapi()["paths"].items()
        if path == "/api/conversations/{conversation_id}/messages"
        for method in methods
    }

    assert ("/api/conversations/{conversation_id}/messages", "get") in message_routes
    assert ("/api/conversations/{conversation_id}/messages", "post") in message_routes
    assert ("/api/conversations/{conversation_id}/messages", "patch") not in message_routes
    assert ("/api/conversations/{conversation_id}/messages", "delete") not in message_routes
