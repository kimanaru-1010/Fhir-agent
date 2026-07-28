"""Tests for conversation management API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from unittest.mock import AsyncMock, MagicMock

from app.db.models import Conversation, Message, User

# Pre-imports for override
from app import database as _db_module
from app.dependencies import auth as _auth_dep


# ===========================================================================
# Helpers
# ===========================================================================


def _build_mock_session() -> tuple[MagicMock, MagicMock]:
    """Build a mock AsyncSession.

    execute / commit / rollback / refresh / delete are async.
    add is sync.
    """
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mock_result = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_result)

    # sync methods
    mock_session.add = MagicMock()

    # async methods
    mock_session.delete = AsyncMock()
    async def _flush():
        added_objects = [call.args[0] for call in mock_session.add.call_args_list]
        for obj in added_objects:
            if getattr(obj, "id", None) is None:
                obj.id = uuid4()

    mock_session.flush = AsyncMock(side_effect=_flush)
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()

    async def _refresh(obj):
        now = datetime.now(timezone.utc)
        if getattr(obj, "id", None) is None:
            obj.id = uuid4()
        if getattr(obj, "created_at", None) is None:
            obj.created_at = now
        if hasattr(obj, "updated_at") and getattr(obj, "updated_at", None) is None:
            obj.updated_at = now

    mock_session.refresh = AsyncMock(side_effect=_refresh)

    return mock_session, mock_result


def _make_test_app(
    *,
    current_user: User | None = None,
    use_real_auth: bool = False,
) -> tuple[FastAPI, MagicMock, MagicMock]:
    """Build test app with auth + conversation routers mounted under /api."""
    from app.api.auth import auth_router, users_router
    from app.api.conversations import router as conversations_router

    app = FastAPI()
    app.include_router(auth_router, prefix="/api")
    app.include_router(users_router, prefix="/api")
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


def _make_conversation(
    id: UUID | None = None,
    user_id: UUID | None = None,
    title: str = "Test conversation",
) -> Conversation:
    """Create a Conversation instance suitable for mocking."""
    return Conversation(
        id=id or uuid4(),
        user_id=user_id or uuid4(),
        title=title,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _make_user(
    id: UUID | None = None,
    username: str = "test_user",
) -> User:
    """Create a User instance suitable for mocking."""
    return User(
        id=id or uuid4(),
        username=username,
        password_hash="hashed",
        external_id="EXT-001",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _configure_user_lookup(
    mock_result: MagicMock,
    user: User,
) -> None:
    """Set up mock_result to return *user* for scalar_one_or_none()."""
    mock_result.scalar_one_or_none.return_value = user


def _configure_execute_calls(
    mock_session: MagicMock,
    results: list,
) -> None:
    """Set execute() to return successive result objects."""
    mock_results = [MagicMock() for _ in results]
    for mr, user in zip(mock_results, results):
        mr.scalars().all.return_value = [user] if user is not None else []
        mr.scalar_one_or_none.return_value = user

    async def _side_effect(*args, **kwargs):
        available = [mr for mr in mock_results if not mr._called]
        if available:
            mr = available[0]
            mr._called = True
            return mr
        return mock_results[-1]

    mock_session.execute.reset_mock()
    mock_session.execute = AsyncMock(side_effect=_side_effect)


# ===========================================================================
# Create tests
# ===========================================================================


def test_create_conversation_success():
    user = _make_user(id=uuid4(), username="create_user")
    app, mock_session, mock_result = _make_test_app(current_user=user)
    client = TestClient(app)
    first_message = "Cho tôi thông tin bệnh nhân Nguyễn Văn A"

    resp = client.post(
        "/api/conversations",
        json={"first_message": first_message},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "id" in data
    assert "user_id" in data
    assert data["title"] == first_message
    assert data["first_message"]["role"] == "user"
    assert data["first_message"]["content"] == first_message
    assert data["first_message"]["conversation_id"] == data["id"]


def test_create_conversation_generates_title_from_first_message():
    user = _make_user()
    app, _, _ = _make_test_app(current_user=user)
    client = TestClient(app)

    first_message = "Cho tôi thông tin bệnh nhân Nguyễn Văn A"

    resp = client.post(
        "/api/conversations",
        json={"first_message": first_message},
    )
    assert resp.status_code == 201
    assert resp.json()["title"] == first_message


def test_create_conversation_normalizes_title_whitespace():
    user = _make_user()
    app, _, _ = _make_test_app(current_user=user)
    client = TestClient(app)

    resp = client.post(
        "/api/conversations",
        json={"first_message": "  Cho tôi   thông tin bệnh nhân A  "},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Cho tôi thông tin bệnh nhân A"
    assert data["first_message"]["content"] == "Cho tôi   thông tin bệnh nhân A"


def test_create_conversation_long_title_truncated():
    user = _make_user()
    app, _, _ = _make_test_app(current_user=user)
    client = TestClient(app)

    first_message = "a" * 80

    resp = client.post(
        "/api/conversations",
        json={"first_message": first_message},
    )
    assert resp.status_code == 201
    title = resp.json()["title"]
    assert len(title) <= 60
    assert title.endswith("...")


def test_create_conversation_uses_current_user_id():
    """user_id comes from JWT, not from request body."""
    user = _make_user(id=uuid4(), username="id_user")
    app, mock_session, mock_result = _make_test_app(current_user=user)
    client = TestClient(app)

    resp = client.post(
        "/api/conversations",
        json={"first_message": "Should be linked to current user"},
    )
    assert resp.status_code == 201

    # Verify the Conversation was created with the current user's id
    call_args = mock_session.add.call_args_list[0]
    conversation_obj = call_args.args[0]
    assert conversation_obj.user_id == user.id


def test_create_conversation_role_not_controlled_by_client():
    user = _make_user()
    app, mock_session, mock_result = _make_test_app(current_user=user)
    client = TestClient(app)

    resp = client.post(
        "/api/conversations",
        json={"first_message": "Hello", "role": "assistant"},
    )
    assert resp.status_code == 422


def test_create_conversation_title_not_controlled_by_client():
    app, _, _ = _make_test_app()
    client = TestClient(app)

    resp = client.post(
        "/api/conversations",
        json={"first_message": "Hello", "title": "Hacked title"},
    )
    assert resp.status_code == 422


def test_create_conversation_empty_first_message():
    app, _, _ = _make_test_app()
    client = TestClient(app)

    resp = client.post("/api/conversations", json={"first_message": ""})
    assert resp.status_code == 422


def test_create_conversation_blank_first_message():
    app, _, _ = _make_test_app()
    client = TestClient(app)

    resp = client.post("/api/conversations", json={"first_message": "   "})
    assert resp.status_code == 422


def test_create_conversation_first_message_too_long():
    app, _, _ = _make_test_app()
    client = TestClient(app)

    resp = client.post("/api/conversations", json={"first_message": "a" * 10_001})
    assert resp.status_code == 422


def test_create_conversation_adds_conversation_and_message():
    user = _make_user()
    app, mock_session, mock_result = _make_test_app(current_user=user)
    client = TestClient(app)
    first_message = "Hello"

    resp = client.post("/api/conversations", json={"first_message": first_message})
    assert resp.status_code == 201

    added_objects = [call.args[0] for call in mock_session.add.call_args_list]
    assert len(added_objects) == 2
    conversation_obj, message_obj = added_objects
    assert isinstance(conversation_obj, Conversation)
    assert isinstance(message_obj, Message)
    assert message_obj.conversation_id == conversation_obj.id
    assert message_obj.role == "user"
    assert message_obj.content == first_message


def test_create_conversation_flushes_before_commit():
    user = _make_user()
    app, mock_session, mock_result = _make_test_app(current_user=user)
    client = TestClient(app)

    resp = client.post("/api/conversations", json={"first_message": "Hello"})
    assert resp.status_code == 201
    mock_session.flush.assert_awaited_once()
    mock_session.commit.assert_awaited_once()


def test_create_conversation_commit_error():
    """Commit fails → rollback → re-raise."""
    user = _make_user()
    app, mock_session, mock_result = _make_test_app(current_user=user)
    client = TestClient(app)

    mock_session.commit = AsyncMock(
        side_effect=IntegrityError(
            statement=None, params=None, orig=Exception("duplicate"),
        ),
    )

    resp = client.post("/api/conversations", json={"first_message": "Should fail"})
    assert resp.status_code == 500
    mock_session.rollback.assert_awaited_once()


def test_create_conversation_flush_error():
    """Flush fails -> rollback -> 500."""
    user = _make_user()
    app, mock_session, mock_result = _make_test_app(current_user=user)
    client = TestClient(app)

    mock_session.flush = AsyncMock(
        side_effect=IntegrityError(
            statement=None, params=None, orig=Exception("constraint"),
        ),
    )

    resp = client.post("/api/conversations", json={"first_message": "Should fail"})
    assert resp.status_code == 500
    mock_session.rollback.assert_awaited_once()


# ===========================================================================
# List tests
# ===========================================================================


def test_list_conversations_success():
    user = _make_user()
    app, mock_session, mock_result = _make_test_app(current_user=user)
    client = TestClient(app)

    now = datetime.now(timezone.utc)
    own_conv = _make_conversation(id=uuid4(), title="My conv", user_id=user.id)
    other_conv = _make_conversation(id=uuid4(), title="Other conv")

    mock_result.scalar_one.return_value = 1
    mock_result.scalars().all.return_value = [own_conv]

    resp = client.get("/api/conversations")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["title"] == "My conv"


def test_list_conversations_only_own():
    """User sees only their own conversations."""
    own_user = _make_user(id=uuid4(), username="own")
    app, mock_session, mock_result = _make_test_app(current_user=own_user)
    client = TestClient(app)

    other_user = _make_user(id=uuid4(), username="other")
    own_conv = _make_conversation(id=uuid4(), user_id=own_user.id)
    other_conv = _make_conversation(id=uuid4(), user_id=other_user.id)

    mock_result.scalar_one.return_value = 1
    mock_result.scalars().all.return_value = [own_conv]

    resp = client.get("/api/conversations")
    assert resp.status_code == 200
    ids = [item["id"] for item in resp.json()["items"]]
    assert str(own_conv.id) in ids
    assert str(other_conv.id) not in ids


def test_list_conversations_empty():
    user = _make_user()
    app, mock_session, mock_result = _make_test_app(current_user=user)
    client = TestClient(app)

    mock_result.scalar_one.return_value = 0
    mock_result.scalars().all.return_value = []

    resp = client.get("/api/conversations")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0
    assert resp.json()["items"] == []


def test_list_conversations_no_token():
    app, _, _ = _make_test_app(use_real_auth=True)
    client = TestClient(app)
    resp = client.get("/api/conversations")
    assert resp.status_code == 401


def test_list_conversations_sorted_by_updated_at_desc():
    user = _make_user()
    app, mock_session, mock_result = _make_test_app(current_user=user)
    client = TestClient(app)

    later = _make_conversation(id=uuid4(), user_id=user.id)
    earlier = _make_conversation(id=uuid4(), user_id=user.id)
    earlier.updated_at = datetime.now(timezone.utc)
    later.updated_at = earlier.updated_at + timedelta(seconds=1)

    # Order in response should be desc
    mock_result.scalar_one.return_value = 1
    mock_result.scalars().all.return_value = [later, earlier]

    resp = client.get("/api/conversations")
    assert resp.status_code == 200
    # Later (first in list) should have greater updated_at
    assert resp.json()["items"][0]["updated_at"] >= resp.json()["items"][1]["updated_at"]


# ===========================================================================
# Get detail tests
# ===========================================================================


def test_get_conversation_owned():
    user = _make_user()
    app, mock_session, mock_result = _make_test_app(current_user=user)
    client = TestClient(app)

    conv = _make_conversation(id=uuid4(), user_id=user.id, title="Detail conv")

    mock_result.scalar_one_or_none.return_value = conv

    resp = client.get(f"/api/conversations/{conv.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == str(conv.id)
    assert data["title"] == "Detail conv"
    assert data["user_id"] == str(user.id)


def test_get_conversation_not_found():
    user = _make_user()
    app, mock_session, mock_result = _make_test_app(current_user=user)
    client = TestClient(app)

    mock_result.scalar_one_or_none.return_value = None  # not found

    fake_id = uuid4()

    resp = client.get(f"/api/conversations/{fake_id}")
    assert resp.status_code == 404


def test_get_conversation_other_user():
    """User B cannot see user A's conversation."""
    user_b = _make_user(id=uuid4(), username="bob")
    app, mock_session, mock_result = _make_test_app(current_user=user_b)
    client = TestClient(app)

    user_a = _make_user(id=uuid4(), username="alice")
    conv = _make_conversation(id=uuid4(), user_id=user_a.id)

    # DB would find the conv, but query also filters by user_b.id → None
    mock_result.scalar_one_or_none.return_value = None

    resp = client.get(f"/api/conversations/{conv.id}")
    assert resp.status_code == 404


def test_get_conversation_invalid_uuid():
    app, _, _ = _make_test_app()
    client = TestClient(app)
    resp = client.get("/api/conversations/not-a-uuid")
    assert resp.status_code == 422


# ===========================================================================
# Delete tests
# ===========================================================================


def test_delete_conversation_success():
    user = _make_user()
    app, mock_session, mock_result = _make_test_app(current_user=user)
    client = TestClient(app)

    conv = _make_conversation(id=uuid4(), user_id=user.id)

    mock_result.scalar_one_or_none.return_value = conv
    mock_session.commit = AsyncMock()

    resp = client.delete(f"/api/conversations/{conv.id}")
    assert resp.status_code == 204
    assert resp.content == b""
    mock_session.delete.assert_awaited_once_with(conv)


def test_delete_conversation_not_found():
    user = _make_user()
    app, mock_session, mock_result = _make_test_app(current_user=user)
    client = TestClient(app)

    mock_result.scalar_one_or_none.return_value = None

    fake_id = uuid4()

    resp = client.delete(f"/api/conversations/{fake_id}")
    assert resp.status_code == 404


def test_delete_conversation_other_user():
    """User B cannot delete user A's conversation."""
    user_b = _make_user(id=uuid4(), username="bob")
    app, mock_session, mock_result = _make_test_app(current_user=user_b)
    client = TestClient(app)

    mock_result.scalar_one_or_none.return_value = None  # filtered by bob

    fake_id = uuid4()

    resp = client.delete(f"/api/conversations/{fake_id}")
    assert resp.status_code == 404


def test_delete_conversation_commit_error():
    user = _make_user()
    app, mock_session, mock_result = _make_test_app(current_user=user)
    client = TestClient(app)

    conv = _make_conversation(id=uuid4(), user_id=user.id)

    mock_result.scalar_one_or_none.return_value = conv
    mock_session.commit = AsyncMock(
        side_effect=IntegrityError(
            statement=None, params=None, orig=Exception("fail"),
        ),
    )

    resp = client.delete(f"/api/conversations/{conv.id}")
    assert resp.status_code == 500
    mock_session.rollback.assert_awaited_once()


# ===========================================================================
# Security
# ===========================================================================


def test_security_no_user_id_in_request():
    """No endpoint accepts user_id from client to override ownership."""
    from app.schemas.conversation import ConversationCreateRequest

    # Check request schema doesn't expose user_id
    create_fields = ConversationCreateRequest.model_fields
    assert "user_id" not in create_fields


def test_security_user_a_cannot_see_user_b_conversation():
    """User A queries with user B's conversation ID → 404."""
    user_a = _make_user(id=uuid4(), username="alice")
    app, mock_session, mock_result = _make_test_app(current_user=user_a)
    client = TestClient(app)

    user_b = _make_user(id=uuid4(), username="bob")
    conv = _make_conversation(id=uuid4(), user_id=user_b.id)

    mock_result.scalar_one_or_none.return_value = None  # alice can't see bob's

    resp = client.get(f"/api/conversations/{conv.id}")
    assert resp.status_code == 404


def test_security_user_a_cannot_delete_user_b_conversation():
    user_a = _make_user(id=uuid4(), username="alice")
    app, mock_session, mock_result = _make_test_app(current_user=user_a)
    client = TestClient(app)

    user_b = _make_user(id=uuid4(), username="bob")
    conv = _make_conversation(id=uuid4(), user_id=user_b.id)

    mock_result.scalar_one_or_none.return_value = None

    resp = client.delete(f"/api/conversations/{conv.id}")
    assert resp.status_code == 404
