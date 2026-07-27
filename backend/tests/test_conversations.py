"""Tests for conversation management API."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pytest_mock import MockerFixture
from sqlalchemy.exc import IntegrityError
from unittest.mock import AsyncMock, MagicMock

from app.db.models import Conversation, User
from app.services.auth import create_access_token

# Pre-imports for override
from app import database as _db_module
from app.dependencies import auth as _auth_dep


# ===========================================================================
# Helpers
# ===========================================================================


def _build_mock_session() -> tuple[MagicMock, MagicMock]:
    """Build a mock AsyncSession.

    execute / commit / rollback / refresh are async.
    add / delete are sync.
    """
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mock_result = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_result)

    # sync methods
    mock_session.add = MagicMock()
    mock_session.delete = MagicMock()

    # async methods
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.refresh = AsyncMock()

    return mock_session, mock_result


def _make_test_app(
    *,
    override_current_user: bool = True,
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

    if override_current_user:
        async def _default_current_user():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        app.dependency_overrides[_auth_dep.get_current_user] = (
            _default_current_user
        )

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


def test_create_conversation_success(mocker: MockerFixture):
    app, mock_session, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    user = _make_user(id=uuid4(), username="create_user")
    mock_result.scalar_one_or_none.return_value = user

    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    resp = client.post("/api/conversations", json={
        "title": "Patient A review",
    }, headers={"Authorization": f"Bearer fake"})
    # Will hit override that returns user, so 201
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Patient A review"
    assert "id" in data
    assert "user_id" in data


def test_create_conversation_uses_current_user_id(mocker: MockerFixture):
    """user_id comes from JWT, not from request body."""
    app, mock_session, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    user = _make_user(id=uuid4(), username="id_user")
    mock_result.scalar_one_or_none.return_value = user

    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    resp = client.post("/api/conversations", json={
        "title": "Should be linked to current user",
    }, headers={"Authorization": "Bearer fake"})
    assert resp.status_code == 201

    # Verify the Conversation was created with the current user's id
    call_args = mock_session.add.call_args
    conversation_obj = call_args.args[0]
    assert conversation_obj.user_id == user.id


def test_create_conversation_title_trimmed():
    """Title whitespace is trimmed by validator before reaching endpoint."""
    app, mock_session, mock_result = _make_test_app()
    client = TestClient(app)

    user = _make_user()
    mock_result.scalar_one_or_none.return_value = user

    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    resp = client.post("/api/conversations", json={
        "title": "  trimmed title  ",
    }, headers={"Authorization": "Bearer fake"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "trimmed title"


def test_create_conversation_empty_title(mocker: MockerFixture):
    app, _, _ = _make_test_app(mocker)
    client = TestClient(app)

    resp = client.post("/api/conversations", json={
        "title": "   ",
    })
    assert resp.status_code == 422


def test_create_conversation_title_too_long():
    app, _, _ = _make_test_app()
    client = TestClient(app)

    long_title = "a" * 201
    resp = client.post("/api/conversations", json={
        "title": long_title,
    })
    assert resp.status_code == 422


def test_create_conversation_default_title(mocker: MockerFixture):
    app, mock_session, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    user = _make_user()
    mock_result.scalar_one_or_none.return_value = user

    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    resp = client.post("/api/conversations", json={})
    assert resp.status_code == 201
    assert resp.json()["title"] == "New conversation"


def test_create_conversation_commit_error(mocker: MockerFixture):
    """Commit fails → rollback → re-raise."""
    app, mock_session, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    user = _make_user()
    mock_result.scalar_one_or_none.return_value = user

    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock(
        side_effect=IntegrityError(
            statement=None, params=None, orig=Exception("duplicate"),
        ),
    )
    mock_session.refresh = AsyncMock()

    resp = client.post("/api/conversations", json={
        "title": "Should fail",
    }, headers={"Authorization": "Bearer fake"})
    # Should raise HTTPException from re-raised IntegrityError
    assert resp.status_code == 500 or resp.status_code == 409


# ===========================================================================
# List tests
# ===========================================================================


def test_list_conversations_success():
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    now = datetime.now(timezone.utc)
    own_conv = _make_conversation(id=uuid4(), title="My conv", user_id=_make_user().id)
    other_conv = _make_conversation(id=uuid4(), title="Other conv")

    mock_result.scalars().all.return_value = [own_conv]
    mock_result.scalar_one_or_none.return_value = _make_user()

    user = _make_user()
    token = create_access_token(user.id)

    resp = client.get(
        "/api/conversations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["title"] == "My conv"


def test_list_conversations_only_own():
    """User sees only their own conversations."""
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    own_user = _make_user(id=uuid4(), username="own")
    other_user = _make_user(id=uuid4(), username="other")
    own_conv = _make_conversation(id=uuid4(), user_id=own_user.id)
    other_conv = _make_conversation(id=uuid4(), user_id=other_user.id)

    mock_result.scalars().all.return_value = [own_conv]
    mock_result.scalar_one_or_none.return_value = own_user

    token = create_access_token(own_user.id)

    resp = client.get(
        "/api/conversations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    ids = [item["id"] for item in resp.json()["items"]]
    assert str(own_conv.id) in ids
    assert str(other_conv.id) not in ids


def test_list_conversations_empty():
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user = _make_user()
    mock_result.scalars().all.return_value = []
    mock_result.scalar_one_or_none.return_value = user

    token = create_access_token(user.id)

    resp = client.get(
        "/api/conversations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["total"] == 0
    assert resp.json()["items"] == []


def test_list_conversations_no_token():
    app, _, _ = _make_test_app()
    client = TestClient(app)
    resp = client.get("/api/conversations")
    assert resp.status_code == 401


def test_list_conversations_sorted_by_updated_at_desc():
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user = _make_user()
    later = _make_conversation(id=uuid4(), user_id=user.id)
    earlier = _make_conversation(id=uuid4(), user_id=user.id)

    # Order in response should be desc
    mock_result.scalars().all.return_value = [later, earlier]
    mock_result.scalar_one_or_none.return_value = user

    token = create_access_token(user.id)

    resp = client.get(
        "/api/conversations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    # Later (first in list) should have greater updated_at
    assert resp.json()["items"][0]["updated_at"] >= resp.json()["items"][1]["updated_at"]


# ===========================================================================
# Get detail tests
# ===========================================================================


def test_get_conversation_owned():
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user = _make_user()
    conv = _make_conversation(id=uuid4(), user_id=user.id, title="Detail conv")

    mock_result.scalar_one_or_none.return_value = conv

    token = create_access_token(user.id)

    resp = client.get(
        f"/api/conversations/{conv.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == str(conv.id)
    assert data["title"] == "Detail conv"
    assert data["user_id"] == str(user.id)


def test_get_conversation_not_found():
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user = _make_user()
    mock_result.scalar_one_or_none.return_value = None  # not found

    token = create_access_token(user.id)
    fake_id = uuid4()

    resp = client.get(
        f"/api/conversations/{fake_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


def test_get_conversation_other_user():
    """User B cannot see user A's conversation."""
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user_a = _make_user(id=uuid4(), username="alice")
    user_b = _make_user(id=uuid4(), username="bob")
    conv = _make_conversation(id=uuid4(), user_id=user_a.id)

    # DB would find the conv, but query also filters by user_b.id → None
    mock_result.scalar_one_or_none.return_value = None

    token = create_access_token(user_b.id)

    resp = client.get(
        f"/api/conversations/{conv.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


def test_get_conversation_invalid_uuid():
    app, _, _ = _make_test_app()
    client = TestClient(app)
    resp = client.get("/api/conversations/not-a-uuid")
    assert resp.status_code == 422


# ===========================================================================
# Update tests
# ===========================================================================


def test_update_conversation_success():
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user = _make_user()
    conv = _make_conversation(id=uuid4(), user_id=user.id, title="Old title")

    mock_result.scalar_one_or_none.return_value = conv
    mock_session.refresh = AsyncMock()
    mock_session.commit = AsyncMock()

    token = create_access_token(user.id)

    resp = client.patch(
        f"/api/conversations/{conv.id}",
        json={"title": "New title"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["title"] == "New title"


def test_update_conversation_trims_title():
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user = _make_user()
    conv = _make_conversation(id=uuid4(), user_id=user.id, title="Old")

    mock_result.scalar_one_or_none.return_value = conv
    mock_session.refresh = AsyncMock()
    mock_session.commit = AsyncMock()

    token = create_access_token(user.id)

    resp = client.patch(
        f"/api/conversations/{conv.id}",
        json={"title": "  padded  "},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["title"] == "padded"


def test_update_conversation_empty_title():
    app, _, _ = _make_test_app()
    client = TestClient(app)
    resp = client.patch("/api/conversations/some-id", json={"title": "  "})
    assert resp.status_code == 422


def test_update_conversation_other_user():
    """User B cannot update user A's conversation."""
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user_b = _make_user(id=uuid4(), username="bob")
    mock_result.scalar_one_or_none.return_value = None  # query filters by bob

    token = create_access_token(user_b.id)
    fake_id = uuid4()

    resp = client.patch(
        f"/api/conversations/{fake_id}",
        json={"title": "Hacked"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


def test_update_conversation_commit_error():
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user = _make_user()
    conv = _make_conversation(id=uuid4(), user_id=user.id)

    mock_result.scalar_one_or_none.return_value = conv
    mock_session.commit = AsyncMock(
        side_effect=IntegrityError(
            statement=None, params=None, orig=Exception("fail"),
        ),
    )
    mock_session.refresh = AsyncMock()

    token = create_access_token(user.id)

    resp = client.patch(
        f"/api/conversations/{conv.id}",
        json={"title": "Broken"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 500 or resp.status_code == 409


# ===========================================================================
# Delete tests
# ===========================================================================


def test_delete_conversation_success():
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user = _make_user()
    conv = _make_conversation(id=uuid4(), user_id=user.id)

    mock_result.scalar_one_or_none.return_value = conv
    mock_session.commit = AsyncMock()

    token = create_access_token(user.id)

    resp = client.delete(
        f"/api/conversations/{conv.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 204
    assert resp.content == b""


def test_delete_conversation_not_found():
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user = _make_user()
    mock_result.scalar_one_or_none.return_value = None

    token = create_access_token(user.id)
    fake_id = uuid4()

    resp = client.delete(
        f"/api/conversations/{fake_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


def test_delete_conversation_other_user():
    """User B cannot delete user A's conversation."""
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user_b = _make_user(id=uuid4(), username="bob")
    mock_result.scalar_one_or_none.return_value = None  # filtered by bob

    token = create_access_token(user_b.id)
    fake_id = uuid4()

    resp = client.delete(
        f"/api/conversations/{fake_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


def test_delete_conversation_commit_error():
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user = _make_user()
    conv = _make_conversation(id=uuid4(), user_id=user.id)

    mock_result.scalar_one_or_none.return_value = conv
    mock_session.commit = AsyncMock(
        side_effect=IntegrityError(
            statement=None, params=None, orig=Exception("fail"),
        ),
    )

    token = create_access_token(user.id)

    resp = client.delete(
        f"/api/conversations/{conv.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 500 or resp.status_code == 409


# ===========================================================================
# Security
# ===========================================================================


def test_security_no_user_id_in_request():
    """No endpoint accepts user_id from client to override ownership."""
    from app.api.conversations import router as conversations_router
    from app.schemas.conversation import (
        ConversationCreateRequest,
        ConversationUpdateRequest,
    )

    # Check request schemas don't expose user_id
    create_fields = ConversationCreateRequest.model_fields
    update_fields = ConversationUpdateRequest.model_fields
    assert "user_id" not in create_fields
    assert "user_id" not in update_fields


def test_security_user_a_cannot_see_user_b_conversation():
    """User A queries with user B's conversation ID → 404."""
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user_a = _make_user(id=uuid4(), username="alice")
    user_b = _make_user(id=uuid4(), username="bob")
    conv = _make_conversation(id=uuid4(), user_id=user_b.id)

    mock_result.scalar_one_or_none.return_value = None  # alice can't see bob's

    token = create_access_token(user_a.id)

    resp = client.get(
        f"/api/conversations/{conv.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


def test_security_user_a_cannot_update_user_b_conversation():
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user_a = _make_user(id=uuid4(), username="alice")
    user_b = _make_user(id=uuid4(), username="bob")
    conv = _make_conversation(id=uuid4(), user_id=user_b.id)

    mock_result.scalar_one_or_none.return_value = None

    token = create_access_token(user_a.id)

    resp = client.patch(
        f"/api/conversations/{conv.id}",
        json={"title": "Hacked"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404


def test_security_user_a_cannot_delete_user_b_conversation():
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user_a = _make_user(id=uuid4(), username="alice")
    user_b = _make_user(id=uuid4(), username="bob")
    conv = _make_conversation(id=uuid4(), user_id=user_b.id)

    mock_result.scalar_one_or_none.return_value = None

    token = create_access_token(user_a.id)

    resp = client.delete(
        f"/api/conversations/{conv.id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 404
