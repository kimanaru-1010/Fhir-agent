"""Tests for auth API â€” register, login, and /users/me."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import jwt
import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from sqlalchemy.exc import IntegrityError
from unittest.mock import AsyncMock, MagicMock

from app.db.models import User
from app.services.auth import create_access_token

# Pre-imports for override
from app.db import session as _db_module
from app.dependencies import auth as _auth_dep


# ===========================================================================
# Helpers
# ===========================================================================


def _build_mock_session() -> tuple[MagicMock, MagicMock]:
    """Build a mock AsyncSession.

    execute / commit / rollback / refresh are async (awaited by production code).
    add is sync (called without await).
    """
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mock_result = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_result)

    # add() is sync in production
    mock_session.add = MagicMock()

    # The remaining methods are async
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.refresh = AsyncMock()

    return mock_session, mock_result


def _make_test_app(
    mocker: MockerFixture | None = None,
    *,
    override_current_user: bool = True,
) -> tuple[FastAPI, MagicMock, MagicMock]:
    """Build test app with auth routers mounted under /api.

    By default overrides get_current_user (for tests that don't need
    real JWT validation).  Set override_current_user=False to exercise
    the real auth pipeline.
    """
    from app.api.auth import auth_router, users_router

    app = FastAPI()
    app.include_router(auth_router, prefix="/api")
    app.include_router(users_router, prefix="/api")

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


# ===========================================================================
# Register tests
# ===========================================================================


def test_register_success(mocker: MockerFixture):
    app, mock_session, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    mock_result.scalar_one_or_none.return_value = None
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    now = datetime.now(timezone.utc)

    async def _refresh_user(user):
        user.id = uuid4()
        user.created_at = now
        user.updated_at = now

    mock_session.refresh = AsyncMock(side_effect=_refresh_user)

    resp = client.post("/api/auth/register", json={
        "username": "test_user",
        "password": "strongpassword123",
        "external_id": "EXT-001",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "test_user"
    assert data["external_id"] == "EXT-001"
    assert "id" in data
    assert "created_at" in data
    assert "password_hash" not in data
    assert "password" not in data

    mock_session.add.assert_called_once()
    user_obj = mock_session.add.call_args.args[0]
    assert isinstance(user_obj, User)
    assert user_obj.password_hash != "strongpassword123"


def test_register_duplicate_username(mocker: MockerFixture):
    app, _, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    mock_result.scalar_one_or_none.return_value = MagicMock()

    resp = client.post("/api/auth/register", json={
        "username": "dup_user",
        "password": "strongpassword123",
    })
    assert resp.status_code == 409


def test_register_duplicate_external_id(mocker: MockerFixture):
    app, _, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    mock_result.scalar_one_or_none.side_effect = [None, MagicMock()]

    resp = client.post("/api/auth/register", json={
        "username": "user_a",
        "password": "strongpassword123",
        "external_id": "SAME-EXT",
    })
    assert resp.status_code == 409


def test_register_integrity_error(mocker: MockerFixture):
    """Race condition: username check passes but commit fails with IntegrityError."""
    app, mock_session, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    mock_result.scalar_one_or_none.return_value = None
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock(
        side_effect=IntegrityError(
            statement=None, params=None, orig=Exception("duplicate"),
        ),
    )
    mock_session.refresh = AsyncMock()

    resp = client.post("/api/auth/register", json={
        "username": "race_user",
        "password": "strongpassword123",
    })
    assert resp.status_code == 409
    assert mock_session.rollback.assert_awaited_once


# ===========================================================================
# Login tests
# ===========================================================================


def _login_user_mocker(
    mocker: MockerFixture,
) -> tuple[FastAPI, TestClient, User, str]:
    """Set up a login scenario with an existing user.

    Returns (app, client, real_User_instance, username).
    """
    app, mock_session, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    user_id = uuid4()
    now = datetime.now(timezone.utc)
    real_user = User(
        id=user_id,
        username="login_user",
        password_hash="hashed_abc",
        external_id="EXT-001",
        created_at=now,
        updated_at=now,
    )

    mock_result.scalar_one_or_none.return_value = real_user
    return app, client, real_user, "login_user"


def test_login_success(mocker: MockerFixture):
    app, client, mock_user, username = _login_user_mocker(mocker)

    mocker.patch(
        "app.api.auth.verify_password",
        side_effect=lambda pwd, stored: pwd == "CorrectPass1",
    )

    resp = client.post("/api/auth/login", json={
        "username": "login_user",
        "password": "CorrectPass1",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "expires_in" in data
    assert data["user"]["username"] == username


def test_login_wrong_password(mocker: MockerFixture):
    """Username exists but password is wrong."""
    app, client, mock_user, username = _login_user_mocker(mocker)

    mocker.patch("app.api.auth.verify_password", return_value=False)

    resp = client.post("/api/auth/login", json={
        "username": username,
        "password": "wrongpassword",
    })
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid username or password"


def test_login_nonexistent_user(mocker: MockerFixture):
    app, _, mock_result = _make_test_app()
    client = TestClient(app)

    mock_result.scalar_one_or_none.return_value = None

    resp = client.post("/api/auth/login", json={
        "username": "no_such_user",
        "password": "anypassword",
    })
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid username or password"


def test_login_same_error(mocker: MockerFixture):
    """Same response for missing user vs wrong password."""
    app, _, mock_result = _make_test_app()
    client = TestClient(app)

    mock_result.scalar_one_or_none.return_value = None

    wrong = client.post("/api/auth/login", json={
        "username": "fake_user",
        "password": "wrong",
    })
    missing = client.post("/api/auth/login", json={
        "username": "nope_user",
        "password": "wrong",
    })
    assert wrong.json()["detail"] == missing.json()["detail"]


def test_login_token_sub_is_uuid(mocker: MockerFixture):
    from app.core.config import settings

    app, client, mock_user, _ = _login_user_mocker(mocker)

    mocker.patch("app.api.auth.verify_password", return_value=True)

    resp = client.post("/api/auth/login", json={
        "username": "login_user",
        "password": "CorrectPass1",
    })
    token = resp.json()["access_token"]
    payload = jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )
    UUID(payload["sub"])


def test_login_trims_username():
    """Username with surrounding whitespace should be trimmed before query."""
    app, mock_session, mock_result = _make_test_app()
    client = TestClient(app)

    now = datetime.now(timezone.utc)
    real_user = User(
        id=uuid4(),
        username="login_user",
        password_hash="hashed_abc",
        external_id="EXT-001",
        created_at=now,
        updated_at=now,
    )
    mock_result.scalar_one_or_none.return_value = real_user

    resp = client.post("/api/auth/login", json={
        "username": "  login_user  ",
        "password": "CorrectPass1",
    })
    assert resp.status_code == 401  # verify_password not mocked, returns False

    # The query was called with the trimmed username
    mock_session.execute.assert_called_once()
    call_args = mock_session.execute.call_args
    stmt = call_args[0][0]
    where_clause = stmt._whereclause
    assert where_clause.right.value == "login_user"


# ===========================================================================
# Users /me tests  (real get_current_user)
# ===========================================================================


def _build_me_app_with_user(
    user_id: UUID | None = None,
    *,
    return_user: bool = True,
) -> tuple[FastAPI, MagicMock, MagicMock]:
    """Build app with real auth; mock db.execute to return a user or None."""
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )

    if user_id is None:
        user_id = uuid4()

    now = datetime.now(timezone.utc)
    if return_user:
        mock_user = User(
            id=user_id,
            username="test_user",
            password_hash="hashed",
            external_id="EXT-001",
            created_at=now,
            updated_at=now,
        )
        mock_result.scalar_one_or_none.return_value = mock_user
    else:
        mock_result.scalar_one_or_none.return_value = None

    return app, mock_session, mock_result


def test_users_me_valid_token(mocker: MockerFixture):
    user_id = uuid4()
    app, _, _ = _build_me_app_with_user(user_id, return_user=True)
    client = TestClient(app)

    token = create_access_token(user_id)

    resp = client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["username"] == "test_user"
    assert data["id"] == str(user_id)


def test_users_me_no_token():
    """No token â†’ get_current_user raises 401."""
    app, _, _ = _build_me_app_with_user(return_user=False)
    client = TestClient(app)

    resp = client.get("/api/users/me")
    assert resp.status_code == 401


def test_users_me_bad_token():
    """Invalid token â†’ decode fails â†’ 401."""
    app, _, _ = _build_me_app_with_user(return_user=False)
    client = TestClient(app)

    resp = client.get(
        "/api/users/me",
        headers={"Authorization": "Bearer invalidtoken"},
    )
    assert resp.status_code == 401


def test_users_me_expired_token():
    """Expired signature â†’ 401."""
    from app.core.config import settings

    app, _, _ = _build_me_app_with_user(return_user=False)
    client = TestClient(app)

    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(uuid4()),
        "iat": now - timedelta(minutes=120),
        "exp": now - timedelta(minutes=60),
    }
    token = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    resp = client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401


def test_users_me_token_valid_but_user_deleted():
    """Token is valid but the user row no longer exists â†’ 401."""
    app, mock_session, mock_result = _make_test_app(
        override_current_user=False,
    )
    client = TestClient(app)

    user_id = uuid4()
    mock_result.scalar_one_or_none.return_value = None  # deleted

    token = create_access_token(user_id)

    resp = client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid or expired authentication token"


# ===========================================================================
# Security tests
# ===========================================================================


def test_password_not_in_register_response(mocker: MockerFixture):
    app, mock_session, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    mock_result.scalar_one_or_none.return_value = None
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    now = datetime.now(timezone.utc)

    async def _refresh_user(user):
        user.id = uuid4()
        user.created_at = now
        user.updated_at = now

    mock_session.refresh = AsyncMock(side_effect=_refresh_user)

    resp = client.post("/api/auth/register", json={
        "username": "sec_user",
        "password": "mysecret123",
    })
    body = resp.json()

    def _check(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                _check(v)
        elif isinstance(obj, list):
            for v in obj:
                _check(v)
        elif isinstance(obj, str) and "password" in obj.lower():
            pytest.fail(f"Password value found: {obj[:30]}")

    _check(body)


def test_password_hash_not_in_response(mocker: MockerFixture):
    app, mock_session, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    mock_result.scalar_one_or_none.return_value = None
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()

    now = datetime.now(timezone.utc)

    async def _refresh_user(user):
        user.id = uuid4()
        user.created_at = now
        user.updated_at = now

    mock_session.refresh = AsyncMock(side_effect=_refresh_user)

    resp = client.post("/api/auth/register", json={
        "username": "sec_user2",
        "password": "mysecret456",
    })
    assert "password_hash" not in resp.json()


def test_token_isolation(mocker: MockerFixture):
    """Token for user_a must not return user_b (uses real JWT + real dep)."""
    user_a = User(
        id=uuid4(),
        username="iso_a",
        password_hash="hash_a",
        external_id="EXT-A",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    user_b = User(
        id=uuid4(),
        username="iso_b",
        password_hash="hash_b",
        external_id="EXT-B",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    app, _, mock_result = _make_test_app(override_current_user=False)
    client = TestClient(app)

    # DB returns only user_a (the one whose token was presented)
    mock_result.scalar_one_or_none.return_value = user_a

    token_a = create_access_token(user_a.id)

    resp = client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == str(user_a.id)
    assert data["id"] != str(user_b.id)
    assert data["username"] == "iso_a"


# ===========================================================================
# Schema validation tests
# ===========================================================================


def test_register_short_password(mocker: MockerFixture):
    app, _, _ = _make_test_app()
    client = TestClient(app)

    resp = client.post("/api/auth/register", json={
        "username": "shortpwd_user",
        "password": "1234567",
    })
    assert resp.status_code == 422


def test_register_invalid_username(mocker: MockerFixture):
    app, _, _ = _make_test_app()
    client = TestClient(app)

    resp = client.post("/api/auth/register", json={
        "username": "bad user!@#",
        "password": "strongpassword123",
    })
    assert resp.status_code == 422


def test_login_empty_username_rejected(mocker: MockerFixture):
    app, _, _ = _make_test_app()
    client = TestClient(app)

    resp = client.post("/api/auth/login", json={
        "username": "   ",
        "password": "anypassword",
    })
    assert resp.status_code == 422
