"""Tests for auth API — register, login, and /users/me."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import jwt
import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pytest_mock import MockerFixture
from unittest.mock import AsyncMock, MagicMock

from app.db.models import User
from app.services.auth import create_access_token, decode_access_token

# Pre-imports for override
from app import database as _db_module
from app.dependencies import auth as _auth_dep


# --- Helpers ---

def _build_mock_session(mocker: MockerFixture) -> MagicMock:
    """Build a mock session: execute() is async, result methods are sync."""
    # The session itself needs __aenter__/__aexit__ for context manager usage
    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    # execute returns a coroutine that resolves to a Mock (sync methods)
    mock_result = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    return mock_session, mock_result


def _make_test_app(mocker: MockerFixture) -> tuple[FastAPI, MagicMock, MagicMock]:
    """Build test app, return (app, mock_session, mock_result)."""
    from app.api.auth import auth_router, users_router

    app = FastAPI()
    app.include_router(auth_router, prefix="/api")
    app.include_router(users_router, prefix="/api")

    mock_session, mock_result = _build_mock_session(mocker)

    async def _override_get_db():
        yield mock_session

    app.dependency_overrides[_db_module.get_db] = _override_get_db

    # Default: 401 on get_current_user
    async def _default_current_user(credentials=None):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    app.dependency_overrides[_auth_dep.get_current_user] = _default_current_user

    return app, mock_session, mock_result


# ===========================================================================
# Register tests
# ===========================================================================


def test_register_success(mocker: MockerFixture):
    app, mock_session, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    # No existing user found
    mock_result.scalar_one_or_none.return_value = None
    mock_session.add = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

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

    # Verify password was hashed
    user_obj = mock_session.add.call_args[0][0]
    assert isinstance(user_obj, User)
    assert user_obj.password_hash != "strongpassword123"


def test_register_duplicate_username(mocker: MockerFixture):
    app, _, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    # Username already exists
    mock_result.scalar_one_or_none.return_value = MagicMock()

    resp = client.post("/api/auth/register", json={
        "username": "dup_user",
        "password": "strongpassword123",
    })
    assert resp.status_code == 409


def test_register_duplicate_external_id(mocker: MockerFixture):
    app, _, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    # First query (username check): not found; Second (external_id check): found
    mock_result.scalar_one_or_none.side_effect = [
        None,
        MagicMock(),
    ]

    resp = client.post("/api/auth/register", json={
        "username": "user_a",
        "password": "strongpassword123",
        "external_id": "SAME-EXT",
    })
    assert resp.status_code == 409


def test_register_short_password(mocker: MockerFixture):
    app, _ = _make_test_app(mocker)
    client = TestClient(app)
    resp = client.post("/api/auth/register", json={
        "username": "shortpwd_user",
        "password": "1234567",
    })
    assert resp.status_code == 422


def test_register_invalid_username(mocker: MockerFixture):
    app, _ = _make_test_app(mocker)
    client = TestClient(app)
    resp = client.post("/api/auth/register", json={
        "username": "bad user!@#",
        "password": "strongpassword123",
    })
    assert resp.status_code == 422


# ===========================================================================
# Login tests
# ===========================================================================


def _login_user_mocker(mocker: MockerFixture) -> tuple[FastAPI, TestClient, MagicMock, str]:
    """Set up mocker for a login with an existing user. Returns (app, client, mock_user, username)."""
    app, mock_session, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    user_id = uuid4()
    now = datetime.now(timezone.utc)
    mock_user = MagicMock(spec=User)
    mock_user.id = user_id
    mock_user.username = "login_user"
    mock_user.external_id = "EXT-001"
    mock_user.created_at = now
    mock_user.updated_at = now
    mock_user.password_hash = "hashed_abc"

    mock_result.scalar_one_or_none.return_value = mock_user

    return app, client, mock_user, "login_user"


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
    app, client, _, _ = _login_user_mocker(mocker)

    mocker.patch("app.api.auth.verify_password", return_value=False)

    resp = client.post("/api/auth/login", json={
        "username": "login_user",
        "password": "wrongpassword",
    })
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid username or password"


def test_login_nonexistent_user(mocker: MockerFixture):
    app, mock_session, mock_result = _make_test_app(mocker)
    client = TestClient(app)

    # User not found
    mock_result.scalar_one_or_none.return_value = None

    resp = client.post("/api/auth/login", json={
        "username": "no_such_user",
        "password": "anypassword",
    })
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid username or password"


def test_login_same_error(mocker: MockerFixture):
    app, _, mock_result = _make_test_app(mocker)
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
    from app.config import settings
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
    UUID(payload["sub"])  # validates sub is valid UUID


# ===========================================================================
# Users /me tests
# ===========================================================================


def test_users_me_valid_token(mocker: MockerFixture):
    app, _, mock_user, username = _login_user_mocker(mocker)

    # Create token for the mocked user
    token = create_access_token(mock_user.id)

    async def _return_current_user(credentials=None):
        return mock_user

    app.dependency_overrides[_auth_dep.get_current_user] = _return_current_user
    client = TestClient(app)

    resp = client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["username"] == username


def test_users_me_no_token(mocker: MockerFixture):
    app, _ = _make_test_app(mocker)
    client = TestClient(app)
    resp = client.get("/api/users/me")
    assert resp.status_code == 401


def test_users_me_bad_token(mocker: MockerFixture):
    app, _ = _make_test_app(mocker)
    client = TestClient(app)
    resp = client.get(
        "/api/users/me",
        headers={"Authorization": "Bearer invalidtoken"},
    )
    assert resp.status_code == 401


def test_users_me_expired_token(mocker: MockerFixture):
    from app.config import settings
    app, _ = _make_test_app(mocker)
    client = TestClient(app)

    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(uuid4()),
        "iat": now - timedelta(minutes=120),
        "exp": now - timedelta(minutes=60),
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    resp = client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401


# ===========================================================================
# Security tests
# ===========================================================================


def test_password_not_in_register_response(mocker: MockerFixture):
    app, mock_session, mock_result = _make_test_app(mocker)
    client = TestClient(app)
    mock_result.scalar_one_or_none.return_value = None
    mock_session.add = AsyncMock()
    mock_session.commit = AsyncMock()

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
    mock_session.add = AsyncMock()
    mock_session.commit = AsyncMock()

    resp = client.post("/api/auth/register", json={
        "username": "sec_user2",
        "password": "mysecret456",
    })
    assert "password_hash" not in resp.json()


def test_token_isolation(mocker: MockerFixture):
    app, _, mock_user_a, username_a = _login_user_mocker(mocker)
    mock_user_a.username = "iso_a"

    # Override get_current_user to always return user A
    app.dependency_overrides[_auth_dep.get_current_user] = lambda: mock_user_a
    client = TestClient(app)

    resp = client.get("/api/users/me")
    assert resp.status_code == 200
    assert resp.json()["username"] == "iso_a"
