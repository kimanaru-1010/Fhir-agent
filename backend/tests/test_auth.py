"""Tests for auth API — register, login, and /users/me."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import User
from app.services.auth import create_access_token, decode_access_token

# --- Test database setup ---
_TEST_DB_URL = "sqlite:///:memory:"
_engine = create_engine(_TEST_DB_URL, connect_args={"check_same_thread": False})
_factory = sessionmaker(bind=_engine, class_=Session)

# Pre-import to override correctly
from app import database as _db_module
from app.dependencies import auth as _auth_dep


@pytest.fixture()
def db_session() -> Session:
    """Fresh in-memory SQLite session per test."""
    Base.metadata.create_all(_engine)
    session = _factory()
    yield session
    session.close()
    Base.metadata.drop_all(_engine)


@pytest.fixture()
def client(db_session: Session):
    """FastAPI test client with real DB and dependency overrides."""
    from app.api.auth import auth_router, users_router

    app = FastAPI()
    app.include_router(auth_router, prefix="/api")
    app.include_router(users_router, prefix="/api")

    async def _override_get_db():
        # Yield the session in async fashion — TestClient runs synchronously
        # so we just yield directly
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[_db_module.get_db] = _override_get_db

    # Override get_current_user to use sync session
    from fastapi import Depends, HTTPException, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    security = HTTPBearer(auto_error=False)

    async def _get_current_user(
        credentials: HTTPAuthorizationCredentials | None = Depends(security),
    ):
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if credentials.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            user_id = decode_access_token(credentials.credentials)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user = db_session.execute(
            select(User).where(User.id == user_id)
        ).scalar_one_or_none()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user

    app.dependency_overrides[_auth_dep.get_current_user] = _get_current_user

    return TestClient(app)


# ===========================================================================
# Register tests
# ===========================================================================


def test_register_success(client: TestClient, db_session: Session):
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
    assert "updated_at" in data
    assert "password_hash" not in data
    assert "password" not in data

    # Verify user in DB with hashed password
    user = db_session.execute(
        select(User).where(User.username == "test_user")
    ).scalar_one_or_none()
    assert user is not None
    assert user.password_hash != "strongpassword123"
    assert len(user.password_hash) > 50


def test_register_duplicate_username(client: TestClient):
    client.post("/api/auth/register", json={
        "username": "dup_user",
        "password": "strongpassword123",
    })
    resp = client.post("/api/auth/register", json={
        "username": "dup_user",
        "password": "anotherpassword1",
    })
    assert resp.status_code == 409


def test_register_duplicate_external_id(client: TestClient):
    client.post("/api/auth/register", json={
        "username": "user_a",
        "password": "strongpassword123",
        "external_id": "SAME-EXT",
    })
    resp = client.post("/api/auth/register", json={
        "username": "user_b",
        "password": "anotherpassword1",
        "external_id": "SAME-EXT",
    })
    assert resp.status_code == 409


def test_register_short_password(client: TestClient):
    resp = client.post("/api/auth/register", json={
        "username": "shortpwd_user",
        "password": "1234567",
    })
    assert resp.status_code == 422


def test_register_invalid_username(client: TestClient):
    resp = client.post("/api/auth/register", json={
        "username": "bad user!@#",
        "password": "strongpassword123",
    })
    assert resp.status_code == 422


# ===========================================================================
# Login tests
# ===========================================================================


def _register(client: TestClient) -> tuple[str, str]:
    """Register and return (username, password)."""
    username = f"user_{uuid4().hex[:8]}"
    password = "CorrectPass1"
    client.post("/api/auth/register", json={
        "username": username,
        "password": password,
    })
    return username, password


def test_login_success(client: TestClient):
    username, password = _register(client)
    resp = client.post("/api/auth/login", json={
        "username": username,
        "password": password,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "expires_in" in data
    assert "user" in data
    assert data["user"]["username"] == username


def test_login_wrong_password(client: TestClient):
    _register(client)
    resp = client.post("/api/auth/login", json={
        "username": "user_xxx",  # wrong user
        "password": "wrongpassword",
    })
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid username or password"


def test_login_nonexistent_user(client: TestClient):
    resp = client.post("/api/auth/login", json={
        "username": "no_such_user_" + uuid4().hex[:8],
        "password": "anypassword",
    })
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid username or password"


def test_login_same_error(client: TestClient):
    wrong = client.post("/api/auth/login", json={
        "username": "fake_" + uuid4().hex[:8],
        "password": "wrong",
    })
    missing = client.post("/api/auth/login", json={
        "username": "nope_" + uuid4().hex[:8],
        "password": "wrong",
    })
    assert wrong.json()["detail"] == missing.json()["detail"]


def test_login_token_sub_is_uuid(client: TestClient):
    from app.config import settings
    username, password = _register(client)
    resp = client.post("/api/auth/login", json={
        "username": username,
        "password": password,
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


def test_users_me_valid_token(client: TestClient):
    username, password = _register(client)
    login_resp = client.post("/api/auth/login", json={
        "username": username,
        "password": password,
    })
    token = login_resp.json()["access_token"]
    resp = client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["username"] == username
    assert "password" not in data


def test_users_me_no_token(client: TestClient):
    resp = client.get("/api/users/me")
    assert resp.status_code == 401


def test_users_me_bad_token(client: TestClient):
    resp = client.get(
        "/api/users/me",
        headers={"Authorization": "Bearer invalidtoken"},
    )
    assert resp.status_code == 401


def test_users_me_expired_token(client: TestClient):
    from app.config import settings
    # Create an expired token for a fake UUID
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


def test_password_not_in_register_response(client: TestClient):
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
            pytest.fail(f"Password-like value found: {obj[:30]}")

    _check(body)


def test_password_hash_not_in_response(client: TestClient):
    resp = client.post("/api/auth/register", json={
        "username": "sec_user2",
        "password": "mysecret456",
    })
    assert "password_hash" not in resp.json()


def test_token_isolation(client: TestClient):
    """Token from user A returns user A, not user B."""
    client.post("/api/auth/register", json={
        "username": "iso_a",
        "password": "passwordAAA1",
    })
    client.post("/api/auth/register", json={
        "username": "iso_b",
        "password": "passwordBBB1",
    })
    login_a = client.post("/api/auth/login", json={
        "username": "iso_a",
        "password": "passwordAAA1",
    })
    token_a = login_a.json()["access_token"]
    resp = client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert resp.status_code == 200
    assert resp.json()["username"] == "iso_a"
