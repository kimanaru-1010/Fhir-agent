"""Authentication — in-memory placeholder for now."""

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=4)


class UserResponse(BaseModel):
    username: str


# ---------------------------------------------------------------------------
# Simple in-memory store (replace with DB when real user system is ready)
# ---------------------------------------------------------------------------

_users: dict[str, str] = {}  # username -> hashed_password


async def register(req: RegisterRequest) -> UserResponse:
    if req.username in _users:
        raise ValueError(f"User '{req.username}' already exists")
    # TODO: hash password before storing
    _users[req.username] = req.password
    return UserResponse(username=req.username)


async def login(req: LoginRequest) -> UserResponse:
    if req.username not in _users or _users[req.username] != req.password:
        raise ValueError("Invalid username or password")
    return UserResponse(username=req.username)
