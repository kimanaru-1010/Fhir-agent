from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class RegisterRequest(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=100,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    password: str = Field(
        min_length=8,
        max_length=128,
    )
    external_id: str | None = Field(
        default=None,
        max_length=255,
    )

    @field_validator("username")
    @classmethod
    def _trim_username(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("username must not be blank after trimming")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def _trim_username(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("username must not be blank after trimming")
        return v


class UserResponse(BaseModel):
    id: UUID
    username: str
    external_id: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True,
    }


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse
