"""Pydantic schemas for conversation management."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ConversationCreateRequest(BaseModel):
    title: str = Field(
        default="New conversation",
        min_length=1,
        max_length=200,
    )

    @field_validator("title")
    @classmethod
    def trim_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title must not be blank")
        return value


class ConversationUpdateRequest(BaseModel):
    title: str = Field(
        min_length=1,
        max_length=200,
    )

    @field_validator("title")
    @classmethod
    def trim_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title must not be blank")
        return value


class ConversationResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse]
    total: int
