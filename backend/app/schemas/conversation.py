"""Pydantic schemas for conversation management."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ConversationCreateRequest(BaseModel):
    first_message: str = Field(
        min_length=1,
        max_length=10_000,
    )

    model_config = {"extra": "forbid"}

    @field_validator("first_message")
    @classmethod
    def trim_first_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("first_message must not be blank")
        return value


class FirstMessageResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationCreateResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    first_message: FirstMessageResponse
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


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
