"""Pydantic schemas for conversation messages."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MessageCreateRequest(BaseModel):
    content: str = Field(
        min_length=1,
        max_length=10_000,
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("content")
    @classmethod
    def trim_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content must not be blank")
        return value


class MessageResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageListResponse(BaseModel):
    items: list[MessageResponse]
    total: int


class MessageExchangeResponse(BaseModel):
    conversation_id: UUID
    user_message: MessageResponse
    assistant_message: MessageResponse
