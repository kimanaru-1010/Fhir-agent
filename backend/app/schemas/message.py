"""Pydantic schemas for conversation messages."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.skin_images.references import build_image_api_url


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


class ChatImageAttachment(BaseModel):
    type: Literal["image"] = "image"
    patient_id: str
    diagnostic_report_id: str
    media_id: str
    binary_id: str
    url: str | None = None
    content_type: str | None = None
    created_at: str | None = None
    title: str | None = None
    description: str | None = None


class MessageResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    role: str
    content: str
    message_type: str = "text"
    attachments: list[ChatImageAttachment] = Field(default_factory=list)
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @field_validator("attachments", mode="before")
    @classmethod
    def normalize_attachments(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        normalized = []
        for item in value:
            if isinstance(item, dict) and item.get("binary_id") and not item.get("url"):
                item = {**item, "url": build_image_api_url(str(item["binary_id"]))}
            normalized.append(item)
        return normalized

    @field_validator("message_type", mode="before")
    @classmethod
    def normalize_message_type(cls, value):
        return "text" if value is None else value


class MessageListResponse(BaseModel):
    items: list[MessageResponse]
    total: int


class MessageExchangeResponse(BaseModel):
    conversation_id: UUID
    user_message: MessageResponse
    assistant_message: MessageResponse
    attachments: list[ChatImageAttachment] = Field(default_factory=list)
