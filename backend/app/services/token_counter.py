"""Token counting abstractions for short-term conversation memory."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from app.core.config import settings


@dataclass(frozen=True)
class ConversationMessage:
    id: UUID
    role: str
    content: str
    created_at: datetime
    message_type: str = "text"
    attachments: list[dict[str, Any]] = field(default_factory=list)


def format_conversation_message(message: ConversationMessage) -> str:
    label = message.role.upper()
    message_type = (message.message_type or "text").strip()
    if message_type and message_type != "text":
        label = f"{label} [{message_type}]"

    text = f"{label}: {message.content}"
    attachment_blocks: list[str] = []
    for attachment in message.attachments or []:
        if not isinstance(attachment, dict) or attachment.get("type") != "image":
            continue
        fields = []
        for key in (
            "patient_id",
            "binary_id",
            "media_id",
            "diagnostic_report_id",
            "content_type",
            "created_at",
        ):
            value = attachment.get(key)
            if value:
                fields.append(f"{key}={value}")
        if fields:
            attachment_blocks.append("IMAGE_ATTACHMENT:\n" + "\n".join(fields))

    if attachment_blocks:
        text = text + "\n\n" + "\n\n".join(attachment_blocks)
    return text


class TokenCounter(Protocol):
    def count_text(self, text: str) -> int:
        ...

    def count_messages(self, messages: list[ConversationMessage]) -> int:
        ...


class ApproximateTokenCounter:
    message_overhead_tokens = 4

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return max(
            1,
            math.ceil(len(text) / settings.short_term_chars_per_token),
        )

    def count_messages(self, messages: list[ConversationMessage]) -> int:
        return sum(
            self.count_text(
                format_conversation_message(message)
                if message.attachments or (message.message_type or "text") != "text"
                else message.content
            ) + self.message_overhead_tokens
            for message in messages
        )
