"""Token counting abstractions for short-term conversation memory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.core.config import settings


@dataclass(frozen=True)
class ConversationMessage:
    id: UUID
    role: str
    content: str
    created_at: datetime


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
            self.count_text(message.content) + self.message_overhead_tokens
            for message in messages
        )
