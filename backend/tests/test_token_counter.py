"""Tests for short-term memory token counting."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.services.token_counter import ApproximateTokenCounter, ConversationMessage


def _message(content: str) -> ConversationMessage:
    return ConversationMessage(
        id=uuid4(),
        role="user",
        content=content,
        created_at=datetime.now(timezone.utc),
    )


def test_count_empty_text_is_zero():
    assert ApproximateTokenCounter().count_text("") == 0


def test_count_non_empty_text_is_at_least_one():
    assert ApproximateTokenCounter().count_text("a") >= 1


def test_count_messages_adds_content_and_role_overhead():
    counter = ApproximateTokenCounter()
    messages = [_message("abcd"), _message("abcdefgh")]

    assert counter.count_messages(messages) == (
        counter.count_text("abcd")
        + counter.count_text("abcdefgh")
        + counter.message_overhead_tokens * 2
    )


def test_count_messages_includes_attachment_metadata():
    counter = ApproximateTokenCounter()
    message = ConversationMessage(
        id=uuid4(),
        role="user",
        content="Analyze this image",
        created_at=datetime.now(timezone.utc),
        message_type="image_upload",
        attachments=[
            {
                "type": "image",
                "patient_id": "12261",
                "binary_id": "binary-123",
                "data": "must-not-count",
            }
        ],
    )

    count = counter.count_messages([message])

    assert count > counter.count_text("Analyze this image")
