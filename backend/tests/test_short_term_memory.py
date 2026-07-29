"""Tests for short-term memory splitting and context building."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from app.services.short_term_memory import (
    build_conversation_context,
    split_messages_for_compaction,
)
from app.services.token_counter import ConversationMessage, TokenCounter


class FixedTokenCounter:
    def count_text(self, text: str) -> int:
        return len(text)

    def count_messages(self, messages: list[ConversationMessage]) -> int:
        return sum(len(message.content) for message in messages)


def _message(
    content: str,
    *,
    id: UUID | None = None,
    offset: int = 0,
) -> ConversationMessage:
    return ConversationMessage(
        id=id or uuid4(),
        role="user",
        content=content,
        created_at=datetime(2026, 7, 29, tzinfo=timezone.utc) + timedelta(seconds=offset),
    )


def test_split_below_budget_keeps_all_messages_recent():
    messages = [_message("aa", offset=1), _message("bb", offset=2)]

    old, recent = split_messages_for_compaction(
        messages,
        recent_token_budget=10,
        token_counter=FixedTokenCounter(),
    )

    assert old == []
    assert recent == messages


def test_split_over_budget_moves_old_messages_to_summarize_and_preserves_order():
    messages = [
        _message("aaaa", offset=1),
        _message("bbbb", offset=2),
        _message("cccc", offset=3),
    ]

    old, recent = split_messages_for_compaction(
        messages,
        recent_token_budget=8,
        token_counter=FixedTokenCounter(),
    )

    assert old == messages[:1]
    assert recent == messages[1:]
    assert old + recent == messages


def test_split_does_not_duplicate_messages_between_old_and_recent():
    messages = [_message("aaaa", offset=1), _message("bbbb", offset=2)]

    old, recent = split_messages_for_compaction(
        messages,
        recent_token_budget=4,
        token_counter=FixedTokenCounter(),
    )

    assert {message.id for message in old}.isdisjoint(
        {message.id for message in recent}
    )
    assert sorted([*old, *recent], key=lambda message: message.created_at) == messages


def test_split_keeps_protected_message_even_when_over_budget():
    protected = _message("x" * 20, offset=2)
    messages = [_message("aaaa", offset=1), protected]

    old, recent = split_messages_for_compaction(
        messages,
        recent_token_budget=4,
        token_counter=FixedTokenCounter(),
        protected_message_ids={protected.id},
    )

    assert protected in recent
    assert old == [messages[0]]


def test_build_conversation_context_omits_current_request_and_empty_sections():
    recent = [
        ConversationMessage(
            id=uuid4(),
            role="assistant",
            content="Patient/123 was found.",
            created_at=datetime.now(timezone.utc),
        )
    ]

    context = build_conversation_context(
        summary="The user is checking Patient/123.",
        recent_messages=recent,
    )

    assert "SHORT-TERM CONVERSATION SUMMARY" in context
    assert "RECENT CONVERSATION" in context
    assert "ASSISTANT: Patient/123 was found." in context
    assert "CURRENT USER REQUEST" not in context


def test_token_counter_protocol_is_satisfied_by_fixed_counter():
    counter: TokenCounter = FixedTokenCounter()
    assert counter.count_text("abc") == 3
