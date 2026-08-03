"""Tests for the short-term LLM summarizer wrapper."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import anyio
import pytest

from app.core.config import settings
from app.services.short_term_summarizer import LLMConversationSummarizer
from app.services.token_counter import ConversationMessage


class _Completions:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.outputs.pop(0)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                )
            ]
        )


class _Client:
    def __init__(self, outputs: list[str]):
        self.completions = _Completions(outputs)
        self.chat = SimpleNamespace(completions=self.completions)


def _message(content: str, role: str = "user") -> ConversationMessage:
    return ConversationMessage(
        id=uuid4(),
        role=role,
        content=content,
        created_at=datetime(2026, 7, 29, 8, 0, tzinfo=timezone.utc),
    )


def test_summarizer_builds_fhir_prompt_and_uses_configured_generation_settings():
    async def run_test():
        client = _Client(["Patient/123 was found."])
        summarizer = LLMConversationSummarizer(client=client)

        result = await summarizer.summarize(
            previous_summary="Existing Patient/123 context.",
            messages=[_message("Find Nguyen Van A")],
        )

        assert result == "Patient/123 was found."
        call = client.completions.calls[0]
        assert call["model"] == settings.internal_llm_model
        assert call["temperature"] == 0
        assert call["max_tokens"] == settings.short_term_summary_max_tokens
        prompt = call["messages"][0]["content"]
        assert "Existing Patient/123 context." in prompt
        assert "Find Nguyen Van A" in prompt
        assert "Do not invent clinical facts" in prompt

    anyio.run(run_test)


def test_summarizer_rejects_empty_summary_for_non_empty_messages():
    async def run_test():
        summarizer = LLMConversationSummarizer(client=_Client([""]))

        with pytest.raises(ValueError):
            await summarizer.summarize(
                previous_summary="",
                messages=[_message("Hello")],
            )

    anyio.run(run_test)


def test_summarizer_runs_one_compression_pass_for_over_budget_summary(monkeypatch):
    async def run_test():
        monkeypatch.setattr(settings, "short_term_summary_max_tokens", 2)
        client = _Client(["A very long summary", "Short"])
        summarizer = LLMConversationSummarizer(client=client)

        result = await summarizer.summarize(
            previous_summary="",
            messages=[_message("Long conversation")],
        )

        assert result == "Short"
        assert len(client.completions.calls) == 2

    anyio.run(run_test)
