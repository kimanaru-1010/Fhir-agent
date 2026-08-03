"""Conversation summarization for short-term memory compaction."""

from __future__ import annotations

from typing import Protocol

from openai import AsyncOpenAI

from app.core.config import settings
from app.services.token_counter import ApproximateTokenCounter, ConversationMessage, TokenCounter


class ConversationSummarizer(Protocol):
    async def summarize(
        self,
        *,
        previous_summary: str,
        messages: list[ConversationMessage],
    ) -> str:
        ...


class LLMConversationSummarizer:
    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.client = client or AsyncOpenAI(
            base_url=settings.internal_llm_base_url,
            api_key=settings.internal_llm_api_key or "internal",
        )
        self.token_counter = token_counter or ApproximateTokenCounter()

    async def summarize(
        self,
        *,
        previous_summary: str,
        messages: list[ConversationMessage],
    ) -> str:
        if not messages:
            return previous_summary.strip()

        summary = await self._run_completion(
            self._build_prompt(previous_summary=previous_summary, messages=messages),
            max_tokens=settings.short_term_summary_max_tokens,
        )
        summary = self._validate_summary(summary, messages=messages)
        if self.token_counter.count_text(summary) <= settings.short_term_summary_max_tokens:
            return summary

        compressed = await self._run_completion(
            (
                "Compress the conversation summary below. Preserve IDs, clinical facts, "
                "constraints, unresolved questions, and uncertainty. Do not invent facts.\n\n"
                f"{summary}"
            ),
            max_tokens=settings.short_term_summary_max_tokens,
        )
        compressed = self._validate_summary(compressed, messages=messages)
        if self.token_counter.count_text(compressed) <= settings.short_term_summary_max_tokens:
            return compressed

        return self._truncate_to_summary_budget(compressed)

    async def _run_completion(self, prompt: str, *, max_tokens: int) -> str:
        response = await self.client.chat.completions.create(
            model=settings.internal_llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _build_prompt(
        self,
        *,
        previous_summary: str,
        messages: list[ConversationMessage],
    ) -> str:
        formatted_messages = "\n\n".join(
            f"[{message.created_at.isoformat()}] {message.role.upper()}:\n{message.content}"
            for message in messages
        )
        return f"""Update the existing conversation summary using only the messages below.

Preserve:
- the user's current clinical or data-retrieval goal;
- important conclusions that were explicitly stated;
- FHIR resource types and resource IDs;
- patient, encounter, medication, condition, observation, and other named entities;
- relevant references, node IDs, and paths when they appeared in the conversation;
- user corrections, selections, constraints, and decisions;
- unresolved questions or unfinished tasks;
- uncertainty, missing data, and conflicting evidence.

Remove:
- greetings and acknowledgements;
- repeated wording;
- presentation details;
- long raw JSON when a concise factual description is sufficient;
- internal reasoning;
- unsupported assumptions.

Rules:
- Be concise and factual.
- Do not invent clinical facts.
- Do not change IDs, codes, dates, quantities, or statuses.
- Treat Neo4j/FHIR data as time-sensitive; record what was previously found,
  not that it is permanently true.
- Do not include information that is absent from the existing summary and messages.

Existing summary:
{previous_summary.strip() or "(empty)"}

Messages:
{formatted_messages}
"""

    def _validate_summary(
        self,
        summary: str,
        *,
        messages: list[ConversationMessage],
    ) -> str:
        summary = summary.strip()
        if messages and not summary:
            raise ValueError("Summarizer returned an empty summary")
        return summary

    def _truncate_to_summary_budget(self, summary: str) -> str:
        max_chars = int(
            settings.short_term_summary_max_tokens
            * settings.short_term_chars_per_token
        )
        if len(summary) <= max_chars:
            return summary
        return summary[:max_chars].rstrip()
