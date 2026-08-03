"""OpenAI-compatible model helpers for the skin diagnostic workflow."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import urllib.request
from pathlib import Path

from app.core.config import settings


_model_cache: dict[tuple[str, str], str] = {}
_NON_LLM_KEYWORDS = ("bge", "embed", "rerank", "e5", "nomic", "minilm", "all-mpnet")


def _normalize_openai_base_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    return clean if clean.endswith("/v1") else f"{clean}/v1"


def _model_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _is_embedding_model(model_id: str) -> bool:
    name = model_id.lower()
    return any(keyword in name for keyword in _NON_LLM_KEYWORDS)


def _resolve_chat_model(base_url: str, configured_model: str) -> str:
    """Resolve configured model against an OpenAI-compatible /models list.

    The upstream skin repo dynamically chooses an active LLM. This keeps that
    behavior so values like "gemma-4-26B-A4B-it" still work when the server
    exposes the id as "google/gemma-4-26B-A4B-it".
    """
    cache_key = (base_url.rstrip("/"), configured_model)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    resolved = configured_model
    try:
        models_url = f"{_normalize_openai_base_url(base_url)}/models"
        with urllib.request.urlopen(models_url, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        llm_models = [
            item.get("id", "")
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id") and not _is_embedding_model(item["id"])
        ]
        if llm_models:
            configured_key = _model_key(configured_model)
            exact = [item for item in llm_models if item == configured_model]
            fuzzy = [item for item in llm_models if configured_key and configured_key in _model_key(item)]
            resolved = (exact or fuzzy or llm_models)[0]
    except Exception:
        resolved = configured_model

    _model_cache[cache_key] = resolved
    return resolved


def image_to_data_uri(image_path: str) -> str:
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


async def call_llm(
    *,
    prompt: str,
    base_url: str,
    model: str,
    image_path: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=_normalize_openai_base_url(base_url),
        api_key=settings.internal_llm_api_key or "internal",
    )

    content: list[dict] = [{"type": "text", "text": prompt}]
    if image_path:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}})

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    resolved_model = _resolve_chat_model(base_url, model)
    response = await client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


async def embed_text(texts: list[str]) -> list[list[float]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=_normalize_openai_base_url(
            settings.skin_embedding_base_url or settings.internal_embedding_base_url
        ),
        api_key=settings.skin_embedding_api_key or settings.internal_embedding_api_key or "internal",
    )
    response = await client.embeddings.create(
        model=settings.skin_embedding_model or settings.internal_embedding_model,
        input=texts,
    )
    data = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in data]
