"""Optional Qdrant-backed symptom knowledge base for skin diseases."""

from __future__ import annotations

import json
import unicodedata
import uuid
from pathlib import Path

from app.core.config import settings
from app.skin_diagnostic.llm_client import embed_text


KB_FILE = Path(__file__).resolve().parent / "data" / "diseases.json"
_UUID_NAMESPACE = uuid.UUID("6f2f9a2e-9c0a-4b3e-8f0a-0e6a4d5b7c11")
_client = None
_indexed_signature = None


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", (text or "").lower())
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn").strip()


def _point_id(disease: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, _normalize(disease)))


def _load_entries() -> list[dict]:
    if not KB_FILE.exists():
        return []
    try:
        data = json.loads(KB_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict) and item.get("disease")]


def _symptom_text(entry: dict) -> str:
    keywords = entry.get("symptom_keywords", [])
    return ". ".join(str(item) for item in keywords if item)


def _signature() -> tuple[int, float]:
    if not KB_FILE.exists():
        return (0, 0.0)
    stat = KB_FILE.stat()
    return (stat.st_size, stat.st_mtime)


def _get_client():
    global _client
    if _client is not None:
        return _client
    from qdrant_client import QdrantClient

    _client = QdrantClient(url=settings.skin_qdrant_url)
    return _client


async def ensure_index() -> None:
    global _indexed_signature

    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = _get_client()
    signature = _signature()
    collection = settings.skin_qdrant_collection
    if signature == _indexed_signature and client.collection_exists(collection):
        return

    entries = _load_entries()
    if not entries:
        _indexed_signature = signature
        return

    vectors = await embed_text([_symptom_text(entry) for entry in entries])
    if not vectors:
        return

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )
    client.upsert(
        collection,
        points=[
            PointStruct(
                id=_point_id(entry["disease"]),
                vector=vector,
                payload={"disease": entry["disease"]},
            )
            for entry, vector in zip(entries, vectors)
        ],
    )
    _indexed_signature = signature


async def match_kb_candidates(
    *,
    query_text: str,
    max_new: int = 5,
    min_score: float | None = None,
) -> list[dict]:
    if not query_text.strip() or not settings.skin_kb_enabled:
        return []

    threshold = settings.skin_kb_min_score if min_score is None else min_score
    await ensure_index()

    client = _get_client()
    collection = settings.skin_qdrant_collection
    if not client.collection_exists(collection):
        return []

    query_vector = (await embed_text([query_text]))[0]
    count = client.count(collection).count
    if count == 0:
        return []

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=min(max_new, count),
    ).points

    matches = []
    for point in results:
        if point.score < threshold:
            continue
        disease = (point.payload or {}).get("disease", "")
        if disease:
            matches.append({"disease": disease, "score": point.score})
    return matches

