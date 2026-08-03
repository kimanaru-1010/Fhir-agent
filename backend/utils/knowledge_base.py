"""Symptom knowledge base — supplements the differential list using vector
(embedding) search against a JSON-file KB, indexed in Qdrant.

Each disease entry has exactly two fields: `disease` (Vietnamese name) and
`symptom_keywords` (the only field that gets embedded and searched).
Keywords are intentionally restricted to visual and tactile observations
(morphology, color, texture, distribution) so the embeddings align closely
with the vision model's observations and can be queried with anamnesis +
observations combined.

Matches on the patient's initial complaint (anamnesis) combined with the
vision model's observations (visual_observations), both captured by the time
this runs — not on the later interview Q&A. Runs a single time, right after
the vision step, so the augmented differential list is already in place
before the planner generates its first round of questions.

All diseases live together in a single compact file, knowledge_base/diseases.json
(a JSON array of entries). See knowledge_base/README.md for the schema. The
Qdrant collection is rebuilt automatically whenever this file changes
(detected via a size + mtime signature).

Runs in Qdrant "local" mode by default (embedded, on-disk, no server needed —
see config/settings.py QDRANT_MODE). Switch to "server" mode by pointing
QDRANT_URL at a running Qdrant instance when the KB grows large enough to
need a real server / concurrent workers — no other code changes required.
"""

from __future__ import annotations

import json
import unicodedata
import uuid
from pathlib import Path

_KB_FILE = Path(__file__).parent.parent / "knowledge_base" / "diseases.json"
_UUID_NAMESPACE = uuid.UUID("6f2f9a2e-9c0a-4b3e-8f0a-0e6a4d5b7c11")  # fixed, arbitrary

_client = None
_indexed_signature = None  # (file_count, max_mtime) of the last successful index build
_vector_size = None


def _normalize(text: str) -> str:
    """Lowercase + strip Vietnamese accents — used only for disease-name dedup,
    not for the embedding search itself."""
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.strip()


def _point_id(disease: str) -> str:
    """Deterministic UUID from the normalized disease name — Qdrant point IDs
    must be int or UUID, not arbitrary strings."""
    return str(uuid.uuid5(_UUID_NAMESPACE, _normalize(disease)))


def _load_kb_files() -> list[dict]:
    """Load every disease entry from knowledge_base/diseases.json (a single
    JSON array). Malformed or incomplete entries are skipped rather than
    crashing the pipeline."""
    entries = []
    if not _KB_FILE.exists():
        return entries
    try:
        data = json.loads(_KB_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return entries
    if not isinstance(data, list):
        return entries
    for item in data:
        if isinstance(item, dict) and item.get("disease"):
            entries.append(item)
    return entries


def _symptom_text(entry: dict) -> str:
    """Build the text used for embedding — symptom_keywords ONLY.
    This is the sole field that gets searched (see module docstring)."""
    parts = list(entry.get("symptom_keywords", []))
    return ". ".join(p for p in parts if p)


def _kb_signature() -> tuple:
    """(size, mtime) of knowledge_base/diseases.json — used to detect edits
    without re-reading/parsing the whole file every call."""
    if not _KB_FILE.exists():
        return (0, 0.0)
    stat = _KB_FILE.stat()
    return (stat.st_size, stat.st_mtime)


def _get_client():
    global _client
    if _client is not None:
        return _client

    from qdrant_client import QdrantClient
    from config.settings import QDRANT_MODE, QDRANT_PATH, QDRANT_URL

    if QDRANT_MODE == "server":
        _client = QdrantClient(url=QDRANT_URL)
    else:
        path = Path(__file__).parent.parent / QDRANT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        _client = QdrantClient(path=str(path))
    return _client


def _ensure_index():
    """(Re)build the Qdrant collection if the KB files on disk have changed
    since the last build. No-op if nothing changed."""
    global _indexed_signature, _vector_size

    from qdrant_client.models import Distance, VectorParams, PointStruct
    from models.shared_client import embed_text
    from config.settings import QDRANT_COLLECTION, EMBEDDING_URL, EMBEDDING_MODEL, EMBEDDING_TIMEOUT

    client = _get_client()
    signature = _kb_signature()

    if signature == _indexed_signature and client.collection_exists(QDRANT_COLLECTION):
        return  # KB unchanged since last build — reuse existing index

    entries = _load_kb_files()
    if not entries:
        _indexed_signature = signature
        return

    texts = [_symptom_text(e) for e in entries]
    vectors = embed_text(texts, base_url=EMBEDDING_URL, model=EMBEDDING_MODEL, timeout=EMBEDDING_TIMEOUT)
    _vector_size = len(vectors[0])

    # Full rebuild (KB is small enough that this is simplest & handles
    # edits/removals correctly; revisit with incremental upserts if the KB
    # grows into the thousands of entries).
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        QDRANT_COLLECTION,
        vectors_config=VectorParams(size=_vector_size, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=_point_id(e["disease"]),
            vector=vec,
            payload={
                "disease": e["disease"],
            },
        )
        for e, vec in zip(entries, vectors)
    ]
    client.upsert(QDRANT_COLLECTION, points=points)

    _indexed_signature = signature


def warm_up_index() -> None:
    """Public entry point to (re)build the Qdrant index without needing a
    real query — call this once at server startup so the first real request
    doesn't pay the embedding/build cost. Safe to call repeatedly: it's a
    no-op if the KB files haven't changed since the last build (see
    _ensure_index)."""
    _ensure_index()


def match_kb_candidates(
    query_text: str,
    max_new: int = 5,
    min_score: float = 0.6,
) -> list[dict]:
    """Vector-search the KB using query_text (typically the patient's initial
    complaint combined with the vision model's observations) and return the
    top matching KB diseases. No deduplication against the LLM's own
    differential list — the two lists are simply combined by the caller,
    duplicates and all.

    min_score is cosine similarity (higher = more similar, max 1.0); results
    below this are considered too weak a match to add. Tune based on your
    embedding model — 0.7 is a reasonable starting point for bge-m3, but
    check a few real cases before trusting it in production.
    """
    if not query_text or not query_text.strip():
        return []

    from config.settings import QDRANT_COLLECTION, EMBEDDING_URL, EMBEDDING_MODEL, EMBEDDING_TIMEOUT
    from models.shared_client import embed_text

    _ensure_index()
    client = _get_client()
    if not client.collection_exists(QDRANT_COLLECTION):
        return []

    query_vector = embed_text(
        [query_text], base_url=EMBEDDING_URL, model=EMBEDDING_MODEL, timeout=EMBEDDING_TIMEOUT
    )[0]

    count = client.count(QDRANT_COLLECTION).count
    if count == 0:
        return []

    limit = min(count, max_new)
    results = client.query_points(
        collection_name=QDRANT_COLLECTION, query=query_vector, limit=limit
    ).points

    candidates = []
    for point in results:
        if point.score < min_score:
            continue
        payload = point.payload or {}
        disease = payload.get("disease", "")
        if not disease:
            continue

        candidates.append({
            "disease": disease,
            "score": point.score,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:max_new]