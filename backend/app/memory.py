"""Mem0 conversational memory backed by PostgreSQL + pgvector.

Neo4j remains the authoritative source of FHIR data. Mem0 stores only
sanitized user/assistant exchanges and never receives raw tool results.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import psycopg
from mem0 import Memory

from app.config import settings
from app.debug_trace import trace

logger = logging.getLogger(__name__)

_memory: Memory | None = None


def _collection_name() -> str:
    """Use a dimension-specific collection to avoid incompatible old vectors."""
    return (
        f"{settings.mem0_collection_name}_"
        f"{settings.internal_embedding_dims}d"
    )


def check_pgvector_connection() -> tuple[bool, bool]:
    """Return (postgres_available, pgvector_extension_enabled)."""
    try:
        conn = psycopg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            dbname="postgres",
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT EXISTS ("
            "  SELECT 1 FROM pg_database WHERE datname = %s"
            ")",
            (settings.postgres_db,),
        )
        db_exists = cur.fetchone()[0]
        cur.close()
        conn.close()

        if not db_exists:
            logger.warning(
                'Database "%s" does not exist in PostgreSQL.',
                settings.postgres_db,
            )
            return (True, False)

        conn2 = psycopg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            dbname=settings.postgres_db,
        )
        cur2 = conn2.cursor()
        cur2.execute(
            "SELECT EXISTS ("
            "  SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            ")"
        )
        has_vector = cur2.fetchone()[0]
        cur2.close()
        conn2.close()

        if not has_vector:
            logger.warning(
                'pgvector extension is not enabled for database "%s". '
                "Run: CREATE EXTENSION IF NOT EXISTS vector;",
                settings.postgres_db,
            )
            return (True, False)

        return (True, True)
    except Exception:
        logger.exception("Failed to connect to PostgreSQL/pgvector")
        return (False, False)


def _build_mem0_config() -> dict[str, Any]:
    """Build Mem0 configuration for internal OpenAI-compatible services."""
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": settings.internal_llm_model,
                "api_key": settings.internal_llm_api_key or "internal",
                "openai_base_url": settings.internal_llm_base_url,
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": settings.internal_embedding_model,
                "api_key": settings.internal_embedding_api_key or "internal",
                "openai_base_url": settings.internal_embedding_base_url,
                # Do not pass embedding_dims here.
                # BAAI/bge-m3 rejects the OpenAI `dimensions` parameter.
            },
        },
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "host": settings.postgres_host,
                "port": settings.postgres_port,
                "user": settings.postgres_user,
                "password": settings.postgres_password,
                "dbname": settings.postgres_db,
                "collection_name": _collection_name(),
                "embedding_model_dims": settings.internal_embedding_dims,
            },
        },
    }


def _normalize_mem0_results(result: Any) -> list[dict[str, Any]]:
    """Normalize common Mem0 result shapes into a list."""
    if isinstance(result, dict):
        values = result.get("results", result.get("memories", []))
        return values if isinstance(values, list) else []
    if isinstance(result, list):
        return result
    return []


def _get_all_memories_sync(
    mem: Memory,
    *,
    user_id: str,
    session_id: str,
) -> Any:
    """Read the current Mem0/pgvector contents for one scoped conversation.

    Mem0 has used more than one get_all signature across releases, so this
    helper tries the current filters form first and then the older keyword form.
    """
    filters = {
        "user_id": user_id,
        "agent_id": settings.mem0_agent_id,
        "run_id": session_id,
    }

    try:
        return mem.get_all(filters=filters)
    except TypeError:
        return mem.get_all(
            user_id=user_id,
            agent_id=settings.mem0_agent_id,
            run_id=session_id,
        )


async def log_memory_snapshot(
    *,
    user_id: str,
    session_id: str,
    reason: str,
) -> list[dict[str, Any]]:
    """Log all memory records currently visible in this Mem0/pgvector scope."""
    mem = get_memory()
    if mem is None:
        trace(
            "memory",
            "pgvector_snapshot_skipped",
            reason=reason,
            detail="Mem0 is not initialized",
            user_id=user_id,
            session_id=session_id,
        )
        return []

    try:
        raw_snapshot = _get_all_memories_sync(
            mem,
            user_id=user_id,
            session_id=session_id,
        )
        normalized = _normalize_mem0_results(raw_snapshot)

        trace(
            "memory",
            "pgvector_snapshot",
            reason=reason,
            user_id=user_id,
            session_id=session_id,
            record_count=len(normalized),
            raw_result=raw_snapshot,
            records=normalized,
        )
        return normalized
    except Exception as exc:
        trace(
            "memory",
            "pgvector_snapshot_error",
            reason=reason,
            user_id=user_id,
            session_id=session_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        logger.warning("Unable to read Mem0/pgvector snapshot", exc_info=True)
        return []


async def init_memory() -> bool:
    """Initialize Mem0 and return whether startup succeeded."""
    global _memory

    try:
        config = _build_mem0_config()
        _memory = await asyncio.to_thread(Memory.from_config, config)

        logger.info(
            "Mem0 initialized: llm_model=%s embedding_model=%s "
            "embedding_dims=%s vector_store=%s collection=%s",
            settings.internal_llm_model,
            settings.internal_embedding_model,
            settings.internal_embedding_dims,
            settings.mem0_vector_store_provider,
            _collection_name(),
        )
        return True
    except Exception:
        _memory = None
        logger.exception(
            "Failed to initialize Mem0; conversational memory disabled"
        )
        return False


def get_memory() -> Memory | None:
    """Return the initialized Mem0 instance, if available."""
    return _memory


def _sanitize(text: str) -> str:
    """Apply a conservative size boundary before persistence."""
    return (text or "").strip()[:]

def _strip_reasoning(text: str) -> str:
    """Remove model reasoning blocks before saving to Mem0."""
    text = text or ""

    if "</think>" in text:
        text = text.split("</think>", 1)[-1]

    return text.strip()

async def search_memories(
    *,
    query: str,
    user_id: str,
    session_id: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Search relevant memories and log the full retrieval result."""
    mem = get_memory()
    clean_query = _sanitize(query)

    if mem is None or not clean_query:
        trace(
            "memory",
            "search_skipped",
            user_id=user_id,
            session_id=session_id,
            reason="Mem0 unavailable or query empty",
        )
        return []

    filters = {
        "user_id": user_id,
        "agent_id": settings.mem0_agent_id,
        "run_id": session_id,
    }

    try:
        trace(
            "memory",
            "search_start",
            query=clean_query,
            filters=filters,
            top_k=max(1, min(limit, 20)),
        )

        # mem.search() is synchronous (handles blocking I/O internally).
        result = mem.search(
            query=clean_query,
            filters=filters,
            top_k=max(1, min(limit, 20)),
        )

        # This is the exact object returned by mem.search().
        trace(
            "memory",
            "search_raw_result",
            query=clean_query,
            filters=filters,
            raw_result=result,
        )

        normalized = _normalize_mem0_results(result)

        trace(
            "memory",
            "search_success",
            query=clean_query,
            filters=filters,
            result_count=len(normalized),
            results=normalized,
        )

        # Optional full snapshot lets you compare retrieved matches with all
        # records currently stored for the same user/session.
        await log_memory_snapshot(
            user_id=user_id,
            session_id=session_id,
            reason="after_search",
        )

        return normalized
    except Exception as exc:
        trace(
            "memory",
            "search_error",
            query=clean_query,
            filters=filters,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        logger.warning("Mem0 search failed", exc_info=True)
        return []


async def save_conversation_memory(
    *,
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> list[dict[str, Any]]:
    """Store one exchange and log both Mem0's result and pgvector contents."""
    mem = get_memory()
    clean_user = _sanitize(user_message)
    clean_assistant = _strip_reasoning(_sanitize(assistant_message))

    if mem is None or not clean_user or not clean_assistant:
        trace(
            "memory",
            "save_skipped",
            user_id=user_id,
            session_id=session_id,
            reason="Mem0 unavailable or message empty",
        )
        return []

    messages = [
        {"role": "user", "content": clean_user},
        {"role": "assistant", "content": clean_assistant},
    ]

    try:
        trace(
            "memory",
            "save_start",
            user_id=user_id,
            session_id=session_id,
            agent_id=settings.mem0_agent_id,
            messages=messages,
        )

        # mem.add() is synchronous (handles blocking I/O internally).
        result = mem.add(
            messages,
            user_id=user_id,
            agent_id=settings.mem0_agent_id,
            run_id=session_id,
        )

        # Exact return value from mem.add(), including ADD/UPDATE/DELETE events.
        trace(
            "memory",
            "save_raw_result",
            user_id=user_id,
            session_id=session_id,
            raw_result=result,
        )

        normalized = _normalize_mem0_results(result)

        trace(
            "memory",
            "save_success",
            user_id=user_id,
            session_id=session_id,
            result_count=len(normalized),
            results=normalized,
        )

        # Read back the scoped records after writing so the log shows what is
        # actually visible through Mem0's pgvector-backed store.
        await log_memory_snapshot(
            user_id=user_id,
            session_id=session_id,
            reason="after_save",
        )

        return normalized
    except Exception as exc:
        trace(
            "memory",
            "save_error",
            user_id=user_id,
            session_id=session_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        logger.warning("Mem0 save failed", exc_info=True)
        return []
