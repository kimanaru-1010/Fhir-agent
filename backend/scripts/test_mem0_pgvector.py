"""Smoke test for Mem0 + PostgreSQL pgvector integration.

Usage:
    uv run python scripts/test_mem0_pgvector.py

Exit code 0 = success, non-zero = failure.
"""

import asyncio
import os
import sys

# Ensure the backend root is on sys.path so `app` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.services.long_term_memory import init_memory, search_memories, save_conversation_memory


TEST_USER_ID = "smoke-test-user"
TEST_SESSION_ID = "smoke-test-session"
ADD_MSG = "The user prefers to receive answers in Vietnamese language."
SEARCH_QUERY = "What language does the user prefer to receive answers in?"


async def run() -> int:
    """Run the full add/search smoke test. Return 0 on success."""
    # -- Step 1: initialise Mem0 --
    print("[1/5] Initialising Mem0 with pgvector config ...", end=" ")
    ok = await init_memory()
    if not ok:
        print("FAIL â€” Mem0 initialisation failed")
        return 1
    print("OK")

    # -- Step 2: add a memory --
    print("[2/5] Adding memory (via save_conversation_memory) ...", end=" ")
    added = await save_conversation_memory(
        user_id=TEST_USER_ID,
        session_id=TEST_SESSION_ID,
        user_message="User prefers to receive answers in Vietnamese language",
        assistant_message=ADD_MSG,
    )
    if not added:
        print("FAIL â€” save_conversation_memory returned empty")
        return 1
    print(f"OK ({len(added)} event(s))")

    # -- Step 3: search for the memory --
    print("[3/5] Searching for the stored memory ...", end=" ")
    results = await search_memories(
        query="What language does the user prefer?",
        user_id=TEST_USER_ID,
        session_id=TEST_SESSION_ID,
        limit=5,
    )
    if not results:
        print("FAIL â€” search returned no results")
        return 1
    print(f"OK ({len(results)} result(s))")

    # -- Step 4: verify the added memory is found --
    print("[4/5] Verifying result contains the added memory ...", end=" ")
    found = False
    for r in results:
        # mem0 2.0.x returns 'memory' at top level or 'data' in payload
        content = r.get("memory", r.get("data", r.get("message", "")))
        if "Vietnamese" in content:
            found = True
            break
    if not found:
        print("FAIL â€” added memory not found in search results")
        print(f"DEBUG results were: {results}", file=sys.stderr)
        return 1
    print("OK")

    # -- Step 5: try to delete the test memory (best-effort) --
    print("[5/5] Attempting to delete test memory (best-effort) ...", end=" ")
    try:
        from mem0 import Memory

        mem: Memory = await asyncio.to_thread(
            Memory.from_config,
            {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": settings.internal_llm_model,
                        "api_key": settings.internal_llm_api_key or "internal",
                        "openai_base_url": settings.internal_llm_base_url,
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": settings.internal_embedding_model,
                        "api_key": settings.internal_embedding_api_key or "internal",
                        "openai_base_url": settings.internal_embedding_base_url,
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
                        "collection_name": f"{settings.mem0_collection_name}_{settings.internal_embedding_dims}d",
                        "embedding_model_dims": settings.internal_embedding_dims,
                    },
                },
            },
        )
        filters = {
            "user_id": TEST_USER_ID,
            "agent_id": settings.mem0_agent_id,
            "run_id": TEST_SESSION_ID,
        }
        all_mems = mem.get_all(filters=filters)
        if isinstance(all_mems, dict):
            items = all_mems.get("results", [])
        else:
            items = all_mems
        for item in items:
            mem_id = item.get("id") or item.get("memory_id")
            if mem_id:
                try:
                    mem.delete(mem_id)
                except Exception:
                    pass
        print("OK (cleaned up)")
    except Exception:
        print("OK (skip â€” cleanup best-effort)")

    print("\nAll smoke tests passed!")
    return 0


if __name__ == "__main__":
    rc = asyncio.run(run())
    sys.exit(rc)
