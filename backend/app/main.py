"""Healthcare Context Graph â€” FastAPI Application."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.graph.client import connect_neo4j, close_neo4j, is_connected
from app.services.long_term_memory import check_pgvector_connection, init_memory
from app.api.graph import router

# Auth routes
from app.api.auth import auth_router, users_router
from app.api.conversations import router as conversations_router
from app.api.messages import router as messages_router
from app.skin_diagnostic.router import router as skin_diagnostic_router

logger = logging.getLogger(__name__)

_neo4j_available: bool = False
_memory_available: bool = False
_postgres_available: bool = False
_pgvector_available: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global _neo4j_available, _memory_available, _postgres_available, _pgvector_available

    # Check PostgreSQL and pgvector extension first
    _postgres_available, _pgvector_available = await asyncio.to_thread(
        check_pgvector_connection
    )

    # Connect to Neo4j (FHIR graph)
    try:
        await connect_neo4j()
        _neo4j_available = True
        logger.info("Neo4j connected successfully")
    except Exception as e:
        _neo4j_available = False
        logger.warning("Neo4j unavailable â€” starting in degraded mode: %s", e)

    if _neo4j_available:
        try:
            from app.graph.vector import create_vector_index
            await create_vector_index()
        except Exception as e:
            logger.warning("Vector index creation failed (non-fatal): %s", e)

    # Initialise Mem0 conversational memory (requires pgvector)
    if _postgres_available and _pgvector_available:
        _memory_available = await init_memory()
        if _memory_available:
            logger.info("Mem0 conversational memory initialized")
        else:
            logger.warning(
                "Mem0 unavailable; chat will run without conversational memory"
            )
    else:
        logger.warning(
            "PostgreSQL or pgvector unavailable; Mem0 not initialized"
        )

    try:
        from app.skin_diagnostic.session_store import get_store as get_skin_store

        skin_store = await get_skin_store()
        loaded_skin_runs = await skin_store.load_from_disk()
        if loaded_skin_runs:
            logger.info("Loaded %s persisted skin diagnostic run(s)", loaded_skin_runs)
    except Exception as e:
        logger.warning("Skin diagnostic session restore failed: %s", e)

    try:
        from utils.knowledge_base import warm_up_index

        await asyncio.sleep(2)
        await asyncio.to_thread(warm_up_index)
        logger.info("Skin diagnostic knowledge base index ready")
    except Exception as e:
        logger.warning(
            "Skin diagnostic knowledge base warm-up failed; will retry lazily: %s",
            e,
        )

    yield

    if _neo4j_available:
        await close_neo4j()


def get_neo4j_status() -> bool:
    """Check if Neo4j is available."""
    return _neo4j_available


def get_memory_status() -> bool:
    """Check if Mem0 is available."""
    return _memory_available


app = FastAPI(
    title="Healthcare Context Graph",
    description="Patient care, clinical encounters, diagnoses, treatments, and provider networks",
    version="0.1.0",
    lifespan=lifespan,
)


CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    f"http://localhost:{settings.frontend_port}",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(users_router, prefix="/api")
app.include_router(conversations_router, prefix="/api")
app.include_router(messages_router, prefix="/api")
app.include_router(skin_diagnostic_router, prefix="/api")


@app.get("/health")
async def health():
    """Return the current status of all required backend services."""
    neo4j_ok = is_connected()

    services_ok = (
        neo4j_ok
        and _postgres_available
        and _pgvector_available
        and _memory_available
    )

    return {
        "status": "ok" if services_ok else "degraded",
        "neo4j": neo4j_ok,
        "postgres": _postgres_available,
        "pgvector": _pgvector_available,
        "memory": "mem0-pgvector" if _memory_available else "disabled",
        "domain": "healthcare",
        "version": "0.2.0",
    }
