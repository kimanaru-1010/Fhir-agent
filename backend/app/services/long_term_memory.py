"""Mem0 conversational memory backed by PostgreSQL + pgvector.

Neo4j remains the authoritative source of FHIR data. Mem0 stores only
sanitized user/assistant exchanges and never receives raw tool results.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mem0 import Memory
else:
    Memory = Any

from app.core.config import settings
from app.core.debug_trace import trace

logger = logging.getLogger(__name__)


def _import_psycopg():
    import psycopg

    return psycopg


def _import_memory_class():
    from mem0 import Memory

    return Memory

_LEGACY_FHIR_MEMORY_EXTRACTION_PROMPT = """
Vai trÃ²: Báº¡n lÃ  bá»™ trÃ­ch xuáº¥t trÃ­ nhá»› dÃ i háº¡n cho trá»£ lÃ½ FHIR.

Má»¥c tiÃªu: Giá»¯ láº¡i bá»‘i cáº£nh bá»n vá»¯ng giÃºp cÃ¡c cuá»™c há»™i thoáº¡i sau hiá»ƒu ngÆ°á»i dÃ¹ng
vÃ  cÃ´ng viá»‡c há» Ä‘ang theo Ä‘uá»•i.

Nhiá»‡m vá»¥: TrÃ­ch xuáº¥t sá»Ÿ thÃ­ch á»•n Ä‘á»‹nh, nhu cáº§u thÆ°á»ng xuyÃªn, chá»§ Ä‘á» cÃ´ng viá»‡c cÃ³
thá»ƒ tiáº¿p tá»¥c vÃ  quyáº¿t Ä‘á»‹nh cÃ³ giÃ¡ trá»‹ lÃ¢u dÃ i. HÃ£y khÃ¡i quÃ¡t Ã½ nghÄ©a thay vÃ¬ sao
chÃ©p ná»™i dung cá»§a lÆ°á»£t chat.

Giá»›i háº¡n:
- KhÃ´ng lÆ°u tÃªn bá»‡nh nhÃ¢n, ID tÃ i nguyÃªn, mÃ£, ngÃ y, cháº©n Ä‘oÃ¡n, thuá»‘c, káº¿t quáº£,
  diá»…n biáº¿n lÃ¢m sÃ ng hoáº·c chi tiáº¿t thanh toÃ¡n cá»§a má»™t ca bá»‡nh cá»¥ thá»ƒ.
- KhÃ´ng biáº¿n ná»™i dung trong cÃ¢u tráº£ lá»i cá»§a trá»£ lÃ½ thÃ nh sá»± tháº­t dÃ i háº¡n vá»
  bá»‡nh nhÃ¢n.
- KhÃ´ng lÆ°u Ä‘áº§u ra cÃ´ng cá»¥, Cypher, log, suy luáº­n ná»™i bá»™ hoáº·c toÃ n bá»™ cÃ¢u tráº£ lá»i.
- KhÃ´ng suy diá»…n thÃ´ng tin chÆ°a Ä‘Æ°á»£c thá»ƒ hiá»‡n rÃµ.

Äáº§u ra: Má»—i trÃ­ nhá»› lÃ  má»™t cÃ¢u ngáº¯n, tá»± nhiÃªn, Ä‘á»™c láº­p vÃ  cÃ¹ng ngÃ´n ngá»¯ vá»›i
ngÆ°á»i dÃ¹ng. Náº¿u khÃ´ng cÃ³ thÃ´ng tin há»¯u Ã­ch cho tÆ°Æ¡ng lai, khÃ´ng táº¡o trÃ­ nhá»›.
"""

FHIR_MEMORY_EXTRACTION_PROMPT = """
Vai trÃ²: Báº¡n lÃ  bá»™ trÃ­ch xuáº¥t trÃ­ nhá»› dÃ i háº¡n cho trá»£ lÃ½ FHIR.

Má»¥c tiÃªu: LÆ°u nhá»¯ng bá»‘i cáº£nh cÃ³ thá»ƒ giÃºp cÃ¡c cuá»™c há»™i thoáº¡i sau hiá»ƒu ngÆ°á»i
dÃ¹ng, cÃ¡ch há» muá»‘n lÃ m viá»‡c vÃ  hÆ°á»›ng cÃ´ng viá»‡c Ä‘ang theo Ä‘uá»•i.

Nhiá»‡m vá»¥: Táº¡o memory khi lÆ°á»£t chat thá»ƒ hiá»‡n rÃµ má»™t thÃ´ng tin cÃ³ kháº£ nÄƒng tÃ¡i sá»­
dá»¥ng. Giá»¯ Ä‘á»§ ngá»¯ cáº£nh Ä‘á»ƒ memory cÃ²n cÃ³ nghÄ©a sau nÃ y, nhÆ°ng khÃ´ng sao chÃ©p toÃ n
bá»™ cÃ¢u há»i hoáº·c cÃ¢u tráº£ lá»i.

NÃªn lÆ°u:
- Sá»Ÿ thÃ­ch á»•n Ä‘á»‹nh cá»§a ngÆ°á»i dÃ¹ng vá» ngÃ´n ngá»¯, Ä‘á»™ dÃ i, Ä‘á»‹nh dáº¡ng, má»©c chi tiáº¿t
  hoáº·c cÃ¡ch trÃ¬nh bÃ y.
- Má»¥c tiÃªu cÃ´ng viá»‡c Ä‘ang theo Ä‘uá»•i á»Ÿ má»©c khÃ¡i quÃ¡t vá»«a Ä‘á»§.
- Pháº¡m vi phÃ¢n tÃ­ch, tiÃªu chÃ­ lá»c, loáº¡i dá»¯ liá»‡u hoáº·c loáº¡i tÃ i nguyÃªn FHIR mÃ 
  ngÆ°á»i dÃ¹ng thÆ°á»ng quan tÃ¢m.
- Quyáº¿t Ä‘á»‹nh hoáº·c quy Æ°á»›c cÃ³ thá»ƒ áº£nh hÆ°á»Ÿng Ä‘áº¿n cÃ¡c lÆ°á»£t há»i sau.
- Viá»‡c cÃ²n dang dá»Ÿ hoáº·c ngá»¯ cáº£nh cáº§n nhá»› Ä‘á»ƒ tiáº¿p tá»¥c cÃ´ng viá»‡c.

Má»©c Ä‘á»™ cá»¥ thá»ƒ:
- CÃ³ thá»ƒ giá»¯ cÃ¡c khÃ¡i niá»‡m miá»n nhÆ° bá»‡nh nhÃ¢n, lÆ°á»£t khÃ¡m, cháº©n Ä‘oÃ¡n, chá»‰ Ä‘á»‹nh,
  káº¿t quáº£, thuá»‘c, thanh toÃ¡n, claim, payment hoáº·c timeline náº¿u chÃºng mÃ´ táº£ loáº¡i
  cÃ´ng viá»‡c ngÆ°á»i dÃ¹ng muá»‘n lÃ m.
- KhÃ´ng lÆ°u giÃ¡ trá»‹ ca bá»‡nh cá»¥ thá»ƒ nhÆ° tÃªn bá»‡nh nhÃ¢n, FHIR id, mÃ£ bá»‡nh, ngÃ y,
  thuá»‘c, káº¿t quáº£ lÃ¢m sÃ ng, sá»‘ tiá»n hoáº·c káº¿t luáº­n thanh toÃ¡n nhÆ° má»™t sá»± tháº­t dÃ i
  háº¡n.
- Khi cáº§n nháº¯c Ä‘áº¿n má»™t ca bá»‡nh, hÃ£y mÃ´ táº£ á»Ÿ má»©c "má»™t bá»‡nh nhÃ¢n/ca bá»‡nh/lÆ°á»£t
  khÃ¡m Ä‘ang Ä‘Æ°á»£c phÃ¢n tÃ­ch" thay vÃ¬ Ä‘á»‹nh danh cá»¥ thá»ƒ.

KhÃ´ng lÆ°u:
- Lá»i chÃ o, cáº£m Æ¡n, xÃ¡c nháº­n ngáº¯n, cÃ¢u há»i má»™t láº§n khÃ´ng táº¡o bá»‘i cáº£nh má»›i.
- Ná»™i dung tool output, Cypher, log, lá»—i ká»¹ thuáº­t, suy luáº­n ná»™i bá»™ hoáº·c toÃ n bá»™
  cÃ¢u tráº£ lá»i cá»§a trá»£ lÃ½.
- ThÃ´ng tin do trá»£ lÃ½ nÃªu ra náº¿u ngÆ°á»i dÃ¹ng khÃ´ng xÃ¡c nháº­n hoáº·c nÃ³ chá»‰ lÃ  dá»¯
  liá»‡u lÃ¢m sÃ ng/financial cá»§a má»™t ca cá»¥ thá»ƒ.

NguyÃªn táº¯c suy luáº­n:
- Chá»‰ lÆ°u Ä‘iá»u Ä‘Æ°á»£c thá»ƒ hiá»‡n rÃµ trong user message hoáº·c Ä‘Æ°á»£c user xÃ¡c nháº­n.
- KhÃ´ng thÃªm vai trÃ², tÃ¡c nhÃ¢n, Ã½ Ä‘á»‹nh, quan há»‡, Ä‘á»‘i tÆ°á»£ng hoáº·c khÃ¡i niá»‡m mÃ 
  ngÆ°á»i dÃ¹ng khÃ´ng nÃ³i rÃµ.
- Náº¿u má»™t cÃ¢u cÃ³ thá»ƒ chá»‰ lÃ  há»i láº¡i ngá»¯ cáº£nh hiá»‡n táº¡i, chá»‰ lÆ°u khi nÃ³ bá»™c lá»™
  nhu cáº§u tÃ¡i sá»­ dá»¥ng rÃµ rÃ ng.
- Náº¿u khÃ´ng cháº¯c memory cÃ³ há»¯u Ã­ch lÃ¢u dÃ i hay khÃ´ng, khÃ´ng táº¡o memory.

Äáº§u ra: Má»—i memory lÃ  má»™t cÃ¢u ngáº¯n, tá»± nhiÃªn, Ä‘á»™c láº­p vÃ  cÃ¹ng ngÃ´n ngá»¯ vá»›i
ngÆ°á»i dÃ¹ng. Æ¯u tiÃªn 1-3 memory tháº­t sá»± há»¯u Ã­ch. Náº¿u khÃ´ng cÃ³ thÃ´ng tin dÃ i háº¡n
má»›i, khÃ´ng táº¡o memory.
"""
_LEGACY_FHIR_MEMORY_EXTRACTION_PROMPT_V2 = FHIR_MEMORY_EXTRACTION_PROMPT

FHIR_MEMORY_EXTRACTION_PROMPT = """
Vai trÃ²: Báº¡n lÃ  bá»™ trÃ­ch xuáº¥t trÃ­ nhá»› dÃ i háº¡n cho trá»£ lÃ½ FHIR.

Má»¥c tiÃªu: LÆ°u bá»‘i cáº£nh cÃ³ thá»ƒ tÃ¡i sá»­ dá»¥ng vá» ngÆ°á»i dÃ¹ng vÃ  cÃ´ng viá»‡c Ä‘ang lÃ m,
Ä‘á»§ cá»¥ thá»ƒ Ä‘á»ƒ nháº­n ra hoáº¡t Ä‘á»™ng lá»‹ch sá»­ nhÆ°ng khÃ´ng biáº¿n dá»¯ liá»‡u ca bá»‡nh thÃ nh
sá»± tháº­t dÃ i háº¡n.

NÃªn lÆ°u khi thÃ´ng tin rÃµ rÃ ng:
- Sá»Ÿ thÃ­ch á»•n Ä‘á»‹nh vá» ngÃ´n ngá»¯, Ä‘á»™ dÃ i, Ä‘á»‹nh dáº¡ng, má»©c chi tiáº¿t hoáº·c cÃ¡ch trÃ¬nh bÃ y.
- Hoáº¡t Ä‘á»™ng Ä‘ang diá»…n ra, vÃ­ dá»¥ Ä‘ang phÃ¢n tÃ­ch má»™t hÃ nh trÃ¬nh chÄƒm sÃ³c, danh sÃ¡ch
  cháº©n Ä‘oÃ¡n, chá»‰ Ä‘á»‹nh, káº¿t quáº£, thuá»‘c, claim/payment hoáº·c nhÃ³m tÃ i nguyÃªn FHIR.
- Pháº¡m vi, tiÃªu chÃ­ lá»c, quy Æ°á»›c hoáº·c quyáº¿t Ä‘á»‹nh mÃ  user muá»‘n Ã¡p dá»¥ng vá» sau.
- Má»™t má»‘c neo nháº¹ do user tá»± nÃªu Ä‘á»ƒ nháº­n diá»‡n hoáº¡t Ä‘á»™ng lá»‹ch sá»­, nhÆ° tÃªn ngÆ°á»i,
  tÃªn bá»‡nh nhÃ¢n, tÃªn case hoáº·c nhÃ£n chá»§ Ä‘á», chá»‰ khi má»‘c Ä‘Ã³ giÃºp tráº£ lá»i "Ä‘ang lÃ m
  gÃ¬/vá»›i ai".

Má»©c cá»¥ thá»ƒ cho phÃ©p:
- CÃ³ thá»ƒ giá»¯ má»™t tÃªn hoáº·c nhÃ£n Ä‘á»‹nh danh do user nÃ³i rÃµ, vÃ­ dá»¥ "Ä‘ang phÃ¢n tÃ­ch
  hÃ nh trÃ¬nh chÄƒm sÃ³c cá»§a bá»‡nh nhÃ¢n Ä‘Æ°á»£c nÃªu tÃªn".
- KhÃ´ng lÆ°u FHIR id, MRN, mÃ£ bá»‡nh, ngÃ y, thuá»‘c, káº¿t quáº£ lÃ¢m sÃ ng, sá»‘ tiá»n,
  káº¿t luáº­n thanh toÃ¡n hoáº·c chuá»—i Ä‘á»‹nh danh nháº¡y cáº£m.
- KhÃ´ng ghÃ©p má»‘c neo vá»›i dá»¯ kiá»‡n lÃ¢m sÃ ng/financial cá»¥ thá»ƒ thÃ nh memory bá»‡nh Ã¡n.

PhÃ¢n biá»‡t:
- Chá»‰ viáº¿t "user muá»‘n/Æ°a thÃ­ch" khi user nÃªu preference rÃµ hoáº·c láº·p láº¡i cÃ¹ng
  Ä‘á»‹nh dáº¡ng/cÃ¡ch lÃ m.
- Vá»›i má»™t nhiá»‡m vá»¥ Ä‘Æ¡n láº» nhÆ°ng cÃ³ thá»ƒ tiáº¿p tá»¥c, viáº¿t trung tÃ­nh nhÆ° "user Ä‘ang
  phÃ¢n tÃ­ch..." hoáº·c "user Ä‘ang truy váº¿t...".
- CÃ¢u tráº£ lá»i cá»§a assistant chá»‰ lÃ  báº±ng chá»©ng vá» ngá»¯ cáº£nh Ä‘ang lÃ m, khÃ´ng pháº£i
  preference hay fact dÃ i háº¡n trá»« khi user xÃ¡c nháº­n.

TrÃ¡nh suy diá»…n:
- KhÃ´ng thÃªm actor, vai trÃ², Ä‘á»‘i tÃ¡c, quan há»‡, nguyÃªn nhÃ¢n, má»¥c Ä‘Ã­ch hoáº·c pháº¡m vi
  náº¿u user khÃ´ng nÃ³i rÃµ.
- KhÃ´ng suy tá»« má»™t cÃ¢u há»i kiá»ƒm tra lá»‹ch sá»­ thÃ nh nhu cáº§u sáº£n pháº©m lÃ¢u dÃ i.
- KhÃ´ng sao chÃ©p toÃ n bá»™ cÃ¢u há»i, cÃ¢u tráº£ lá»i, tool output, Cypher, log, lá»—i ká»¹
  thuáº­t hoáº·c reasoning ná»™i bá»™.
- Náº¿u chá»‰ lÃ  lá»i chÃ o, cáº£m Æ¡n, xÃ¡c nháº­n ngáº¯n hoáº·c cÃ¢u há»i nháº¥t thá»i khÃ´ng thÃªm
  bá»‘i cáº£nh má»›i, khÃ´ng táº¡o memory.
- Náº¿u khÃ´ng cháº¯c memory cÃ³ há»¯u Ã­ch vá» sau hay khÃ´ng, khÃ´ng táº¡o memory.

Äáº§u ra: Táº¡o tá»‘i Ä‘a 1-3 memory ngáº¯n, tá»± nhiÃªn, Ä‘á»™c láº­p vÃ  cÃ¹ng ngÃ´n ngá»¯ vá»›i user.
Æ¯u tiÃªn memory khÃ¡i quÃ¡t vá»«a Ä‘á»§, cÃ³ má»‘c neo nháº¹ khi cáº§n. Náº¿u khÃ´ng cÃ³ thÃ´ng tin
dÃ i háº¡n má»›i, khÃ´ng táº¡o memory.
"""
_memory: Memory | None = None
_MEMORY_SEARCH_MAX_RESULTS = 5
_MEMORY_SEARCH_MIN_SCORE = 0.5


def _collection_name() -> str:
    """Use a dimension-specific collection to avoid incompatible old vectors."""
    return (
        f"{settings.mem0_collection_name}_"
        f"{settings.internal_embedding_dims}d"
    )


def check_pgvector_connection() -> tuple[bool, bool]:
    """Return (postgres_available, pgvector_extension_enabled)."""
    try:
        psycopg = _import_psycopg()
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


def _filter_relevant_memories(
    memories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep strong matches while tolerating Mem0 versions without scores."""
    relevant: list[dict[str, Any]] = []

    for memory in memories:
        score = memory.get("score")
        if score is None:
            relevant.append(memory)
            continue

        try:
            if float(score) >= _MEMORY_SEARCH_MIN_SCORE:
                relevant.append(memory)
        except (TypeError, ValueError):
            relevant.append(memory)

    return relevant[:_MEMORY_SEARCH_MAX_RESULTS]


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
        raw_snapshot = await asyncio.to_thread(
            _get_all_memories_sync,
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
        memory_class = _import_memory_class()
        _memory = await asyncio.to_thread(memory_class.from_config, config)

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
    limit: int = _MEMORY_SEARCH_MAX_RESULTS,
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
    }

    try:
        top_k = max(
            1,
            min(limit, _MEMORY_SEARCH_MAX_RESULTS),
        )

        trace(
            "memory",
            "search_start",
            query=clean_query,
            filters=filters,
            top_k=top_k,
            min_score=_MEMORY_SEARCH_MIN_SCORE,
        )

        # Mem0 exposes a synchronous API. Run it in a worker thread so the
        # FastAPI event loop remains responsive while Mem0 performs blocking I/O.
        result = await asyncio.to_thread(
            mem.search,
            query=clean_query,
            filters=filters,
            top_k=top_k,
        )

        # This is the exact object returned by mem.search().
        trace(
            "memory",
            "search_raw_result",
            query=clean_query,
            filters=filters,
            raw_result=result,
        )

        candidates = _normalize_mem0_results(result)
        normalized = _filter_relevant_memories(candidates)

        trace(
            "memory",
            "search_success",
            query=clean_query,
            filters=filters,
            result_count=len(normalized),
            candidate_count=len(candidates),
            dropped_count=len(candidates) - len(normalized),
            min_score=_MEMORY_SEARCH_MIN_SCORE,
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

        # Mem0 exposes a synchronous API. Run it in a worker thread so the
        # FastAPI event loop remains responsive while Mem0 performs blocking I/O.
        result = await asyncio.to_thread(
            mem.add,
            messages,
            user_id=user_id,
            agent_id=settings.mem0_agent_id,
            run_id=session_id,
            prompt=FHIR_MEMORY_EXTRACTION_PROMPT,
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
