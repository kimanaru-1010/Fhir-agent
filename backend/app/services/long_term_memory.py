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

FHIR_MEMORY_EXTRACTION_PROMPT = """
Vai trò: Bạn là bộ trích xuất trí nhớ dài hạn cho trợ lý FHIR.

Mục tiêu: Lưu bối cảnh có thể tái sử dụng về người dùng và công việc đang làm,
đủ cụ thể để nhận ra hoạt động lịch sử nhưng không biến dữ liệu ca bệnh thành
sự thật dài hạn.

Nên lưu khi thông tin rõ ràng:
- Sở thích ổn định về ngôn ngữ, độ dài, định dạng, mức chi tiết hoặc cách trình bày.
- Hoạt động đang diễn ra, ví dụ đang phân tích một hành trình chăm sóc, danh sách
  chẩn đoán, chỉ định, kết quả, thuốc, claim/payment hoặc nhóm tài nguyên FHIR.
- Phạm vi, tiêu chí lọc, quy ước hoặc quyết định mà user muốn áp dụng về sau.
- Một mốc neo nhẹ do user tự nêu để nhận diện hoạt động lịch sử, như tên người,
  tên bệnh nhân, Patient ID, case ID hoặc nhãn chủ đề, chỉ khi mốc đó giúp trả lời
  "đang làm gì/với ai".

Mức cụ thể cho phép:
- Có thể giữ một tên hoặc nhãn định danh do user nói rõ, ví dụ "đang phân tích
  hành trình chăm sóc của bệnh nhân được nêu tên".
- Có thể lưu một ID định danh tối thiểu do user nêu rõ, ví dụ Patient/10796 hoặc
  "patient_id=10796", khi ID đó chỉ dùng để nối tiếp đúng đối tượng/case đang nghiên cứu.
- Không lưu MRN, mã bệnh, ngày, thuốc, kết quả lâm sàng, số tiền, kết luận thanh toán
  hoặc chuỗi định danh nhạy cảm khác.
- Không ghép mốc neo hoặc ID với dữ kiện lâm sàng/financial cụ thể thành memory bệnh án.

Phân biệt:
- Chỉ viết "user muốn/ưa thích" khi user nêu preference rõ hoặc lặp lại cùng
  định dạng/cách làm.
- Với một nhiệm vụ đơn lẻ nhưng có thể tiếp tục, viết trung tính như "user đang
  phân tích..." hoặc "user đang truy vết...".
- Câu trả lời của assistant chỉ là bằng chứng về ngữ cảnh đang làm, không phải
  preference hay fact dài hạn trừ khi user xác nhận.

Tránh suy diễn:
- Không thêm actor, vai trò, đối tác, quan hệ, nguyên nhân, mục đích hoặc phạm vi
  nếu user không nói rõ.
- Không suy từ một câu hỏi kiểm tra lịch sử thành nhu cầu sản phẩm lâu dài.
- Không sao chép toàn bộ câu hỏi, câu trả lời, tool output, Cypher, log, lỗi kỹ
  thuật hoặc reasoning nội bộ.
- Nếu chỉ là lời chào, cảm ơn, xác nhận ngắn hoặc câu hỏi nhất thời không thêm
  bối cảnh mới, không tạo memory.
- Nếu không chắc memory có hữu ích về sau hay không, không tạo memory.

Đầu ra: Tạo tối đa 1-3 memory ngắn, tự nhiên, độc lập và cùng ngôn ngữ với user.
Ưu tiên memory khái quát vừa đủ, có mốc neo nhẹ khi cần. Nếu không có thông tin
dài hạn mới, không tạo memory.
"""
_memory: Memory | None = None
_MEMORY_INPUT_MAX_CHARS = 12000
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


def _memory_ids(memories: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for memory in memories:
        memory_id = memory.get("id")
        if memory_id:
            ids.append(str(memory_id))
    return ids


def _memory_event_types(memories: list[dict[str, Any]]) -> list[str]:
    events: list[str] = []
    for memory in memories:
        event = memory.get("event")
        if event:
            events.append(str(event))
    return events


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
            result_ids=_memory_ids(normalized),
            event_types=_memory_event_types(normalized),
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
    return (text or "").strip()[:_MEMORY_INPUT_MAX_CHARS]

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
            query_length=len(clean_query),
            user_id=user_id,
            agent_id=settings.mem0_agent_id,
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

        trace(
            "memory",
            "search_result",
            query_length=len(clean_query),
            user_id=user_id,
            agent_id=settings.mem0_agent_id,
            result_count=len(_normalize_mem0_results(result)),
        )

        candidates = _normalize_mem0_results(result)
        normalized = _filter_relevant_memories(candidates)

        trace(
            "memory",
            "search_success",
            query_length=len(clean_query),
            user_id=user_id,
            agent_id=settings.mem0_agent_id,
            result_count=len(normalized),
            candidate_count=len(candidates),
            dropped_count=len(candidates) - len(normalized),
            min_score=_MEMORY_SEARCH_MIN_SCORE,
            result_ids=_memory_ids(normalized),
            event_types=_memory_event_types(normalized),
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
            query_length=len(clean_query),
            user_id=user_id,
            agent_id=settings.mem0_agent_id,
            error_type=type(exc).__name__,
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
            user_message_length=len(clean_user),
            assistant_message_length=len(clean_assistant),
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

        trace(
            "memory",
            "save_result",
            user_id=user_id,
            session_id=session_id,
            result_count=len(_normalize_mem0_results(result)),
        )

        normalized = _normalize_mem0_results(result)

        trace(
            "memory",
            "save_success",
            user_id=user_id,
            session_id=session_id,
            result_count=len(normalized),
            result_ids=_memory_ids(normalized),
            event_types=_memory_event_types(normalized),
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
        )
        logger.warning("Mem0 save failed", exc_info=True)
        return []
