"""Unit tests for the Mem0 memory adapter (backend/app/memory.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest_mock import MockerFixture

from app.services.long_term_memory import (
    FHIR_MEMORY_EXTRACTION_PROMPT,
    _build_mem0_config,
    _sanitize,
    check_pgvector_connection,
    init_memory,
    save_conversation_memory,
    search_memories,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_mem0():
    """Return a mock Memory instance and patch get_memory to return it."""
    mem = MagicMock()
    mem.search = MagicMock(return_value={"results": []})
    mem.add = MagicMock(return_value={"results": [{"id": "1", "memory": "test", "event": "ADD"}]})

    with patch("app.services.long_term_memory._memory", mem):
        yield mem


@pytest.fixture()
def mock_mem0_error():
    """Return a mock Memory that raises on search/add."""
    mem = MagicMock()
    mem.search = MagicMock(side_effect=RuntimeError("mem0 unavailable"))
    mem.add = MagicMock(side_effect=RuntimeError("mem0 unavailable"))
    mem.get_all = MagicMock(return_value={"results": []})

    with patch("app.services.long_term_memory._memory", mem):
        yield mem


# ---------------------------------------------------------------------------
# _sanitize
# ---------------------------------------------------------------------------


def test_sanitize_empty():
    assert _sanitize("") == ""
    assert _sanitize("   ") == ""
    assert _sanitize(None) == ""


def test_sanitize_trims():
    assert _sanitize("  hello  ") == "hello"
    # _sanitize trims whitespace but does not truncate
    long = "x" * 10000
    result = _sanitize(long)
    assert len(result) == 10000


# ---------------------------------------------------------------------------
# init_memory â€” graceful failure
# ---------------------------------------------------------------------------


def test_init_memory_logs_on_failure():
    with patch("app.services.long_term_memory.asyncio.to_thread", side_effect=ValueError("bad config")), \
         patch("app.services.long_term_memory.logger") as mock_logger:
        pytest.importorskip("mem0")  # skip if mem0 not installed
        import asyncio

        asyncio.run(init_memory())
        mock_logger.exception.assert_called_once()


# ---------------------------------------------------------------------------
# _build_mem0_config â€” pgvector provider
# ---------------------------------------------------------------------------


def test_build_mem0_config_uses_pgvector():
    config = _build_mem0_config()
    vs = config["vector_store"]
    assert vs["provider"] == "pgvector"


def test_build_mem0_config_has_pgvector_fields():
    config = _build_mem0_config()
    cfg = config["vector_store"]["config"]
    assert "host" in cfg
    assert "port" in cfg
    assert "user" in cfg
    assert "password" in cfg
    assert "dbname" in cfg
    assert "collection_name" in cfg
    assert "embedding_model_dims" in cfg


def test_build_mem0_config_no_qdrant_path():
    config = _build_mem0_config()
    cfg = config["vector_store"]["config"]
    assert "path" not in cfg


def test_build_mem0_config_embedding_model_dims_positive():
    config = _build_mem0_config()
    dims = config["embedder"]["config"]
    # embedding_model_dims is only in vector_store config, not embedder
    vs_dims = config["vector_store"]["config"]["embedding_model_dims"]
    assert vs_dims > 0


def test_build_mem0_config_collection_name_includes_dims():
    config = _build_mem0_config()
    cn = config["vector_store"]["config"]["collection_name"]
    assert "768d" in cn or "d" in cn.split("_")[-1]


# ---------------------------------------------------------------------------
# search_memories â€” correct parameters
# ---------------------------------------------------------------------------


def test_search_memories_uses_correct_filters(mock_mem0, mocker: MockerFixture):
    mocker.patch("app.services.long_term_memory.get_memory", return_value=mock_mem0)

    # Patch get_memory in the module scope
    import asyncio

    asyncio.run(
        search_memories(
            query="Patient/123",
            user_id="doctor-1",
            session_id="chat-1",
            limit=8,
        )
    )

    mock_mem0.search.assert_called_once_with(
        query="Patient/123",
        filters={
            "user_id": "doctor-1",
            "agent_id": "fhir-clinical-agent",
        },
        top_k=5,
    )


def test_search_memories_filters_low_scores(
    mock_mem0,
    mocker: MockerFixture,
):
    mock_mem0.search.return_value = {
        "results": [
            {"id": "strong", "memory": "Relevant", "score": 0.7},
            {"id": "weak", "memory": "Noise", "score": 0.49},
            {"id": "unscored", "memory": "Compatible"},
        ]
    }
    mocker.patch("app.services.long_term_memory.get_memory", return_value=mock_mem0)

    import asyncio

    result = asyncio.run(
        search_memories(
            query="Patient/123",
            user_id="doctor-1",
            session_id="chat-1",
        )
    )

    assert [item["id"] for item in result] == ["strong", "unscored"]


def test_search_memories_empty_when_no_memory():
    with patch("app.services.long_term_memory._memory", None):
        import asyncio

        result = asyncio.run(
            search_memories(query="x", user_id="u", session_id="s")
        )
    assert result == []


def test_search_memories_returns_empty_on_error(mock_mem0_error):
    mock_mem0_error.search.side_effect = RuntimeError("mem0 unavailable")
    import asyncio

    result = asyncio.run(
        search_memories(query="x", user_id="u", session_id="s")
    )
    assert result == []


# ---------------------------------------------------------------------------
# save_conversation_memory â€” correct parameters
# ---------------------------------------------------------------------------


def test_save_conversation_memory_passes_correct_args(mock_mem0):
    import asyncio

    asyncio.run(
        save_conversation_memory(
            user_id="doctor-1",
            session_id="chat-1",
            user_message="Patient/123 has fever",
            assistant_message="Fever in Patient/123: 38.5 C",
        )
    )

    mock_mem0.add.assert_called_once_with(
        [
            {"role": "user", "content": "Patient/123 has fever"},
            {"role": "assistant", "content": "Fever in Patient/123: 38.5 C"},
        ],
        user_id="doctor-1",
        agent_id="fhir-clinical-agent",
        run_id="chat-1",
        prompt=FHIR_MEMORY_EXTRACTION_PROMPT,
    )


def test_save_conversation_memory_passes_custom_fhir_prompt(mock_mem0):
    import asyncio

    user_id = "doctor-1"
    session_id = "chat-1"

    asyncio.run(
        save_conversation_memory(
            user_id=user_id,
            session_id=session_id,
            user_message="  Tim Nguyen Van A.  ",
            assistant_message="  Da xac dinh Patient/123.  ",
        )
    )

    args, kwargs = mock_mem0.add.call_args
    assert args[0] == [
        {"role": "user", "content": "Tim Nguyen Van A."},
        {"role": "assistant", "content": "Da xac dinh Patient/123."},
    ]
    assert kwargs["prompt"] == FHIR_MEMORY_EXTRACTION_PROMPT
    assert kwargs["user_id"] == user_id
    assert kwargs["agent_id"] == "fhir-clinical-agent"
    assert kwargs["run_id"] == session_id


def test_save_conversation_memory_strips_reasoning_before_saving(mock_mem0):
    import asyncio

    asyncio.run(
        save_conversation_memory(
            user_id="doctor-1",
            session_id="chat-1",
            user_message="Tim Nguyen Van A.",
            assistant_message="<think>internal reasoning</think>\nDa xac dinh Patient/123.",
        )
    )

    messages = mock_mem0.add.call_args.args[0]
    assert messages == [
        {"role": "user", "content": "Tim Nguyen Van A."},
        {"role": "assistant", "content": "Da xac dinh Patient/123."},
    ]


def test_save_conversation_memory_no_memory_instance_returns_empty():
    import asyncio

    with patch("app.services.long_term_memory.get_memory", return_value=None):
        result = asyncio.run(
            save_conversation_memory(
                user_id="doctor-1",
                session_id="chat-1",
                user_message="Tim Nguyen Van A.",
                assistant_message="Da xac dinh Patient/123.",
            )
        )

    assert result == []


def test_save_conversation_memory_normalizes_results(mock_mem0):
    import asyncio

    memory = "[entity_context] The currently selected patient is Patient/123."
    mock_mem0.add.return_value = {"results": [{"memory": memory}]}

    result = asyncio.run(
        save_conversation_memory(
            user_id="doctor-1",
            session_id="chat-1",
            user_message="Tim Nguyen Van A.",
            assistant_message="Da xac dinh Patient/123.",
        )
    )

    assert result == [{"memory": memory}]


def test_save_conversation_memory_different_sessions(mock_mem0):
    """Two different sessions produce separate run_ids."""
    import asyncio

    asyncio.run(
        save_conversation_memory(
            user_id="doctor-1",
            session_id="chat-a",
            user_message="msg1",
            assistant_message="ans1",
        )
    )
    call_args_1 = mock_mem0.add.call_args
    assert call_args_1.kwargs["run_id"] == "chat-a"

    mock_mem0.add.reset_mock()
    asyncio.run(
        save_conversation_memory(
            user_id="doctor-1",
            session_id="chat-b",
            user_message="msg2",
            assistant_message="ans2",
        )
    )
    call_args_2 = mock_mem0.add.call_args
    assert call_args_2.kwargs["run_id"] == "chat-b"


def test_save_conversation_memory_different_users_same_session(mock_mem0):
    """Same session, different users â€” agent_id still matches but run_id is shared."""
    import asyncio

    asyncio.run(
        save_conversation_memory(
            user_id="doctor-1",
            session_id="chat-1",
            user_message="msg1",
            assistant_message="ans1",
        )
    )
    call_args_1 = mock_mem0.add.call_args
    assert call_args_1.kwargs["user_id"] == "doctor-1"

    mock_mem0.add.reset_mock()
    asyncio.run(
        save_conversation_memory(
            user_id="doctor-2",
            session_id="chat-1",
            user_message="msg2",
            assistant_message="ans2",
        )
    )
    call_args_2 = mock_mem0.add.call_args
    assert call_args_2.kwargs["user_id"] == "doctor-2"
    # run_id is the same session_id
    assert call_args_2.kwargs["run_id"] == "chat-1"


def test_save_conversation_memory_empty_messages_skipped(mock_mem0):
    import asyncio

    result = asyncio.run(
        save_conversation_memory(
            user_id="doctor-1",
            session_id="chat-1",
            user_message="   ",
            assistant_message="ans",
        )
    )
    mock_mem0.add.assert_not_called()
    assert result == []


def test_save_conversation_memory_empty_assistant_skipped(mock_mem0):
    import asyncio

    result = asyncio.run(
        save_conversation_memory(
            user_id="doctor-1",
            session_id="chat-1",
            user_message="msg",
            assistant_message="",
        )
    )
    mock_mem0.add.assert_not_called()
    assert result == []


def test_save_conversation_memory_returns_empty_on_error(mock_mem0_error):
    mock_mem0_error.add.side_effect = RuntimeError("fail")
    import asyncio

    result = asyncio.run(
        save_conversation_memory(
            user_id="u",
            session_id="s",
            user_message="msg",
            assistant_message="ans",
        )
    )
    assert result == []


# ---------------------------------------------------------------------------
# check_pgvector_connection â€” unit tests (mocked)
# ---------------------------------------------------------------------------


def test_check_pgvector_connection_success():
    """When PostgreSQL is reachable and vector extension exists, return (True, True)."""
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.fetchone.return_value = (True,)

    mock_psycopg = MagicMock()
    mock_psycopg.connect.return_value = mock_conn

    with patch("app.services.long_term_memory._import_psycopg", return_value=mock_psycopg):
        result = check_pgvector_connection()
    assert result == (True, True)


def test_check_pgvector_connection_no_vector_ext():
    """When vector extension is missing, return (True, False)."""
    mock_conn = MagicMock()
    # First call: db_exists = True
    # Second call: has_vector = False
    mock_conn.cursor.return_value.fetchone.side_effect = [
        (True,),   # db exists
        (False,),  # no vector extension
    ]

    mock_psycopg = MagicMock()
    mock_psycopg.connect.return_value = mock_conn

    with patch("app.services.long_term_memory._import_psycopg", return_value=mock_psycopg):
        result = check_pgvector_connection()
    assert result == (True, False)


def test_check_pgvector_connection_connect_fails():
    """When connection raises, return (False, False)."""
    mock_psycopg = MagicMock()
    mock_psycopg.connect.side_effect = RuntimeError("conn refused")

    with patch("app.services.long_term_memory._import_psycopg", return_value=mock_psycopg):
        result = check_pgvector_connection()
    assert result == (False, False)
