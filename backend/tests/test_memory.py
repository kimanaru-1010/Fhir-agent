"""Unit tests for the Mem0 memory adapter (backend/app/memory.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest_mock import MockerFixture

from app.memory import (
    _sanitize,
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
    mem.add = MagicMock(return_value={"results": []})

    with patch("app.memory._memory", mem):
        yield mem


@pytest.fixture()
def mock_mem0_error():
    """Return a mock Memory that raises on search/add."""
    mem = MagicMock()
    mem.search = MagicMock(side_effect=RuntimeError("mem0 unavailable"))
    mem.add = MagicMock(side_effect=RuntimeError("mem0 unavailable"))

    with patch("app.memory._memory", mem):
        yield mem


# ---------------------------------------------------------------------------
# _sanitize
# ---------------------------------------------------------------------------


def test_sanitize_empty():
    assert _sanitize("") == ""
    assert _sanitize("   ") == ""
    assert _sanitize(None) == ""


def test_sanitize_trims_and_truncates():
    assert _sanitize("  hello  ") == "hello"
    long = "x" * 10000
    result = _sanitize(long)
    assert len(result) == 4000


# ---------------------------------------------------------------------------
# init_memory — graceful failure
# ---------------------------------------------------------------------------


def test_init_memory_logs_on_failure():
    with patch("app.memory.asyncio.to_thread", side_effect=ValueError("bad config")), \
         patch("app.memory.logger") as mock_logger:
        pytest.importorskip("mem0")  # skip if mem0 not installed
        import asyncio

        asyncio.run(init_memory())
        mock_logger.exception.assert_called_once()


# ---------------------------------------------------------------------------
# search_memories — correct parameters
# ---------------------------------------------------------------------------


def test_search_memories_uses_correct_filters(mock_mem0, mocker: MockerFixture):
    mocker.patch("app.memory.get_memory", return_value=mock_mem0)

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
            "run_id": "chat-1",
        },
        top_k=8,
    )


def test_search_memories_empty_when_no_memory():
    with patch("app.memory._memory", None):
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
# save_conversation_memory — correct parameters
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
    )


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
    """Same session, different users — agent_id still matches but run_id is shared."""
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
