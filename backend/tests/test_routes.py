"""Tests for Healthcare Context Graph API."""

import os
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

os.environ.setdefault("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")

from app.main import app


@pytest.fixture(autouse=True)
def mock_backend():
    """Mock Neo4j connection so tests don't need a real database."""
    with patch("app.graph.client.connect_neo4j", new_callable=AsyncMock), \
         patch("app.graph.client.close_neo4j", new_callable=AsyncMock), \
         patch("app.main.is_connected", return_value=True), \
         patch("app.main.init_memory", new_callable=AsyncMock), \
         patch("app.main._postgres_available", True), \
         patch("app.main._pgvector_available", True), \
         patch("app.main._memory_available", True), \
         patch("app.graph.vector.create_vector_index", new_callable=AsyncMock):
        yield


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["domain"] == "healthcare"
    assert "neo4j" in data
    assert "postgres" in data
    assert "pgvector" in data
    assert "memory" in data


def test_scenarios():
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert "domain" in data
    assert "scenarios" in data
    assert isinstance(data["scenarios"], list)


def test_health_with_pgvector_status():
    """Health should include postgres and pgvector booleans."""
    from app.main import _postgres_available, _pgvector_available

    response = client.get("/health")
    data = response.json()
    assert isinstance(data["postgres"], bool)
    assert isinstance(data["pgvector"], bool)
    assert data["memory"] in ("mem0-pgvector", "disabled")


def test_health_degraded_when_neo4j_down():
    """When Neo4j is down, status should be degraded."""
    with patch("app.main.is_connected", return_value=False):
        response = client.get("/health")
    data = response.json()
    assert data["status"] == "degraded"
    assert data["neo4j"] is False


def test_health_status_ok_when_neo4j_up():
    """When Neo4j is up, status should be ok."""
    with patch("app.main.is_connected", return_value=True):
        response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert data["neo4j"] is True
