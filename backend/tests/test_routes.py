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
    with patch("app.context_graph_client.connect_neo4j", new_callable=AsyncMock), \
         patch("app.context_graph_client.close_neo4j", new_callable=AsyncMock), \
         patch("app.main.is_connected", return_value=True), \
         patch("app.main.init_memory", new_callable=AsyncMock), \
         patch("app.vector_client.create_vector_index", new_callable=AsyncMock):
        yield


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["domain"] == "healthcare"
    assert "neo4j" in data


def test_scenarios():
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert "domain" in data
    assert "scenarios" in data
    assert isinstance(data["scenarios"], list)
