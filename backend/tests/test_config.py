"""Unit tests for application configuration (backend/app/config.py)."""

from __future__ import annotations

import os

import pytest

from app.core.config import Settings


@pytest.fixture()
def clean_env(monkeypatch):
    """Ensure environment has only minimal required keys for local defaults."""
    monkeypatch.setenv("NEO4J_URI", "neo4j://localhost:7687")
    monkeypatch.setenv("INTERNAL_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("INTERNAL_LLM_MODEL", "test-model")
    monkeypatch.setenv("INTERNAL_EMBEDDING_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("INTERNAL_EMBEDDING_MODEL", "test-embed")
    # PostgreSQL defaults
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_USER", "postgres")
    monkeypatch.setenv("POSTGRES_DB", "fhir_agent")
    monkeypatch.setenv("MEM0_VECTOR_STORE_PROVIDER", "pgvector")
    monkeypatch.setenv("MEM0_COLLECTION_NAME", "fhir_agent_memories")
    monkeypatch.setenv("MEM0_AGENT_ID", "fhir-clinical-agent")


def test_postgres_settings_defaults():
    s = Settings()
    assert s.postgres_host == "localhost"
    assert s.postgres_port == 5432
    assert s.postgres_user == "postgres"
    assert s.postgres_db == "fhir_agent"


def test_postgres_settings_from_env(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "db.example.com")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_USER", "admin")
    monkeypatch.setenv("POSTGRES_DB", "custom_db")
    s = Settings()
    assert s.postgres_host == "db.example.com"
    assert s.postgres_port == 5433
    assert s.postgres_user == "admin"
    assert s.postgres_db == "custom_db"


def test_mem0_vector_store_provider_default():
    s = Settings()
    assert s.mem0_vector_store_provider == "pgvector"


def test_mem0_collection_name_not_empty():
    s = Settings()
    assert s.mem0_collection_name.strip()


def test_embedding_dims_must_be_positive():
    with pytest.raises(ValueError, match="INTERNAL_EMBEDDING_DIMS"):
        Settings.model_validate(
            {
                "neo4j_uri": "x",
                "internal_llm_base_url": "x",
                "internal_llm_model": "x",
                "internal_embedding_base_url": "x",
                "internal_embedding_model": "x",
                "internal_embedding_dims": 0,
                "postgres_host": "localhost",
                "postgres_port": 5432,
                "postgres_user": "postgres",
                "postgres_db": "fhir_agent",
                "mem0_vector_store_provider": "pgvector",
                "mem0_collection_name": "memories",
                "mem0_agent_id": "agent",
            }
        )


def test_postgres_port_range_valid():
    s = Settings()
    assert s.postgres_port == 5432  # default is valid


def test_postgres_port_range_invalid_high(monkeypatch):
    monkeypatch.setenv("POSTGRES_PORT", "70000")
    with pytest.raises(ValueError, match="POSTGRES_PORT"):
        Settings()


def test_postgres_port_range_invalid_low(monkeypatch):
    monkeypatch.setenv("POSTGRES_PORT", "0")
    with pytest.raises(ValueError, match="POSTGRES_PORT"):
        Settings()
