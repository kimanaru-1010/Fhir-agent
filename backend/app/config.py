"""Application configuration from environment variables."""

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from ``.env``."""

    # Neo4j — authoritative FHIR graph
    neo4j_uri: str = ""
    neo4j_username: str = ""
    neo4j_password: str = ""

    # Internal OpenAI-compatible chat API used by the main agent and Mem0
    internal_llm_base_url: str = ""
    internal_llm_api_key: str = ""
    internal_llm_model: str = ""

    # Internal OpenAI-compatible embedding API.
    # It may be the same endpoint as the chat API or a separate service.
    internal_embedding_base_url: str = ""
    internal_embedding_api_key: str = ""
    internal_embedding_model: str = ""
    internal_embedding_dims: int = 768

    # PostgreSQL — Mem0 pgvector store
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "fhir_agent"

    # Mem0 conversational memory
    mem0_agent_id: str = "fhir-clinical-agent"
    mem0_vector_store_provider: str = "pgvector"
    mem0_collection_name: str = "fhir_agent_memories"

    # Application
    domain_id: str = "healthcare"
    backend_port: int = 8000
    frontend_port: int = 3000

    model_config = {
        "env_file": "../.env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @model_validator(mode="after")
    def _validate_required_settings(self):
        required = {
            "NEO4J_URI": self.neo4j_uri,
            "INTERNAL_LLM_BASE_URL": self.internal_llm_base_url,
            "INTERNAL_LLM_MODEL": self.internal_llm_model,
            "INTERNAL_EMBEDDING_BASE_URL": self.internal_embedding_base_url,
            "INTERNAL_EMBEDDING_MODEL": self.internal_embedding_model,
            "POSTGRES_HOST": self.postgres_host,
            "POSTGRES_PORT": str(self.postgres_port),
            "POSTGRES_USER": self.postgres_user,
            "POSTGRES_DB": self.postgres_db,
            "MEM0_VECTOR_STORE_PROVIDER": self.mem0_vector_store_provider,
            "MEM0_COLLECTION_NAME": self.mem0_collection_name,
            "MEM0_AGENT_ID": self.mem0_agent_id,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"Missing required settings: {', '.join(missing)}")
        if self.internal_embedding_dims <= 0:
            raise ValueError("INTERNAL_EMBEDDING_DIMS must be greater than zero")
        if self.postgres_port <= 0 or self.postgres_port > 65535:
            raise ValueError("POSTGRES_PORT must be between 1 and 65535")
        return self


settings = Settings()
