# Backend Package Layout

The backend is organized by responsibility. Keep new logic inside the package
that owns the behavior; the `app` package root should stay thin.

## Packages

- `app.core`
  - application settings, constants, and tracing helpers
- `app.agents`
  - LLM/agent implementations
- `app.graph`
  - Neo4j/FHIR graph access, graph algorithms, vector index helpers, and graph-domain models
- `app.api`
  - FastAPI routers
- `app.db`
  - SQLAlchemy models and database session wiring
- `app.dependencies`
  - FastAPI dependency providers
- `app.schemas`
  - request/response schemas shared by routers
- `app.services`
  - application services, chat orchestration, memory, auth, and streaming helpers
- `app.skin_diagnostic`
  - skin-lesion diagnostic workflow and its local knowledge base

## Root Package

The root contains only:

- `main.py` for FastAPI app creation and lifespan wiring
- `__init__.py`
- this README

Do not add feature modules directly under `app/`.
