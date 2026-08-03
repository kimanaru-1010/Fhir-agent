"""Structured runtime tracing for the FHIR agent.

Writes one JSON object per line to component-specific files under backend/logs/debug.
Payloads are truncated and obvious secrets are redacted.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_DIR = Path(
    os.getenv(
        "FHIR_DEBUG_LOG_DIR",
        str(Path(__file__).resolve().parents[1] / "logs" / "debug"),
    )
)
_ENABLED = os.getenv("FHIR_DEBUG_TRACE", "true").lower() in {"1", "true", "yes", "on"}
_MAX_CHARS = int(os.getenv("FHIR_DEBUG_MAX_CHARS", "12000"))
_LOCK = threading.Lock()
logger = logging.getLogger(__name__)

_SECRET_KEYS = {
    "api_key", "authorization", "password", "token", "access_token",
    "refresh_token", "neo4j_password",
}


def _redact(value: Any, key: str = "") -> Any:
    if key.lower() in _SECRET_KEYS:
        return "***REDACTED***"

    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]

    if isinstance(value, str):
        if len(value) > _MAX_CHARS:
            return value[:_MAX_CHARS] + f"...[truncated {len(value) - _MAX_CHARS} chars]"
        return value

    return value


def trace(component: str, event: str, **data: Any) -> None:
    """Append a structured event to `<component>.jsonl` and `all.jsonl`."""
    if not _ENABLED:
        return

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "event": event,
        **_redact(data),
    }

    try:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            for filename in (f"{component}.jsonl", "all.jsonl"):
                with (_LOG_DIR / filename).open("a", encoding="utf-8") as file:
                    file.write(line)
    except Exception:
        logger.debug("Unable to write debug trace", exc_info=True)


def get_debug_log_dir() -> str:
    return str(_LOG_DIR.resolve())
