"""In-memory run store for the skin diagnostic workflow.

Runs are persisted as JSON snapshots so a browser refresh does not lose the
diagnostic state. This is intentionally separate from the chat Conversation
tables; it can be migrated into Postgres after the workflow stabilizes.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings


DATA_DIR = Path(__file__).resolve().parent / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SkinDiagnosticRun:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    status: str = "idle"
    current_step: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    pending_question: dict | None = None
    pending_questions: list[dict] | None = None
    pending_answers: list[dict] = field(default_factory=list)
    pending_answer: str | None = None
    image_path: str = ""
    image_url: str = ""
    anamnesis: str = ""
    error: str | None = None
    step_history: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str = ""

    def __post_init__(self) -> None:
        if not self.expires_at:
            expires = datetime.now(timezone.utc) + timedelta(hours=settings.skin_session_ttl_hours)
            self.expires_at = expires.isoformat()


class SkinDiagnosticStore:
    def __init__(self) -> None:
        self._runs: dict[str, SkinDiagnosticRun] = {}
        self._lock = asyncio.Lock()

    def _path(self, run_id: str) -> Path:
        return SESSIONS_DIR / f"{run_id}.json"

    def _persist(self, run: SkinDiagnosticRun) -> None:
        try:
            self._path(run.id).write_text(
                json.dumps(asdict(run), ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _remove_snapshot(self, run_id: str) -> None:
        try:
            self._path(run_id).unlink(missing_ok=True)
        except OSError:
            pass

    async def load_from_disk(self) -> int:
        loaded = 0
        async with self._lock:
            for path in SESSIONS_DIR.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    run = SkinDiagnosticRun(**data)
                    self._runs[run.id] = run
                    loaded += 1
                except (OSError, TypeError, json.JSONDecodeError):
                    continue
        return loaded

    async def create(
        self,
        *,
        user_id: str,
        image_path: str,
        image_url: str,
        anamnesis: str,
        run_id: str | None = None,
    ) -> SkinDiagnosticRun:
        async with self._lock:
            run = SkinDiagnosticRun(
                id=run_id or str(uuid.uuid4()),
                user_id=user_id,
                image_path=image_path,
                image_url=image_url,
                anamnesis=anamnesis,
            )
            self._runs[run.id] = run
            self._persist(run)
            return run

    async def get(self, run_id: str, *, user_id: str | None = None) -> SkinDiagnosticRun | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            if user_id is not None and run.user_id != user_id:
                return None
            return run

    async def update(self, run_id: str, **kwargs: Any) -> bool:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return False
            new_step = kwargs.get("current_step")
            for key, value in kwargs.items():
                if hasattr(run, key):
                    setattr(run, key, value)
            if new_step and (not run.step_history or run.step_history[-1] != new_step):
                run.step_history.append(new_step)
            self._persist(run)
            return True

    async def delete(self, run_id: str, *, user_id: str) -> bool:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.user_id != user_id:
                return False
            self._runs.pop(run_id)
            self._remove_snapshot(run_id)
            if run.image_path:
                try:
                    Path(run.image_path).unlink(missing_ok=True)
                except OSError:
                    pass
            return True


_store = SkinDiagnosticStore()


async def get_store() -> SkinDiagnosticStore:
    return _store
