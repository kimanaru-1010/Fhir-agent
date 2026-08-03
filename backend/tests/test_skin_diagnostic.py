"""Tests for the skin diagnostic workflow API."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.db.models import User
from app.dependencies import auth as auth_dep
from app.skin_diagnostic.answer_validation import normalize_yes_no
from app.skin_diagnostic.llm_client import _normalize_openai_base_url, _resolve_chat_model
from app.skin_diagnostic.prompts import DIAGNOSE_PROMPT, PLANNER_PROMPT
from app.skin_diagnostic.router import router as skin_router
from app.skin_diagnostic.session_store import _store


def _make_user(user_id=None) -> User:
    return User(
        id=user_id or uuid4(),
        username="skin_user",
        password_hash="hashed",
        external_id=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture(autouse=True)
def clear_skin_store():
    _store._runs.clear()
    yield
    _store._runs.clear()


def _make_app(user: User | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(skin_router, prefix="/api")

    async def _current_user():
        return user or _make_user()

    app.dependency_overrides[auth_dep.get_current_user] = _current_user
    return app


def test_normalize_yes_no_accepts_english_and_vietnamese():
    assert normalize_yes_no("yes") == "Yes"
    assert normalize_yes_no("co") == "Yes"
    assert normalize_yes_no("khong") == "No"
    assert normalize_yes_no("no") == "No"


def test_normalize_yes_no_rejects_ambiguous_answer():
    with pytest.raises(Exception):
        normalize_yes_no("maybe", question_num=3)


def test_openai_base_url_adds_v1_when_missing():
    assert _normalize_openai_base_url("http://localhost:8000") == "http://localhost:8000/v1"
    assert _normalize_openai_base_url("http://localhost:8000/v1") == "http://localhost:8000/v1"


def test_resolve_chat_model_accepts_provider_prefixed_id(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"data":[{"id":"BAAI/bge-m3"},{"id":"google/gemma-4-26B-A4B-it"}]}'

    monkeypatch.setattr("app.skin_diagnostic.llm_client._model_cache", {})
    monkeypatch.setattr(
        "app.skin_diagnostic.llm_client.urllib.request.urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )

    assert (
        _resolve_chat_model("http://localhost:8000", "gemma-4-26B-A4B-it")
        == "google/gemma-4-26B-A4B-it"
    )


def test_planner_prompt_formats_json_example():
    prompt = PLANNER_PROMPT.format(
        complaint="itchy light patch",
        visual_findings="hypopigmented patch",
        visual_differentials="vitiligo",
        exam_only_hints="",
    )

    assert '"verified_findings"' in prompt
    assert '"questions"' in prompt


def test_diagnose_prompt_formats_json_example():
    prompt = DIAGNOSE_PROMPT.format(
        complaint="itchy light patch",
        image_path="/tmp/image.jpg",
        updated_visual_findings="hypopigmented patch",
        visual_differentials="vitiligo",
        exam_only_hints="",
        qa_history="Q: itchy? A: Yes",
    )

    assert '"ranked_diagnoses"' in prompt
    assert '"overall_reasoning"' in prompt


def test_start_skin_diagnostic_creates_run():
    user = _make_user()
    client = TestClient(_make_app(user))

    with patch(
        "app.skin_diagnostic.router.run_pipeline_background",
        new=AsyncMock(),
    ) as runner:
        response = client.post(
            "/api/skin-diagnostics/start",
            files={"image": ("lesion.jpg", b"fake-image", "image/jpeg")},
            data={"anamnesis": "itchy red lesion"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["current_step"] == "visual_extract"
    assert data["run_id"]
    assert data["run_id"] in _store._runs
    assert _store._runs[data["run_id"]].user_id == str(user.id)
    assert runner.called


def test_get_status_returns_only_owner_run():
    owner = _make_user()
    other = _make_user()
    client = TestClient(_make_app(owner))

    with patch("app.skin_diagnostic.router.run_pipeline_background", new=AsyncMock()):
        started = client.post(
            "/api/skin-diagnostics/start",
            files={"image": ("lesion.jpg", b"fake-image", "image/jpeg")},
            data={"anamnesis": "itchy red lesion"},
        ).json()

    response = client.get(f"/api/skin-diagnostics/{started['run_id']}/status")
    assert response.status_code == 200

    other_client = TestClient(_make_app(other))
    response = other_client.get(f"/api/skin-diagnostics/{started['run_id']}/status")
    assert response.status_code == 404


def test_submit_answers_is_idempotent_when_not_waiting():
    user = _make_user()
    client = TestClient(_make_app(user))

    with patch("app.skin_diagnostic.router.run_pipeline_background", new=AsyncMock()):
        run_id = client.post(
            "/api/skin-diagnostics/start",
            files={"image": ("lesion.jpg", b"fake-image", "image/jpeg")},
            data={"anamnesis": "itchy red lesion"},
        ).json()["run_id"]

    response = client.post(
        f"/api/skin-diagnostics/{run_id}/answers",
        json={"answers": [{"question_num": 1, "answer": "yes"}]},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "idle"


def test_submit_answers_queues_normalized_answers():
    user = _make_user()
    client = TestClient(_make_app(user))

    with patch("app.skin_diagnostic.router.run_pipeline_background", new=AsyncMock()):
        run_id = client.post(
            "/api/skin-diagnostics/start",
            files={"image": ("lesion.jpg", b"fake-image", "image/jpeg")},
            data={"anamnesis": "itchy red lesion"},
        ).json()["run_id"]

    run = _store._runs[run_id]
    run.status = "interrupt"
    run.current_step = "user_interview_round1"
    run.pending_questions = [{"question_num": 1, "question": "Does it itch?"}]

    with patch("app.skin_diagnostic.router.resume_pipeline", new=AsyncMock()) as resume:
        response = client.post(
            f"/api/skin-diagnostics/{run_id}/answers",
            json={"answers": [{"question_num": 1, "answer": "co"}]},
        )

    assert response.status_code == 200
    assert run.pending_answers == [{"question_num": 1, "answer": "Yes"}]
    assert run.status == "running"
    assert run.pending_questions is None
    assert resume.called


def test_submit_answers_ignores_duplicate_submit_while_processing():
    user = _make_user()
    client = TestClient(_make_app(user))

    with patch("app.skin_diagnostic.router.run_pipeline_background", new=AsyncMock()):
        run_id = client.post(
            "/api/skin-diagnostics/start",
            files={"image": ("lesion.jpg", b"fake-image", "image/jpeg")},
            data={"anamnesis": "itchy red lesion"},
        ).json()["run_id"]

    run = _store._runs[run_id]
    run.status = "interrupt"
    run.current_step = "user_interview_round1"
    run.pending_questions = [{"question_num": 1, "question": "Does it itch?"}]
    run.pending_answers = [{"question_num": 1, "answer": "Yes"}]

    with patch("app.skin_diagnostic.router.resume_pipeline", new=AsyncMock()) as resume:
        response = client.post(
            f"/api/skin-diagnostics/{run_id}/answers",
            json={"answers": [{"question_num": 1, "answer": "no"}]},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert run.pending_answers == [{"question_num": 1, "answer": "Yes"}]
    assert not resume.called
