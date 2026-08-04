from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.db.models import User
from app.dependencies import auth as auth_dep
from app.skin_diagnostic.llm_client import strip_hidden_reasoning
from app.skin_images.fhir_builders import build_skin_analysis_bundle
from app.skin_images.image_processing import ProcessedImage
from app.skin_images import neo4j_repository
from app.skin_images.router import router


def _make_user(external_id: str | None = None) -> User:
    return User(
        id=uuid4(),
        username="doctor_user",
        password_hash="hashed",
        external_id=external_id,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _make_app(user: User | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api")

    async def _current_user():
        return user or _make_user()

    app.dependency_overrides[auth_dep.get_current_user] = _current_user
    return app


def test_analyze_requires_linked_patient_in_neo4j(mocker):
    app = _make_app(_make_user())
    client = TestClient(app)

    mocker.patch("app.skin_images.router.patient_exists", new=AsyncMock(return_value=False))
    process_mock = mocker.patch("app.skin_images.router.normalize_uploaded_skin_image", new=AsyncMock())
    save_mock = mocker.patch("app.skin_images.router.save_skin_analysis", new=AsyncMock())

    response = client.post(
        "/api/skin-images/analyze",
        data={"patient_id": "missing-patient"},
        files={"image": ("skin.jpg", b"fake-image", "image/jpeg")},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Patient was not found in Neo4j"
    process_mock.assert_not_awaited()
    save_mock.assert_not_awaited()


def test_analyze_saves_skin_resources_for_selected_patient(mocker):
    app = _make_app(_make_user())
    client = TestClient(app)

    patient_mock = mocker.patch("app.skin_images.router.patient_exists", new=AsyncMock(return_value=True))
    mocker.patch(
        "app.skin_images.router.normalize_uploaded_skin_image",
        new=AsyncMock(
            return_value=ProcessedImage(
                raw=b"normalized-image",
                content_type="image/jpeg",
                data_uri="data:image/jpeg;base64,bm9ybWFsaXplZC1pbWFnZQ==",
                size=16,
            )
        ),
    )
    mocker.patch(
        "app.skin_images.router.classify_skin_modality",
        new=AsyncMock(return_value=("XC", "Dermatology")),
    )
    mocker.patch(
        "app.skin_images.router.analyze_skin_image",
        new=AsyncMock(return_value="AI skin analysis"),
    )
    save_mock = mocker.patch(
        "app.skin_images.router.save_skin_analysis",
        new=AsyncMock(
            return_value={
                "binary_id": "binary-1",
                "media_id": "media-1",
                "diagnostic_report_id": "report-1",
            }
        ),
    )

    response = client.post(
        "/api/skin-images/analyze",
        data={"patient_id": "10796"},
        files={"image": ("skin.jpg", b"fake-image", "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["binary_id"] == "binary-1"
    assert data["media_id"] == "media-1"
    assert data["diagnostic_report_id"] == "report-1"
    assert data["analysis_text"] == "AI skin analysis"
    save_mock.assert_awaited_once()
    _, kwargs = save_mock.await_args
    assert kwargs["patient_id"] == "10796"
    assert kwargs["patient_already_validated"] is True
    patient_mock.assert_awaited_once_with("10796")
    resources = save_mock.await_args.args[0]
    binary = next(resource for resource in resources if resource["resourceType"] == "Binary")
    assert binary["contentType"] == "image/jpeg"
    assert binary["data"] == "bm9ybWFsaXplZC1pbWFnZQ=="
    assert "storagePath" not in binary


def test_list_images_allows_doctor_to_list_all(mocker):
    app = _make_app(_make_user())
    client = TestClient(app)

    list_mock = mocker.patch(
        "app.skin_images.router.list_skin_images",
        new=AsyncMock(
            return_value=[
                {
                    "diagnostic_report_id": "report-a",
                    "media_id": "media-a",
                    "binary_id": "binary-a",
                    "modality": "XC",
                    "conclusion": "Only patient A data",
                    "image_url": "/api/skin-images/files/image-a",
                    "created_at": "2026-08-04T00:00:00Z",
                }
            ]
        ),
    )

    response = client.get("/api/skin-images")

    assert response.status_code == 200
    assert response.json()["items"][0]["diagnostic_report_id"] == "report-a"
    list_mock.assert_awaited_once_with(None)


def test_list_images_can_filter_by_patient_id(mocker):
    app = _make_app(_make_user())
    client = TestClient(app)

    list_mock = mocker.patch(
        "app.skin_images.router.list_skin_images",
        new=AsyncMock(return_value=[]),
    )

    response = client.get("/api/skin-images?patient_id=patient-a")

    assert response.status_code == 200
    list_mock.assert_awaited_once_with("patient-a")


def test_get_image_file_returns_binary_data_from_neo4j(mocker):
    app = _make_app(_make_user())
    client = TestClient(app)

    mocker.patch(
        "app.skin_images.router.get_binary_for_skin_image",
        new=AsyncMock(
            return_value={
                "binary_id": "binary-1",
                "data": "bm9ybWFsaXplZC1pbWFnZQ==",
                "content_type": "image/jpeg",
            }
        ),
    )

    response = client.get("/api/skin-images/files/binary-1")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"normalized-image"


def test_build_skin_analysis_bundle_preserves_resource_order():
    binary = {"resourceType": "Binary", "id": "binary-1"}
    media = {"resourceType": "Media", "id": "media-1"}
    report = {"resourceType": "DiagnosticReport", "id": "report-1"}

    bundle = build_skin_analysis_bundle([binary, media, report])

    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert len(bundle["entry"]) == 3
    assert bundle["entry"][0]["resource"] is binary
    assert bundle["entry"][1]["resource"] is media
    assert bundle["entry"][2]["resource"] is report


def test_strip_hidden_reasoning_removes_think_blocks():
    text = "<think>internal analysis</think>\n1. Lesion description\n2. Differential diagnosis"

    assert strip_hidden_reasoning(text) == "1. Lesion description\n2. Differential diagnosis"


@pytest.mark.anyio
async def test_repository_saves_resources_through_cyfhir_bundle(mocker):
    execute_mock = mocker.patch(
        "app.skin_images.neo4j_repository.execute_cypher",
        new=AsyncMock(
            return_value=[
                {
                    "value": {
                        "loadedResources": 3,
                        "skippedEntries": 0,
                        "referencesResolved": 3,
                        "referencesPending": 0,
                        "referencesAmbiguous": 0,
                    }
                }
            ]
        ),
    )

    result = await neo4j_repository.save_skin_analysis(
        [
            {"resourceType": "Binary", "id": "binary-1"},
            {"resourceType": "Media", "id": "media-1"},
            {"resourceType": "DiagnosticReport", "id": "report-1"},
        ],
        patient_id="10796",
        patient_already_validated=True,
    )

    assert result == {
        "binary_id": "binary-1",
        "media_id": "media-1",
        "diagnostic_report_id": "report-1",
    }
    execute_mock.assert_awaited_once()
    query = execute_mock.await_args.args[0]
    params = execute_mock.await_args.args[1]
    assert "CALL cyfhir.bundle.load" in query
    assert "cyfhir.resource.load" not in query
    assert "cyfhir.resource.resolve" not in query
    bundle_json = params["json"]
    assert '"resourceType": "Bundle"' in bundle_json
    assert '"resourceType": "Binary"' in bundle_json
    assert '"resourceType": "Media"' in bundle_json
    assert '"resourceType": "DiagnosticReport"' in bundle_json
    assert "MERGE (binary" not in query
    assert "CREATE (subject" not in query
