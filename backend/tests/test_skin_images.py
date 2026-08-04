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
from app.skin_images.references import build_image_api_url, extract_binary_id
from app.skin_images.router import router
from app.skin_images.schemas import ResolvedSkinImageSearchFilters, SkinImageSearchFilters
from app.skin_images.search_filters import LOCAL_TZ, resolve_skin_image_filters
from app.services.skin_image_chat import maybe_answer_skin_image_request


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

    mocker.patch("app.skin_images.service.patient_exists", new=AsyncMock(return_value=False))
    process_mock = mocker.patch("app.skin_images.service.normalize_uploaded_skin_image", new=AsyncMock())
    save_mock = mocker.patch("app.skin_images.service.save_skin_analysis", new=AsyncMock())

    response = client.post(
        "/api/skin-images/analyze",
        data={"patient_id": "missing-patient"},
        files={"image": ("skin.jpg", b"fake-image", "image/jpeg")},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Patient was not found in Neo4j"
    process_mock.assert_not_awaited()
    save_mock.assert_not_awaited()


def test_extract_binary_id_accepts_id_relative_and_absolute_url():
    assert extract_binary_id("17761") == "17761"
    assert extract_binary_id("Binary/17761") == "17761"
    assert extract_binary_id("https://fhir.example.test/fhir/Binary/17761") == "17761"


def test_build_image_api_url_uses_binary_id_only():
    assert build_image_api_url("Binary/17761") == "/api/skin-images/files/17761"


def test_analyze_saves_skin_resources_for_selected_patient(mocker):
    app = _make_app(_make_user())
    client = TestClient(app)

    patient_mock = mocker.patch("app.skin_images.service.patient_exists", new=AsyncMock(return_value=True))
    mocker.patch(
        "app.skin_images.service.normalize_uploaded_skin_image",
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
        "app.skin_images.service.classify_skin_modality",
        new=AsyncMock(return_value=("XC", "Dermatology")),
    )
    mocker.patch(
        "app.skin_images.service.analyze_skin_image",
        new=AsyncMock(return_value="AI skin analysis"),
    )
    save_mock = mocker.patch(
        "app.skin_images.service.save_skin_analysis",
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
    assert data["image_url"].startswith("/api/skin-images/files/binary-")
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


def test_list_images_allows_doctor_with_external_id_to_filter_any_patient(mocker):
    app = _make_app(_make_user("patient-a"))
    client = TestClient(app)

    list_mock = mocker.patch(
        "app.skin_images.router.list_skin_images",
        new=AsyncMock(return_value=[]),
    )

    response = client.get("/api/skin-images?patient_id=patient-b")

    assert response.status_code == 200
    list_mock.assert_awaited_once_with("patient-b")


def test_get_image_file_returns_binary_data_from_neo4j(mocker):
    app = _make_app(_make_user())
    client = TestClient(app)

    mocker.patch(
        "app.skin_images.router.get_binary_for_skin_image",
        new=AsyncMock(
            return_value={
                "patient_id": "10796",
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


@pytest.mark.anyio
async def test_get_binary_query_keeps_patient_scope(mocker):
    execute_mock = mocker.patch(
        "app.skin_images.neo4j_repository.execute_cypher",
        new=AsyncMock(return_value=[]),
    )

    await neo4j_repository.get_binary_for_skin_image("binary-1")

    query = execute_mock.await_args.args[0]
    assert "WITH patient, content, code" in query
    assert "WITH patient, content, resolved" in query
    assert "RETURN patient.id AS patient_id" in query


def test_resolve_skin_image_filters_converts_today_to_utc_range():
    filters = SkinImageSearchFilters(patient_id="10796", date_range="today")
    now = datetime(2026, 8, 4, 8, 0, tzinfo=LOCAL_TZ)

    resolved = resolve_skin_image_filters(filters, now=now)

    assert resolved.patient_id == "10796"
    assert resolved.from_datetime.isoformat() == "2026-08-03T17:00:00+00:00"
    assert resolved.to_datetime.isoformat().startswith("2026-08-04T16:59:59")


@pytest.mark.anyio
async def test_search_patient_skin_images_starts_from_patient_and_limits(mocker):
    execute_mock = mocker.patch(
        "app.skin_images.neo4j_repository.execute_cypher",
        new=AsyncMock(return_value=[]),
    )
    filters = ResolvedSkinImageSearchFilters(patient_id="10796", modality="XC", count=3)

    await neo4j_repository.search_patient_skin_images(filters)

    query = execute_mock.await_args.args[0]
    params = execute_mock.await_args.args[1]
    assert "MATCH (patient:FHIRResource:Patient" in query
    assert "id: $patient_id" in query
    assert "LIMIT $count" in query
    assert "binary.data" not in query
    assert params["patient_id"] == "10796"
    assert params["modality"] == "XC"
    assert params["count"] == 3


@pytest.mark.anyio
async def test_skin_image_chat_request_returns_attachments(mocker):
    mocker.patch(
        "app.services.skin_image_chat.search_patient_skin_images",
        new=AsyncMock(
            return_value=[
                {
                    "patient_id": "10796",
                    "diagnostic_report_id": "report-1",
                    "media_id": "media-1",
                    "binary_id": "binary-1",
                    "url": "/api/skin-images/files/binary-1",
                    "content_type": "image/jpeg",
                    "created_at": "2026-08-04T02:00:00Z",
                    "conclusion": "AI conclusion",
                }
            ]
        ),
    )

    result = await maybe_answer_skin_image_request("Lấy ảnh da gần nhất của bệnh nhân 10796.")

    assert result.handled is True
    assert result.attachments
    assert result.attachments[0].binary_id == "binary-1"
    assert result.attachments[0].url == "/api/skin-images/files/binary-1"


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

