from __future__ import annotations

import uuid
import base64
import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import Response

from app.db.models import User
from app.dependencies.auth import get_current_user
from app.skin_images.fhir_builders import (
    build_binary_resource,
    build_diagnostic_report_resource,
    build_media_resource,
    utc_now_fhir,
)
from app.skin_images.image_processing import binary_data, normalize_uploaded_skin_image
from app.skin_images.modality import classify_skin_modality
from app.skin_images.neo4j_repository import (
    get_binary_for_skin_image,
    get_skin_image_detail,
    list_skin_images,
    patient_exists,
    save_skin_analysis,
)
from app.skin_images.schemas import (
    SkinImageAnalyzeResponse,
    SkinImageDetailResponse,
    SkinImageListResponse,
    SkinImageSummary,
)
from app.skin_images.vision import analyze_skin_image


router = APIRouter(prefix="/skin-images", tags=["skin images"])
logger = logging.getLogger(__name__)


def _normalize_patient_id(patient_id: str) -> str:
    patient_id = patient_id.strip()
    if not patient_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Patient ID is required",
        )
    return patient_id


def _assert_user_can_access_patient(current_user: User, patient_id: str) -> None:
    scoped_patient_id = (current_user.external_id or "").strip()
    if scoped_patient_id and scoped_patient_id != patient_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not allowed to access this Patient",
        )


@router.post("/analyze", response_model=SkinImageAnalyzeResponse)
async def analyze_image(
    patient_id: str = Form(...),
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    request_started = time.perf_counter()
    patient_id = _normalize_patient_id(patient_id)
    step_started = time.perf_counter()
    patient_found = await patient_exists(patient_id)
    logger.info(
        "skin_image.patient_lookup duration=%.3fs patient_id=%s",
        time.perf_counter() - step_started,
        patient_id,
    )
    if not patient_found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient was not found in Neo4j",
        )

    created_at = utc_now_fhir()
    created_dt = datetime.now(timezone.utc)
    image_uuid = str(uuid.uuid4())
    step_started = time.perf_counter()
    processed_image = await normalize_uploaded_skin_image(image)
    logger.info(
        "skin_image.image_processing duration=%.3fs size=%d content_type=%s",
        time.perf_counter() - step_started,
        processed_image.size,
        processed_image.content_type,
    )
    image_url = f"/api/skin-images/files/{image_uuid}"

    step_started = time.perf_counter()
    modality, modality_display = await classify_skin_modality(processed_image)
    logger.info(
        "skin_image.modality duration=%.3fs modality=%s",
        time.perf_counter() - step_started,
        modality,
    )
    if modality != "XC":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded image is classified as {modality}, not dermatology (XC)",
        )

    step_started = time.perf_counter()
    analysis_text = await analyze_skin_image(processed_image)
    logger.info(
        "skin_image.vision duration=%.3fs",
        time.perf_counter() - step_started,
    )
    if not analysis_text:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Vision model returned an empty analysis",
        )

    binary_id = f"binary-{image_uuid}"
    media_id = f"media-{image_uuid}"
    report_id = f"report-{image_uuid}"

    step_started = time.perf_counter()
    resources = [
        build_binary_resource(
            binary_id=binary_id,
            content_type=processed_image.content_type,
            data=binary_data(processed_image.raw),
            size=processed_image.size,
            created_at=created_at,
        ),
        build_media_resource(
            media_id=media_id,
            patient_id=patient_id,
            binary_id=binary_id,
            content_type=processed_image.content_type,
            modality=modality,
            modality_display=modality_display,
            analysis_text=analysis_text,
            created_at=created_at,
        ),
        build_diagnostic_report_resource(
            report_id=report_id,
            patient_id=patient_id,
            media_id=media_id,
            analysis_text=analysis_text,
            created_at=created_at,
        ),
    ]
    logger.info(
        "skin_image.resource_building duration=%.3fs report_id=%s",
        time.perf_counter() - step_started,
        report_id,
    )

    step_started = time.perf_counter()
    try:
        ids = await save_skin_analysis(
            resources,
            patient_id=patient_id,
            patient_already_validated=True,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Failed to persist skin image analysis")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to save skin image analysis to Neo4j",
        ) from exc
    logger.info(
        "skin_image.persistence duration=%.3fs report_id=%s",
        time.perf_counter() - step_started,
        ids["diagnostic_report_id"],
    )
    logger.info(
        "skin_image.total duration=%.3fs patient_id=%s report_id=%s",
        time.perf_counter() - request_started,
        patient_id,
        ids["diagnostic_report_id"],
    )
    return SkinImageAnalyzeResponse(
        binary_id=ids["binary_id"],
        media_id=ids["media_id"],
        diagnostic_report_id=ids["diagnostic_report_id"],
        modality=modality,
        analysis_text=analysis_text,
        image_url=image_url,
        created_at=created_dt,
    )


@router.get("", response_model=SkinImageListResponse)
async def list_images(
    patient_id: str | None = Query(default=None),
    current_user: User = Depends(get_current_user),
):
    patient_id = patient_id.strip() if patient_id else None
    rows = await list_skin_images(patient_id)
    return SkinImageListResponse(items=[SkinImageSummary(**row) for row in rows])


@router.get("/files/{image_id}")
async def get_image_file(
    image_id: str,
    current_user: User = Depends(get_current_user),
):
    binary_id = image_id if image_id.startswith("binary-") else f"binary-{image_id}"
    row = await get_binary_for_skin_image(binary_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    _assert_user_can_access_patient(current_user, str(row.get("patient_id") or ""))

    encoded = str(row.get("data") or "")
    if not encoded:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image data not found")

    try:
        content = base64.b64decode(encoded)
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid image data")

    return Response(content=content, media_type=row.get("content_type") or "image/jpeg")


@router.get("/{report_id}", response_model=SkinImageDetailResponse)
async def get_image_detail(
    report_id: str,
    current_user: User = Depends(get_current_user),
):
    row = await get_skin_image_detail(report_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skin image report not found")
    return SkinImageDetailResponse(**row)
