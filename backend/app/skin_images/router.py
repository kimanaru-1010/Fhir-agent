from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import Response

from app.db.models import User
from app.dependencies.auth import get_current_user
from app.skin_images.neo4j_repository import (
    get_binary_for_skin_image,
    get_skin_image_detail,
    list_skin_images,
)
from app.skin_images.schemas import (
    SkinImageAnalyzeResponse,
    SkinImageDetailResponse,
    SkinImageListResponse,
    SkinImageSummary,
)
from app.skin_images.service import analyze_and_save_skin_image, normalize_patient_id


router = APIRouter(prefix="/skin-images", tags=["skin images"])
logger = logging.getLogger(__name__)


def _assert_user_can_access_patient(current_user: User, patient_id: str) -> None:
    # Doctor-mode access: authenticated users can work with any Patient record.
    # Patient-scoped authorization was removed because this app currently uses a
    # single doctor role for cross-patient clinical lookup and image upload.
    return None


@router.post("/analyze", response_model=SkinImageAnalyzeResponse)
async def analyze_image(
    patient_id: str = Form(...),
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    patient_id = normalize_patient_id(patient_id)
    _assert_user_can_access_patient(current_user, patient_id)
    return await analyze_and_save_skin_image(patient_id=patient_id, image=image)


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
    _assert_user_can_access_patient(current_user, str(row.get("patient_id") or ""))
    return SkinImageDetailResponse(**row)
