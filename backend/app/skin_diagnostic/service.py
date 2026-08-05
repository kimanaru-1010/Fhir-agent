"""Shared services for starting skin diagnostic runs."""

from __future__ import annotations

import asyncio
import base64
import uuid

from app.skin_diagnostic.pipeline_runner import run_pipeline_background
from app.skin_diagnostic.session_store import SkinDiagnosticRun, get_store
from app.skin_diagnostic.uploads import UPLOADS_DIR
from app.skin_images.neo4j_repository import get_binary_for_skin_image
from app.skin_images.references import build_image_api_url


def _extension_from_content_type(content_type: str | None) -> str:
    normalized = (content_type or "").lower()
    if "png" in normalized:
        return ".png"
    if "webp" in normalized:
        return ".webp"
    return ".jpg"


def _decode_binary_data(value) -> bytes:
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Binary.data is empty")
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError("Binary.data is not valid base64") from exc


async def start_skin_diagnostic_from_binary(
    *,
    user_id: str,
    conversation_id: str = "",
    patient_id: str,
    binary_id: str,
    initial_complaint: str,
) -> SkinDiagnosticRun:
    row = await get_binary_for_skin_image(binary_id)
    if row is None:
        raise ValueError("Skin image Binary was not found")

    resolved_patient_id = str(row.get("patient_id") or "")
    if patient_id and resolved_patient_id and resolved_patient_id != patient_id:
        raise ValueError("Binary does not belong to the requested Patient")

    image_bytes = _decode_binary_data(row.get("data"))
    run_id = str(uuid.uuid4())
    ext = _extension_from_content_type(row.get("content_type"))
    image_path = UPLOADS_DIR / f"{run_id}{ext}"
    image_path.write_bytes(image_bytes)

    store = await get_store()
    run = await store.create(
        run_id=run_id,
        user_id=user_id,
        conversation_id=conversation_id,
        image_path=str(image_path),
        image_url=build_image_api_url(binary_id),
        anamnesis=initial_complaint,
    )
    asyncio.create_task(
        run_pipeline_background(run.id, str(image_path), initial_complaint)
    )
    return run
