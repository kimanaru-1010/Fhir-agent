"""API routes for skin diagnostic runs."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.db.models import User
from app.dependencies.auth import get_current_user
from app.skin_diagnostic.answer_validation import normalize_yes_no
from app.skin_diagnostic.pipeline_runner import resume_pipeline, run_pipeline_background
from app.skin_diagnostic.schemas import (
    BulkAnswerRequest,
    SkinDiagnosticAnswersResponse,
    SkinDiagnosticDetailResponse,
    SkinDiagnosticStartResponse,
    SkinDiagnosticStatusResponse,
)
from app.skin_diagnostic.session_store import get_store
from app.skin_diagnostic.session_view import build_result, get_pending_questions, get_step_progress
from app.skin_diagnostic.uploads import UPLOADS_DIR, save_upload


router = APIRouter(prefix="/skin-diagnostics", tags=["skin diagnostics"])


@router.post("/start", response_model=SkinDiagnosticStartResponse)
async def start_skin_diagnostic(
    image: UploadFile = File(...),
    anamnesis: str = Form(""),
    current_user: User = Depends(get_current_user),
):
    store = await get_store()
    run_id = str(uuid.uuid4())
    image_path, image_url = await save_upload(image, run_id)
    run = await store.create(
        run_id=run_id,
        user_id=str(current_user.id),
        image_path=image_path,
        image_url=image_url,
        anamnesis=anamnesis,
    )
    asyncio.create_task(run_pipeline_background(run.id, image_path, anamnesis))
    return SkinDiagnosticStartResponse(
        run_id=run.id,
        status="running",
        current_step="visual_extract",
    )


@router.get("/{run_id}/status", response_model=SkinDiagnosticStatusResponse)
async def get_skin_diagnostic_status(
    run_id: str,
    current_user: User = Depends(get_current_user),
):
    run = await _get_run_or_404(run_id, str(current_user.id))
    result = build_result(run.state) if run.status == "completed" else {}
    return SkinDiagnosticStatusResponse(
        run_id=run.id,
        status=run.status,
        current_step=run.current_step,
        progress=get_step_progress(run.current_step),
        pending_questions=get_pending_questions(run),
        result=result,
        error=run.error,
    )


@router.get("/{run_id}", response_model=SkinDiagnosticDetailResponse)
async def get_skin_diagnostic_detail(
    run_id: str,
    current_user: User = Depends(get_current_user),
):
    run = await _get_run_or_404(run_id, str(current_user.id))
    result = build_result(run.state) if run.status == "completed" else None
    return SkinDiagnosticDetailResponse(
        run_id=run.id,
        status=run.status,
        current_step=run.current_step,
        image_url=run.image_url,
        anamnesis=run.anamnesis,
        step_history=run.step_history,
        pending_questions=get_pending_questions(run),
        result=result,
        error=run.error,
    )


@router.post("/{run_id}/answers", response_model=SkinDiagnosticAnswersResponse)
async def submit_skin_diagnostic_answers(
    run_id: str,
    request: BulkAnswerRequest,
    current_user: User = Depends(get_current_user),
):
    run = await _get_run_or_404(run_id, str(current_user.id))
    store = await get_store()
    if run.status != "interrupt":
        return SkinDiagnosticAnswersResponse(status=run.status, current_step=run.current_step)

    if run.pending_answers or run.pending_answer:
        return SkinDiagnosticAnswersResponse(status="running", current_step=run.current_step)

    normalized_answers = []
    for item in request.answers:
        normalized_answers.append(
            {
                "question_num": item.question_num,
                "answer": normalize_yes_no(item.answer, question_num=item.question_num),
            }
        )
    run.pending_answers.extend(normalized_answers)
    await store.update(
        run_id,
        status="running",
        pending_questions=None,
        pending_answer=None,
    )
    asyncio.create_task(resume_pipeline(run_id))
    return SkinDiagnosticAnswersResponse(status="running", current_step=run.current_step)


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_skin_diagnostic(
    run_id: str,
    current_user: User = Depends(get_current_user),
):
    store = await get_store()
    deleted = await store.delete(run_id, user_id=str(current_user.id))
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return None


@router.get("/uploads/{filename}")
async def get_uploaded_skin_image(
    filename: str,
    current_user: User = Depends(get_current_user),
):
    path = (UPLOADS_DIR / filename).resolve()
    uploads_root = UPLOADS_DIR.resolve()
    if uploads_root not in path.parents or not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    store = await get_store()
    run = await store.get(Path(filename).stem, user_id=str(current_user.id))
    if run is None or Path(run.image_path).resolve() != path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    return FileResponse(path)


async def _get_run_or_404(run_id: str, user_id: str):
    store = await get_store()
    run = await store.get(run_id, user_id=user_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return run
