"""Persist completed skin diagnostic results into chat messages."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select

from app.db.models import Conversation, Message
from app.db.session import AsyncSessionFactory
from app.skin_diagnostic.session_store import SkinDiagnosticRun
from app.skin_diagnostic.session_view import build_result

logger = logging.getLogger(__name__)


def _diagnosis_lines(result: dict) -> list[str]:
    lines: list[str] = []
    ranked = result.get("ranked_diagnoses") or []
    if not isinstance(ranked, list):
        return lines

    for index, diagnosis in enumerate(ranked, start=1):
        if not isinstance(diagnosis, dict):
            continue
        rank = diagnosis.get("rank") or index
        disease = str(diagnosis.get("disease") or "Chưa xác định").strip()
        confidence = str(diagnosis.get("confidence") or "Low").strip()
        evidence_for = str(diagnosis.get("evidence_for") or "").strip()
        evidence_against = str(diagnosis.get("evidence_against") or "").strip()

        lines.append(f"{rank}. {disease} - {confidence}")
        if evidence_for:
            lines.append(f"   Bằng chứng hỗ trợ: {evidence_for}")
        if evidence_against:
            lines.append(f"   Điểm cần loại trừ: {evidence_against}")

    return lines


def format_skin_diagnostic_result_message(run: SkinDiagnosticRun) -> str:
    """Format a completed run as markdown suitable for chat and memory."""

    result = build_result(run.state)
    lines = [
        "## Phiếu chẩn đoán da",
        "",
        f"Run ID: {run.id}",
    ]
    if run.anamnesis.strip():
        lines.extend(["", f"Triệu chứng ban đầu: {run.anamnesis.strip()}"])

    diagnosis_lines = _diagnosis_lines(result)
    if diagnosis_lines:
        lines.extend(["", "### Chẩn đoán xếp hạng", *diagnosis_lines])

    visual = str(result.get("visual_observations") or "").strip()
    if visual:
        lines.extend(["", "### Phân tích hình ảnh tổn thương", visual])

    reasoning = str(result.get("reasoning") or "").strip()
    if reasoning:
        lines.extend(["", "### Biện luận y khoa", reasoning])

    uncertainty = str(result.get("remaining_uncertainty") or "").strip()
    if uncertainty:
        lines.extend(["", "### Thông tin cần bổ sung", uncertainty])

    qa_history = str(result.get("qa_history") or "").strip()
    if qa_history:
        lines.extend(["", "### Hỏi bệnh đã ghi nhận", qa_history])

    lines.extend(
        [
            "",
            "_Kết quả chỉ có mục đích hỗ trợ quyết định lâm sàng, không thay thế việc khám trực tiếp và chẩn đoán của bác sĩ._",
        ]
    )
    return "\n".join(lines).strip()


async def persist_skin_diagnostic_result_message(run: SkinDiagnosticRun) -> Message | None:
    """Persist a completed diagnostic result as an assistant message once."""

    if run.status != "completed" or not run.conversation_id:
        return None

    try:
        conversation_id = UUID(run.conversation_id)
    except ValueError:
        logger.warning("Invalid skin diagnostic conversation_id=%s", run.conversation_id)
        return None

    result = build_result(run.state)
    attachment = {
        "type": "skin_diagnostic_result",
        "run_id": run.id,
        "result": result,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    content = format_skin_diagnostic_result_message(run)

    async with AsyncSessionFactory() as session:
        existing_result = await session.execute(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.message_type == "skin_diagnostic_result",
                Message.content.contains(f"Run ID: {run.id}"),
            )
            .limit(1)
        )
        existing = existing_result.scalar_one_or_none()
        if existing is not None:
            return existing

        conversation_result = await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = conversation_result.scalar_one_or_none()
        if conversation is None:
            return None

        message = Message(
            conversation_id=conversation_id,
            role="assistant",
            message_type="skin_diagnostic_result",
            content=content,
            attachments=[attachment],
        )
        session.add(message)
        conversation.updated_at = datetime.now(timezone.utc)
        try:
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception(
                "Failed to persist skin diagnostic result message for run_id=%s",
                run.id,
            )
            return None
        await session.refresh(message)
        return message
