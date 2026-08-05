"""Conversation CRUD API â€” protected by JWT authentication."""

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from app.db.session import get_db
from app.db.models import Conversation, Message, User
from app.dependencies.auth import get_current_user
from app.schemas.conversation import (
    ConversationCreateRequest,
    ConversationInitialExchangeResponse,
    ConversationListResponse,
    ConversationResponse,
)
from app.schemas.message import ChatImageAttachment, MessageExchangeResponse, MessageResponse
from app.services.chat import generate_assistant_exchange, persist_chat_memory
from app.services.chat_stream import (
    serialize_conversation,
    serialize_message,
    stream_persisted_exchange,
)
from app.skin_images.service import (
    analyze_and_save_skin_image,
    build_upload_attachment,
    normalize_patient_id,
)
from app.skin_diagnostic.service import start_skin_diagnostic_from_binary

router = APIRouter(prefix="/conversations", tags=["conversations"])
logger = logging.getLogger(__name__)


def generate_title(first_message: str, max_length: int = 60) -> str:
    normalized = " ".join(first_message.strip().split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


async def create_conversation_with_user_message(
    *,
    db: AsyncSession,
    user_id: UUID,
    first_message: str,
    message_type: str = "text",
    attachments: list[dict] | None = None,
) -> tuple[Conversation, Message]:
    conversation = Conversation(
        user_id=user_id,
        title=generate_title(first_message),
    )
    try:
        db.add(conversation)
        await db.flush()

        user_message = Message(
            conversation_id=conversation.id,
            role="user",
            content=first_message,
            message_type=message_type,
            attachments=attachments or [],
        )
        db.add(user_message)
        await db.commit()
    except Exception:
        await db.rollback()
        raise

    await db.refresh(conversation)
    await db.refresh(user_message)
    return conversation, user_message


def _stored_image_attachment(attachment: ChatImageAttachment) -> dict:
    return {
        "type": "image",
        "patient_id": attachment.patient_id,
        "binary_id": attachment.binary_id,
        "media_id": attachment.media_id,
        "diagnostic_report_id": attachment.diagnostic_report_id,
        "content_type": attachment.content_type,
    }


def _upload_assistant_content(patient_id: str, analysis_text: str) -> str:
    return f"Đã phân tích và lưu ảnh da cho bệnh nhân {patient_id}.\n\n{analysis_text}"


def _diagnostic_pending_content(patient_id: str, complaint: str) -> str:
    if not complaint:
        return _upload_assistant_content(patient_id, "")
    return (
        "Đã lưu ảnh da vào hồ sơ bệnh nhân "
        f"{patient_id}. Tôi đang bắt đầu quy trình phân tích hình ảnh da liễu."
    )


def _diagnostic_started_content(
    content: str,
    run_id: str | None,
    warning: str | None,
    complaint: str,
) -> str:
    if warning:
        return (
            f"{content}\n\n"
            f"Lưu ý: ảnh đã được lưu nhưng chưa thể bắt đầu chẩn đoán tự động: {warning}"
        )
    if run_id:
        complaint_text = f" cho tình trạng {complaint}" if complaint else ""
        return (
            "Tôi đã bắt đầu quy trình phân tích hình ảnh da liễu"
            f"{complaint_text} dựa trên hình ảnh đã cung cấp.\n\n"
            "**Trạng thái phân tích:**\n\n"
            f"- **Mã số phiên làm việc (Run ID):** `{run_id}`\n"
            "- **Trạng thái hiện tại:** Đang thực hiện (Running)\n\n"
            "Tôi sẽ thông báo cho bạn ngay khi có kết quả phân tích từ hệ thống."
        )
    return content


def _memory_user_content(complaint: str) -> str:
    return complaint or "User uploaded a skin image without a complaint."


def _memory_assistant_content(*, diagnostic_run_id: str | None, warning: str | None) -> str:
    if warning:
        return "A skin image was saved, but automatic skin diagnosis could not be started."
    if diagnostic_run_id:
        return "A skin image was saved and automatic skin diagnosis was started."
    return "A skin image was saved without starting automatic skin diagnosis."


async def _get_owned_conversation(
    db: AsyncSession,
    conversation_id: UUID,
    user_id: UUID,
) -> Conversation:
    """Fetch a conversation owned by the given user, or raise 404."""
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return conversation


@router.post(
    "",
    response_model=ConversationInitialExchangeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    req: ConversationCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a conversation and persist the initial message exchange."""
    try:
        conversation, user_message = await create_conversation_with_user_message(
            db=db,
            user_id=current_user.id,
            first_message=req.first_message,
        )
    except Exception:
        logger.exception(
            "Failed to create conversation user message for user_id=%s",
            current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to create conversation",
        )

    try:
        assistant_content, attachments = await generate_assistant_exchange(
            content=req.first_message,
            user_id=str(current_user.id),
            conversation_id=str(conversation.id),
            current_user_message_id=user_message.id,
        )
        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=assistant_content,
            attachments=[item.model_dump(mode="json") for item in attachments],
        )
        db.add(assistant_message)
        conversation.updated_at = datetime.now(timezone.utc)

        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception(
            "Failed to create conversation assistant message for user_id=%s",
            current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to create conversation",
        )
    await db.refresh(conversation)
    await db.refresh(user_message)
    await db.refresh(assistant_message)

    try:
        await persist_chat_memory(
            user_id=str(current_user.id),
            conversation_id=str(conversation.id),
            user_message=req.first_message,
            assistant_message=assistant_content,
        )
    except Exception:
        logger.exception(
            "Failed to persist memory for conversation_id=%s",
            conversation.id,
        )

    return ConversationInitialExchangeResponse(
        conversation=ConversationResponse.model_validate(conversation),
        user_message=MessageResponse.model_validate(user_message),
        assistant_message=MessageResponse.model_validate(assistant_message),
    )


@router.post("/stream")
async def create_conversation_stream(
    req: ConversationCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        conversation, user_message = await create_conversation_with_user_message(
            db=db,
            user_id=current_user.id,
            first_message=req.first_message,
        )
    except Exception:
        logger.exception(
            "Failed to create streaming conversation for user_id=%s",
            current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to create conversation",
        )

    conversation_payload = serialize_conversation(conversation)
    user_message_payload = serialize_message(user_message)

    async def event_generator():
        async for event in stream_persisted_exchange(
            conversation_id=conversation.id,
            user_id=current_user.id,
            user_message_id=user_message.id,
            content=req.first_message,
            start_event="conversation_started",
            start_payload={
                "conversation": conversation_payload,
                "user_message": user_message_payload,
            },
        ):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/image-upload",
    response_model=MessageExchangeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation_image_upload(
    patient_id: str = Form(...),
    content: str = Form(default=""),
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    patient_id = normalize_patient_id(patient_id)
    complaint = content.strip()
    user_content = complaint or f"Upload skin image for Patient/{patient_id}"
    result = await analyze_and_save_skin_image(patient_id=patient_id, image=image)
    attachment = build_upload_attachment(patient_id=patient_id, result=result)
    stored_attachment = _stored_image_attachment(attachment)

    try:
        conversation, user_message = await create_conversation_with_user_message(
            db=db,
            user_id=current_user.id,
            first_message=user_content,
            message_type="image_upload",
            attachments=[stored_attachment],
        )
        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=_diagnostic_pending_content(patient_id, complaint)
            if complaint
            else _upload_assistant_content(patient_id, result.analysis_text),
            message_type="text",
        )
        db.add(assistant_message)
        conversation.updated_at = datetime.now(timezone.utc)
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("Failed to persist image upload conversation for user_id=%s", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to persist image upload message",
        )

    await db.refresh(conversation)
    await db.refresh(user_message)
    await db.refresh(assistant_message)

    diagnostic_run_id: str | None = None
    diagnostic_warning: str | None = None
    if complaint:
        try:
            run = await start_skin_diagnostic_from_binary(
                user_id=str(current_user.id),
                conversation_id=str(conversation.id),
                patient_id=patient_id,
                binary_id=attachment.binary_id,
                initial_complaint=complaint,
            )
            diagnostic_run_id = run.id
        except Exception as exc:
            logger.exception(
                "Failed to start automatic skin diagnostic for conversation_id=%s user_id=%s",
                conversation.id,
                current_user.id,
            )
            diagnostic_warning = str(exc)

        assistant_message.content = _diagnostic_started_content(
            _upload_assistant_content(patient_id, result.analysis_text),
            diagnostic_run_id,
            diagnostic_warning,
            complaint,
        )
        conversation.updated_at = datetime.now(timezone.utc)
        try:
            await db.commit()
            await db.refresh(conversation)
            await db.refresh(assistant_message)
        except Exception:
            await db.rollback()
            logger.exception(
                "Failed to update image upload diagnostic status for conversation_id=%s user_id=%s",
                conversation.id,
                current_user.id,
            )

    try:
        await persist_chat_memory(
            user_id=str(current_user.id),
            conversation_id=str(conversation.id),
            user_message=_memory_user_content(complaint),
            assistant_message=_memory_assistant_content(
                diagnostic_run_id=diagnostic_run_id,
                warning=diagnostic_warning,
            ),
        )
    except Exception:
        logger.exception(
            "Failed to persist memory for conversation_id=%s",
            conversation.id,
        )

    return MessageExchangeResponse(
        conversation_id=conversation.id,
        user_message=MessageResponse.model_validate(user_message),
        assistant_message=MessageResponse.model_validate(assistant_message),
        attachments=[attachment],
        diagnostic_run_id=diagnostic_run_id,
    )


@router.get(
    "",
    response_model=ConversationListResponse,
)
async def list_conversations(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List conversations owned by the authenticated user."""
    count_stmt = select(func.count()).select_from(Conversation).where(
        Conversation.user_id == current_user.id,
    )
    count_result = await db.execute(count_stmt)
    total = count_result.scalar_one()

    # Fetch page
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .offset(skip)
        .limit(limit)
    )
    conversations = result.scalars().all()
    return ConversationListResponse(
        items=conversations,
        total=total,
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
async def get_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single conversation owned by the authenticated user."""
    return await _get_owned_conversation(db, conversation_id, current_user.id)


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a conversation owned by the authenticated user."""
    conversation = await _get_owned_conversation(
        db, conversation_id, current_user.id
    )
    try:
        await db.delete(conversation)
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception(
            "Failed to delete conversation_id=%s for user_id=%s",
            conversation_id,
            current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
    return None
