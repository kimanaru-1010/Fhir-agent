"""Conversation CRUD API â€” protected by JWT authentication."""

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
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
from app.schemas.message import MessageResponse
from app.services.chat import generate_assistant_response, persist_chat_memory
from app.services.chat_stream import (
    serialize_conversation,
    serialize_message,
    stream_persisted_exchange,
)

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
        )
        db.add(user_message)
        await db.commit()
    except Exception:
        await db.rollback()
        raise

    await db.refresh(conversation)
    await db.refresh(user_message)
    return conversation, user_message


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
        assistant_content = await generate_assistant_response(
            content=req.first_message,
            user_id=str(current_user.id),
            conversation_id=str(conversation.id),
            current_user_message_id=user_message.id,
        )
        assistant_message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=assistant_content,
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
