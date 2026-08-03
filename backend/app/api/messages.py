"""Message API for authenticated conversations."""

from __future__ import annotations

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
from app.schemas.message import (
    MessageCreateRequest,
    MessageExchangeResponse,
    MessageListResponse,
    MessageResponse,
)
from app.services.chat import generate_assistant_response, persist_chat_memory
from app.services.chat_stream import (
    serialize_message,
    stream_persisted_exchange,
)

router = APIRouter(
    prefix="/conversations/{conversation_id}/messages",
    tags=["messages"],
)
logger = logging.getLogger(__name__)


async def get_owned_conversation(
    db: AsyncSession,
    conversation_id: UUID,
    user_id: UUID,
) -> Conversation:
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


async def create_user_message(
    *,
    db: AsyncSession,
    conversation: Conversation,
    content: str,
) -> Message:
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=content,
    )
    try:
        db.add(user_message)
        conversation.updated_at = datetime.now(timezone.utc)
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    await db.refresh(user_message)
    await db.refresh(conversation)
    return user_message


@router.get(
    "",
    response_model=MessageListResponse,
)
async def list_messages(
    conversation_id: UUID,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await get_owned_conversation(db, conversation_id, current_user.id)

    count_stmt = (
        select(func.count())
        .select_from(Message)
        .where(Message.conversation_id == conversation_id)
    )
    count_result = await db.execute(count_stmt)
    total = count_result.scalar_one()

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(
            Message.created_at.asc(),
            Message.id.asc(),
        )
        .offset(skip)
        .limit(limit)
    )
    return MessageListResponse(
        items=result.scalars().all(),
        total=total,
    )


@router.post(
    "",
    response_model=MessageExchangeResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_message(
    conversation_id: UUID,
    req: MessageCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conversation = await get_owned_conversation(
        db,
        conversation_id,
        current_user.id,
    )
    try:
        user_message = await create_user_message(
            db=db,
            conversation=conversation,
            content=req.content,
        )
    except Exception:
        logger.exception(
            "Failed to persist user message for conversation_id=%s user_id=%s",
            conversation_id,
            current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process message",
        )

    try:
        assistant_content = await generate_assistant_response(
            content=req.content,
            user_id=str(current_user.id),
            conversation_id=str(conversation.id),
            current_user_message_id=user_message.id,
        )
        if not assistant_content.strip():
            raise RuntimeError("Agent returned an empty response")

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
            "Failed to process message for conversation_id=%s user_id=%s",
            conversation_id,
            current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process message",
        )

    await db.refresh(user_message)
    await db.refresh(assistant_message)
    await db.refresh(conversation)

    try:
        await persist_chat_memory(
            user_id=str(current_user.id),
            conversation_id=str(conversation.id),
            user_message=req.content,
            assistant_message=assistant_content,
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
    )


@router.post("/stream")
async def create_message_stream(
    conversation_id: UUID,
    req: MessageCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conversation = await get_owned_conversation(
        db,
        conversation_id,
        current_user.id,
    )
    try:
        user_message = await create_user_message(
            db=db,
            conversation=conversation,
            content=req.content,
        )
    except Exception:
        logger.exception(
            "Failed to persist stream user message for conversation_id=%s user_id=%s",
            conversation_id,
            current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to process message",
        )

    user_message_payload = serialize_message(user_message)

    async def event_generator():
        async for event in stream_persisted_exchange(
            conversation_id=conversation.id,
            user_id=current_user.id,
            user_message_id=user_message.id,
            content=req.content,
            start_event="message_started",
            start_payload={
                "conversation_id": str(conversation.id),
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
