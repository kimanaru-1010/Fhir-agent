"""Message API for authenticated conversations."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import handle_message
from app.database import get_db
from app.db.models import Conversation, Message, User
from app.dependencies.auth import get_current_user
from app.schemas.message import (
    MessageCreateRequest,
    MessageExchangeResponse,
    MessageListResponse,
    MessageResponse,
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


async def run_agent_for_message(
    *,
    content: str,
    user_id: str,
    conversation_id: str,
) -> str:
    result = await handle_message(
        content,
        session_id=conversation_id,
        user_id=user_id,
    )
    assistant_content = _extract_agent_text(result)
    if not assistant_content.strip():
        raise RuntimeError("Agent returned an empty response")
    return assistant_content


def _extract_agent_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("response", "content", "message", "text", "output"):
            value = result.get(key)
            if value is not None:
                return str(value)
        return str(result)
    for attr in ("response", "content", "message", "text", "output"):
        value = getattr(result, attr, None)
        if value is not None:
            return str(value)
    return str(result)


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
    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=req.content,
    )

    try:
        db.add(user_message)
        await db.flush()

        assistant_content = await run_agent_for_message(
            content=req.content,
            user_id=str(current_user.id),
            conversation_id=str(conversation.id),
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

    return MessageExchangeResponse(
        conversation_id=conversation.id,
        user_message=MessageResponse.model_validate(user_message),
        assistant_message=MessageResponse.model_validate(assistant_message),
    )
