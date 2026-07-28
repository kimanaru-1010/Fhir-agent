"""Conversation CRUD API — protected by JWT authentication."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.db.models import Conversation, Message, User
from app.dependencies.auth import get_current_user
from app.schemas.conversation import (
    ConversationCreateRequest,
    ConversationCreateResponse,
    ConversationListResponse,
    ConversationResponse,
    FirstMessageResponse,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])
logger = logging.getLogger(__name__)


def generate_title(first_message: str, max_length: int = 60) -> str:
    normalized = " ".join(first_message.strip().split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


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
    response_model=ConversationCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    req: ConversationCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a conversation and persist the first user message."""
    conversation = Conversation(
        user_id=current_user.id,
        title=generate_title(req.first_message),
    )
    try:
        db.add(conversation)
        await db.flush()

        message = Message(
            conversation_id=conversation.id,
            role="user",
            content=req.first_message,
        )
        db.add(message)

        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("Failed to create conversation for user_id=%s", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
    await db.refresh(conversation)
    await db.refresh(message)
    return ConversationCreateResponse(
        id=conversation.id,
        user_id=conversation.user_id,
        title=conversation.title,
        first_message=FirstMessageResponse.model_validate(message),
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
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
