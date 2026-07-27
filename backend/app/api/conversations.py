"""Conversation CRUD API — protected by JWT authentication."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.db.models import Conversation, User
from app.dependencies.auth import get_current_user
from app.schemas.conversation import (
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationResponse,
    ConversationUpdateRequest,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


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
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    req: ConversationCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new conversation for the authenticated user."""
    conversation = Conversation(
        user_id=current_user.id,
        title=req.title,
    )
    db.add(conversation)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    await db.refresh(conversation)
    return conversation


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
    # Count total
    count_stmt = select(Conversation).where(
        Conversation.user_id == current_user.id,
    )
    count_result = await db.execute(count_stmt)
    total = len(count_result.scalars().all())

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


@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
async def update_conversation(
    conversation_id: UUID,
    req: ConversationUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update the title of a conversation owned by the authenticated user."""
    conversation = await _get_owned_conversation(
        db, conversation_id, current_user.id
    )
    conversation.title = req.title
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    await db.refresh(conversation)
    return conversation


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
    db.delete(conversation)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    return None
