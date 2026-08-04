"""add message type

Revision ID: b4e2d8ab6a10
Revises: aa91c8f2d304
Create Date: 2026-08-04 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "b4e2d8ab6a10"
down_revision = "aa91c8f2d304"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column("message_type", sa.String(length=50), server_default="text", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("messages", "message_type")
