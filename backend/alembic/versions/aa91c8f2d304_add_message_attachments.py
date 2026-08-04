"""add message attachments

Revision ID: aa91c8f2d304
Revises: f5b7c8d9e0a1
Create Date: 2026-08-04 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "aa91c8f2d304"
down_revision = "f5b7c8d9e0a1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column(
            "attachments",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("messages", "attachments")
