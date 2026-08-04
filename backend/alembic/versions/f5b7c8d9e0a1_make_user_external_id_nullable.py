"""Make user external_id nullable for doctor accounts.

Revision ID: f5b7c8d9e0a1
Revises: 9c2d1a7e4f10
Create Date: 2026-08-04 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "f5b7c8d9e0a1"
down_revision = "9c2d1a7e4f10"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "users",
        "external_id",
        existing_type=sa.String(length=255),
        nullable=True,
    )


def downgrade() -> None:
    op.execute("UPDATE users SET external_id = id::text WHERE external_id IS NULL")
    op.alter_column(
        "users",
        "external_id",
        existing_type=sa.String(length=255),
        nullable=False,
    )
