"""require user external id

Revision ID: 9c2d1a7e4f10
Revises: 6f8d4b2a9c13
Create Date: 2026-08-03 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9c2d1a7e4f10"
down_revision: Union[str, Sequence[str], None] = "6f8d4b2a9c13"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("UPDATE users SET external_id = id::text WHERE external_id IS NULL")
    op.alter_column(
        "users",
        "external_id",
        existing_type=sa.String(length=255),
        nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "users",
        "external_id",
        existing_type=sa.String(length=255),
        nullable=True,
    )
