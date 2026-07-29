"""add short term memory to conversations

Revision ID: 6f8d4b2a9c13
Revises: eb4557cb4c80
Create Date: 2026-07-29 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "6f8d4b2a9c13"
down_revision: Union[str, Sequence[str], None] = "eb4557cb4c80"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("summary", sa.Text(), server_default="", nullable=False),
    )
    op.add_column(
        "conversations",
        sa.Column("summary_through_message_id", sa.UUID(), nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column("summary_updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column("memory_version", sa.Integer(), server_default="0", nullable=False),
    )
    op.create_foreign_key(
        "fk_conversations_summary_through_message_id_messages",
        "conversations",
        "messages",
        ["summary_through_message_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_conversations_summary_through_message_id_messages",
        "conversations",
        type_="foreignkey",
    )
    op.drop_column("conversations", "memory_version")
    op.drop_column("conversations", "summary_updated_at")
    op.drop_column("conversations", "summary_through_message_id")
    op.drop_column("conversations", "summary")
