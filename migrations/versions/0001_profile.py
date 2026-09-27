"""pgvector extension and profile tables

Revision ID: 0001
Revises:
Create Date: 2026-09-27
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "profile_snapshots",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("content_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("content", postgresql.JSONB, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_table(
        "profile_items",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("section", sa.String, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("facts", postgresql.JSONB, nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.String), nullable=False),
        sa.Column("snapshot_id", sa.Integer, sa.ForeignKey("profile_snapshots.id"), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_table("profile_items")
    op.drop_table("profile_snapshots")
