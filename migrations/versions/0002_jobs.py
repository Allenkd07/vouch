"""jobs table

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-27
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source", sa.String, nullable=False),
        sa.Column("url", sa.Text, nullable=False, unique=True),
        sa.Column("external_id", sa.String),
        sa.Column("company", sa.String),
        sa.Column("title", sa.String),
        sa.Column("location", sa.String),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False, index=True),
        sa.Column("posted_at", sa.String),
        sa.Column(
            "fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("analysis", postgresql.JSONB),
        sa.Column("analysis_model", sa.String),
        sa.Column("analysis_hash", sa.String(64)),
        sa.Column("analyzed_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    op.drop_table("jobs")
