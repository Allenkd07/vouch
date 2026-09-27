"""job discovery: last_seen_at, job_embeddings, matches

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-27
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("last_seen_at", sa.DateTime(timezone=True)))
    op.create_table(
        "job_embeddings",
        sa.Column("job_id", sa.Integer, sa.ForeignKey("jobs.id"), primary_key=True),
        sa.Column("model", sa.String, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        # No fixed dimension: different embedding models can be compared within their own rows.
        sa.Column("embedding", Vector(), nullable=False),
    )
    op.create_table(
        "matches",
        sa.Column("job_id", sa.Integer, sa.ForeignKey("jobs.id"), primary_key=True),
        sa.Column("score", sa.Float),
        sa.Column("similarity", sa.Float),
        sa.Column("fit", postgresql.JSONB),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_table("matches")
    op.drop_table("job_embeddings")
    op.drop_column("jobs", "last_seen_at")
