"""resume_versions table

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-27
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "resume_versions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("job_id", sa.Integer, sa.ForeignKey("jobs.id"), nullable=False, index=True),
        sa.Column(
            "profile_snapshot_id",
            sa.Integer,
            sa.ForeignKey("profile_snapshots.id"),
            nullable=False,
        ),
        sa.Column("model", sa.String, nullable=False),
        sa.Column("content", postgresql.JSONB, nullable=False),
        sa.Column("report", postgresql.JSONB, nullable=False),
        sa.Column("evidence", postgresql.JSONB, nullable=False),
        sa.Column("pdf_path", sa.Text),
        sa.Column("docx_path", sa.Text),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("approved_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    op.drop_table("resume_versions")
