"""applications table

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-27
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "applications",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("job_id", sa.Integer, sa.ForeignKey("jobs.id"), nullable=False, unique=True),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("resume_version_id", sa.Integer, sa.ForeignKey("resume_versions.id")),
        sa.Column("applied_at", sa.DateTime(timezone=True)),
        sa.Column("notes", sa.Text, nullable=False, server_default=""),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_table("applications")
