"""background runs

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-02
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("job_id", sa.Integer, sa.ForeignKey("jobs.id")),
        sa.Column("state", sa.String(10), nullable=False),
        sa.Column("message", sa.Text, nullable=False, server_default=""),
        sa.Column("result_id", sa.Integer),
        sa.Column("owner", sa.String(200), nullable=False),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_runs_kind_job", "runs", ["kind", "job_id", "id"])


def downgrade() -> None:
    op.drop_index("ix_runs_kind_job", table_name="runs")
    op.drop_table("runs")
