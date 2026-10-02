from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from vouch.db import Base


class Run(Base):
    """One piece of background work (tailoring a resume, a job search) and how it ended.

    Stored in the database rather than in memory, so status survives restarts and is the same
    whichever server instance answers."""

    __tablename__ = "runs"
    __table_args__ = (Index("ix_runs_kind_job", "kind", "job_id", "id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(20))  # tailor | discover
    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id"))
    state: Mapped[str] = mapped_column(String(10))  # running | done | error
    message: Mapped[str] = mapped_column(Text, default="")
    result_id: Mapped[int | None]  # e.g. the resume version a tailor run created
    owner: Mapped[str] = mapped_column(String(200))  # host:pid of the process doing the work
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
