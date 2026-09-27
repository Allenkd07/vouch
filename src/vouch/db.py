from datetime import datetime
from functools import lru_cache

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from vouch.config import get_settings


class Base(DeclarativeBase):
    pass


class ProfileSnapshot(Base):
    """Every distinct version of profile.yaml, so each resume version can point at its source."""

    __tablename__ = "profile_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)
    content: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProfileItem(Base):
    """One bullet from the current profile, flattened for querying and evidence links."""

    __tablename__ = "profile_items"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # bullet id
    section: Mapped[str] = mapped_column(String)  # e.g. "experience:acme"
    text: Mapped[str] = mapped_column(Text)
    facts: Mapped[dict] = mapped_column(JSONB)
    tags: Mapped[list[str]] = mapped_column(ARRAY(String))
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("profile_snapshots.id"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(
        String
    )  # workday | greenhouse | lever | ashby | web | manual
    url: Mapped[str] = mapped_column(Text, unique=True)
    external_id: Mapped[str | None] = mapped_column(String)
    company: Mapped[str | None] = mapped_column(String)
    title: Mapped[str | None] = mapped_column(String)
    location: Mapped[str | None] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    posted_at: Mapped[str | None] = mapped_column(String)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Output of jobs.requirements.analyze_job (VerifiedAnalysis), for the description it was run on.
    analysis: Mapped[dict | None] = mapped_column(JSONB)
    analysis_model: Mapped[str | None] = mapped_column(String)
    analysis_hash: Mapped[str | None] = mapped_column(String(64))
    analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ResumeVersion(Base):
    """One tailored resume for one job, with the profile snapshot it was built from."""

    __tablename__ = "resume_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)
    profile_snapshot_id: Mapped[int] = mapped_column(ForeignKey("profile_snapshots.id"))
    model: Mapped[str] = mapped_column(String)
    content: Mapped[dict] = mapped_column(JSONB)  # tailoring.document.ResumeDoc
    report: Mapped[dict] = mapped_column(JSONB)  # tailoring.pipeline.TailorReport
    evidence: Mapped[dict] = mapped_column(JSONB)  # tailoring.evidence.EvidenceMap
    pdf_path: Mapped[str | None] = mapped_column(Text)
    docx_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Application(Base):
    """Tracker row for a job I'm pursuing; status is one of services.STATUSES."""

    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), unique=True)
    status: Mapped[str] = mapped_column(String)
    resume_version_id: Mapped[int | None] = mapped_column(ForeignKey("resume_versions.id"))
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def get_sessionmaker() -> sessionmaker:
    return sessionmaker(get_engine(), expire_on_commit=False)
