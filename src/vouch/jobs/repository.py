"""Reading and writing jobs."""

import hashlib
import re
from datetime import UTC, datetime, timedelta

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from vouch.boards import FetchedJob
from vouch.db import Job
from vouch.jobs.requirements import VerifiedAnalysis

ACTIVE_DAYS = 3  # a discovered job not seen on any board for this long counts as closed


def content_hash(description: str) -> str:
    """Whitespace/case-insensitive, so cosmetic re-renders of a posting don't count as changes."""
    canonical = re.sub(r"\s+", " ", description).strip().lower()
    return hashlib.sha256(canonical.encode()).hexdigest()


def get(session: Session, job_id: int) -> Job | None:
    return session.get(Job, job_id)


def recent(session: Session, limit: int) -> list[Job]:
    """Most recently fetched first."""
    return list(session.scalars(select(Job).order_by(Job.fetched_at.desc()).limit(limit)))


def active_filter(now: datetime):
    """Manually added jobs always; discovered ones while they're still listed."""
    return or_(Job.last_seen_at.is_(None), Job.last_seen_at >= now - timedelta(days=ACTIVE_DAYS))


def active(session: Session, now: datetime) -> list[Job]:
    return list(session.scalars(select(Job).where(active_filter(now))))


def upsert_job(session: Session, fetched: FetchedJob) -> tuple[Job, bool]:
    """Insert or refresh a job by URL. Returns (job, changed); changed is False when the
    description is identical to what's stored, so the existing analysis stays valid."""
    digest = content_hash(fetched.description)
    url = fetched.url or f"manual:{digest[:16]}"
    job = session.scalar(select(Job).where(Job.url == url))
    changed = job is None or job.content_hash != digest
    if job is None:
        job = Job(url=url)
        session.add(job)
    job.source = fetched.source
    job.external_id = fetched.external_id
    job.company = fetched.company or job.company
    job.title = fetched.title or job.title
    job.location = fetched.location or job.location
    job.description = fetched.description
    job.content_hash = digest
    job.posted_at = fetched.posted_at
    job.fetched_at = datetime.now(UTC)
    session.flush()
    return job, changed


def needs_analysis(job: Job) -> bool:
    """True when requirements were never extracted or the posting changed since. Switching the
    extraction model doesn't count: re-reading every job would waste quota (use --reanalyze)."""
    return job.analysis is None or job.analysis_hash != job.content_hash


def save_analysis(job: Job, result: VerifiedAnalysis, model: str) -> None:
    job.analysis = result.model_dump(mode="json")
    job.analysis_model = model
    job.analysis_hash = job.content_hash
    job.analyzed_at = datetime.now(UTC)
    a = result.analysis
    job.title = job.title or a.title
    job.company = job.company or a.company
    job.location = job.location or a.location
