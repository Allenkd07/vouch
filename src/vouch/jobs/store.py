import hashlib
import re
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from vouch.db import Job
from vouch.jobs.requirements import VerifiedAnalysis
from vouch.jobs.sources import FetchedJob


def content_hash(description: str) -> str:
    """Whitespace/case-insensitive, so cosmetic re-renders of a posting don't count as changes."""
    canonical = re.sub(r"\s+", " ", description).strip().lower()
    return hashlib.sha256(canonical.encode()).hexdigest()


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


def needs_analysis(job: Job, model: str) -> bool:
    return (
        job.analysis is None or job.analysis_hash != job.content_hash or job.analysis_model != model
    )


def save_analysis(job: Job, result: VerifiedAnalysis, model: str) -> None:
    job.analysis = result.model_dump(mode="json")
    job.analysis_model = model
    job.analysis_hash = job.content_hash
    job.analyzed_at = datetime.now(UTC)
    a = result.analysis
    job.title = job.title or a.title
    job.company = job.company or a.company
    job.location = job.location or a.location
