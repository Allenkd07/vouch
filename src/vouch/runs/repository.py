"""Reading and writing background runs."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from vouch.runs.models import Run


def get(session: Session, run_id: int) -> Run | None:
    return session.get(Run, run_id)


def latest(session: Session, kind: str, job_id: int | None = None) -> Run | None:
    """The most recent run of `kind` for a job (or, with no job, the most recent global one)."""
    query = select(Run).where(Run.kind == kind)
    query = query.where(Run.job_id.is_(None) if job_id is None else Run.job_id == job_id)
    return session.scalar(query.order_by(Run.id.desc()).limit(1))


def add(session: Session, run: Run) -> Run:
    session.add(run)
    session.flush()
    return run
