"""Reading and writing tracked applications (one per job)."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from vouch.db import Application, Job


def for_job(session: Session, job_id: int) -> Application | None:
    return session.scalar(select(Application).where(Application.job_id == job_id))


def add(session: Session, application: Application) -> Application:
    session.add(application)
    session.flush()
    return application


def delete(session: Session, application: Application) -> None:
    session.delete(application)
    session.flush()


def with_jobs(session: Session) -> list[tuple[Application, Job]]:
    """Every application with its job, most recently updated first."""
    rows = session.execute(
        select(Application, Job)
        .join(Job, Job.id == Application.job_id)
        .order_by(Application.updated_at.desc())
    ).all()
    return [tuple(r) for r in rows]
