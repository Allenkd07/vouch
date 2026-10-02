"""Reading and writing tailored resume versions."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from vouch.db import ResumeVersion


def get(session: Session, version_id: int) -> ResumeVersion | None:
    return session.get(ResumeVersion, version_id)


def for_job(session: Session, job_id: int) -> list[ResumeVersion]:
    """Newest first."""
    return list(
        session.scalars(
            select(ResumeVersion)
            .where(ResumeVersion.job_id == job_id)
            .order_by(ResumeVersion.id.desc())
        )
    )


def add(session: Session, version: ResumeVersion) -> ResumeVersion:
    """Store a new version and assign its id."""
    session.add(version)
    session.flush()
    return version
