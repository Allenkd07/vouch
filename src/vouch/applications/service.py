"""The application tracker: one status (and notes) per job I'm pursuing."""

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from vouch.applications import repository as applications_db
from vouch.db import Application
from vouch.transaction import transaction

STATUSES = ["saved", "applied", "interviewing", "offer", "rejected"]


def set_status(session: Session, job_id: int, status: str | None) -> Application | None:
    """status None (or "") removes the job from the tracker."""
    with transaction(session):
        return apply_status(session, job_id, status)


def set_notes(session: Session, job_id: int, notes: str) -> Application | None:
    """None when the job isn't tracked."""
    with transaction(session):
        app = applications_db.for_job(session, job_id)
        if app:
            app.notes = notes
    return app


def apply_status(session: Session, job_id: int, status: str | None) -> Application | None:
    """`set_status` without committing, for services that change more in the same transaction."""
    app = applications_db.for_job(session, job_id)
    if not status:
        if app:
            applications_db.delete(session, app)
        return None
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if app is None:
        app = applications_db.add(session, Application(job_id=job_id, status=status))
    app.status = status
    if status == "applied" and app.applied_at is None:
        app.applied_at = datetime.now(UTC)
    session.flush()
    return app
