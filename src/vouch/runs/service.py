"""Start background work, record how it ended, and notice runs a crash or restart abandoned."""

import logging
import os
import socket
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from vouch.llm import LLMError
from vouch.runs import repository as runs_db
from vouch.runs.models import Run
from vouch.runs.runner import TaskRunner
from vouch.transaction import transaction

log = logging.getLogger(__name__)

HOST = socket.gethostname()
OWNER = f"{HOST}:{os.getpid()}"
# A run on another machine (e.g. another Cloud Run instance) that hasn't finished by then has
# died with its instance. Tailoring takes about a minute and a job search a few.
STALE_AFTER = timedelta(minutes=15)
INTERRUPTED = "Interrupted: the server restarted before it finished. Try again."
# Failures with a message worth showing as is; anything else is logged with its traceback.
EXPECTED = (LLMError, ValidationError, ValueError, OSError)

# The work itself: gets its own session, returns (message, result id).
Work = Callable[[Session], tuple[str, int | None]]


def start(
    session: Session,
    sessions: sessionmaker,
    runner: TaskRunner,
    kind: str,
    work: Work,
    *,
    job_id: int | None = None,
) -> Run:
    """Start `work` unless the same kind of run (for the same job) is already going."""
    current = latest(session, kind, job_id)
    if current and current.state == "running":
        return current
    with transaction(session):
        run = runs_db.add(session, Run(kind=kind, job_id=job_id, state="running", owner=OWNER))
    run_id = run.id
    runner.submit(lambda: _execute(sessions, run_id, work))
    return run


def latest(session: Session, kind: str, job_id: int | None = None) -> Run | None:
    """The latest run; one whose process is gone is marked interrupted first."""
    run = runs_db.latest(session, kind, job_id)
    if run and run.state == "running" and _abandoned(run):
        with transaction(session):
            _finish(run, "error", INTERRUPTED, None)
    return run


def _abandoned(run: Run, now: datetime | None = None) -> bool:
    if run.owner == OWNER:
        return False  # ours, still going (failures are always recorded, see _execute)
    if run.owner.rsplit(":", 1)[0] == HOST:
        return True  # an earlier process on this machine, i.e. before a restart
    return run.started_at < (now or datetime.now(UTC)) - STALE_AFTER


def _execute(sessions: sessionmaker, run_id: int, work: Work) -> None:
    with sessions() as session:
        try:
            message, result_id = work(session)
            state = "done"
        except EXPECTED as e:
            session.rollback()
            state, message, result_id = "error", str(e), None
        except Exception as e:  # noqa: BLE001 - a background run must never die silently
            log.exception("run %s failed", run_id)
            session.rollback()
            state, message = "error", f"Unexpected error ({type(e).__name__}): {e}"
            result_id = None
        with transaction(session):
            _finish(runs_db.get(session, run_id), state, message, result_id)


def _finish(run: Run, state: str, message: str, result_id: int | None) -> None:
    run.state, run.message, run.result_id = state, message, result_id
    run.finished_at = datetime.now(UTC)
