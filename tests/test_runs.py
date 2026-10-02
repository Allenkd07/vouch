"""Background runs are recorded in the database and survive restarts."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session, sessionmaker

from vouch.runs import repository as runs_db
from vouch.runs import service as runs
from vouch.runs.models import Run
from vouch.runs.runner import InlineRunner


class HeldRunner:
    """Keeps tasks instead of running them, so a run stays "running"."""

    def __init__(self):
        self.tasks = []

    def submit(self, task):
        self.tasks.append(task)


@pytest.fixture
def sessions(session):
    # The worker gets its own session on the test's connection (rolled back afterwards).
    return sessionmaker(bind=session.connection(), join_transaction_mode="create_savepoint")


def test_finished_run_records_result(session, sessions):
    run = runs.start(session, sessions, InlineRunner(), "test", lambda s: ("ok", 42))
    session.refresh(run)
    assert (run.state, run.message, run.result_id) == ("done", "ok", 42)
    assert run.finished_at is not None


def test_failures_are_recorded_with_their_message(session, sessions):
    def expected(s: Session):
        raise ValueError("job 7 has not been analysed")

    def unexpected(s: Session):
        raise RuntimeError("disk full")

    first = runs.start(session, sessions, InlineRunner(), "test", expected)
    session.refresh(first)
    assert (first.state, first.message) == ("error", "job 7 has not been analysed")

    second = runs.start(session, sessions, InlineRunner(), "test", unexpected)
    session.refresh(second)
    assert second.message == "Unexpected error (RuntimeError): disk full"


def test_a_running_run_is_not_started_twice(session, sessions):
    held = HeldRunner()
    first = runs.start(session, sessions, held, "test", lambda s: ("", None))
    again = runs.start(session, sessions, held, "test", lambda s: ("", None))
    assert again.id == first.id and len(held.tasks) == 1


def _running(session, owner, started_at):
    run = Run(kind="test", state="running", owner=owner, started_at=started_at)
    return runs_db.add(session, run)


def test_runs_left_by_an_earlier_process_are_marked_interrupted(session):
    now = datetime.now(UTC)
    _running(session, f"{runs.HOST}:0", now)  # this machine, a process that's gone
    run = runs.latest(session, "test")
    assert (run.state, run.message) == ("error", runs.INTERRUPTED)


def test_runs_on_other_machines_count_as_abandoned_only_when_stale(session):
    now = datetime.now(UTC)
    _running(session, "other-host:1", now - timedelta(minutes=1))
    assert runs.latest(session, "test").state == "running"
    _running(session, "other-host:1", now - runs.STALE_AFTER - timedelta(minutes=1))
    assert runs.latest(session, "test").state == "error"
