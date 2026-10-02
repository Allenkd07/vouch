"""Services own the transaction: they commit on success and store nothing on failure."""

from pathlib import Path

import pytest
from fakes import make_analysis

from vouch.applications import repository as applications_db
from vouch.applications.service import set_notes, set_status
from vouch.config import Settings
from vouch.discovery import repository as matches_db
from vouch.jobs import repository as jobs_db
from vouch.jobs.service import Scoring, add_job, load_scoring
from vouch.llm import FakeLLM, LLMError
from vouch.profile.schema import load_profile

EXAMPLE = Path(__file__).parent.parent / "profile" / "profile.example.yaml"
SETTINGS = Settings(llm_provider="fake")
POSTING = "Backend Engineer. Python, Celery and PostgreSQL. 3+ years building REST APIs."


def test_add_job_stores_nothing_when_extraction_fails(session):
    with pytest.raises(LLMError):
        add_job(session, FakeLLM(), SETTINGS, text=POSTING, company="Rollback Co")
    assert all(j.company != "Rollback Co" for j in jobs_db.recent(session, 50))


def test_add_job_reads_requirements_and_scores_fit(session):
    llm = FakeLLM(lambda prompt, schema: make_analysis())
    scoring = Scoring(load_profile(EXAMPLE))
    job, changed = add_job(session, llm, SETTINGS, text=POSTING, scoring=scoring)

    assert changed and job.analysis is not None
    match = matches_db.get_match(session, job.id)
    assert match is not None and match.score is not None


def test_load_scoring_is_none_without_a_profile(tmp_path):
    assert load_scoring(tmp_path / "missing.yaml", tmp_path / "search.yaml") is None
    assert load_scoring(EXAMPLE, tmp_path / "search.yaml").accept_years_up_to == 5


def test_tracker_status_and_notes(session):
    llm = FakeLLM(lambda prompt, schema: make_analysis())
    job, _ = add_job(session, llm, SETTINGS, text=POSTING + " Tracker test.")

    assert set_notes(session, job.id, "not tracked yet") is None
    app = set_status(session, job.id, "applied")
    assert app.status == "applied" and app.applied_at is not None
    assert set_notes(session, job.id, "Referred by a friend").notes == "Referred by a friend"

    with pytest.raises(ValueError, match="unknown status"):
        set_status(session, job.id, "ghosted")
    assert set_status(session, job.id, None) is None
    assert applications_db.for_job(session, job.id) is None
