"""Adding jobs and reading their requirements (used by the CLI and the web app)."""

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy.orm import Session

from vouch.boards import fetch_job, manual_job
from vouch.config import Settings
from vouch.db import Job
from vouch.discovery.config import load_search
from vouch.discovery.run import rescore_job
from vouch.jobs import repository as jobs_db
from vouch.jobs.requirements import JobAnalysis, VerifiedAnalysis, analyze_job
from vouch.llm import LLM
from vouch.profile.schema import Profile, load_profile
from vouch.transaction import transaction


@dataclass
class Scoring:
    """What's needed to score a job's fit right after its requirements are read."""

    profile: Profile
    accept_years_up_to: float = 5


def load_scoring(profile_path: Path, search_path: Path) -> Scoring | None:
    """None when the profile is missing or invalid; the score then appears after the next
    discovery run instead."""
    try:
        years = load_search(search_path).ranking.accept_years_up_to if search_path.exists() else 5
        return Scoring(load_profile(profile_path), years)
    except (OSError, ValidationError):
        return None


def job_analysis(job: Job) -> JobAnalysis | None:
    return VerifiedAnalysis.model_validate(job.analysis).analysis if job.analysis else None


def add_job(
    session: Session,
    llm: LLM,
    settings: Settings,
    *,
    url: str | None = None,
    text: str | None = None,
    company: str | None = None,
    title: str | None = None,
    reanalyze: bool = False,
    scoring: Scoring | None = None,
) -> tuple[Job, bool]:
    """Fetch (or take pasted text), store, read requirements and score a job.
    Returns (job, changed). `llm` should be the extraction LLM (llm.get_extraction_llm).
    Raises FetchError, LLMError or ValueError, and then stores nothing."""
    if not url and not text:
        raise ValueError("give a job link or paste the job description")
    fetched = manual_job(text, url, company, title) if text else fetch_job(url)
    fetched.company = company or fetched.company
    fetched.title = title or fetched.title
    with transaction(session):
        job, changed = jobs_db.upsert_job(session, fetched)
        if reanalyze or jobs_db.needs_analysis(job):
            _analyze(session, job, llm, settings, scoring)
    return job, changed


def analyze(
    session: Session, job: Job, llm: LLM, settings: Settings, scoring: Scoring | None = None
) -> None:
    """Read (or re-read) a stored job's requirements and update its fit score. Raises LLMError,
    and then stores nothing."""
    with transaction(session):
        _analyze(session, job, llm, settings, scoring)


def _analyze(
    session: Session, job: Job, llm: LLM, settings: Settings, scoring: Scoring | None
) -> None:
    result = analyze_job(job.description, llm, title=job.title)
    jobs_db.save_analysis(job, result, settings.extraction_model)
    if scoring:
        rescore_job(session, job, scoring.profile, accept_years_up_to=scoring.accept_years_up_to)
