"""Discovery: run the pipeline in `discovery.steps`, and rescore single jobs between runs."""

import time
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
from sqlalchemy.orm import Session

from vouch.config import Settings
from vouch.db import Job
from vouch.discovery import repository as matches_db
from vouch.discovery.config import SearchConfig
from vouch.discovery.fit import score_fit
from vouch.discovery.steps import PIPELINE, DiscoveryContext, DiscoveryResult, Step
from vouch.jobs.requirements import VerifiedAnalysis
from vouch.llm import LLM
from vouch.profile.schema import Profile


def discover(
    session: Session,
    config: SearchConfig,
    profile: Profile,
    llm: LLM,
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    analyze: int | None = None,
    now: datetime | None = None,
    log: Callable[[str], None] = lambda _: None,
    steps: list[Step] = PIPELINE,
) -> DiscoveryResult:
    """Fetch, store, embed, extract and score; commits after each step."""
    ctx = DiscoveryContext(
        session=session,
        config=config,
        profile=profile,
        llm=llm,
        settings=settings,
        now=now or datetime.now(UTC),
        analyze=config.ranking.analyze_per_run if analyze is None else analyze,
        client=client,
        log=log,
    )
    for step in steps:
        started = time.perf_counter()
        step(ctx)
        session.commit()
        ctx.result.seconds[step.__name__] = round(time.perf_counter() - started, 2)
    return ctx.result


def rescore_job(
    session: Session,
    job: Job,
    profile: Profile,
    now: datetime | None = None,
    accept_years_up_to: float = 5,
) -> None:
    """Recompute one job's fit (e.g. right after its requirements were extracted), keeping the
    similarity from the last discovery run."""
    if not job.analysis:
        return
    match = matches_db.get_match(session, job.id)
    similarity = match.similarity if match else None
    analysis = VerifiedAnalysis.model_validate(job.analysis).analysis
    fit = score_fit(profile, analysis, similarity, now, accept_years_up_to=accept_years_up_to)
    matches_db.upsert_match(session, job.id, fit, similarity)
