"""One discovery run: fetch boards -> filter -> store -> embed -> analyse the top few -> score.

Cost grows left to right, so each step only sees what the previous one kept: fetching and
filtering are free, embeddings are one batched request, and only `analyze_per_run` jobs reach
the LLM, `batch_size` postings per request."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
from sqlalchemy.orm import Session

from vouch.boards import BOARDS, FetchedJob, list_jobs, new_client
from vouch.config import Settings
from vouch.db import Job, JobEmbedding
from vouch.discovery import repository as matches_db
from vouch.discovery.config import Company, SearchConfig
from vouch.discovery.filters import rejection_reason
from vouch.discovery.fit import score_fit
from vouch.jobs import repository as jobs_db
from vouch.jobs.requirements import VerifiedAnalysis, analyze_jobs
from vouch.llm import LLM, LLMError
from vouch.profile.schema import Profile

EMBED_DIMENSIONS = 768


@dataclass
class CompanyResult:
    name: str
    listed: int = 0
    kept: list[FetchedJob] = field(default_factory=list)
    error: str | None = None


@dataclass
class DiscoveryResult:
    companies: list[CompanyResult]
    new_or_changed: int = 0
    embedded: int = 0
    analyzed: list[int] = field(default_factory=list)
    requests: int = 0  # extraction requests made
    scored: int = 0
    notes: list[str] = field(default_factory=list)


def fetch_company(company: Company, config: SearchConfig, client: httpx.Client) -> CompanyResult:
    result = CompanyResult(company.name)
    try:
        jobs = list_jobs(company, client)
        result.listed = len(jobs)
        kept = [j for j in jobs if rejection_reason(j, config.filters) is None]
        board = BOARDS[company.ats]
        if board.partial_listing:  # descriptions come from a second request per posting
            kept = [board.details(j, company, client) for j in kept]
            kept = [j for j in kept if rejection_reason(j, config.filters) is None]
        result.kept = kept
    except (httpx.HTTPError, KeyError, ValueError) as e:
        result.error = f"{type(e).__name__}: {e}"
    return result


def fetch_all(config: SearchConfig, client: httpx.Client | None = None) -> list[CompanyResult]:
    client = client or new_client()
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda c: fetch_company(c, config, client), config.companies))


def profile_text(profile: Profile) -> str:
    parts = [profile.summary or "", "Skills: " + ", ".join(profile.skills.all())]
    parts += [b.text for _, b in profile.iter_bullets()]
    return "\n".join(parts)


def job_text(job: Job) -> str:
    # The opening of a posting says what the role is; 2,000 characters keeps token use low.
    return f"{job.title}\n{job.company}\n{job.description[:2000]}"


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
) -> DiscoveryResult:
    now = now or datetime.now(UTC)
    analyze = config.ranking.analyze_per_run if analyze is None else analyze

    # 1. Fetch and filter (free).
    companies = fetch_all(config, client)
    result = DiscoveryResult(companies=companies)
    for c in companies:
        log(f"{c.name}: {c.error or f'{len(c.kept)} of {c.listed} jobs kept'}")

    # 2. Store what passed the filters.
    for c in companies:
        for fetched in c.kept:
            job, changed = jobs_db.upsert_job(session, fetched)
            job.last_seen_at = now
            result.new_or_changed += changed
    session.commit()

    active = jobs_db.active(session, now)

    # 3. Embed jobs whose text changed since their last embedding (one batched request).
    similarity: dict[int, float] = {}
    try:
        similarity, result.embedded = _embed_and_rank(session, active, profile, llm, settings)
    except LLMError as e:
        # Keep the similarity from earlier runs rather than wiping it; new jobs go without.
        similarity = matches_db.stored_similarities(session, [j.id for j in active])
        result.notes.append(f"Similarity not updated (kept earlier values): {e}")

    # 4. Extract requirements for the most similar unanalysed jobs (the LLM step), several
    #    postings per request so the daily quota covers many more jobs.
    candidates = sorted(
        (j for j in active if jobs_db.needs_analysis(j)), key=lambda j: -similarity.get(j.id, 0.0)
    )
    todo = candidates[:analyze]
    batch = config.ranking.batch_size
    skipped = 0
    for start in range(0, len(todo), batch):
        chunk = todo[start : start + batch]
        try:
            analyses = analyze_jobs([(str(j.id), j.title, j.description) for j in chunk], llm)
        except LLMError as e:
            result.notes.append(f"Stopped extracting requirements: {e}")
            break
        for job in chunk:
            if (analysis := analyses.get(str(job.id))) is None:
                skipped += 1  # the model left it out; a later run retries it
                continue
            jobs_db.save_analysis(job, analysis, settings.extraction_model)
            result.analyzed.append(job.id)
            log(f"Analysed: {job.title} @ {job.company}")
        session.commit()
        result.requests += 1
    if skipped:
        result.notes.append(
            f"{skipped} job(s) were left out of a batch response; retried next run."
        )
    if len(candidates) > len(result.analyzed):
        result.notes.append(
            f"{len(candidates) - len(result.analyzed)} job(s) still need requirement extraction; "
            "they'll be picked up by later runs, most similar first."
        )

    # 5. Score every active job (free).
    for job in active:
        fit = None
        if job.analysis:
            analysis = VerifiedAnalysis.model_validate(job.analysis).analysis
            fit = score_fit(
                profile,
                analysis,
                similarity.get(job.id),
                now,
                accept_years_up_to=config.ranking.accept_years_up_to,
            )
        matches_db.upsert_match(session, job.id, fit, similarity.get(job.id))
        result.scored += 1
    session.commit()
    return result


def _embed_and_rank(
    session: Session, jobs: list[Job], profile: Profile, llm: LLM, settings: Settings
) -> tuple[dict[int, float], int]:
    model = f"{settings.embedding_model}@{EMBED_DIMENSIONS}"
    job_ids = [j.id for j in jobs]
    existing = matches_db.embeddings_for(session, job_ids)
    stale = [
        j
        for j in jobs
        if (e := existing.get(j.id)) is None or e.model != model or e.content_hash != j.content_hash
    ]
    vectors = llm.embed(
        [profile_text(profile)] + [job_text(j) for j in stale], dimensions=EMBED_DIMENSIONS
    )
    profile_vec, job_vecs = vectors[0], vectors[1:]
    if len(profile_vec) != EMBED_DIMENSIONS:
        # Vectors of different sizes can't be compared with the stored ones.
        raise LLMError(
            f"{settings.embedding_model} returned {len(profile_vec)}-dimension vectors, "
            f"expected {EMBED_DIMENSIONS}"
        )
    rows = []
    for job, vec in zip(stale, job_vecs, strict=True):
        row = existing.get(job.id) or JobEmbedding(job_id=job.id)
        row.model, row.content_hash, row.embedding = model, job.content_hash, vec
        rows.append(row)
    matches_db.save_embeddings(session, rows)
    return matches_db.similarities(session, model, job_ids, profile_vec), len(stale)


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
