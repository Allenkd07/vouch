"""One discovery run: fetch boards -> filter -> store -> embed -> analyse the top few -> score.

Cost grows left to right, so each step only sees what the previous one kept: fetching and
filtering are free, embeddings are one batched request, and only `analyze_per_run` jobs reach
the LLM."""

from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from vouch.config import Settings
from vouch.db import Application, Job, JobEmbedding, Match
from vouch.discovery.boards import list_jobs, workday_details
from vouch.discovery.config import Company, SearchConfig
from vouch.discovery.filters import rejection_reason
from vouch.discovery.fit import Fit, score_fit
from vouch.jobs.requirements import VerifiedAnalysis, analyze_job
from vouch.jobs.sources import USER_AGENT, FetchedJob
from vouch.jobs.store import needs_analysis, save_analysis, upsert_job
from vouch.llm import LLM, LLMError
from vouch.profile.schema import Profile

EMBED_DIMENSIONS = 768
ACTIVE_DAYS = 3  # a discovered job not seen on any board for this long counts as closed


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
    scored: int = 0
    notes: list[str] = field(default_factory=list)


def fetch_company(company: Company, config: SearchConfig, client: httpx.Client) -> CompanyResult:
    result = CompanyResult(company.name)
    try:
        jobs = list_jobs(company, client)
        result.listed = len(jobs)
        kept = [j for j in jobs if rejection_reason(j, config.filters) is None]
        if company.ats == "workday":
            kept = [workday_details(j, company, client) for j in kept]
            kept = [j for j in kept if rejection_reason(j, config.filters) is None]
        result.kept = kept
    except (httpx.HTTPError, KeyError, ValueError) as e:
        result.error = f"{type(e).__name__}: {e}"
    return result


def fetch_all(config: SearchConfig, client: httpx.Client | None = None) -> list[CompanyResult]:
    client = client or httpx.Client(
        timeout=30, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda c: fetch_company(c, config, client), config.companies))


def profile_text(profile: Profile) -> str:
    parts = [profile.summary or "", "Skills: " + ", ".join(profile.skills.all())]
    parts += [b.text for _, b in profile.iter_bullets()]
    return "\n".join(parts)


def job_text(job: Job) -> str:
    # The opening of a posting says what the role is; 2,000 characters keeps token use low.
    return f"{job.title}\n{job.company}\n{job.description[:2000]}"


def active_jobs_filter(now: datetime):
    """Manually added jobs always; discovered ones while they're still listed."""
    return or_(Job.last_seen_at.is_(None), Job.last_seen_at >= now - timedelta(days=ACTIVE_DAYS))


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
            job, changed = upsert_job(session, fetched)
            job.last_seen_at = now
            result.new_or_changed += changed
    session.commit()

    active = session.scalars(select(Job).where(active_jobs_filter(now))).all()

    # 3. Embed jobs whose text changed since their last embedding (one batched request).
    similarity: dict[int, float] = {}
    try:
        similarity, result.embedded = _embed_and_rank(session, active, profile, llm, settings)
    except LLMError as e:
        result.notes.append(f"Similarity ranking skipped: {e}")

    # 4. Extract requirements for the most similar unanalysed jobs (the LLM step).
    candidates = sorted(
        (j for j in active if needs_analysis(j, settings.llm_model)),
        key=lambda j: -similarity.get(j.id, 0.0),
    )
    for job in candidates[:analyze]:
        try:
            save_analysis(
                job, analyze_job(job.description, llm, title=job.title), settings.llm_model
            )
            session.commit()
            result.analyzed.append(job.id)
            log(f"Analysed: {job.title} @ {job.company}")
        except LLMError as e:
            result.notes.append(f"Stopped extracting requirements: {e}")
            break
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
            fit = score_fit(profile, analysis, similarity.get(job.id), now)
        _upsert_match(session, job.id, fit, similarity.get(job.id))
        result.scored += 1
    session.commit()
    return result


def _embed_and_rank(
    session: Session, jobs: list[Job], profile: Profile, llm: LLM, settings: Settings
) -> tuple[dict[int, float], int]:
    model = f"{settings.embedding_model}@{EMBED_DIMENSIONS}"
    existing = {
        e.job_id: e
        for e in session.scalars(
            select(JobEmbedding).where(JobEmbedding.job_id.in_([j.id for j in jobs]))
        )
    }
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
    for job, vec in zip(stale, job_vecs, strict=True):
        row = existing.get(job.id) or JobEmbedding(job_id=job.id)
        row.model, row.content_hash, row.embedding = model, job.content_hash, vec
        session.add(row)
    session.flush()

    # Cosine similarity in Postgres via pgvector (vectors are unit length).
    rows = session.execute(
        select(JobEmbedding.job_id, 1 - JobEmbedding.embedding.cosine_distance(profile_vec)).where(
            JobEmbedding.model == model, JobEmbedding.job_id.in_([j.id for j in jobs])
        )
    ).all()
    return {job_id: float(sim) for job_id, sim in rows}, len(stale)


def _upsert_match(session: Session, job_id: int, fit: Fit | None, similarity: float | None):
    values = {
        "job_id": job_id,
        "score": fit.score if fit else None,
        "similarity": similarity,
        "fit": fit.model_dump() if fit else None,
        "updated_at": datetime.now(UTC),
    }
    stmt = insert(Match).values(values)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Match.job_id],
            set_={k: stmt.excluded[k] for k in ("score", "similarity", "fit", "updated_at")},
        )
    )


def ranked_matches(session: Session, limit: int = 50, now: datetime | None = None):
    """[(Job, Match, Application | None)] best first: scored jobs by score, then the rest by
    similarity. Closed postings and rejected applications are left out."""
    now = now or datetime.now(UTC)
    return session.execute(
        select(Job, Match, Application)
        .join(Match, Match.job_id == Job.id)
        .outerjoin(Application, Application.job_id == Job.id)
        .where(active_jobs_filter(now))
        .where(or_(Application.status.is_(None), Application.status != "rejected"))
        .order_by(Match.score.desc().nulls_last(), Match.similarity.desc().nulls_last())
        .limit(limit)
    ).all()


def skill_gaps(rows, top: int = 30) -> list[tuple[str, int]]:
    """Missing must-have skills most often asked for across the best-scored matches."""
    counts: Counter[str] = Counter()
    scored = [m for _, m, _ in rows if m.fit][:top]
    for m in scored:
        counts.update({name.strip(): 1 for name in m.fit["missing_must"]})
    return counts.most_common(10)


def rescore_job(session: Session, job: Job, profile: Profile, now: datetime | None = None) -> None:
    """Recompute one job's fit (e.g. right after its requirements were extracted), keeping the
    similarity from the last discovery run."""
    if not job.analysis:
        return
    match = session.get(Match, job.id)
    similarity = match.similarity if match else None
    analysis = VerifiedAnalysis.model_validate(job.analysis).analysis
    _upsert_match(session, job.id, score_fit(profile, analysis, similarity, now), similarity)
