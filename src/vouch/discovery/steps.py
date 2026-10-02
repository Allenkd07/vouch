"""The steps of a discovery run, in order: fetch -> store -> embed -> extract -> score.

Cost grows left to right, so each step only sees what the previous one kept: fetching and
filtering are free, embeddings are one batched request, and only `analyze_per_run` jobs reach
the LLM, `batch_size` postings per request. Each step reads and updates a shared
`DiscoveryContext`; `discovery.service.discover` runs them and commits after each one."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from sqlalchemy.orm import Session

from vouch.config import Settings
from vouch.db import Job, JobEmbedding
from vouch.discovery import repository as matches_db
from vouch.discovery.config import SearchConfig
from vouch.discovery.fetch import CompanyResult, fetch_all
from vouch.discovery.fit import score_fit
from vouch.jobs import repository as jobs_db
from vouch.jobs.requirements import VerifiedAnalysis, analyze_jobs
from vouch.llm import LLM, LLMError
from vouch.profile.schema import Profile

EMBED_DIMENSIONS = 768


@dataclass
class DiscoveryResult:
    companies: list[CompanyResult] = field(default_factory=list)
    new_or_changed: int = 0
    embedded: int = 0
    analyzed: list[int] = field(default_factory=list)
    requests: int = 0  # extraction requests made
    scored: int = 0
    notes: list[str] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)  # time spent in each step


@dataclass
class DiscoveryContext:
    session: Session
    config: SearchConfig
    profile: Profile
    llm: LLM
    settings: Settings
    now: datetime
    analyze: int  # jobs to send for requirement extraction
    client: httpx.Client | None = None
    log: Callable[[str], None] = lambda _: None
    result: DiscoveryResult = field(default_factory=DiscoveryResult)
    active: list[Job] = field(default_factory=list)  # jobs still listed, after `store`
    similarity: dict[int, float] = field(default_factory=dict)  # job id -> similarity


Step = Callable[[DiscoveryContext], None]


def fetch(ctx: DiscoveryContext) -> None:
    """Read every board and keep what passes the title/location/age filters (free)."""
    ctx.result.companies = fetch_all(ctx.config, ctx.client)
    for c in ctx.result.companies:
        ctx.log(f"{c.name}: {c.error or f'{len(c.kept)} of {c.listed} jobs kept'}")


def store(ctx: DiscoveryContext) -> None:
    """Save the kept postings and note which jobs are still listed."""
    for c in ctx.result.companies:
        for fetched in c.kept:
            job, changed = jobs_db.upsert_job(ctx.session, fetched)
            job.last_seen_at = ctx.now
            ctx.result.new_or_changed += changed
    ctx.session.flush()
    ctx.active = jobs_db.active(ctx.session, ctx.now)


def embed(ctx: DiscoveryContext) -> None:
    """Embed jobs whose text changed since their last embedding (one batched request) and rank
    every active job by similarity to the profile."""
    try:
        ctx.similarity, ctx.result.embedded = _embed_and_rank(ctx)
    except LLMError as e:
        # Keep the similarity from earlier runs rather than wiping it; new jobs go without.
        ctx.similarity = matches_db.stored_similarities(ctx.session, [j.id for j in ctx.active])
        ctx.result.notes.append(f"Similarity not updated (kept earlier values): {e}")


def extract(ctx: DiscoveryContext) -> None:
    """Read requirements for the most similar unanalysed jobs (the LLM step), several postings
    per request so the daily quota covers many more jobs. Commits after each request, so a
    quota error part-way keeps what was already read."""
    result = ctx.result
    candidates = sorted(
        (j for j in ctx.active if jobs_db.needs_analysis(j)),
        key=lambda j: -ctx.similarity.get(j.id, 0.0),
    )
    todo = candidates[: ctx.analyze]
    batch = ctx.config.ranking.batch_size
    skipped = 0
    for start in range(0, len(todo), batch):
        chunk = todo[start : start + batch]
        try:
            analyses = analyze_jobs([(str(j.id), j.title, j.description) for j in chunk], ctx.llm)
        except LLMError as e:
            result.notes.append(f"Stopped extracting requirements: {e}")
            break
        for job in chunk:
            if (analysis := analyses.get(str(job.id))) is None:
                skipped += 1  # the model left it out; a later run retries it
                continue
            jobs_db.save_analysis(job, analysis, ctx.settings.extraction_model)
            result.analyzed.append(job.id)
            ctx.log(f"Analysed: {job.title} @ {job.company}")
        ctx.session.commit()
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


def score(ctx: DiscoveryContext) -> None:
    """Estimate fit for every active job from its requirements and similarity (free)."""
    for job in ctx.active:
        fit = None
        if job.analysis:
            analysis = VerifiedAnalysis.model_validate(job.analysis).analysis
            fit = score_fit(
                ctx.profile,
                analysis,
                ctx.similarity.get(job.id),
                ctx.now,
                accept_years_up_to=ctx.config.ranking.accept_years_up_to,
            )
        matches_db.upsert_match(ctx.session, job.id, fit, ctx.similarity.get(job.id))
        ctx.result.scored += 1


PIPELINE: list[Step] = [fetch, store, embed, extract, score]


def profile_text(profile: Profile) -> str:
    parts = [profile.summary or "", "Skills: " + ", ".join(profile.skills.all())]
    parts += [b.text for _, b in profile.iter_bullets()]
    return "\n".join(parts)


def job_text(job: Job) -> str:
    # The opening of a posting says what the role is; 2,000 characters keeps token use low.
    return f"{job.title}\n{job.company}\n{job.description[:2000]}"


def _embed_and_rank(ctx: DiscoveryContext) -> tuple[dict[int, float], int]:
    model = f"{ctx.settings.embedding_model}@{EMBED_DIMENSIONS}"
    job_ids = [j.id for j in ctx.active]
    existing = matches_db.embeddings_for(ctx.session, job_ids)
    stale = [
        j
        for j in ctx.active
        if (e := existing.get(j.id)) is None or e.model != model or e.content_hash != j.content_hash
    ]
    vectors = ctx.llm.embed(
        [profile_text(ctx.profile)] + [job_text(j) for j in stale], dimensions=EMBED_DIMENSIONS
    )
    profile_vec, job_vecs = vectors[0], vectors[1:]
    if len(profile_vec) != EMBED_DIMENSIONS:
        # Vectors of different sizes can't be compared with the stored ones.
        raise LLMError(
            f"{ctx.settings.embedding_model} returned {len(profile_vec)}-dimension vectors, "
            f"expected {EMBED_DIMENSIONS}"
        )
    rows = []
    for job, vec in zip(stale, job_vecs, strict=True):
        row = existing.get(job.id) or JobEmbedding(job_id=job.id)
        row.model, row.content_hash, row.embedding = model, job.content_hash, vec
        rows.append(row)
    matches_db.save_embeddings(ctx.session, rows)
    return matches_db.similarities(ctx.session, model, job_ids, profile_vec), len(stale)
