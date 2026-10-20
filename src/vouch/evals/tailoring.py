"""Tailoring eval: for a fixed set of jobs (evals/tailoring.yaml), how much of each job does the
tailored resume cover, how often are rewrites thrown out, and does it gain the job's keywords?

Calls the tailoring pipeline directly, so runs don't create resume versions. Faithfulness of the
final text is checked separately, by a judge independent of the one inside the pipeline."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel

from vouch.evals.faithfulness import Claim, audit, resume_items
from vouch.evals.faithfulness import summary_source as labelled_source
from vouch.jobs.requirements import JobAnalysis
from vouch.llm import LLM, LLMError
from vouch.profile.schema import Profile
from vouch.tailoring.evidence import map_evidence
from vouch.tailoring.pipeline import (
    TailorResult,
    summary_source,
    tailor,
    verify_rewrites,
    vocabulary,
)
from vouch.tailoring.rewrite import rewrite
from vouch.tailoring.select import select_bullets

DEFAULT_SET = Path("evals/tailoring.yaml")
COVERED = {"strong", "partial", "education"}  # on the page; skills-only doesn't count
NOT_RETURNED = {
    "no rewrite returned",
    "no summary returned",
}  # reasons from pipeline.verify_rewrites


class TailoringSet(BaseModel):
    jobs: list[str]  # job URLs, as stored by discovery or `vouch job add`


def load_set(path: Path = DEFAULT_SET) -> TailoringSet:
    return TailoringSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass
class JobScore:
    url: str
    title: str
    company: str
    must_coverage: float | None  # weighted share of must-haves backed on the page
    must_possible: float | None  # ... that the profile could back at all (bullets or education)
    left_off: list[str]  # must-haves the profile backs but the page doesn't
    attempted: int  # bullets on the page, plus the summary
    rejected: int  # of those, rewrites the honesty check threw out (original kept)
    not_returned: int  # ... and ones the rewriter skipped (original kept, nothing to check)
    keywords_before: int
    keywords_after: int
    keywords_total: int
    pages: int
    max_pages: int
    audited: int | None = None  # changed texts the independent judge checked (None: no judge)
    fabrications: list[tuple[str, str, list[Claim]]] = field(default_factory=list)
    # ^ (id, text, unsupported claims) the pipeline's honesty check let through


def measure(
    analysis: JobAnalysis, result: TailorResult, *, url: str = "", max_pages: int = 1
) -> JobScore:
    report, doc = result.report, result.doc
    must = [c for c in report.coverage if c.must_have]
    weights = {r.name: r.weight for r in analysis.requirements}

    def possible(name: str) -> bool:
        item = result.evidence.for_requirement(name)
        return bool(item and (item.bullets or item.in_education))

    def share(ok: Callable[[str, str], bool]) -> float | None:
        total = sum(c.weight for c in must)
        return sum(c.weight for c in must if ok(c.requirement, c.status)) / total if total else None

    on_page = {b.id for e in doc.experience + doc.projects for b in e.bullets}
    summary = {"summary"} if doc.summary or "summary" in report.fallbacks else set()
    fallbacks = {i: reasons for i, reasons in report.fallbacks.items() if i in on_page | summary}
    skipped = {i for i, reasons in fallbacks.items() if set(reasons) <= NOT_RETURNED}
    return JobScore(
        url=url,
        title=analysis.title,
        company=analysis.company or "?",
        must_coverage=share(lambda _, status: status in COVERED),
        must_possible=share(lambda name, status: status in COVERED or possible(name)),
        left_off=[
            c.requirement
            for c in sorted(must, key=lambda c: -weights.get(c.requirement, 0))
            if c.status not in COVERED and possible(c.requirement)
        ],
        attempted=len(on_page | summary),
        rejected=len(fallbacks) - len(skipped),
        not_returned=len(skipped),
        keywords_before=len(report.keywords_before),
        keywords_after=len(report.keywords_after),
        keywords_total=len(analysis.keywords),
        pages=report.pages,
        max_pages=max_pages,
    )


@dataclass
class Run:
    scores: list[JobScore] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # URLs not in the database, or unanalysed
    stopped: str | None = None  # LLM error that ended the run early

    def mean(self, attr: str) -> float | None:
        values = [getattr(s, attr) for s in self.scores if getattr(s, attr) is not None]
        return sum(values) / len(values) if values else None

    @property
    def rejection_rate(self) -> float | None:
        attempted = sum(s.attempted for s in self.scores)
        return sum(s.rejected for s in self.scores) / attempted if attempted else None

    @property
    def audited(self) -> int | None:
        counts = [s.audited for s in self.scores if s.audited is not None]
        return sum(counts) if counts else None


def run_eval(
    jobs: list[tuple[str, JobAnalysis | None]],
    profile: Profile,
    llm: LLM,
    *,
    judge: LLM | None = None,
    max_pages: int = 1,
    log: Callable[[str], None] = lambda _: None,
) -> Run:
    """Tailor each (url, analysis) in turn, then (with `judge`) audit the changed text in one
    request per job. An LLM error (quota, overload) stops the run and keeps the jobs already
    measured."""
    run = Run()
    for url, analysis in jobs:
        if analysis is None:
            run.missing.append(url)
            continue
        log(f"Tailoring for {analysis.title} @ {analysis.company}...")
        try:
            result = tailor(profile, analysis, llm, max_pages=max_pages)
            score = measure(analysis, result, url=url, max_pages=max_pages)
            if judge is not None:
                items = resume_items(profile, result.doc, result.report.dropped_for_space)
                audits = audit(items, judge)
                score.audited = len(items)
                score.fabrications = [
                    (i, text, audits[i].unsupported)
                    for i, _, text in items
                    if audits[i].unsupported
                ]
        except LLMError as e:
            run.stopped = str(e)
            break
        run.scores.append(score)
    return run


class SummaryDraft(BaseModel):
    """One first-draft summary: what the pipeline's honesty check made of it, and what the
    independent judge did."""

    text: str | None  # None: the rewriter returned no summary
    rejected: str | None  # the honesty check's reason; None if it passed
    unsupported: list[Claim] = []  # independent judge: misleading claims
    minor: list[Claim] = []  # ... and unsupported wording it doesn't count

    @property
    def escaped(self) -> bool:
        return self.rejected is None and bool(self.unsupported)


class SummaryJob(BaseModel):
    url: str
    title: str
    company: str
    source: str = ""  # what the drafts were written from, labelled for the judge
    bullet_ids: list[str] = []  # the bullet selection, so a later run can reuse it
    supports: dict[str, list[str]] = {}  # requirement -> bullet ids, as the rewriter sees it
    drafts: list[SummaryDraft] = []


@dataclass
class SummaryRun:
    jobs: list[SummaryJob] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    stopped: str | None = None

    def drafts(self) -> list[SummaryDraft]:
        return [d for j in self.jobs for d in j.drafts if d.text]


def audit_drafts(job: SummaryJob, judge: LLM) -> None:
    """Audit every draft of `job` in one request, replacing any earlier findings."""
    items = [(f"s{n}", job.source, d.text) for n, d in enumerate(job.drafts) if d.text]
    for item_id, result in audit(items, judge).items():
        draft = job.drafts[int(item_id[1:])]
        draft.unsupported, draft.minor = result.unsupported, result.minor


def sample_summaries(
    jobs: list[tuple[str, JobAnalysis | None]],
    profile: Profile,
    llm: LLM,
    judge: LLM,
    *,
    samples: int = 3,
    log: Callable[[str], None] = lambda _: None,
    on_job: Callable[[SummaryJob], None] = lambda _: None,
    selections: dict[str, SummaryJob] | None = None,
) -> SummaryRun:
    """Summaries vary run to run, so one tailoring run can't say whether a change helped.
    For each job, write `samples` first drafts from the same bullet selection, put each
    through the pipeline's honesty check exactly as tailoring does, and audit them all with
    the independent judge (one request per job). About 2 requests per sample, plus 2 per job.
    No retries: this measures how often drafts misstate the work and how many of those the
    check catches.

    selections: saved jobs (by URL) whose bullet selection to reuse, so a before/after
    comparison differs only in the drafts; it also skips evidence mapping, the slowest call."""
    run = SummaryRun()
    bullets = {b.id: b for _, b in profile.iter_bullets()}
    for url, analysis in jobs:
        if analysis is None:
            run.missing.append(url)
            continue
        log(f"Sampling {samples} summaries for {analysis.title} @ {analysis.company}...")
        try:
            if earlier := (selections or {}).get(url):
                ids, supports = earlier.bullet_ids, earlier.supports
            else:
                selection = select_bullets(profile, analysis, map_evidence(profile, analysis, llm))
                ids, supports = selection.bullet_ids(), selection.supports
            chosen = [bullets[i] for i in ids]
            job = SummaryJob(
                url=url,
                title=analysis.title,
                company=analysis.company or "?",
                source=labelled_source(profile, chosen),
                bullet_ids=ids,
                supports=supports,
            )
            source, vocab = summary_source(profile, chosen), vocabulary(profile, analysis)
            for _ in range(samples):
                drafted = rewrite(chosen, supports, profile, analysis, llm)
                _, rejected = verify_rewrites(
                    chosen, drafted, source, vocab, llm, check_summary=True
                )
                reasons = rejected.get("summary")
                job.drafts.append(
                    SummaryDraft(
                        text=(drafted.summary or "").strip() or None,
                        rejected="; ".join(reasons) if reasons else None,
                    )
                )
            audit_drafts(job, judge)
        except LLMError as e:
            run.stopped = str(e)
            break
        run.jobs.append(job)
        on_job(job)
    return run


def rescore(
    jobs: list[SummaryJob],
    judge: LLM,
    *,
    log: Callable[[str], None] = lambda _: None,
    on_job: Callable[[SummaryJob], None] = lambda _: None,
) -> SummaryRun:
    """Audit saved drafts again (after changing the judge), without drafting new ones."""
    run = SummaryRun()
    for job in jobs:
        log(f"Re-auditing {len(job.drafts)} summaries for {job.title} @ {job.company}...")
        try:
            audit_drafts(job, judge)
        except LLMError as e:
            run.stopped = str(e)
            break
        run.jobs.append(job)
        on_job(job)
    return run


def load_drafts(path: Path) -> list[SummaryJob]:
    """Saved by `vouch eval summaries --save`: one job per line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [SummaryJob.model_validate_json(line) for line in lines if line.strip()]
