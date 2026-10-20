"""Profile + analysed job -> verified, page-limited tailored resume and a coverage report."""

from dataclasses import dataclass

from pydantic import BaseModel

from vouch.jobs.requirements import JobAnalysis, quote_in_text
from vouch.llm import LLM
from vouch.profile.schema import Bullet, Profile
from vouch.tailoring.document import (
    DocBullet,
    DocEntry,
    ResumeDoc,
    date_range,
    format_dates,
    tailor_skills,
)
from vouch.tailoring.evidence import EvidenceMap, map_evidence
from vouch.tailoring.render import page_count, render_pdf
from vouch.tailoring.rewrite import Rewrites, rewrite
from vouch.tailoring.select import Budget, Selection, drop_weakest, select_bullets
from vouch.tailoring.verify import deterministic_issues, judge


class RequirementCoverage(BaseModel):
    requirement: str
    must_have: bool
    weight: int
    status: str  # strong | partial | education | skills-only | gap
    bullet_ids: list[str]
    note: str


class TailorReport(BaseModel):
    coverage: list[RequirementCoverage]
    keywords_before: list[str]
    keywords_after: list[str]
    keywords_missing: list[str]
    fallbacks: dict[str, list[str]]  # bullet id (or "summary") -> reasons its rewrite was rejected
    dropped_for_space: list[str]
    pages: int


@dataclass
class TailorResult:
    doc: ResumeDoc
    report: TailorReport
    pdf: bytes
    evidence: EvidenceMap


def bullet_facts(b: Bullet) -> list[str]:
    f = b.facts
    return [*f.tools, *f.metrics, *f.scope]


def source_text(b: Bullet) -> str:
    """What a rewrite of `b` may draw on, run together for the deterministic check."""
    return " ".join([b.text, *bullet_facts(b)])


def summary_source(profile: Profile, chosen: list[Bullet]) -> str:
    """What the tailored summary may draw on: the profile summary, chosen bullets and skills."""
    return " ".join(
        [profile.summary or "", *(source_text(b) for b in chosen), *profile.skills.all()]
    )


# The same material with its parts named, for the LLM judges. Run together, a bullet's tools
# read like a skills list, and a summary's sources blur, so a tool recorded for one piece of
# work seems to back a claim about another ("Airflow" + "pipelines" -> "ETL with Airflow").
def labelled(text: str, facts: list[str]) -> str:
    return f"{text}\nFacts recorded for this work: {', '.join(facts)}" if facts else text


def labelled_summary(
    summary: str | None, bullets: list[tuple[str, list[str]]], skills: list[str]
) -> str:
    """bullets: (text, facts) for each bullet the summary may draw on."""
    return "\n".join(
        [
            f"Profile summary: {summary or '(none)'}",
            "Resume bullets (each tool or number backs only the work in its own bullet):",
            *(f"- {labelled(text, facts)}" for text, facts in bullets),
            f"Skills list (knowing a skill, not using it for any work above): {', '.join(skills)}",
        ]
    )


# Requirement categories whose options name technologies. Others hold phrases ("backend
# engineering", "debugging") that a fair summary may use and only the LLM judge can weigh.
TECHNICAL = {"language", "framework", "database", "cloud", "tool"}


def vocabulary(profile: Profile, analysis: JobAnalysis) -> set[str]:
    """Named technologies the verifier watches for: anything in the profile or the posting."""
    vocab = set(profile.skills.all())
    vocab |= {t for _, b in profile.iter_bullets() for t in b.facts.tools}
    vocab |= {o for r in analysis.requirements if r.category in TECHNICAL for o in r.options}
    return {v for v in vocab if len(v) > 1}


def tailor(
    profile: Profile,
    analysis: JobAnalysis,
    llm: LLM,
    *,
    max_pages: int = 1,
    budget: Budget | None = None,
) -> TailorResult:
    bullets = {b.id: b for _, b in profile.iter_bullets()}
    evidence = map_evidence(profile, analysis, llm)
    selection = select_bullets(profile, analysis, evidence, budget)
    chosen = [bullets[i] for i in selection.bullet_ids()]

    vocab = vocabulary(profile, analysis)
    summary_from = summary_source(profile, chosen)

    # Rewrite, verify, and give rejected items one retry with the reviewer's reasons.
    rewrites = rewrite(chosen, selection.supports, profile, analysis, llm)
    accepted, rejected = verify_rewrites(
        chosen, rewrites, summary_from, vocab, llm, check_summary=True
    )
    if rejected:
        retry_bullets = [bullets[i] for i in rejected if i in bullets]
        retry = rewrite(
            retry_bullets,
            selection.supports,
            profile,
            analysis,
            llm,
            with_summary="summary" in rejected,
            feedback={i: "; ".join(reasons) for i, reasons in rejected.items()},
        )
        accepted_retry, rejected = verify_rewrites(
            retry_bullets, retry, summary_from, vocab, llm, check_summary="summary" in rejected
        )
        accepted |= accepted_retry
    fallbacks = rejected

    final = {b.id: accepted.get(b.id, b.text) for b in chosen}
    final_summary = accepted.get("summary", profile.summary)

    # Render, dropping the weakest bullets until the page limit holds.
    dropped: list[str] = []
    while True:
        doc = build_doc(profile, analysis, selection, final, fallbacks, final_summary)
        pdf = render_pdf(doc)
        pages = page_count(pdf)
        if pages <= max_pages or len(selection.bullet_ids()) <= len(profile.experience):
            break
        removed = drop_weakest(selection, profile, analysis)
        if removed is None:
            break
        dropped.append(removed)

    report = build_report(profile, analysis, evidence, selection, doc, fallbacks, dropped, pages)
    return TailorResult(doc=doc, report=report, pdf=pdf, evidence=evidence)


def verify_rewrites(
    chosen: list[Bullet],
    rewrites: Rewrites,
    summary_source: str,
    vocab: set[str],
    llm: LLM,
    *,
    check_summary: bool,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Deterministic checks first, then the LLM judge on everything that changed.
    Returns (accepted text by id, rejection reasons by id); "summary" is one of the ids."""
    proposed = {r.id: r.text.strip() for r in rewrites.bullets}
    accepted: dict[str, str] = {}
    rejected: dict[str, list[str]] = {}
    to_judge: list[tuple[str, str, str]] = []
    for b in chosen:
        new = proposed.get(b.id)
        if not new:
            rejected[b.id] = ["no rewrite returned"]
        elif new == b.text:
            accepted[b.id] = new
        elif issues := deterministic_issues(new, source_text(b), vocab):
            rejected[b.id] = issues
        else:
            to_judge.append((b.id, source_text(b), new))
    if check_summary:
        summary = (rewrites.summary or "").strip()
        if not summary:
            rejected["summary"] = ["no summary returned"]
        elif issues := deterministic_issues(summary, summary_source, vocab):
            rejected["summary"] = issues
        else:
            to_judge.append(("summary", summary_source, summary))
    for item_id, verdict in judge(to_judge, llm).items():
        if verdict.supported:
            accepted[item_id] = next(text for i, _, text in to_judge if i == item_id)
        else:
            rejected[item_id] = [verdict.issue or "judge: unsupported"]
    return accepted, rejected


def build_doc(
    profile: Profile,
    analysis: JobAnalysis,
    selection: Selection,
    final: dict[str, str],
    fallbacks: dict[str, list[str]],
    summary: str | None,
) -> ResumeDoc:
    bullets = {b.id: b for _, b in profile.iter_bullets()}

    def doc_bullets(key: str) -> list[DocBullet]:
        return [
            DocBullet(
                id=i,
                text=final[i],
                original=bullets[i].text,
                proposed=final[i],
                flags=fallbacks.get(i, []),
                choice="original" if i in fallbacks else "tailored",
            )
            for i in selection.sections.get(key, [])
        ]

    experience = [
        DocEntry(
            heading=e.company,
            subheading=e.title,
            location=e.location,
            dates=date_range(e.start, e.end, current=e.end is None),
            description=e.description,
            bullets=doc_bullets(f"experience:{e.id}"),
        )
        for e in profile.experience
    ]
    projects = [
        DocEntry(
            heading=p.name,
            dates=format_dates(p.dates),
            description=p.description,
            bullets=doc_bullets(f"project:{p.id}"),
        )
        for p in profile.projects
        if selection.sections.get(f"project:{p.id}")
    ]
    education = [
        DocEntry(
            heading=ed.institution,
            subheading=", ".join(x for x in (ed.degree, ed.field) if x),
            dates=date_range(ed.start, ed.end, current=False),
            bullets=[
                DocBullet(id=f"{ed.id}-{n}", text=t, original=t) for n, t in enumerate(ed.details)
            ],
        )
        for ed in profile.education
    ]
    used_tools = {t for i in selection.bullet_ids() for t in bullets[i].facts.tools}
    return ResumeDoc(
        contact=profile.contact,
        summary=summary,
        summary_original=profile.summary,
        summary_proposed=summary,
        skills=tailor_skills(profile, analysis, used_tools),
        experience=experience,
        projects=projects,
        education=education,
    )


def original_resume_text(profile: Profile) -> str:
    """The resume as it was before this tool: resume-sourced bullets, summary and skills."""
    parts = [profile.summary or "", *profile.skills.all()]
    parts += [b.text for _, b in profile.iter_bullets() if b.source in (None, "resume")]
    return "\n".join(parts)


def build_report(
    profile: Profile,
    analysis: JobAnalysis,
    evidence: EvidenceMap,
    selection: Selection,
    doc: ResumeDoc,
    fallbacks: dict[str, list[str]],
    dropped: list[str],
    pages: int,
) -> TailorReport:
    selected = set(selection.bullet_ids())
    coverage = []
    for r in analysis.requirements:
        item = evidence.for_requirement(r.name)
        on_page = [m for m in (item.bullets if item else []) if m.bullet_id in selected]
        if any(m.strength == "strong" for m in on_page):
            status = "strong"
        elif on_page:
            status = "partial"
        elif item and item.in_education:
            status = "education"
        elif item and item.in_skills:
            status = "skills-only"
        else:
            status = "gap"
        coverage.append(
            RequirementCoverage(
                requirement=r.name,
                must_have=r.must_have,
                weight=r.weight,
                status=status,
                bullet_ids=[m.bullet_id for m in on_page],
                note=item.note if item else "",
            )
        )
    before_text, after_text = original_resume_text(profile), doc.all_text()
    before = [k for k in analysis.keywords if quote_in_text(k, before_text)]
    after = [k for k in analysis.keywords if quote_in_text(k, after_text)]
    return TailorReport(
        coverage=coverage,
        keywords_before=before,
        keywords_after=after,
        keywords_missing=[k for k in analysis.keywords if k not in after],
        fallbacks=fallbacks,
        dropped_for_space=dropped,
        pages=pages,
    )


def report_markdown(result: TailorResult, analysis: JobAnalysis) -> str:
    r, doc = result.report, result.doc
    icon = {"strong": "✅", "partial": "🟡", "education": "✅", "skills-only": "🟠", "gap": "❌"}
    lines = [
        f"# Tailoring report: {analysis.title} @ {analysis.company}",
        "",
        f"Pages: {r.pages} | ATS keywords: {len(r.keywords_before)} → {len(r.keywords_after)}"
        f" of {len(analysis.keywords)}",
        "",
        "## Requirement coverage",
        "",
        "| | Requirement | Type | Evidence on the page |",
        "|---|---|---|---|",
    ]
    for c in r.coverage:
        kind = f"{'must' if c.must_have else 'nice'} ({'*' * c.weight})"
        evidence = ", ".join(c.bullet_ids) or c.note
        lines.append(f"| {icon[c.status]} {c.status} | {c.requirement} | {kind} | {evidence} |")
    lines += ["", f"Missing keywords: {', '.join(r.keywords_missing) or 'none'}", ""]
    lines += ["## Summary", "", f"**Tailored:** {doc.summary}", ""]
    if doc.summary != doc.summary_original:
        lines += [f"**Original:** {doc.summary_original}", ""]
    if "summary" in r.fallbacks:
        lines += [f"⚠️ Rewrite rejected ({'; '.join(r.fallbacks['summary'])}), original kept.", ""]
    lines += ["## Bullets", ""]
    for e in [*doc.experience, *doc.projects]:
        if not e.bullets:
            continue
        lines += [f"### {e.heading}", ""]
        for b in e.bullets:
            if b.flags:
                lines.append(f"- ⚠️ `{b.id}` kept original ({'; '.join(b.flags)}): {b.text}")
            elif b.changed:
                lines.append(f"- ✏️ `{b.id}` {b.text}\n  - was: {b.original}")
            else:
                lines.append(f"- `{b.id}` {b.text}")
        lines.append("")
    if r.dropped_for_space:
        lines += [f"Dropped to fit {r.pages} page(s): {', '.join(r.dropped_for_space)}", ""]
    return "\n".join(lines)
