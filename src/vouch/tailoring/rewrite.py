"""Rephrase selected bullets (and write a summary) in the job's language, without new claims."""

from pydantic import BaseModel, Field

from vouch.jobs.requirements import JobAnalysis
from vouch.llm import LLM
from vouch.profile.schema import Bullet, Profile


class RewrittenBullet(BaseModel):
    id: str
    text: str


class Rewrites(BaseModel):
    summary: str | None = Field(description="2 sentences, at most 50 words; null if not asked")
    bullets: list[RewrittenBullet]


SYSTEM = """You tailor resume bullets to a job posting. Honesty is the hard constraint.
You MAY: reorder clauses to lead with what the job cares about; use the posting's terminology when
it names the same thing the source says (e.g. source "Postgres" -> "PostgreSQL"); tighten wording;
drop details irrelevant to this job.
You MUST NOT: add any technology, tool, number, metric, team size, title, scale claim ("millions of
users", "high-traffic") or responsibility that the source bullet and its facts do not state;
generalise into a claim the source doesn't make (e.g. do not call something "microservices" or
"distributed" unless the source says so); use result verbs ("scaled", "optimized", "improved",
"reduced") unless the source states that result; relabel it as a different posting keyword (RBAC is
authorization, not "authentication"; CI is not "CI/CD"); change what the candidate did.
When the source clearly describes something the posting names, use the posting's exact phrase
so ATS keyword matching works (e.g. source "unit tests" -> posting "unit testing").
Style: start with a strong past-tense verb; one sentence; at most 30 words; numbers as digits
("9", not "nine"); no first person; do not
start more than two bullets with the same verb. If a bullet is already good for this job, return it
nearly unchanged.
Summary: 2 sentences, at most 50 words, built only from the profile summary and the given bullets;
keep years of experience exactly as the profile states."""


def rewrite(
    bullets: list[Bullet],
    supports: dict[str, list[str]],
    profile: Profile,
    analysis: JobAnalysis,
    llm: LLM,
    *,
    with_summary: bool = True,
    feedback: dict[str, str] | None = None,
) -> Rewrites:
    """feedback: id (or "summary") -> why the previous attempt was rejected, for a retry."""
    targets = {b.id: [req for req, ids in supports.items() if b.id in ids] for b in bullets}
    lines = []
    for b in bullets:
        facts = b.facts
        lines.append(
            f"[{b.id}] {b.text}\n"
            f"    facts: tools={facts.tools} metrics={facts.metrics} scope={facts.scope}\n"
            f"    supports: {', '.join(targets[b.id]) or '-'}"
        )
    prompt = (
        f"Job: {analysis.title} at {analysis.company}\n{analysis.summary}\n"
        f"Posting keywords: {', '.join(analysis.keywords)}\n\n"
        f"Profile summary: {profile.summary or '-'}\n\n"
        "Bullets to tailor (return every id):\n" + "\n".join(lines)
    )
    if not with_summary:
        prompt += "\n\nDo not write a summary this time (return null)."
    if feedback:
        prompt += (
            "\n\nYour previous attempt was rejected by an honesty review. Fix exactly these "
            "problems and stay closer to the source:\n"
            + "\n".join(f"- [{i}] {reason}" for i, reason in feedback.items())
        )
    return llm.generate_json(prompt, Rewrites, system=SYSTEM)
