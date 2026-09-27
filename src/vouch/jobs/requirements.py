"""Job description -> structured requirements, each backed by a verbatim quote from the posting."""

import re
from typing import Literal

from pydantic import BaseModel, Field

from vouch.llm import LLM

Category = Literal[
    "language",
    "framework",
    "database",
    "cloud",
    "tool",
    "practice",
    "domain",
    "soft_skill",
    "education",
    "experience",
]


class Requirement(BaseModel):
    name: str = Field(description="Short label, e.g. 'Kafka', 'Backend language', 'Mentoring'")
    category: Category
    options: list[str] = Field(
        description="Concrete skills that satisfy it; several when any one is enough "
        "(e.g. ['Java', 'Go', 'Python']); empty for non-technical requirements"
    )
    must_have: bool
    weight: int = Field(ge=1, le=3, description="3 = core to the role, 2 = important, 1 = minor")
    quote: str = Field(description="Exact phrase copied from the posting that states this")


class JobAnalysis(BaseModel):
    title: str
    company: str | None
    seniority: Literal["intern", "junior", "mid", "senior", "staff", "principal", "manager"]
    years_experience_min: int | None = Field(description="Only if stated")
    years_experience_max: int | None = Field(description="Only if stated")
    location: str | None
    work_mode: Literal["onsite", "hybrid", "remote", "unknown"]
    summary: str = Field(description="Two sentences: what the role does and for which product")
    responsibilities: list[str] = Field(description="Main duties, concise")
    requirements: list[Requirement]
    keywords: list[str] = Field(
        description="10-30 terms an ATS would scan for, spelled exactly as in the posting: "
        "technologies, practices (e.g. 'unit testing', 'code reviews') and domain terms "
        "(e.g. 'payments', 'subscriptions')"
    )


class VerifiedAnalysis(BaseModel):
    analysis: JobAnalysis
    unverified_requirements: list[str] = Field(default_factory=list)
    dropped_keywords: list[str] = Field(default_factory=list)


SYSTEM = """You extract structured hiring requirements from a job posting.
Rules:
- Only include requirements the posting actually states. Never add typical or implied ones.
- Split compound lines into separate requirements, e.g. "cloud platforms (AWS preferred),
  messaging systems (Kafka), and CI/CD pipelines" -> three requirements.
- When any one of several skills is enough ("Java, Golang, or Python"), make ONE requirement
  with all of them in options.
- must_have = false when the posting says preferred, plus, bonus, nice to have, or similar;
  otherwise true for items under requirements/qualifications.
- Responsibilities that imply a skill (e.g. "mentor junior engineers") may become requirements
  with must_have = true and weight 1-2.
- quote: copy the exact words from the posting (at least 3 words), no paraphrasing.
- Ignore company boilerplate (about us, equal opportunity statements, benefits)."""


def analyze_job(description: str, llm: LLM, *, title: str | None = None) -> VerifiedAnalysis:
    header = f"Job title: {title}\n\n" if title else ""
    analysis = llm.generate_json(f"{header}Job posting:\n{description}", JobAnalysis, system=SYSTEM)
    return verify_analysis(analysis, description)


def _normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(re.findall(r"[a-z0-9+#/.'-]+", text))


def quote_in_text(quote: str, text: str) -> bool:
    return bool(quote.strip()) and _normalize(quote) in _normalize(text)


def verify_analysis(analysis: JobAnalysis, description: str) -> VerifiedAnalysis:
    """Flag requirements whose quote isn't in the posting; drop keywords that don't appear in it.

    Keywords must appear literally, because their whole purpose is ATS matching."""
    unverified = [r.name for r in analysis.requirements if not quote_in_text(r.quote, description)]
    kept, dropped = [], []
    for kw in analysis.keywords:
        (kept if quote_in_text(kw, description) else dropped).append(kw)
    analysis = analysis.model_copy(update={"keywords": kept})
    return VerifiedAnalysis(
        analysis=analysis, unverified_requirements=unverified, dropped_keywords=dropped
    )
