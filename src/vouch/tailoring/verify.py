"""Check that tailored text claims nothing the source doesn't. Failing text falls back to the
original wording, so a bad rewrite can cost polish but never honesty."""

import re

from pydantic import BaseModel

from vouch.llm import LLM
from vouch.profile.schema import normalize_name

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# A posting often names a tool with its vendor ("Apache Kafka") where the profile says "Kafka".
_VENDOR = re.compile(r"^(apache|amazon|aws|google|microsoft|azure)\s+", re.I)


def _words(text: str) -> str:
    """Normalised, with plural s dropped, so "API integration" matches "API integrations"."""
    words = normalize_name(text).split()
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and w[-2] != "s" else w for w in words)


def _contains(term: str, text: str) -> bool:
    return f" {_words(term)} " in f" {_words(text)} "


def deterministic_issues(rewritten: str, source: str, vocabulary: set[str]) -> list[str]:
    """Tools and numbers in the rewrite must appear in the source text (bullet + its facts)."""
    known = {_words(v) for v in vocabulary}
    issues = []
    for term in sorted(vocabulary):
        if not _contains(term, rewritten) or _contains(term, source):
            continue
        bare = _VENDOR.sub("", term)
        if bare != term and _words(bare) in known and _contains(bare, source):
            continue  # "Apache Kafka" for a source that says "Kafka"
        issues.append(f"adds {term!r}")
    source_numbers = {n.replace(",", "") for n in _NUMBER.findall(source)}
    for n in _NUMBER.findall(rewritten):
        if n.replace(",", "") not in source_numbers:
            issues.append(f"adds number {n!r}")
    return issues


class Verdict(BaseModel):
    id: str
    supported: bool
    issue: str


class Verdicts(BaseModel):
    verdicts: list[Verdict]


JUDGE_SYSTEM = """You audit tailored resume text for honesty. For each item, compare the
REWRITTEN text with its SOURCE. supported = false if the rewrite claims anything the source does
not state or directly imply: a new technology, number, scale, scope, responsibility, outcome,
seniority, or a broader characterisation (e.g. "microservices", "distributed", "high-traffic",
"led") not in the source. Also unsupported: a stronger verb than the source's ("designed" or
"built" -> "architected", "built" -> "engineered" or "spearheaded", "helped" -> "led"), and an
added purpose or benefit ("to ensure reliability", "enabling faster releases") the source doesn't
state. Rewording, reordering, shortening and synonyms for the same thing are fine. Also
unsupported: a compound term where only part is evidenced (e.g. "CI/CD" when only CI is shown,
"full ownership" when only building is shown), and relabeling one thing as a different one
(e.g. calling RBAC/authorization "authentication"). A skills list alone does not
support a claim of experience. issue: short reason if unsupported, else empty. Return one
verdict per id."""


def judge(items: list[tuple[str, str, str]], llm: LLM) -> dict[str, Verdict]:
    """items: (id, source, rewritten). Returns verdicts by id; missing ids count as unsupported."""
    if not items:
        return {}
    prompt = "\n\n".join(
        f"id: {i}\nSOURCE: {source}\nREWRITTEN: {rewritten}" for i, source, rewritten in items
    )
    result = llm.generate_json(prompt, Verdicts, system=JUDGE_SYSTEM)
    by_id = {v.id: v for v in result.verdicts}
    return {
        i: by_id.get(i) or Verdict(id=i, supported=False, issue="not checked by judge")
        for i, _, _ in items
    }
