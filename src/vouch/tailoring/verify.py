"""Check that tailored text claims nothing the source doesn't. Failing text falls back to the
original wording, so a bad rewrite can cost polish but never honesty."""

import re

from pydantic import BaseModel

from vouch.llm import LLM
from vouch.profile.schema import normalize_name

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _contains(term: str, text: str) -> bool:
    return f" {normalize_name(term)} " in f" {normalize_name(text)} "


def deterministic_issues(rewritten: str, source: str, vocabulary: set[str]) -> list[str]:
    """Tools and numbers in the rewrite must appear in the source text (bullet + its facts)."""
    issues = []
    for term in sorted(vocabulary):
        if _contains(term, rewritten) and not _contains(term, source):
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
"led") not in the source. Rewording, reordering, shortening and synonyms for the same thing are
fine. Also unsupported: a compound term where only part is evidenced (e.g. "CI/CD" when only
CI is shown, "full ownership" when only building is shown), and relabeling one thing as a
different one (e.g. calling RBAC/authorization "authentication"). A skills list alone does not
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
