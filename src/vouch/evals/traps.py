"""Trap rewrites for the honesty check (evals/traps.yaml): fabrications it must reject, and
honest rewrites it must accept. Run through the same two layers tailoring uses: the
deterministic check, then the LLM judge on whatever passed it."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from vouch.llm import LLM
from vouch.tailoring.pipeline import labelled, labelled_summary
from vouch.tailoring.verify import deterministic_issues, judge

DEFAULT_TRAPS = Path("evals/traps.yaml")


class Case(BaseModel):
    id: str
    source: str  # a key of TrapSet.sources
    rewrite: str
    caught_by: Literal["deterministic", "judge"] | None = None  # None for controls


class Bullet(BaseModel):
    text: str
    facts: list[str] = []


class Source(Bullet):
    """A bullet; or, with `bullets`, what a summary is written from (`text` is then the
    profile summary)."""

    bullets: list[Bullet] = []
    skills: list[str] = []


class TrapSet(BaseModel):
    vocabulary: list[str]
    sources: dict[str, Source]
    traps: list[Case]
    controls: list[Case]

    def source(self, case: Case) -> str:
        """Run together, as the deterministic check sees it (pipeline.source_text and
        summary_source)."""
        s = self.sources[case.source]
        parts = [s.text, *s.facts, *(p for b in s.bullets for p in [b.text, *b.facts]), *s.skills]
        return " ".join(parts)

    def labelled_source(self, case: Case) -> str:
        """With its parts named, as the LLM judges see it (pipeline.labelled and
        labelled_summary)."""
        s = self.sources[case.source]
        if s.bullets:
            return labelled_summary(s.text, [(b.text, b.facts) for b in s.bullets], s.skills)
        return labelled(s.text, s.facts)


def load_traps(path: Path = DEFAULT_TRAPS) -> TrapSet:
    return TrapSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass
class Outcome:
    case: Case
    is_trap: bool
    rejected_by: Literal["deterministic", "judge", "independent"] | None
    reason: str

    @property
    def correct(self) -> bool:
        return (self.rejected_by is not None) == self.is_trap


def check_deterministic(traps: TrapSet) -> list[Outcome]:
    """The free layer only: what it rejects, and why."""
    vocab = set(traps.vocabulary)
    outcomes = []
    for is_trap, cases in ((True, traps.traps), (False, traps.controls)):
        for case in cases:
            issues = deterministic_issues(case.rewrite, traps.source(case), vocab)
            outcomes.append(
                Outcome(case, is_trap, "deterministic" if issues else None, "; ".join(issues))
            )
    return outcomes


def check_all(traps: TrapSet, llm: LLM) -> list[Outcome]:
    """Both layers, as in tailoring: the judge sees what the deterministic check let through
    (one LLM request for all of it)."""
    outcomes = check_deterministic(traps)
    passed = [o for o in outcomes if o.rejected_by is None]
    verdicts = judge([(o.case.id, traps.source(o.case), o.case.rewrite) for o in passed], llm)
    for o in passed:
        verdict = verdicts[o.case.id]
        if not verdict.supported:
            o.rejected_by, o.reason = "judge", verdict.issue or "unsupported"
    return outcomes


def check_independent(traps: TrapSet, llm: LLM) -> list[Outcome]:
    """Calibrate the tailoring eval's independent judge: it sees every case on its own (no
    deterministic layer in front), so this is its own catch and false-alarm rate."""
    from vouch.evals.faithfulness import audit

    cases = [(True, c) for c in traps.traps] + [(False, c) for c in traps.controls]
    audits = audit([(c.id, traps.labelled_source(c), c.rewrite) for _, c in cases], llm)
    outcomes = []
    for is_trap, case in cases:
        bad = audits[case.id].unsupported
        reason = "; ".join(f"{c.claim}: {c.why}" for c in bad)
        outcomes.append(Outcome(case, is_trap, "independent" if bad else None, reason))
    return outcomes
