"""Trap rewrites for the honesty check (evals/traps.yaml): fabrications it must reject, and
honest rewrites it must accept. Run through the same two layers tailoring uses: the
deterministic check, then the LLM judge on whatever passed it."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from vouch.llm import LLM
from vouch.tailoring.verify import deterministic_issues, judge

DEFAULT_TRAPS = Path("evals/traps.yaml")


class Case(BaseModel):
    id: str
    source: str  # a key of TrapSet.sources
    rewrite: str
    caught_by: Literal["deterministic", "judge"] | None = None  # None for controls


class TrapSet(BaseModel):
    vocabulary: list[str]
    sources: dict[str, str]
    traps: list[Case]
    controls: list[Case]

    def source(self, case: Case) -> str:
        return " ".join(self.sources[case.source].split())


def load_traps(path: Path = DEFAULT_TRAPS) -> TrapSet:
    return TrapSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass
class Outcome:
    case: Case
    is_trap: bool
    rejected_by: Literal["deterministic", "judge"] | None
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
