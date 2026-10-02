"""The free half of the trap eval: the deterministic honesty check against evals/traps.yaml.

Traps marked `judge` must get past it (otherwise the label is stale: relabel them
`deterministic`); `vouch eval traps` checks those with the real LLM judge."""

from pathlib import Path

import pytest

from vouch.evals.traps import check_all, check_deterministic, load_traps
from vouch.llm import FakeLLM
from vouch.tailoring.verify import Verdict, Verdicts

TRAPS = load_traps(Path(__file__).parent.parent / "evals" / "traps.yaml")
OUTCOMES = {o.case.id: o for o in check_deterministic(TRAPS)}


def test_trap_ids_are_unique_and_sources_exist():
    cases = [*TRAPS.traps, *TRAPS.controls]
    assert len({c.id for c in cases}) == len(cases)
    assert all(c.source in TRAPS.sources for c in cases)


@pytest.mark.parametrize(
    "case", [t for t in TRAPS.traps if t.caught_by == "deterministic"], ids=lambda c: c.id
)
def test_deterministic_check_rejects(case):
    assert OUTCOMES[case.id].rejected_by == "deterministic"


@pytest.mark.parametrize(
    "case", [t for t in TRAPS.traps if t.caught_by == "judge"], ids=lambda c: c.id
)
def test_judge_traps_get_past_the_deterministic_check(case):
    assert OUTCOMES[case.id].rejected_by is None, OUTCOMES[case.id].reason


@pytest.mark.parametrize("case", TRAPS.controls, ids=lambda c: c.id)
def test_deterministic_check_accepts_honest_rewrites(case):
    assert OUTCOMES[case.id].rejected_by is None, OUTCOMES[case.id].reason


def test_judge_layer_only_sees_what_passed():
    seen = []

    def perfect_judge(prompt, schema):
        ids = [line.removeprefix("id: ") for line in prompt.splitlines() if line.startswith("id: ")]
        seen.extend(ids)
        traps = {t.id for t in TRAPS.traps}
        return Verdicts(verdicts=[Verdict(id=i, supported=i not in traps, issue="") for i in ids])

    outcomes = check_all(TRAPS, FakeLLM(perfect_judge))
    assert all(o.correct for o in outcomes)
    assert set(seen) == {i for i, o in OUTCOMES.items() if o.rejected_by is None}
