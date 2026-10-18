"""Ranking eval: does the fit score put the jobs I'd apply to at the top?

Compares two orderings of the labelled jobs, the fit score (what the Matches page shows) and
plain profile similarity (the baseline), with precision@k and NDCG@k."""

import hashlib
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from vouch.db import Job
from vouch.discovery.queries import job_rows
from vouch.evals.labels import GAIN, Label


@dataclass
class Item:
    url: str
    title: str
    company: str
    label: Label
    score: float | None  # fit score; None when requirements weren't read
    similarity: float | None


def _last_if_none(value: float | None) -> tuple[bool, float]:
    return (value is None, -(value or 0.0))


# Each ranker orders items best first. "fit" matches the Matches page: score, then similarity.
RANKERS: dict[str, Callable[[Item], tuple]] = {
    "fit": lambda i: (*_last_if_none(i.score), *_last_if_none(i.similarity)),
    "similarity": lambda i: _last_if_none(i.similarity),
}


def precision_at(ranked: list[Item], k: int) -> float:
    """Share of the top k I'd apply to (labelled good)."""
    top = ranked[:k]
    return sum(i.label.grade == "good" for i in top) / len(top) if top else 0.0


def ndcg_at(ranked: list[Item], k: int) -> float:
    """Normalised discounted cumulative gain: 1.0 when the best-labelled jobs come first."""

    def dcg(items: list[Item]) -> float:
        return sum(GAIN[i.label.grade] / math.log2(n + 2) for n, i in enumerate(items[:k]))

    ideal = dcg(sorted(ranked, key=lambda i: -GAIN[i.label.grade]))
    return dcg(ranked) / ideal if ideal else 0.0


@dataclass
class Report:
    k: int
    counts: Counter
    metrics: dict[str, dict[str, float]]  # ranker -> {"precision": .., "ndcg": ..}
    missed_good: list[tuple[int, Item]]  # (position by fit, item): good jobs ranked low
    top_bad: list[tuple[int, Item]]  # bad jobs the fit score put in the top k
    bad_reasons: Counter  # reasons on bad/okay jobs the fit score ranked in the top k


def evaluate(items: list[Item], k: int = 10) -> Report:
    items = [i for i in items if i.label.grade in GAIN]
    metrics = {}
    for name, key in RANKERS.items():
        ranked = sorted(items, key=key)
        metrics[name] = {"precision": precision_at(ranked, k), "ndcg": ndcg_at(ranked, k)}
    by_fit = sorted(items, key=RANKERS["fit"])
    positions = list(enumerate(by_fit, 1))
    half = len(by_fit) // 2
    return Report(
        k=k,
        counts=Counter(i.label.grade for i in items),
        metrics=metrics,
        missed_good=[(n, i) for n, i in positions if i.label.grade == "good" and n > half],
        top_bad=[(n, i) for n, i in positions[:k] if i.label.grade == "bad"],
        bad_reasons=Counter(
            r for _, i in positions[:k] if i.label.grade != "good" for r in i.label.reasons
        ),
    )


def label_queue(session: Session, labels: dict[str, Label]) -> list[Job]:
    """Ranked jobs (those with a match row) not labelled yet, in a shuffled but stable order:
    labelling top-ranked jobs first would leave the eval blind to good jobs ranked low."""
    jobs = [job for job, match, _ in job_rows(session, active_only=False) if match]
    todo = [j for j in jobs if j.url not in labels]
    return sorted(todo, key=lambda j: hashlib.sha1(j.url.encode()).hexdigest())


def labelled_items(session: Session, labels: dict[str, Label]) -> list[Item]:
    """Labelled jobs still in the database, with their current score and similarity."""
    return [
        Item(
            url=job.url,
            title=job.title or "?",
            company=job.company or "?",
            label=labels[job.url],
            score=match.score if match else None,
            similarity=match.similarity if match else None,
        )
        for job, match, _ in job_rows(session, active_only=False)
        if job.url in labels
    ]
