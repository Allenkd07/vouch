"""Hand labels for the ranking eval: would I apply to this job today? (evals/labels.yaml)

Keyed by job URL, so labels survive a database reset and re-fetched postings."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

DEFAULT_LABELS = Path("evals/labels.yaml")

Grade = Literal["good", "okay", "bad", "skip"]
GRADES: dict[str, str] = {
    "good": "Good: I'd apply",
    "okay": "Okay: a stretch, or only if nothing better",
    "bad": "Bad: I wouldn't apply",
    "skip": "Skip: can't tell",
}
# Why a job isn't a good fit; tells the ranking eval what the scorer gets wrong.
REASONS: dict[str, str] = {
    "seniority": "Wrong level",
    "stack": "Wrong stack",
    "domain": "Wrong domain",
    "location": "Location or work mode",
    "role": "Not the kind of role I want",
    "interest": "Not interested",
}
GAIN = {"good": 2, "okay": 1, "bad": 0}  # relevance for NDCG; skips are left out


class Label(BaseModel):
    grade: Grade
    reasons: list[str] = []
    note: str = ""
    title: str | None = None  # for reading the file; the URL is the key
    company: str | None = None
    labelled_at: datetime


def load_labels(path: Path = DEFAULT_LABELS) -> dict[str, Label]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {url: Label.model_validate(v) for url, v in (data.get("labels") or {}).items()}


def save_label(
    url: str,
    grade: Grade,
    *,
    reasons: list[str] = (),
    note: str = "",
    title: str | None = None,
    company: str | None = None,
    path: Path = DEFAULT_LABELS,
) -> Label:
    """Add or replace the label for `url`."""
    if grade not in GRADES:
        raise ValueError(f"unknown grade {grade!r}")
    labels = load_labels(path)
    label = Label(
        grade=grade,
        reasons=[r for r in reasons if r in REASONS],
        note=note.strip(),
        title=title,
        company=company,
        labelled_at=datetime.now(UTC),
    )
    labels[url] = label
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"labels": {u: lb.model_dump(mode="json") for u, lb in labels.items()}}
    path.write_text(
        "# Written by the /label page. Keyed by job URL.\n"
        + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return label
