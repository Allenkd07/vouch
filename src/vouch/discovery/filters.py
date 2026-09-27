"""Free, rule-based filtering that runs before any embedding or LLM call."""

import re
from datetime import UTC, datetime

from vouch.discovery.config import Filters
from vouch.jobs.sources import FetchedJob


def has_phrase(text: str, phrases: list[str]) -> str | None:
    """First phrase found in text as whole words (case-insensitive), else None."""
    lowered = text.lower()
    for phrase in phrases:
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase.lower())}(?![a-z0-9])", lowered):
            return phrase
    return None


def age_days(posted_at: str | None, now: datetime | None = None) -> int | None:
    """Days since an ISO date or datetime; None if missing or unparseable."""
    if not posted_at:
        return None
    try:
        posted = datetime.fromisoformat(posted_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if posted.tzinfo is None:
        posted = posted.replace(tzinfo=UTC)
    return ((now or datetime.now(UTC)) - posted).days


def rejection_reason(job: FetchedJob, filters: Filters, now: datetime | None = None) -> str | None:
    """Why this job doesn't match the filters, or None if it passes."""
    title = job.title or ""
    if filters.title_include and not has_phrase(title, filters.title_include):
        return "title"
    if excluded := has_phrase(title, filters.title_exclude):
        return f"title contains {excluded!r}"
    if filters.locations and not has_phrase(job.location or "", filters.locations):
        return "location"
    age = age_days(job.posted_at, now)
    if filters.max_age_days is not None and age is not None and age > filters.max_age_days:
        return "too old"
    return None
