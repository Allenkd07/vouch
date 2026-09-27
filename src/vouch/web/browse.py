"""Filtering, sorting and facet counts for the job lists (Matches and Jobs pages).

Lists are small (tens to a few hundred jobs), so rows are loaded once and filtered in Python;
facet counts are computed over the rows that match every *other* filter, so each count says
what you'd get by clicking it."""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from urllib.parse import urlencode

from sqlalchemy import select
from sqlalchemy.orm import Session

from vouch.db import Application, Job, Match
from vouch.discovery.filters import age_days
from vouch.discovery.run import active_jobs_filter

# Location spellings -> one display name. Unknown places are shown as written.
CITIES = {
    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",
    "pune": "Pune",
    "mumbai": "Mumbai",
    "hyderabad": "Hyderabad",
    "chennai": "Chennai",
    "gurgaon": "Gurugram",
    "gurugram": "Gurugram",
    "noida": "Noida",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "kochi": "Kochi",
    "remote": "Remote",
}
SORTS = {
    "fit": "Best fit",
    "similar": "Most similar to your profile",
    "newest": "Newest posted",
    "added": "Recently added",
    "company": "Company A–Z",
}
FITS = {"": "Any", "scored": "Scored", "60": "60 or more", "unscored": "Not scored yet"}


@dataclass
class Params:
    q: str = ""
    company: str = ""
    city: str = ""
    status: str = ""  # "" any, "none" not tracking, or a services.STATUSES value
    fit: str = ""
    sort: str = "fit"

    def active(self) -> bool:
        return any((self.q, self.company, self.city, self.status, self.fit))


@dataclass
class Row:
    job: Job
    match: Match | None
    app: Application | None
    cities: list[str]
    age: int | None
    hue: int  # for the company monogram colour
    initials: str

    @property
    def score(self) -> float | None:
        return self.match.score if self.match else None

    @property
    def similarity(self) -> float | None:
        return self.match.similarity if self.match else None

    @property
    def status(self) -> str:
        return self.app.status if self.app else "none"


@dataclass
class Browse:
    rows: list[Row]
    total: int
    params: Params
    companies: list[tuple[str, int]] = field(default_factory=list)
    cities: list[tuple[str, int]] = field(default_factory=list)
    statuses: dict[str, int] = field(default_factory=dict)


def parse_params(query, default_sort: str) -> Params:
    p = Params(
        **{k: (query.get(k) or "").strip() for k in ("q", "company", "city", "status", "fit")}
    )
    p.sort = query.get("sort") if query.get("sort") in SORTS else default_sort
    return p


def load_rows(session: Session, *, active_only: bool, now: datetime | None = None) -> list[Row]:
    now = now or datetime.now(UTC)
    query = (
        select(Job, Match, Application)
        .outerjoin(Match, Match.job_id == Job.id)
        .outerjoin(Application, Application.job_id == Job.id)
    )
    if active_only:
        query = query.where(active_jobs_filter(now))
    return [_row(job, match, app, now) for job, match, app in session.execute(query).all()]


def _row(job: Job, match: Match | None, app: Application | None, now: datetime) -> Row:
    company = job.company or "Unknown company"
    digest = int(hashlib.md5(company.lower().encode()).hexdigest(), 16)
    words = re.findall(r"[A-Za-z0-9]+", company)
    initials = "".join(w[0] for w in words[:2]).upper() or "?"
    return Row(
        job,
        match,
        app,
        job_cities(job.location),
        age_days(job.posted_at, now),
        digest % 360,
        initials,
    )


def job_cities(location: str | None) -> list[str]:
    text = (location or "").lower()
    found = []
    for spelling, name in CITIES.items():
        if re.search(rf"(?<![a-z]){re.escape(spelling)}(?![a-z])", text) and name not in found:
            found.append(name)
    return found or ([location.strip()] if location and location.strip() else [])


def _matches(row: Row, p: Params, skip: str = "") -> bool:
    if p.q and skip != "q":
        hay = f"{row.job.title} {row.job.company} {row.job.location}".lower()
        if not all(word in hay for word in p.q.lower().split()):
            return False
    if p.company and skip != "company" and row.job.company != p.company:
        return False
    if p.city and skip != "city" and p.city not in row.cities:
        return False
    if p.status and skip != "status" and row.status != p.status:
        return False
    if p.fit and skip != "fit":
        if p.fit == "scored" and row.score is None:
            return False
        if p.fit == "unscored" and row.score is not None:
            return False
        if p.fit == "60" and (row.score is None or row.score < 60):
            return False
    return True


def _sort_key(sort: str):
    def last_if_none(value, reverse=True):
        # Sort Nones last whichever direction the values go.
        return (value is None, -value if (value is not None and reverse) else value or 0)

    return {
        "fit": lambda r: (last_if_none(r.score), last_if_none(r.similarity)),
        "similar": lambda r: last_if_none(r.similarity),
        "newest": lambda r: (r.age is None, r.age if r.age is not None else 0),
        "added": lambda r: -r.job.fetched_at.timestamp(),
        "company": lambda r: ((r.job.company or "").lower(), last_if_none(r.score)),
    }[sort]


def browse(rows: list[Row], p: Params) -> Browse:
    shown = sorted((r for r in rows if _matches(r, p)), key=_sort_key(p.sort))
    companies = Counter(
        r.job.company or "Unknown company" for r in rows if _matches(r, p, "company")
    )
    cities = Counter(c for r in rows if _matches(r, p, "city") for c in r.cities)
    statuses = Counter(r.status for r in rows if _matches(r, p, "status"))
    return Browse(
        rows=shown,
        total=len(rows),
        params=p,
        companies=sorted(companies.items(), key=lambda kv: (-kv[1], kv[0].lower())),
        cities=sorted(cities.items(), key=lambda kv: (-kv[1], kv[0])),
        statuses=dict(statuses),
    )


def query_string(p: Params, **changes) -> str:
    """URL query for `p` with some fields changed, e.g. for the "clear" links."""
    updated = replace(p, **changes)
    return urlencode({k: v for k, v in vars(updated).items() if v})
