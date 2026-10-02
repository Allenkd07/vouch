"""Read-only views over jobs, matches and applications, for the job lists and `vouch matches`."""

from collections import Counter
from datetime import UTC, datetime

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from vouch.db import Application, Job, Match
from vouch.jobs.repository import active_filter

Row = tuple[Job, Match | None, Application | None]


def job_rows(session: Session, *, active_only: bool, now: datetime | None = None) -> list[Row]:
    """Every job with its match and application, if any."""
    query = (
        select(Job, Match, Application)
        .outerjoin(Match, Match.job_id == Job.id)
        .outerjoin(Application, Application.job_id == Job.id)
    )
    if active_only:
        query = query.where(active_filter(now or datetime.now(UTC)))
    return [tuple(r) for r in session.execute(query).all()]


def ranked_matches(session: Session, limit: int = 50, now: datetime | None = None) -> list[Row]:
    """Best first: scored jobs by score, then the rest by similarity. Closed postings and
    rejected applications are left out."""
    now = now or datetime.now(UTC)
    return session.execute(
        select(Job, Match, Application)
        .join(Match, Match.job_id == Job.id)
        .outerjoin(Application, Application.job_id == Job.id)
        .where(active_filter(now))
        .where(or_(Application.status.is_(None), Application.status != "rejected"))
        .order_by(Match.score.desc().nulls_last(), Match.similarity.desc().nulls_last())
        .limit(limit)
    ).all()


def skill_gaps(rows: list[Row], top: int = 30) -> list[tuple[str, int]]:
    """Missing must-have skills most often asked for across the best-scored matches."""
    counts: Counter[str] = Counter()
    scored = [m for _, m, _ in rows if m and m.fit][:top]
    for m in scored:
        counts.update({name.strip(): 1 for name in m.fit["missing_must"]})
    return counts.most_common(10)
