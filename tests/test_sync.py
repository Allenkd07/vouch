"""Uses the `session` fixture from conftest.py (Docker Postgres; skips without it)."""

from sqlalchemy import select

from vouch.db import ProfileItem, ProfileSnapshot
from vouch.profile.schema import Profile
from vouch.profile.sync import sync_profile


def _profile(*bullets: tuple[str, str]) -> Profile:
    return Profile.model_validate(
        {"contact": {"name": "X"}, "achievements": [{"id": i, "text": t} for i, t in bullets]}
    )


def test_sync_upserts_removes_and_dedupes_snapshots(session):
    first = sync_profile(session, _profile(("zz-a", "one"), ("zz-b", "two")))
    assert first.new_snapshot and first.items == 2

    again = sync_profile(session, _profile(("zz-a", "one"), ("zz-b", "two")))
    assert not again.new_snapshot and again.snapshot_id == first.snapshot_id

    changed = sync_profile(session, _profile(("zz-a", "one, reworded")))
    assert changed.new_snapshot and changed.removed >= 1

    items = session.scalars(select(ProfileItem)).all()
    assert [(i.id, i.text) for i in items] == [("zz-a", "one, reworded")]
    assert session.scalar(select(ProfileSnapshot).where(ProfileSnapshot.id == first.snapshot_id))


def test_upsert_job_dedupes_by_url_and_content(session):
    from vouch.boards import FetchedJob
    from vouch.jobs.requirements import JobAnalysis, VerifiedAnalysis
    from vouch.jobs.store import needs_analysis, save_analysis, upsert_job

    fetched = FetchedJob("web", "https://x.example/zz-1", "X", "SDE", None, "Build APIs in Go")
    job, changed = upsert_job(session, fetched)
    assert changed and needs_analysis(job)

    analysis = JobAnalysis(
        title="SDE",
        company="X",
        seniority="mid",
        years_experience_min=None,
        years_experience_max=None,
        location=None,
        work_mode="unknown",
        summary="s",
        responsibilities=[],
        requirements=[],
        keywords=[],
    )
    save_analysis(job, VerifiedAnalysis(analysis=analysis), "m1")

    same, changed = upsert_job(session, fetched)
    assert same.id == job.id and not changed and not needs_analysis(same)

    fetched.description = "Build APIs in Go and Rust"
    _, changed = upsert_job(session, fetched)
    assert changed and needs_analysis(same)  # posting changed
