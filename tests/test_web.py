"""Web UI against the Docker Postgres, inside a rolled-back transaction (skips without a DB)."""

from pathlib import Path

import pytest
from fakes import ScriptedLLM, make_analysis
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from vouch.config import get_settings
from vouch.db import Application, Job, ResumeVersion
from vouch.jobs.requirements import VerifiedAnalysis
from vouch.profile.schema import load_profile
from vouch.profile.sync import sync_profile
from vouch.tailoring.document import ResumeDoc
from vouch.tailoring.pipeline import tailor
from vouch.web.app import apply_form, create_app
from vouch.web.diff import highlight_changes

EXAMPLE = Path(__file__).parent.parent / "profile" / "profile.example.yaml"


@pytest.fixture
def db():
    engine = create_engine(get_settings().database_url)
    try:
        conn = engine.connect()
        conn.execute(text("SELECT 1 FROM applications LIMIT 1"))
        conn.rollback()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"database not available: {e}")
    trans = conn.begin()
    yield sessionmaker(bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False)
    trans.rollback()
    conn.close()


@pytest.fixture
def seeded(db, tmp_path):
    """A job with an analysis and one tailored version whose files live in tmp_path."""
    analysis = make_analysis()
    profile = load_profile(EXAMPLE)
    result = tailor(profile, analysis, ScriptedLLM())
    pdf = tmp_path / "Asha_Acme.pdf"
    pdf.write_bytes(result.pdf)
    with db() as s:
        job = Job(
            source="manual",
            url="manual:web-test",
            company="Acme",
            title="Backend Engineer",
            description="Python, Celery",
            content_hash="x" * 64,
            analysis=VerifiedAnalysis(analysis=analysis).model_dump(mode="json"),
        )
        s.add(job)
        s.flush()
        version = ResumeVersion(
            job_id=job.id,
            profile_snapshot_id=sync_profile(s, profile).snapshot_id,
            model="fake",
            content=result.doc.model_dump(mode="json"),
            report=result.report.model_dump(mode="json"),
            evidence=result.evidence.model_dump(mode="json"),
            pdf_path=str(pdf),
            docx_path=str(tmp_path / "Asha_Acme.docx"),
        )
        s.add(version)
        s.commit()
        return job.id, version.id


@pytest.fixture
def client(db):
    return TestClient(create_app(make_session=db, profile_path=EXAMPLE))


def test_jobs_page_and_status_tracking(client, seeded):
    job_id, _ = seeded
    page = client.get("/jobs")
    assert page.status_code == 200 and "Backend Engineer" in page.text

    r = client.post(f"/jobs/{job_id}/status", data={"status": "applied"})
    assert r.status_code == 200 and "Saved" in r.text

    apps = client.get("/applications")
    assert "Applied" in apps.text and "Backend Engineer" in apps.text


def test_review_edit_approve_flow(client, db, seeded):
    job_id, version_id = seeded
    page = client.get(f"/versions/{version_id}")
    assert page.status_code == 200
    assert "<mark>" in page.text  # changed words highlighted
    assert "Kept original" in page.text  # the judge-rejected bullet is flagged

    form = {
        "include:finlytics-1": "on",
        "choice:finlytics-1": "custom",
        "custom:finlytics-1": "Built a reconciliation service in Python",
        # finlytics-2 left unchecked -> excluded
        "choice:finlytics-2": "tailored",
        "include:finlytics-3": "on",
        "choice:finlytics-3": "original",
        "include:docqa-1": "on",
        "choice:summary": "custom",
        "custom:summary": "Backend engineer.",
    }
    r = client.post(f"/versions/{version_id}", data=form, follow_redirects=False)
    assert r.status_code == 303

    client.post(f"/versions/{version_id}/approve")
    with db() as s:
        version = s.get(ResumeVersion, version_id)
        doc = ResumeDoc.model_validate(version.content)
        bullets = {b.id: b for e in doc.experience + doc.projects for b in e.bullets}
        assert bullets["finlytics-1"].text == "Built a reconciliation service in Python"
        assert not bullets["finlytics-2"].included
        assert doc.summary == "Backend engineer."
        assert version.approved_at is not None
        assert Path(version.docx_path).exists()
        app = s.query(Application).filter_by(job_id=job_id).one()
        assert app.resume_version_id == version_id

    pdf = client.get(f"/versions/{version_id}/pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"


def test_apply_form_falls_back_when_custom_is_empty():
    doc = ResumeDoc.model_validate(
        {
            "contact": {"name": "X"},
            "summary": "tailored",
            "summary_original": "orig",
            "summary_proposed": "tailored",
            "skills": [],
            "experience": [
                {
                    "heading": "A",
                    "bullets": [{"id": "a-1", "text": "new", "original": "old", "proposed": "new"}],
                }
            ],
            "projects": [],
            "education": [],
        }
    )
    doc = apply_form(doc, {"include:a-1": "on", "choice:a-1": "custom", "custom:a-1": " "})
    assert doc.experience[0].bullets[0].text == "new"  # empty custom -> tailored
    doc = apply_form(doc, {"include:a-1": "on", "choice:a-1": "original"})
    assert doc.experience[0].bullets[0].text == "old"


def test_highlight_marks_only_new_words_and_escapes():
    original = "Built a service in Python, deployed on GCP"
    html = str(highlight_changes(original, "Deployed a <fast> Python service to GCP quickly"))
    # Reordered words and function words stay plain; new wording is marked and escaped.
    assert html == "Deployed a <mark>&lt;fast&gt;</mark> Python service to GCP <mark>quickly</mark>"


def test_highlight_merges_adjacent_new_words():
    assert str(highlight_changes("Built APIs", "Built scalable backend APIs")) == (
        "Built <mark>scalable backend</mark> APIs"
    )


def test_tailor_button_runs_and_redirects_to_new_version(client, db, seeded, monkeypatch):
    import time

    import vouch.web.app as web

    job_id, version_id = seeded
    page = client.get(f"/jobs/{job_id}")
    assert f'hx-post="/jobs/{job_id}/tailor"' in page.text  # regression: was "/jobs//tailor"

    monkeypatch.setattr(web, "get_llm", lambda settings: None)
    monkeypatch.setattr(
        web,
        "create_version",
        lambda session, job, profile, llm, settings: session.get(ResumeVersion, version_id),
    )
    started = client.post(f"/jobs/{job_id}/tailor")
    assert started.status_code == 200

    for _ in range(50):  # the run happens in a background thread
        status = client.get(f"/jobs/{job_id}/tailor")
        if "HX-Redirect" in status.headers:
            break
        time.sleep(0.05)
    assert status.headers["HX-Redirect"] == f"/versions/{version_id}"
