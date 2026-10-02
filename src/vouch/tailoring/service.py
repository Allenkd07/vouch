"""Tailored resume versions: create, edit, approve (used by the CLI and the web app)."""

import re
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from vouch.applications import repository as applications_db
from vouch.applications.service import apply_status
from vouch.config import Settings
from vouch.db import Job, ResumeVersion
from vouch.jobs import repository as jobs_db
from vouch.jobs.requirements import quote_in_text
from vouch.jobs.service import job_analysis
from vouch.llm import LLM
from vouch.profile.schema import Profile
from vouch.profile.sync import sync_profile
from vouch.tailoring import repository as versions_db
from vouch.tailoring.document import ResumeDoc
from vouch.tailoring.pipeline import TailorReport, report_markdown, tailor
from vouch.tailoring.render import page_count, render_docx, render_pdf
from vouch.transaction import transaction

OUTPUT_DIR = Path("output")


def create_version(
    session: Session,
    job: Job,
    profile: Profile,
    llm: LLM,
    settings: Settings,
    *,
    pages: int = 1,
    out: Path = OUTPUT_DIR,
) -> ResumeVersion:
    """Tailor a resume for `job`, write PDF/DOCX/report, and store the version."""
    analysis = job_analysis(job)
    if analysis is None:
        raise ValueError(f"job {job.id} has not been analysed")
    snapshot_id = sync_profile(session, profile).snapshot_id
    result = tailor(profile, analysis, llm, max_pages=pages)
    with transaction(session):
        version = versions_db.add(
            session,
            ResumeVersion(
                job_id=job.id,
                profile_snapshot_id=snapshot_id,
                model=settings.llm_model,
                content=result.doc.model_dump(mode="json"),
                report=result.report.model_dump(mode="json"),
                evidence=result.evidence.model_dump(mode="json"),
            ),
        )
        folder = version_folder(job, version, out)
        (folder / "report.md").write_text(report_markdown(result, analysis), encoding="utf-8")
        _write_files(version, result.doc, job, folder, result.pdf)
    return version


def version_folder(job: Job, version: ResumeVersion, out: Path = OUTPUT_DIR) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", f"{job.company} {job.title}".lower()).strip("-")[:60]
    folder = out / f"job-{job.id}-{slug}" / f"v{version.id}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_files(
    version: ResumeVersion, doc: ResumeDoc, job: Job, folder: Path, pdf: bytes | None = None
) -> None:
    company = re.sub(r"[^A-Za-z0-9]+", "", (job.company or "Company").split()[0])
    base = f"{doc.contact.name.replace(' ', '_')}_{company}"
    pdf_path, docx_path = folder / f"{base}.pdf", folder / f"{base}.docx"
    pdf_path.write_bytes(pdf if pdf is not None else render_pdf(doc.for_render()))
    render_docx(doc.for_render(), docx_path)
    version.pdf_path, version.docx_path = str(pdf_path), str(docx_path)


def save_edits(session: Session, version: ResumeVersion, doc: ResumeDoc) -> TailorReport:
    """Store an edited document, re-render its files, and refresh page count and keyword
    coverage. Editing clears any previous approval."""
    with transaction(session):
        job = jobs_db.get(session, version.job_id)
        analysis = job_analysis(job)
        pdf = render_pdf(doc.for_render())
        report = TailorReport.model_validate(version.report)
        text = doc.for_render().all_text()
        report.keywords_after = [k for k in analysis.keywords if quote_in_text(k, text)]
        report.keywords_missing = [k for k in analysis.keywords if k not in report.keywords_after]
        report.pages = page_count(pdf)
        version.content = doc.model_dump(mode="json")
        version.report = report.model_dump(mode="json")
        version.approved_at = None
        folder = Path(version.pdf_path).parent if version.pdf_path else version_folder(job, version)
        _write_files(version, doc, job, folder, pdf)
    return report


def approve_version(session: Session, version: ResumeVersion) -> None:
    """Mark a version ready to send and attach it to the job's application."""
    with transaction(session):
        version.approved_at = datetime.now(UTC)
        app = applications_db.for_job(session, version.job_id) or apply_status(
            session, version.job_id, "saved"
        )
        app.resume_version_id = version.id
