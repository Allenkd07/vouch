"""Operations shared by the CLI and the web app."""

import re
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from vouch.boards import FetchedJob, fetch_job, manual_job
from vouch.config import Settings
from vouch.db import Application, Job, ResumeVersion
from vouch.jobs.requirements import JobAnalysis, VerifiedAnalysis, analyze_job, quote_in_text
from vouch.jobs.store import needs_analysis, save_analysis, upsert_job
from vouch.llm import LLM
from vouch.profile.schema import Profile
from vouch.profile.sync import sync_profile
from vouch.tailoring.document import ResumeDoc
from vouch.tailoring.pipeline import TailorReport, report_markdown, tailor
from vouch.tailoring.render import page_count, render_docx, render_pdf

OUTPUT_DIR = Path("output")
STATUSES = ["saved", "applied", "interviewing", "offer", "rejected"]


def add_job(
    session: Session,
    llm: LLM,
    settings: Settings,
    *,
    url: str | None = None,
    text: str | None = None,
    company: str | None = None,
    title: str | None = None,
    reanalyze: bool = False,
) -> tuple[Job, bool]:
    """Fetch (or take pasted text), store, and analyse a job. Returns (job, changed).
    `llm` should be the extraction LLM (llm.get_extraction_llm). Raises FetchError or
    LLMError; the caller commits."""
    if not url and not text:
        raise ValueError("give a job link or paste the job description")
    fetched: FetchedJob = manual_job(text, url, company, title) if text else fetch_job(url)
    fetched.company = company or fetched.company
    fetched.title = title or fetched.title
    job, changed = upsert_job(session, fetched)
    if reanalyze or needs_analysis(job):
        analysis = analyze_job(job.description, llm, title=job.title)
        save_analysis(job, analysis, settings.extraction_model)
    return job, changed


def job_analysis(job: Job) -> JobAnalysis | None:
    return VerifiedAnalysis.model_validate(job.analysis).analysis if job.analysis else None


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
    """Tailor a resume for `job`, write PDF/DOCX/report, and store the version (not committed)."""
    analysis = job_analysis(job)
    if analysis is None:
        raise ValueError(f"job {job.id} has not been analysed")
    snapshot_id = sync_profile(session, profile).snapshot_id
    result = tailor(profile, analysis, llm, max_pages=pages)
    version = ResumeVersion(
        job_id=job.id,
        profile_snapshot_id=snapshot_id,
        model=settings.llm_model,
        content=result.doc.model_dump(mode="json"),
        report=result.report.model_dump(mode="json"),
        evidence=result.evidence.model_dump(mode="json"),
    )
    session.add(version)
    session.flush()
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
    job = session.get(Job, version.job_id)
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


def get_application(session: Session, job_id: int) -> Application | None:
    return session.scalar(select(Application).where(Application.job_id == job_id))


def set_status(session: Session, job_id: int, status: str | None) -> Application | None:
    """status None (or "") removes the job from the tracker."""
    app = get_application(session, job_id)
    if not status:
        if app:
            session.delete(app)
        return None
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if app is None:
        app = Application(job_id=job_id, status=status)
        session.add(app)
    app.status = status
    if status == "applied" and app.applied_at is None:
        app.applied_at = datetime.now(UTC)
    session.flush()
    return app


def approve_version(session: Session, version: ResumeVersion) -> None:
    """Mark a version ready to send and attach it to the job's application."""
    version.approved_at = datetime.now(UTC)
    app = get_application(session, version.job_id) or set_status(session, version.job_id, "saved")
    app.resume_version_id = version.id
