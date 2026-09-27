"""Local web UI: add jobs, tailor and review resumes, track applications.

Single user, runs on localhost (`vouch web`). Server-rendered Jinja2 + htmx; no build step."""

import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from vouch.config import get_settings
from vouch.db import Application, Job, Match, ResumeVersion, get_sessionmaker
from vouch.discovery.config import DEFAULT_SEARCH, load_search
from vouch.discovery.run import discover, rescore_job, skill_gaps
from vouch.jobs.requirements import analyze_job
from vouch.jobs.sources import FetchError
from vouch.jobs.store import save_analysis
from vouch.llm import LLMError, get_llm
from vouch.profile.schema import lint, load_profile
from vouch.services import (
    STATUSES,
    add_job,
    approve_version,
    create_version,
    get_application,
    job_analysis,
    save_edits,
    set_status,
)
from vouch.tailoring.document import ResumeDoc
from vouch.tailoring.pipeline import TailorReport
from vouch.web import browse as browsing
from vouch.web.diff import highlight_changes

HERE = Path(__file__).parent
log = logging.getLogger(__name__)
DEFAULT_PROFILE = Path("profile/profile.yaml")


@dataclass
class TailorRun:
    state: str  # running | done | error
    message: str = ""
    version_id: int | None = None


def create_app(
    make_session: sessionmaker | None = None,
    profile_path: Path = DEFAULT_PROFILE,
    search_path: Path = DEFAULT_SEARCH,
) -> FastAPI:
    app = FastAPI(title="Vouch")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters["changes"] = lambda b: highlight_changes(b.original, b.text)
    templates.env.globals["STATUSES"] = STATUSES
    # Cache-busting: /static/style.css?v=<mtime> changes whenever the file does, so browsers
    # never keep an old stylesheet or script.
    templates.env.globals["static_version"] = lambda name: int(
        (HERE / "static" / name).stat().st_mtime
    )
    templates.env.globals["SORTS"] = browsing.SORTS
    templates.env.globals["FITS"] = browsing.FITS
    templates.env.globals["query_string"] = browsing.query_string
    runs: dict[int, TailorRun] = {}
    discovery: dict[str, TailorRun] = {}  # single entry under "run"

    def get_session() -> Iterator[Session]:
        with (make_session or get_sessionmaker())() as session:
            yield session

    app.state.get_session = get_session

    def page(request: Request, name: str, **context) -> HTMLResponse:
        return templates.TemplateResponse(request, name, context)

    def coverage_counts(version: ResumeVersion | None) -> dict | None:
        if version is None:
            return None
        report = TailorReport.model_validate(version.report)
        met = sum(c.status in ("strong", "education") for c in report.coverage)
        return {
            "met": met,
            "partial": sum(c.status == "partial" for c in report.coverage),
            "total": len(report.coverage),
        }

    def latest_version(session: Session, job_id: int) -> ResumeVersion | None:
        return session.scalar(
            select(ResumeVersion)
            .where(ResumeVersion.job_id == job_id)
            .order_by(ResumeVersion.id.desc())
        )

    # --- jobs -------------------------------------------------------------------------------

    @app.get("/")
    def home() -> RedirectResponse:
        return RedirectResponse("/jobs", status_code=303)

    def jobs_page(request: Request, session: Session, error: str = "", form: dict | None = None):
        params = browsing.parse_params(request.query_params, default_sort="added")
        rows = browsing.load_rows(session, active_only=False)
        return page(
            request,
            "jobs.html",
            b=browsing.browse(rows, params),
            error=error,
            form=form or {},
        )

    @app.get("/jobs")
    def jobs(request: Request, session: Session = Depends(get_session)):
        return jobs_page(request, session)

    @app.post("/jobs")
    def jobs_add(
        request: Request,
        url: str = Form(""),
        text: str = Form(""),
        company: str = Form(""),
        title: str = Form(""),
        session: Session = Depends(get_session),
    ):
        settings = get_settings()
        form = {"url": url, "text": text, "company": company, "title": title}
        try:
            job, _ = add_job(
                session,
                get_llm(settings),
                settings,
                url=url.strip() or None,
                text=text.strip() or None,
                company=company.strip() or None,
                title=title.strip() or None,
            )
        except (FetchError, LLMError, ValueError) as e:
            session.rollback()
            return jobs_page(request, session, error=str(e), form=form)
        session.commit()
        return RedirectResponse(f"/jobs/{job.id}", status_code=303)

    @app.get("/jobs/{job_id}")
    def job_detail(
        job_id: int, request: Request, session: Session = Depends(get_session), error: str = ""
    ):
        job = session.get(Job, job_id) or _not_found()
        versions = session.scalars(
            select(ResumeVersion)
            .where(ResumeVersion.job_id == job_id)
            .order_by(ResumeVersion.id.desc())
        ).all()
        latest = versions[0] if versions else None
        coverage = TailorReport.model_validate(latest.report).coverage if latest else []
        return page(
            request,
            "job.html",
            job=job,
            analysis=job_analysis(job),
            app=get_application(session, job_id),
            versions=[(v, coverage_counts(v)) for v in versions],
            coverage={c.requirement: c for c in coverage},
            run=runs.get(job_id),
            match=session.get(Match, job_id),
            error=error,
        )

    @app.post("/jobs/{job_id}/analyze")
    def job_analyze(job_id: int, request: Request, session: Session = Depends(get_session)):
        job = session.get(Job, job_id) or _not_found()
        settings = get_settings()
        inline = request.headers.get("HX-Request") == "true"  # the button on a list row
        try:
            result = analyze_job(job.description, get_llm(settings), title=job.title)
        except LLMError as e:
            if inline:
                return page(request, "_job_row.html", row=_row_for(session, job), error=str(e))
            return job_detail(job_id, request, session, error=str(e))
        save_analysis(job, result, settings.llm_model)
        try:
            years = (
                load_search(search_path).ranking.accept_years_up_to if search_path.exists() else 5
            )
            rescore_job(session, job, load_profile(profile_path), accept_years_up_to=years)
        except (OSError, ValidationError):
            pass  # the score appears after the next discovery run instead
        session.commit()
        if inline:
            return page(request, "_job_row.html", row=_row_for(session, job))
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    def _row_for(session: Session, job: Job) -> browsing.Row:
        return next(r for r in browsing.load_rows(session, active_only=False) if r.job.id == job.id)

    @app.post("/jobs/{job_id}/status")
    def job_status(
        job_id: int,
        request: Request,
        status: str = Form(""),
        session: Session = Depends(get_session),
    ):
        session.get(Job, job_id) or _not_found()
        app_row = set_status(session, job_id, status or None)
        session.commit()
        return page(request, "_status.html", job_id=job_id, app=app_row, saved=True)

    # --- tailoring --------------------------------------------------------------------------

    def run_tailor(job_id: int) -> None:
        settings = get_settings()
        try:
            profile = load_profile(profile_path)
            with (make_session or get_sessionmaker())() as session:
                job = session.get(Job, job_id)
                version = create_version(session, job, profile, get_llm(settings), settings)
                session.commit()
                runs[job_id] = TailorRun("done", version_id=version.id)
        except (LLMError, ValidationError, ValueError, OSError) as e:
            runs[job_id] = TailorRun("error", message=str(e))
        except Exception as e:  # noqa: BLE001 - a background run must never die silently
            log.exception("tailoring job %s failed", job_id)
            runs[job_id] = TailorRun("error", message=f"Unexpected error ({type(e).__name__}): {e}")

    @app.post("/jobs/{job_id}/tailor")
    def tailor_start(job_id: int, request: Request, session: Session = Depends(get_session)):
        job = session.get(Job, job_id) or _not_found()
        if job_analysis(job) is None:
            raise HTTPException(400, "Job has no extracted requirements yet")
        if runs.get(job_id, TailorRun("idle")).state != "running":
            runs[job_id] = TailorRun("running")
            threading.Thread(target=run_tailor, args=(job_id,), daemon=True).start()
        return page(request, "_run.html", job_id=job_id, run=runs[job_id])

    @app.get("/jobs/{job_id}/tailor")
    def tailor_status(job_id: int, request: Request):
        run = runs.get(job_id)
        if run and run.state == "done":
            runs.pop(job_id)
            return Response(headers={"HX-Redirect": f"/versions/{run.version_id}"})
        return page(request, "_run.html", job_id=job_id, run=run)

    # --- resume versions --------------------------------------------------------------------

    def version_page(request: Request, session: Session, version: ResumeVersion, notice=""):
        job = session.get(Job, version.job_id)
        return page(
            request,
            "version.html",
            job=job,
            version=version,
            doc=ResumeDoc.model_validate(version.content),
            report=TailorReport.model_validate(version.report),
            analysis=job_analysis(job),
            app=get_application(session, job.id),
            notice=notice,
        )

    @app.get("/versions/{version_id}")
    def version_view(version_id: int, request: Request, session: Session = Depends(get_session)):
        version = session.get(ResumeVersion, version_id) or _not_found()
        notice = {"saved": "Changes saved and resume rebuilt.", "approved": "Approved."}.get(
            request.query_params.get("done", ""), ""
        )
        return version_page(request, session, version, notice)

    @app.post("/versions/{version_id}")
    async def version_save(
        version_id: int, request: Request, session: Session = Depends(get_session)
    ):
        version = session.get(ResumeVersion, version_id) or _not_found()
        form = await request.form()
        doc = apply_form(ResumeDoc.model_validate(version.content), form)
        save_edits(session, version, doc)
        session.commit()
        return RedirectResponse(f"/versions/{version_id}?done=saved", status_code=303)

    @app.post("/versions/{version_id}/approve")
    def version_approve(version_id: int, session: Session = Depends(get_session)):
        version = session.get(ResumeVersion, version_id) or _not_found()
        approve_version(session, version)
        session.commit()
        return RedirectResponse(f"/versions/{version_id}?done=approved", status_code=303)

    @app.get("/versions/{version_id}/{kind}")
    def version_file(version_id: int, kind: str, session: Session = Depends(get_session)):
        version = session.get(ResumeVersion, version_id) or _not_found()
        path = {"pdf": version.pdf_path, "docx": version.docx_path}.get(kind)
        if not path or not Path(path).exists():
            _not_found()
        inline = kind == "pdf"
        return FileResponse(
            path,
            filename=Path(path).name,
            content_disposition_type="inline" if inline else "attachment",
        )

    # --- applications & profile -------------------------------------------------------------

    # --- discovery ----------------------------------------------------------------------------

    @app.get("/matches")
    def matches(request: Request, session: Session = Depends(get_session)):
        params = browsing.parse_params(request.query_params, default_sort="fit")
        rows = browsing.load_rows(session, active_only=True)
        rows = [r for r in rows if r.match is not None and r.status != "rejected"]
        ranked = sorted(rows, key=lambda r: (r.score is None, -(r.score or 0)))
        return page(
            request,
            "matches.html",
            b=browsing.browse(rows, params),
            gaps=skill_gaps([(r.job, r.match, r.app) for r in ranked]),
            run=discovery.get("run"),
            has_search=search_path.exists(),
        )

    def run_discovery() -> None:
        settings = get_settings()
        try:
            config = load_search(search_path)
            profile = load_profile(profile_path)
            with (make_session or get_sessionmaker())() as session:
                result = discover(session, config, profile, get_llm(settings), settings)
            kept = sum(len(c.kept) for c in result.companies)
            failed = [c.name for c in result.companies if c.error]
            message = (
                f"Checked {len(result.companies)} companies: {kept} jobs fit your filters, "
                f"{result.new_or_changed} new or changed, {len(result.analyzed)} analysed."
            )
            if failed:
                message += f" Couldn't reach: {', '.join(failed)}."
            discovery["run"] = TailorRun("done", message=" ".join([message, *result.notes]))
        except (LLMError, ValidationError, ValueError, OSError) as e:
            discovery["run"] = TailorRun("error", message=str(e))
        except Exception as e:  # noqa: BLE001 - a background run must never die silently
            log.exception("job search failed")
            discovery["run"] = TailorRun(
                "error", message=f"Unexpected error ({type(e).__name__}): {e}"
            )

    @app.post("/discover")
    def discover_start(request: Request):
        if discovery.get("run", TailorRun("idle")).state != "running":
            discovery["run"] = TailorRun("running")
            threading.Thread(target=run_discovery, daemon=True).start()
        return page(request, "_discover.html", run=discovery["run"])

    @app.get("/discover")
    def discover_status(request: Request):
        run = discovery.get("run")
        if run and run.state == "done":
            return Response(headers={"HX-Redirect": "/matches"})
        return page(request, "_discover.html", run=run)

    @app.get("/applications")
    def applications(request: Request, session: Session = Depends(get_session)):
        rows = session.execute(
            select(Application, Job)
            .join(Job, Job.id == Application.job_id)
            .order_by(Application.updated_at.desc())
        ).all()
        groups = {s: [(a, j) for a, j in rows if a.status == s] for s in STATUSES}
        return page(request, "applications.html", groups=groups)

    @app.post("/applications/{job_id}/notes")
    def application_notes(
        job_id: int, notes: str = Form(""), session: Session = Depends(get_session)
    ):
        app_row = get_application(session, job_id) or _not_found()
        app_row.notes = notes
        session.commit()
        return HTMLResponse('<span class="saved" role="status">Saved</span>')

    @app.get("/profile")
    def profile_view(request: Request):
        try:
            profile = load_profile(profile_path)
        except (OSError, ValidationError) as e:
            return page(request, "profile.html", profile=None, warnings=[], error=str(e))
        return page(request, "profile.html", profile=profile, warnings=lint(profile), error="")

    return app


def apply_form(doc: ResumeDoc, form) -> ResumeDoc:
    """Apply the review form: per bullet, include/exclude and tailored/original/custom text."""
    for entry in [*doc.experience, *doc.projects]:
        for b in entry.bullets:
            b.included = form.get(f"include:{b.id}") == "on"
            choice = form.get(f"choice:{b.id}", b.choice)
            custom = (form.get(f"custom:{b.id}") or "").strip()
            if choice == "custom" and custom:
                b.choice, b.text = "custom", custom
            elif choice == "original":
                b.choice, b.text = "original", b.original
            else:
                b.choice, b.text = "tailored", b.proposed or b.text
    choice = form.get("choice:summary", "tailored")
    custom = (form.get("custom:summary") or "").strip()
    if choice == "custom" and custom:
        doc.summary = custom
    elif choice == "original":
        doc.summary = doc.summary_original
    else:
        doc.summary = doc.summary_proposed or doc.summary
    return doc


def _not_found():
    raise HTTPException(404, "Not found")


app = create_app()
