"""Local web UI: add jobs, tailor and review resumes, track applications.

Single user, runs on localhost (`vouch web`). Server-rendered Jinja2 + htmx; no build step."""

from collections.abc import Iterator
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy.orm import Session

from vouch.applications import repository as applications_db
from vouch.applications import service as application_service
from vouch.applications.service import STATUSES
from vouch.boards import FetchError
from vouch.bootstrap import Deps, build
from vouch.db import Job, ResumeVersion
from vouch.discovery import repository as matches_db
from vouch.discovery.queries import skill_gaps
from vouch.discovery.run import discover
from vouch.jobs import repository as jobs_db
from vouch.jobs import service as job_service
from vouch.jobs.service import job_analysis
from vouch.llm import LLMError
from vouch.profile.schema import lint
from vouch.runs import service as runs
from vouch.tailoring import repository as versions_db
from vouch.tailoring import service as tailoring_service
from vouch.tailoring.document import ResumeDoc
from vouch.tailoring.pipeline import TailorReport
from vouch.web import browse as browsing
from vouch.web.diff import highlight_changes

HERE = Path(__file__).parent


def create_app(deps: Deps) -> FastAPI:
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

    def get_session() -> Iterator[Session]:
        with deps.sessions() as session:
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
        form = {"url": url, "text": text, "company": company, "title": title}
        try:
            job, _ = job_service.add_job(
                session,
                deps.llm("extraction"),
                deps.settings,
                url=url.strip() or None,
                text=text.strip() or None,
                company=company.strip() or None,
                title=title.strip() or None,
                scoring=deps.scoring(),
            )
        except (FetchError, LLMError, ValueError) as e:
            return jobs_page(request, session, error=str(e), form=form)
        return RedirectResponse(f"/jobs/{job.id}", status_code=303)

    @app.get("/jobs/{job_id}")
    def job_detail(
        job_id: int, request: Request, session: Session = Depends(get_session), error: str = ""
    ):
        job = jobs_db.get(session, job_id) or _not_found()
        versions = versions_db.for_job(session, job_id)
        latest = versions[0] if versions else None
        coverage = TailorReport.model_validate(latest.report).coverage if latest else []
        return page(
            request,
            "job.html",
            job=job,
            analysis=job_analysis(job),
            app=applications_db.for_job(session, job_id),
            versions=[(v, coverage_counts(v)) for v in versions],
            coverage={c.requirement: c for c in coverage},
            run=runs.latest(session, "tailor", job_id),
            match=matches_db.get_match(session, job_id),
            error=error,
        )

    @app.post("/jobs/{job_id}/analyze")
    def job_analyze(job_id: int, request: Request, session: Session = Depends(get_session)):
        job = jobs_db.get(session, job_id) or _not_found()
        inline = request.headers.get("HX-Request") == "true"  # the button on a list row
        try:
            job_service.analyze(session, job, deps.llm("extraction"), deps.settings, deps.scoring())
        except LLMError as e:
            if inline:
                return page(request, "_job_row.html", row=_row_for(session, job), error=str(e))
            return job_detail(job_id, request, session, error=str(e))
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
        jobs_db.get(session, job_id) or _not_found()
        app_row = application_service.set_status(session, job_id, status or None)
        return page(request, "_status.html", job_id=job_id, app=app_row, saved=True)

    # --- tailoring --------------------------------------------------------------------------

    def tailor_work(job_id: int) -> runs.Work:
        def work(session: Session) -> tuple[str, int]:
            job = jobs_db.get(session, job_id)
            version = tailoring_service.create_version(
                session, job, deps.profile(), deps.llm("tailoring"), deps.settings
            )
            return "", version.id

        return work

    @app.post("/jobs/{job_id}/tailor")
    def tailor_start(job_id: int, request: Request, session: Session = Depends(get_session)):
        job = jobs_db.get(session, job_id) or _not_found()
        if job_analysis(job) is None:
            raise HTTPException(400, "Job has no extracted requirements yet")
        run = runs.start(
            session, deps.sessions, deps.runner, "tailor", tailor_work(job_id), job_id=job_id
        )
        return page(request, "_run.html", job_id=job_id, run=run)

    @app.get("/jobs/{job_id}/tailor")
    def tailor_status(job_id: int, request: Request, session: Session = Depends(get_session)):
        run = runs.latest(session, "tailor", job_id)
        if run and run.state == "done":
            return Response(headers={"HX-Redirect": f"/versions/{run.result_id}"})
        return page(request, "_run.html", job_id=job_id, run=run)

    # --- resume versions --------------------------------------------------------------------

    def version_page(request: Request, session: Session, version: ResumeVersion, notice=""):
        job = jobs_db.get(session, version.job_id)
        return page(
            request,
            "version.html",
            job=job,
            version=version,
            doc=ResumeDoc.model_validate(version.content),
            report=TailorReport.model_validate(version.report),
            analysis=job_analysis(job),
            app=applications_db.for_job(session, job.id),
            notice=notice,
        )

    @app.get("/versions/{version_id}")
    def version_view(version_id: int, request: Request, session: Session = Depends(get_session)):
        version = versions_db.get(session, version_id) or _not_found()
        notice = {"saved": "Changes saved and resume rebuilt.", "approved": "Approved."}.get(
            request.query_params.get("done", ""), ""
        )
        return version_page(request, session, version, notice)

    @app.post("/versions/{version_id}")
    async def version_save(
        version_id: int, request: Request, session: Session = Depends(get_session)
    ):
        version = versions_db.get(session, version_id) or _not_found()
        form = await request.form()
        doc = apply_form(ResumeDoc.model_validate(version.content), form)
        tailoring_service.save_edits(session, version, doc)
        return RedirectResponse(f"/versions/{version_id}?done=saved", status_code=303)

    @app.post("/versions/{version_id}/approve")
    def version_approve(version_id: int, session: Session = Depends(get_session)):
        version = versions_db.get(session, version_id) or _not_found()
        tailoring_service.approve_version(session, version)
        return RedirectResponse(f"/versions/{version_id}?done=approved", status_code=303)

    @app.get("/versions/{version_id}/{kind}")
    def version_file(version_id: int, kind: str, session: Session = Depends(get_session)):
        version = versions_db.get(session, version_id) or _not_found()
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
            run=runs.latest(session, "discover"),
            has_search=deps.search_path.exists(),
        )

    def discovery_work(session: Session) -> tuple[str, None]:
        result = discover(
            session, deps.search(), deps.profile(), deps.llm("extraction"), deps.settings
        )
        kept = sum(len(c.kept) for c in result.companies)
        failed = [c.name for c in result.companies if c.error]
        message = (
            f"Checked {len(result.companies)} companies: {kept} jobs fit your filters, "
            f"{result.new_or_changed} new or changed, {len(result.analyzed)} analysed."
        )
        if failed:
            message += f" Couldn't reach: {', '.join(failed)}."
        return " ".join([message, *result.notes]), None

    @app.post("/discover")
    def discover_start(request: Request, session: Session = Depends(get_session)):
        run = runs.start(session, deps.sessions, deps.runner, "discover", discovery_work)
        return page(request, "_discover.html", run=run)

    @app.get("/discover")
    def discover_status(request: Request, session: Session = Depends(get_session)):
        run = runs.latest(session, "discover")
        if run and run.state == "done":
            return Response(headers={"HX-Redirect": "/matches"})
        return page(request, "_discover.html", run=run)

    @app.get("/applications")
    def applications(request: Request, session: Session = Depends(get_session)):
        rows = applications_db.with_jobs(session)
        groups = {s: [(a, j) for a, j in rows if a.status == s] for s in STATUSES}
        return page(request, "applications.html", groups=groups)

    @app.post("/applications/{job_id}/notes")
    def application_notes(
        job_id: int, notes: str = Form(""), session: Session = Depends(get_session)
    ):
        application_service.set_notes(session, job_id, notes) or _not_found()
        return HTMLResponse('<span class="saved" role="status">Saved</span>')

    @app.get("/profile")
    def profile_view(request: Request):
        try:
            profile = deps.profile()
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


app = create_app(build())
