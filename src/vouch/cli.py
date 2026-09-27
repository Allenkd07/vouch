from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ValidationError

from vouch.config import get_settings

app = typer.Typer(no_args_is_help=True, help="Vouch: truthful, tailored resumes")
db_app = typer.Typer(no_args_is_help=True, help="Database checks")
llm_app = typer.Typer(no_args_is_help=True, help="LLM checks")
profile_app = typer.Typer(no_args_is_help=True, help="Master profile (profile/profile.yaml)")
app.add_typer(db_app, name="db")
app.add_typer(llm_app, name="llm")
app.add_typer(profile_app, name="profile")
job_app = typer.Typer(no_args_is_help=True, help="Jobs: add from a link, list, show")
app.add_typer(job_app, name="job")

DEFAULT_PROFILE = Path("profile/profile.yaml")
ProfilePath = Annotated[Path, typer.Argument(exists=True)]


@db_app.command("check")
def db_check() -> None:
    """Connect to Postgres and confirm pgvector and migrations are in place."""
    from sqlalchemy import text

    from vouch.db import get_engine

    with get_engine().connect() as conn:
        version = conn.scalar(text("SHOW server_version"))
        vector = conn.scalar(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
        revision = conn.scalar(text("SELECT version_num FROM alembic_version"))
    typer.echo(f"postgres {version}, pgvector {vector or 'MISSING'}, migration {revision}")


@llm_app.command("check")
def llm_check() -> None:
    """One structured generation and one embedding against the configured provider."""
    from vouch.llm import get_llm

    class Ping(BaseModel):
        answer: str

    settings = get_settings()
    llm = get_llm(settings)
    reply = llm.generate_json("Reply with answer='pong'.", Ping)
    vector = llm.embed(["senior backend engineer, Python, PostgreSQL"])[0]
    typer.echo(f"{settings.llm_model}: {reply.answer!r}")
    typer.echo(f"{settings.embedding_model}: {len(vector)} dimensions")


@profile_app.command("bootstrap")
def profile_bootstrap(
    files: Annotated[
        list[Path], typer.Argument(exists=True, help="Resume files (.pdf/.docx/.txt/.md)")
    ],
    out: Annotated[Path, typer.Option("--out", "-o")] = DEFAULT_PROFILE,
    force: Annotated[bool, typer.Option(help="Overwrite an existing profile")] = False,
) -> None:
    """Draft profile.yaml from existing resumes. Review every line before using it."""
    from vouch.llm import get_llm
    from vouch.profile.bootstrap import bootstrap_profile
    from vouch.profile.schema import dump_profile, lint

    if out.exists() and not force:
        raise typer.BadParameter(f"{out} exists; pass --force to overwrite", param_hint="--out")
    profile = bootstrap_profile(files, get_llm())
    header = (
        "# DRAFT generated from: " + ", ".join(f.name for f in files) + "\n"
        "# Review every bullet: fix wording, fill in facts (tools/metrics/scope),\n"
        "# and add missing work.\n"
        "# Everything here is what tailored resumes are allowed to claim.\n\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + dump_profile(profile), encoding="utf-8")
    bullets = sum(1 for _ in profile.iter_bullets())
    typer.echo(
        f"Wrote {out}: {len(profile.experience)} roles, {len(profile.projects)} projects, "
        f"{bullets} bullets"
    )
    _print_warnings(lint(profile))


@profile_app.command("validate")
def profile_validate(path: ProfilePath = DEFAULT_PROFILE) -> None:
    """Check the profile against the schema and list quality warnings."""
    from vouch.profile.schema import lint

    profile = _load(path)
    bullets = sum(1 for _ in profile.iter_bullets())
    typer.echo(
        f"OK: {len(profile.experience)} roles, {len(profile.projects)} projects, {bullets} bullets"
    )
    _print_warnings(lint(profile))


@profile_app.command("sync")
def profile_sync(path: ProfilePath = DEFAULT_PROFILE) -> None:
    """Store the profile in the database (snapshot + one row per bullet)."""
    from vouch.db import get_sessionmaker
    from vouch.profile.sync import sync_profile

    profile = _load(path)
    with get_sessionmaker()() as session:
        result = sync_profile(session, profile)
    state = "new snapshot" if result.new_snapshot else "unchanged"
    typer.echo(
        f"Snapshot {result.snapshot_id} ({state}): {result.items} bullets, {result.removed} removed"
    )


def _load(path: Path):
    from vouch.profile.schema import load_profile

    try:
        return load_profile(path)
    except ValidationError as e:
        typer.echo(f"{path} is invalid:\n{e}", err=True)
        raise typer.Exit(1) from e


def _print_warnings(warnings: list[str]) -> None:
    if warnings:
        typer.echo(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            typer.echo(f"  - {w}")


@job_app.command("add")
def job_add(
    url: Annotated[str | None, typer.Argument(help="Job posting link")] = None,
    text_file: Annotated[
        Path | None,
        typer.Option("--text-file", "-f", exists=True, help="Pasted job description (.txt/.html)"),
    ] = None,
    company: Annotated[str | None, typer.Option(help="Override company name")] = None,
    title: Annotated[str | None, typer.Option(help="Override job title")] = None,
    reanalyze: Annotated[bool, typer.Option(help="Re-run extraction even if unchanged")] = False,
) -> None:
    """Fetch a job (or read pasted text), store it, and extract its requirements."""
    from vouch.db import get_sessionmaker
    from vouch.jobs.sources import FetchError
    from vouch.llm import get_llm
    from vouch.services import add_job

    if not url and not text_file:
        raise typer.BadParameter("give a job URL, or --text-file with the pasted description")
    settings = get_settings()
    with get_sessionmaker()() as session:
        try:
            job, changed = add_job(
                session,
                get_llm(settings),
                settings,
                url=url,
                text=text_file.read_text(encoding="utf-8") if text_file else None,
                company=company,
                title=title,
                reanalyze=reanalyze,
            )
        except FetchError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1) from e
        session.commit()
        state = "new/changed" if changed else "unchanged"
        typer.echo(f"Job {job.id}: {job.title} @ {job.company} ({state})")
        _print_job(job)


@job_app.command("list")
def job_list(limit: int = 20) -> None:
    """Most recently fetched jobs."""
    from sqlalchemy import select

    from vouch.db import Job, get_sessionmaker

    with get_sessionmaker()() as session:
        jobs = session.scalars(select(Job).order_by(Job.fetched_at.desc()).limit(limit)).all()
    for j in jobs:
        typer.echo(f"{j.id:>4}  {j.fetched_at:%Y-%m-%d}  {j.company or '?'} | {j.title or '?'}")


@job_app.command("show")
def job_show(job_id: int, description: bool = False) -> None:
    """Requirements for one job (--description to include the full posting)."""
    from vouch.db import Job, get_sessionmaker

    with get_sessionmaker()() as session:
        job = session.get(Job, job_id)
    if job is None:
        typer.echo(f"no job {job_id}", err=True)
        raise typer.Exit(1)
    _print_job(job)
    if description:
        typer.echo(f"\n--- Posting ---\n{job.description}")


def _print_job(job) -> None:
    from vouch.jobs.requirements import VerifiedAnalysis

    typer.echo(f"\n{job.title} @ {job.company}  [{job.source}]  {job.url}")
    if not job.analysis:
        typer.echo("(not analysed yet)")
        return
    result = VerifiedAnalysis.model_validate(job.analysis)
    a = result.analysis
    years = (
        f"{a.years_experience_min}-{a.years_experience_max} yrs"
        if a.years_experience_max
        else f"{a.years_experience_min}+ yrs"
        if a.years_experience_min
        else "years not stated"
    )
    typer.echo(f"{a.seniority} | {years} | {a.location or job.location} | {a.work_mode}")
    typer.echo(f"\n{a.summary}")
    for label, must in (("Must-have", True), ("Nice-to-have", False)):
        reqs = sorted((r for r in a.requirements if r.must_have == must), key=lambda r: -r.weight)
        if not reqs:
            continue
        typer.echo(f"\n{label}:")
        for r in reqs:
            options = f"  [{' / '.join(r.options)}]" if r.options else ""
            flag = (
                "  (!) quote not found in posting"
                if r.name in result.unverified_requirements
                else ""
            )
            typer.echo(f"  {'*' * r.weight:<3} {r.name}{options}{flag}")
    typer.echo(f"\nATS keywords: {', '.join(a.keywords)}")
    if result.dropped_keywords:
        typer.echo(f"(dropped, not in posting: {', '.join(result.dropped_keywords)})")


@app.command("tailor")
def tailor_cmd(
    job_id: int,
    pages: Annotated[int, typer.Option(help="Page limit")] = 1,
    profile_path: Annotated[Path, typer.Option("--profile", exists=True)] = DEFAULT_PROFILE,
    out: Annotated[Path, typer.Option(help="Output folder")] = Path("output"),
) -> None:
    """Build a tailored, verified resume (PDF + DOCX + report) for a stored job."""

    from vouch.db import Job, get_sessionmaker
    from vouch.llm import get_llm
    from vouch.services import create_version, job_analysis
    from vouch.tailoring.document import ResumeDoc
    from vouch.tailoring.pipeline import TailorReport

    settings = get_settings()
    profile = _load(profile_path)
    with get_sessionmaker()() as session:
        job = session.get(Job, job_id)
        analysis = job_analysis(job) if job else None
        if analysis is None:
            typer.echo(f"job {job_id} not found or not analysed (run `vouch job add`)", err=True)
            raise typer.Exit(1)
        typer.echo(f"Tailoring for {job.title} @ {job.company} with {settings.llm_model}...")
        version = create_version(
            session, job, profile, get_llm(settings), settings, pages=pages, out=out
        )
        session.commit()

    r = TailorReport.model_validate(version.report)
    doc = ResumeDoc.model_validate(version.content)
    pdf_path, docx_path = Path(version.pdf_path), Path(version.docx_path)
    folder = pdf_path.parent
    counts = {
        s: sum(c.status == s for c in r.coverage)
        for s in ("strong", "partial", "education", "skills-only", "gap")
    }
    must_gaps = [c.requirement for c in r.coverage if c.must_have and c.status == "gap"]
    changed = sum(b.changed for e in doc.experience + doc.projects for b in e.bullets)
    typer.echo(
        f"\nVersion {version.id}: {r.pages} page(s), {changed} bullets rewritten, "
        f"{len(r.fallbacks)} rewrite(s) rejected and kept original"
    )
    typer.echo(
        f"Coverage: {counts['strong']} strong, {counts['partial']} partial, "
        f"{counts['education']} via education, {counts['skills-only']} skills-only, "
        f"{counts['gap']} gap"
    )
    if must_gaps:
        typer.echo(f"Must-have gaps: {', '.join(must_gaps)}")
    typer.echo(
        f"ATS keywords: {len(r.keywords_before)} -> {len(r.keywords_after)} "
        f"of {len(analysis.keywords)}"
    )
    typer.echo(f"\n{pdf_path}\n{docx_path}\n{folder / 'report.md'}")


def main() -> None:
    """Entry point: LLM failures (quota, overload) print one line instead of a traceback."""
    from vouch.llm import LLMError

    try:
        app()
    except LLMError as e:
        typer.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@app.command("web")
def web(
    port: Annotated[int, typer.Option(help="Port")] = 8000,
    reload: Annotated[bool, typer.Option(help="Reload on code changes")] = False,
) -> None:
    """Start the web UI on http://localhost:<port>."""
    import uvicorn

    typer.echo(f"Vouch running at http://localhost:{port}")
    uvicorn.run("vouch.web.app:app", host="127.0.0.1", port=port, reload=reload)
