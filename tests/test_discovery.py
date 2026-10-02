import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from vouch.boards import FetchedJob, list_jobs
from vouch.config import Settings
from vouch.db import Job
from vouch.discovery.config import Company, Filters, SearchConfig
from vouch.discovery.filters import rejection_reason
from vouch.discovery.fit import score_fit, years_of_experience
from vouch.discovery.run import discover, ranked_matches, skill_gaps
from vouch.jobs.requirements import JobAnalysis, Requirement
from vouch.llm import FakeLLM
from vouch.profile.schema import load_profile

EXAMPLE = Path(__file__).parent.parent / "profile" / "profile.example.yaml"
NOW = datetime(2026, 9, 27, tzinfo=UTC)
FILTERS = Filters(
    title_include=["engineer", "sde"],
    title_exclude=["senior", "sr", "manager"],
    locations=["india", "bengaluru"],
    max_age_days=30,
)


def _job(title="Software Engineer", location="Bengaluru, India", posted=None, **kw) -> FetchedJob:
    return FetchedJob(
        "greenhouse", kw.get("url", "u"), "Acme", title, location, "desc", posted_at=posted
    )


# --- filters -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "location", "posted", "reason"),
    [
        ("Backend Engineer", "Bengaluru", None, None),
        ("SDE II", "Pune, India", "2026-09-20", None),
        ("Senior Backend Engineer", "Bengaluru", None, "title contains 'senior'"),
        ("Sr. Engineer", "Bengaluru", None, "title contains 'sr'"),
        ("Engineering Manager", "Bengaluru", None, "title"),  # "engineering" isn't "engineer"
        ("Backend Engineer", "Remote - USA", None, "location"),
        ("Backend Engineer", "Bengaluru", "2026-07-01T00:00:00Z", "too old"),
        ("Backend Engineer", "Bengaluru", "Posted 2 Days Ago", None),  # unparseable -> kept
    ],
)
def test_filters(title, location, posted, reason):
    assert rejection_reason(_job(title, location, posted), FILTERS, NOW) == reason


def test_exclude_matches_whole_words_only():
    assert rejection_reason(_job("Srinivas Platform Engineer"), FILTERS, NOW) is None


# --- fit -----------------------------------------------------------------------------------------


def _req(name, options=(), must=True, weight=2, category="language"):
    return Requirement(
        name=name,
        category=category,
        options=list(options),
        must_have=must,
        weight=weight,
        quote=name,
    )


def _analysis(requirements, years=None, seniority="mid"):
    return JobAnalysis(
        title="Backend Engineer",
        company="Acme",
        seniority=seniority,
        years_experience_min=years,
        years_experience_max=None,
        location=None,
        work_mode="onsite",
        summary="s",
        responsibilities=[],
        requirements=requirements,
        keywords=[],
    )


def test_fit_matches_profile_skills_and_bullet_tools():
    profile = load_profile(EXAMPLE)  # skills: Python, PostgreSQL, Redis, ...; tools incl. Celery
    analysis = _analysis(
        [
            _req("Backend language", ["Java", "Python"], weight=3),
            _req("Go", ["Go"], weight=1),
            _req("Kafka", ["Kafka"], must=False, category="tool"),
            _req("Mentoring", category="soft_skill"),  # not checkable -> ignored
            _req("Database fundamentals", category="database"),  # no concrete options -> ignored
        ]
    )
    fit = score_fit(profile, analysis, similarity=None, now=NOW)
    assert fit.matched == ["Backend language"]
    assert fit.missing_must == ["Go"] and fit.missing_nice == ["Kafka"]
    assert fit.unchecked == ["Mentoring", "Database fundamentals"]
    assert (fit.checked, fit.total) == (3, 5)
    assert fit.must_coverage == pytest.approx(0.75)  # weights 3 of 4
    # Only must (0.55) and nice (0.15) are known: (0.55*0.75 + 0.15*0) / 0.70
    assert fit.score == pytest.approx(58.9, abs=0.1)


def test_fit_penalises_experience_beyond_the_accepted_range():
    profile = load_profile(EXAMPLE)  # first role starts 2023-07 -> ~3.2 years at NOW
    assert years_of_experience(profile, NOW) == pytest.approx(3.24, abs=0.02)
    python = _req("Python", ["Python"])

    def fit(**kw):
        return score_fit(profile, _analysis([python], **kw), 0.85, NOW, accept_years_up_to=5)

    assert fit(years=5).score == 100.0 and not fit(years=5).notes  # 2-5 years: no penalty
    assert fit(years=7).score == 100 - 24  # 2 years beyond the accepted range
    assert fit(years=12).score == 60.0  # penalty capped at 40
    assert fit(years=7).notes == ["asks for 7+ years; you have about 3.2"]
    staff = fit(seniority="staff")
    assert staff.score == 70.0 and staff.notes == ["staff-level role"]


# --- boards --------------------------------------------------------------------------------------


def _mock_client(routes: dict) -> httpx.Client:
    """routes: (method, url-without-query) -> JSON body, or a callable(request) -> body."""

    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, str(request.url).split("?")[0])
        body = routes[key]
        return httpx.Response(200, json=body(request) if callable(body) else body)

    return httpx.Client(transport=httpx.MockTransport(handler))


GREENHOUSE = {
    "jobs": [
        {
            "id": 1,
            "title": "Backend Engineer",
            "location": {"name": "Bengaluru, India"},
            "content": "&lt;p&gt;Python and Go&lt;/p&gt;",
            "absolute_url": "https://x/1",
            "updated_at": "2026-09-20T00:00:00Z",
        },
        {
            "id": 2,
            "title": "Senior Backend Engineer",
            "location": {"name": "Bengaluru"},
            "content": "&lt;p&gt;Go&lt;/p&gt;",
            "absolute_url": "https://x/2",
        },
        {
            "id": 3,
            "title": "Backend Engineer",
            "location": {"name": "Remote - USA"},
            "content": "&lt;p&gt;Go&lt;/p&gt;",
            "absolute_url": "https://x/3",
        },
    ]
}
GH_URL = ("GET", "https://boards-api.greenhouse.io/v1/boards/acme/jobs")


def test_workday_pages_past_the_first_response():
    company = Company(name="Wd", ats="workday", url="https://wd.wd1.myworkdayjobs.com/site")
    api = "https://wd.wd1.myworkdayjobs.com/wday/cxs/wd/site/jobs"

    def page(request):
        offset = json.loads(request.content)["offset"]
        # Workday reports the total only on the first page.
        return {
            "total": 25 if offset == 0 else 0,
            "jobPostings": [
                {"title": f"SDE {i}", "externalPath": f"/job/{i}", "locationsText": "Bengaluru"}
                for i in range(offset, min(offset + 20, 25))
            ],
        }

    jobs = list_jobs(company, _mock_client({("POST", api): page}))
    assert len(jobs) == 25 and jobs[-1].url == "https://wd.wd1.myworkdayjobs.com/site/job/24"


# --- batched requirement extraction --------------------------------------------------------------


def batch_responder(analysis, skip=()):
    """Fake LLM answer for analyze_jobs: the same analysis for every posting in the prompt."""
    import re

    from vouch.jobs.requirements import BatchAnalysis, BatchItem

    def respond(prompt, schema):
        assert schema is BatchAnalysis
        refs = re.findall(r"^=== \[([^\]]+)\]", prompt, re.M)
        return BatchAnalysis(
            items=[BatchItem(ref=r, analysis=analysis) for r in refs if r not in skip]
        )

    return respond


def test_analyze_jobs_verifies_each_posting_against_its_own_text():
    from vouch.jobs.requirements import analyze_jobs

    # One analysis quoting "Python" is returned for both postings; only one of them says it.
    analysis = _analysis([_req("Python", ["Python"])]).model_copy(update={"keywords": ["Python"]})
    analysis.requirements[0].quote = "Python"
    postings = [("a", "SDE", "We use Python and Go."), ("b", "SRE", "We use Terraform.")]
    llm = FakeLLM(batch_responder(analysis))

    results = analyze_jobs(postings, llm)

    assert len(llm.prompts) == 1  # one request for both
    assert results["a"].unverified_requirements == []
    # Borrowed from the other posting -> flagged, and the keyword is dropped.
    assert results["b"].unverified_requirements == ["Python"]
    assert results["b"].dropped_keywords == ["Python"]


def test_analyze_jobs_leaves_out_postings_the_model_skipped():
    from vouch.jobs.requirements import analyze_jobs

    llm = FakeLLM(batch_responder(_analysis([]), skip={"b"}))
    results = analyze_jobs([("a", None, "x"), ("b", None, "y")], llm)
    assert set(results) == {"a"}


def test_discovery_batches_extraction_requests(session):
    profile = load_profile(EXAMPLE)
    jobs = {
        "jobs": [
            {
                "id": i,
                "title": f"Backend Engineer {i}",
                "location": {"name": "Bengaluru"},
                "content": f"&lt;p&gt;Posting {i}: Python&lt;/p&gt;",
                "absolute_url": f"https://x/batch-{i}",
            }
            for i in range(14)
        ]
    }
    config = SearchConfig(
        companies=[Company(name="Acme", ats="greenhouse", board="acme")],
        filters=FILTERS.model_copy(update={"max_age_days": None}),
    )
    config.ranking.batch_size = 6
    llm = FakeLLM(batch_responder(_analysis([_req("Python", ["Python"])])))
    settings = Settings(llm_provider="fake")

    result = discover(
        session,
        config,
        profile,
        llm,
        settings,
        client=_mock_client({GH_URL: jobs}),
        now=NOW,
        analyze=1000,
    )
    ours = {j.id for j in session.scalars(select(Job).where(Job.url.like("https://x/batch-%")))}
    assert ours <= set(result.analyzed)
    # 14 postings of ours (plus whatever else is pending in the real DB), 6 per request.
    assert result.requests == len(llm.prompts) == -(-len(result.analyzed) // 6)
    assert all(
        j.analysis_model == settings.extraction_model
        for j in session.scalars(select(Job).where(Job.id.in_(ours)))
    )


# --- a whole run against the database ------------------------------------------------------------


def test_discover_run(session):
    profile = load_profile(EXAMPLE)
    config = SearchConfig(
        companies=[Company(name="Acme", ats="greenhouse", board="acme")],
        filters=FILTERS.model_copy(update={"max_age_days": None}),
    )
    analysis = _analysis([_req("Backend language", ["Python"]), _req("Go", ["Go"])])
    llm = FakeLLM(batch_responder(analysis))
    settings = Settings(llm_provider="fake", llm_model="fake-model")
    client = _mock_client({GH_URL: GREENHOUSE})

    # analyze=1000: other (real) unanalysed jobs in the database must not crowd this one out.
    first = discover(session, config, profile, llm, settings, client=client, now=NOW, analyze=1000)
    assert first.companies[0].listed == 3 and len(first.companies[0].kept) == 1
    job = session.scalar(select(Job).where(Job.url == "https://x/1"))
    # The test runs inside the real database (rolled back), so other jobs may exist too.
    assert first.new_or_changed == 1 and job.id in first.analyzed and first.embedded >= 1
    assert job.company == "Acme" and job.last_seen_at == NOW

    rows = [r for r in ranked_matches(session, now=NOW) if r[0].id == job.id]
    _, match, _ = rows[0]
    assert match.fit["missing_must"] == ["Go"] and match.score is not None
    assert skill_gaps(rows) == [("Go", 1)]

    # Same postings next day: nothing new, and the job isn't re-embedded or re-analysed.
    sent: list[str] = []
    real_embed = llm.embed
    llm.embed = lambda texts, dimensions=None: sent.extend(texts) or real_embed(texts, dimensions)
    second = discover(
        session, config, profile, llm, settings, client=client, now=NOW + timedelta(days=1)
    )
    assert second.new_or_changed == 0 and job.id not in second.analyzed
    assert not any(t.startswith("Backend Engineer\nAcme") for t in sent)  # only the profile

    # Embedding quota gone: similarity from the earlier run is kept, not wiped.
    from vouch.llm import LLMError

    def no_quota(texts, dimensions=None):
        raise LLMError("quota exhausted")

    llm.embed = no_quota
    offline = discover(
        session, config, profile, llm, settings, client=client, now=NOW + timedelta(days=2)
    )
    assert any("kept earlier values" in n for n in offline.notes)
    kept = [r for r in ranked_matches(session, now=NOW + timedelta(days=2)) if r[0].id == job.id]
    assert kept[0][1].similarity == match.similarity is not None
    llm.embed = real_embed

    # Posting closed: once unseen for more than ACTIVE_DAYS it drops out of the ranking.
    later = NOW + timedelta(days=6)  # last seen on day 2; more than ACTIVE_DAYS ago
    empty = _mock_client({GH_URL: {"jobs": []}})
    discover(session, config, profile, llm, settings, client=empty, now=later)
    assert all(r[0].id != job.id for r in ranked_matches(session, now=later))


def test_gemini_embed_falls_back_when_a_batch_comes_back_combined():
    from types import SimpleNamespace

    from google import genai

    from vouch.llm import GeminiLLM

    calls = []

    def embed_content(model, contents, config):
        calls.append(contents)
        # The model merges a whole list into a single embedding.
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0])])

    llm = object.__new__(GeminiLLM)
    llm._genai = genai
    llm._embedding_model = "m"
    llm._client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))

    vectors = llm.embed(["a", "b", "c"], dimensions=2)
    assert vectors == [[0.6, 0.8]] * 3  # one per text, unit length
    assert calls == [["a", "b", "c"], "a", "b", "c"]


# --- company probe -------------------------------------------------------------------------------


def test_slug_variants_and_names_file():
    from vouch.discovery.probe import read_names, slug_variants

    assert slug_variants("Pine Labs") == ["pinelabs", "pine-labs", "pinelabsindia", "pinelabshq"]
    assert slug_variants("Cult.fit") == ["cultfit", "cult-fit", "cultfitindia", "cultfithq"]
    assert slug_variants("CRED") == ["cred", "credindia", "credhq"]  # no duplicate for one word
    assert read_names("# fintech\nCRED\n\nGroww\nCRED\n") == ["CRED", "Groww"]


def test_probe_keeps_boards_that_answer_and_counts_india_jobs():
    from vouch.discovery.probe import probe

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v0/postings/acme":
            return httpx.Response(
                200,
                json=[
                    {"text": "SDE II", "categories": {"location": "Bengaluru"}, "id": "1"},
                    {"text": "Sales", "categories": {"location": "London"}, "id": "2"},
                ],
            )
        return httpx.Response(404, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    found = probe(["Acme"], client=client, workers=2)
    assert [(f.ats, f.board, f.india, f.total, f.sample) for f in found] == [
        ("lever", "acme", 1, 2, ["SDE II"])
    ]
