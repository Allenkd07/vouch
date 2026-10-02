"""Discovery steps run on their own, with in-memory jobs and no database."""

import re
from datetime import UTC, datetime
from pathlib import Path

from fakes import make_analysis

from vouch.config import Settings
from vouch.db import Job
from vouch.discovery.config import SearchConfig
from vouch.discovery.service import discover
from vouch.discovery.steps import DiscoveryContext, extract
from vouch.jobs.requirements import BatchAnalysis, BatchItem
from vouch.llm import FakeLLM
from vouch.profile.schema import load_profile

EXAMPLE = Path(__file__).parent.parent / "profile" / "profile.example.yaml"
NOW = datetime(2026, 9, 27, tzinfo=UTC)


class StubSession:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


def _context(llm, analyze, batch_size=2, jobs=(), similarity=None):
    config = SearchConfig(companies=[])
    config.ranking.batch_size = batch_size
    return DiscoveryContext(
        session=StubSession(),
        config=config,
        profile=load_profile(EXAMPLE),
        llm=llm,
        settings=Settings(llm_provider="fake"),
        now=NOW,
        analyze=analyze,
        active=list(jobs),
        similarity=similarity or {},
    )


def _every_posting(prompt, schema):
    refs = re.findall(r"^=== \[([^\]]+)\]", prompt, re.M)
    return BatchAnalysis(items=[BatchItem(ref=r, analysis=make_analysis()) for r in refs])


def test_extract_reads_the_most_similar_jobs_first_in_batches():
    jobs = [
        Job(id=i, title=f"SDE {i}", description="Python, Celery", content_hash=f"h{i}")
        for i in range(1, 6)
    ]
    llm = FakeLLM(_every_posting)
    ctx = _context(llm, analyze=3, jobs=jobs, similarity={1: 0.1, 2: 0.9, 3: 0.5, 4: 0.7, 5: 0.2})

    extract(ctx)

    assert ctx.result.analyzed == [2, 4, 3]  # top 3 by similarity
    assert ctx.result.requests == len(llm.prompts) == 2  # 2 + 1 postings
    assert ctx.session.commits == 2  # after each request
    assert any("2 job(s) still need requirement extraction" in n for n in ctx.result.notes)


def test_extract_stops_cleanly_when_the_llm_fails():
    jobs = [Job(id=1, title="SDE", description="Python", content_hash="h1")]
    ctx = _context(FakeLLM(), analyze=5, jobs=jobs)  # no responder: raises LLMError

    extract(ctx)

    assert ctx.result.analyzed == [] and ctx.result.requests == 0
    assert any(n.startswith("Stopped extracting requirements") for n in ctx.result.notes)


def test_discover_runs_the_given_steps_in_order_and_times_them():
    order = []

    def first(ctx):
        order.append("first")

    def second(ctx):
        order.append("second")
        ctx.result.scored = 7

    session = StubSession()
    result = discover(
        session,
        SearchConfig(companies=[]),
        load_profile(EXAMPLE),
        FakeLLM(),
        Settings(llm_provider="fake"),
        now=NOW,
        steps=[first, second],
    )

    assert order == ["first", "second"] and result.scored == 7
    assert list(result.seconds) == ["first", "second"]
    assert session.commits == 2  # one per step
