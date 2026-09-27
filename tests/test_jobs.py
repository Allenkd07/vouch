import json
from pathlib import Path

import pytest

from vouch.jobs.html import html_to_text
from vouch.jobs.requirements import JobAnalysis, analyze_job, quote_in_text
from vouch.jobs.sources import (
    FetchError,
    parse_greenhouse,
    parse_web,
    parse_workday,
    workday_api_url,
)
from vouch.jobs.store import content_hash
from vouch.llm import FakeLLM

FIXTURES = Path(__file__).parent / "fixtures"
JIOSTAR_URL = (
    "https://jiostar.wd102.myworkdayjobs.com/jiostar/job/Bengaluru--We-Work/"
    "Software-Development-Engineer-II_JR12388-2"
)


@pytest.fixture
def jiostar():
    data = json.loads((FIXTURES / "workday_jiostar.json").read_text(encoding="utf-8"))
    return parse_workday(data, JIOSTAR_URL)


@pytest.mark.parametrize(
    "url",
    [
        JIOSTAR_URL,
        JIOSTAR_URL.replace("/jiostar/job", "/en-US/jiostar/job"),  # locale prefix
    ],
)
def test_workday_api_url(url):
    assert workday_api_url(url) == (
        "https://jiostar.wd102.myworkdayjobs.com/wday/cxs/jiostar/jiostar/job/"
        "Bengaluru--We-Work/Software-Development-Engineer-II_JR12388-2"
    )


def test_workday_api_url_rejects_non_job_pages():
    with pytest.raises(FetchError):
        workday_api_url("https://jiostar.wd102.myworkdayjobs.com/jiostar")


def test_parse_workday(jiostar):
    assert jiostar.company == "Jiostar India Private Limited"  # "7018 " prefix stripped
    assert jiostar.title == "Software Development Engineer II"
    assert jiostar.external_id == "JR12388"
    assert "- Proficiency in Java, Golang, or Python" in jiostar.description
    assert "<br" not in jiostar.description


def test_html_to_text_lists_and_entities():
    html = "<p>About&nbsp;us</p><ul><li>Go &amp; Python</li><li></li></ul><script>x()</script>"
    assert html_to_text(html) == "About us\n\n- Go & Python"


def test_parse_greenhouse_unescapes_content():
    data = {
        "id": 42,
        "title": "ML Engineer",
        "company_name": "Acme",
        "location": {"name": "Remote"},
        "content": "&lt;p&gt;Build &lt;strong&gt;RAG&lt;/strong&gt; systems&lt;/p&gt;",
        "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/42",
    }
    job = parse_greenhouse(data, "acme")
    assert (job.company, job.location, job.description) == ("Acme", "Remote", "Build RAG systems")


def test_parse_web_prefers_json_ld():
    page = """<html><title>Careers</title><script type="application/ld+json">
    {"@context": "https://schema.org", "@type": "JobPosting", "title": "Backend Engineer",
     "description": "&lt;p&gt;Kafka and Go&lt;/p&gt;", "hiringOrganization": {"name": "Beta"},
     "jobLocation": {"address": {"addressLocality": "Kochi", "addressCountry": "IN"}}}
    </script></html>"""
    job = parse_web(page, "https://beta.example/jobs/1")
    assert (job.title, job.company, job.location) == ("Backend Engineer", "Beta", "Kochi, IN")
    assert job.description == "Kafka and Go"


def test_content_hash_ignores_whitespace_and_case():
    assert content_hash("Build  APIs\n") == content_hash("build apis")
    assert content_hash("Build APIs") != content_hash("Build UIs")


def test_quote_matching_tolerates_punctuation_and_case():
    text = "● Experience working with cloud platforms (AWS preferred), messaging systems (Kafka)"
    assert quote_in_text("cloud platforms (AWS preferred)", text)
    assert quote_in_text("Messaging systems", text)
    assert not quote_in_text("distributed systems", text)
    assert not quote_in_text("", text)


def _analysis(**overrides) -> dict:
    base = {
        "title": "SDE II",
        "company": "JioStar",
        "seniority": "mid",
        "years_experience_min": 2,
        "years_experience_max": 4,
        "location": "Bangalore",
        "work_mode": "onsite",
        "summary": "Backend services.",
        "responsibilities": ["Build microservices"],
        "requirements": [
            {
                "name": "Backend language",
                "category": "language",
                "options": ["Java", "Golang", "Python"],
                "must_have": True,
                "weight": 3,
                "quote": "Proficiency in Java, Golang, or Python",
            },
            {
                "name": "Rust",
                "category": "language",
                "options": ["Rust"],
                "must_have": True,
                "weight": 3,
                "quote": "Expert in Rust",  # hallucinated: not in the posting
            },
        ],
        "keywords": ["Kafka", "microservices", "distributed systems"],
    }
    return {**base, **overrides}


def test_analyze_job_flags_hallucinated_requirements_and_keywords(jiostar):
    llm = FakeLLM(lambda prompt, schema: JobAnalysis.model_validate(_analysis()))

    result = analyze_job(jiostar.description, llm, title=jiostar.title)

    assert result.unverified_requirements == ["Rust"]
    assert result.analysis.keywords == ["Kafka", "microservices"]
    assert result.dropped_keywords == ["distributed systems"]
    assert "Job title: Software Development Engineer II" in llm.prompts[0]
