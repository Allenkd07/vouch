from pathlib import Path

import pytest
from fakes import ScriptedLLM, make_analysis, make_evidence

from vouch.profile.schema import load_profile
from vouch.tailoring.document import format_dates
from vouch.tailoring.pipeline import tailor
from vouch.tailoring.render import page_count, render_docx
from vouch.tailoring.select import Budget, drop_weakest, select_bullets
from vouch.tailoring.verify import deterministic_issues

EXAMPLE = Path(__file__).parent.parent / "profile" / "profile.example.yaml"
VOCAB = {"Python", "PostgreSQL", "Celery", "Redis", "Kubernetes", "Kafka"}


@pytest.fixture
def profile():
    return load_profile(EXAMPLE)


@pytest.fixture
def analysis():
    return make_analysis()


# --- verification ------------------------------------------------------------------------------


def test_deterministic_flags_added_tool_and_number():
    source = "Built a service in Python processing 2M records/day"
    assert deterministic_issues("Built a Python service on Kubernetes", source, VOCAB) == [
        "adds 'Kubernetes'"
    ]
    assert deterministic_issues("Built a Python service for 5M records/day", source, VOCAB) == [
        "adds number '5'"
    ]


def test_deterministic_allows_vendor_prefix_and_plurals():
    vocab = {"Kafka", "Apache Kafka", "Apache Spark", "API integration"}
    source = "Published Kafka job-status events; built third-party API integrations"
    assert deterministic_issues("Produced Apache Kafka events", source, vocab) == []
    assert deterministic_issues("Built API integration for partners", source, vocab) == []
    # The prefix only drops when the bare name is a tool the source names.
    assert deterministic_issues("Ran Apache Spark jobs", source, vocab) == ["adds 'Apache Spark'"]


def test_vocabulary_skips_options_of_non_technical_requirements(profile, analysis):
    from fakes import _req

    from vouch.tailoring.pipeline import vocabulary

    roles = _req("Technical roles", options=["backend engineering", "data engineering"])
    roles.category = "experience"
    queues = _req("Message queues", options=["Apache Kafka"])
    vocab = vocabulary(profile, analysis.model_copy(update={"requirements": [roles, queues]}))
    assert "Apache Kafka" in vocab
    assert "backend engineering" not in vocab and "data engineering" not in vocab


def test_deterministic_allows_rewording_and_formatting():
    source = "Processed 1,200 PDFs with Python and PostgreSQL"
    ok = "Processed 1200 PDFs using PostgreSQL and Python"
    assert deterministic_issues(ok, source, VOCAB) == []


# --- evidence ----------------------------------------------------------------------------------


def test_evidence_keeps_items_named_with_the_whole_prompt_line(profile, analysis):
    from fakes import make_evidence

    from vouch.llm import FakeLLM
    from vouch.tailoring.evidence import map_evidence

    def respond(prompt, schema):
        reply = make_evidence()
        for i in reply.items:  # as the model sometimes answers
            i.requirement += f' (must-have, weight 3) | posting says: "{i.requirement}"'
        reply.items[0].requirement = "Python development"  # an unknown longer name: dropped
        return reply

    emap = map_evidence(profile, analysis, FakeLLM(respond))
    assert [i.requirement for i in emap.items] == [
        "Python",
        "Task queues",
        "Mentoring",
        "Kubernetes",
    ]
    assert emap.for_requirement("Python").note == "not assessed"
    assert [b.bullet_id for b in emap.for_requirement("Task queues").bullets] == ["finlytics-2"]


# --- selection ---------------------------------------------------------------------------------


def test_selection_covers_must_haves_and_skips_irrelevant(profile, analysis):
    sel = select_bullets(profile, analysis, make_evidence())
    chosen = set(sel.bullet_ids())
    assert {"finlytics-1", "finlytics-2", "finlytics-3"} <= chosen
    assert sel.supports["Task queues"] == ["finlytics-2"]


def test_selection_respects_caps(profile, analysis):
    sel = select_bullets(profile, analysis, make_evidence(), Budget(first_role=2, min_per_role=1))
    assert len(sel.sections["experience:finlytics"]) == 2


def test_drop_weakest_keeps_sole_evidence_for_must_haves(profile, analysis):
    sel = select_bullets(profile, analysis, make_evidence())
    dropped = drop_weakest(sel, profile, analysis)
    # docqa-1 only partially supports Python, which finlytics-1 also covers.
    assert dropped == "docqa-1"


def test_format_dates():
    assert format_dates("2024-09 — 2024-12") == "Sep 2024 – Dec 2024"
    assert format_dates("2020-2024") == "2020 – 2024"
    assert format_dates("2025-07") == "Jul 2025"


# --- end to end with a fake LLM -----------------------------------------------------------------


def test_tailor_end_to_end(profile, analysis, tmp_path):
    llm = ScriptedLLM()
    result = tailor(profile, analysis, llm)

    bullets = {b.id: b for e in result.doc.experience + result.doc.projects for b in e.bullets}
    # Fixed on retry.
    assert "Kubernetes" not in bullets["finlytics-1"].text
    assert bullets["finlytics-1"].changed and not bullets["finlytics-1"].flags
    # Judge rejected twice -> original text kept and flagged.
    assert (
        bullets["finlytics-3"].text
        == "Mentored 2 interns and ran weekly code reviews for the payments team"
    )
    assert bullets["finlytics-3"].flags == ["overclaim"]
    # Summary rejected on both attempts -> profile summary (None in the example) kept.
    assert result.doc.summary is None and "summary" in result.report.fallbacks
    assert llm.rewrite_calls == 2

    status = {c.requirement: c.status for c in result.report.coverage}
    assert status == {
        "Python": "strong",
        "Task queues": "strong",
        "Mentoring": "strong",
        "Kubernetes": "gap",
    }
    assert result.report.pages == 1 == page_count(result.pdf)

    docx = tmp_path / "r.docx"
    render_docx(result.doc, docx)
    assert docx.stat().st_size > 0
