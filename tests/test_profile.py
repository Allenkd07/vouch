from pathlib import Path

import pytest
from pydantic import ValidationError

from vouch.llm import FakeLLM
from vouch.profile.bootstrap import bootstrap_profile
from vouch.profile.schema import Profile, dump_profile, lint, load_profile

EXAMPLE = Path(__file__).parent.parent / "profile" / "profile.example.yaml"


def test_example_profile_is_valid_and_clean():
    profile = load_profile(EXAMPLE)
    assert [b.id for _, b in profile.iter_bullets()] == [
        "finlytics-1",
        "finlytics-2",
        "finlytics-3",
        "docqa-1",
    ]
    assert lint(profile) == []


def test_dump_round_trips():
    profile = load_profile(EXAMPLE)
    import yaml

    assert Profile.model_validate(yaml.safe_load(dump_profile(profile))) == profile


def _minimal(**extra) -> dict:
    return {"contact": {"name": "X"}, **extra}


def test_duplicate_ids_rejected():
    bullet = {"id": "a-1", "text": "did a thing"}
    job = {"id": "a", "company": "A", "title": "Eng", "start": 2024, "bullets": [bullet]}
    data = _minimal(
        experience=[job],
        achievements=[bullet],
    )
    with pytest.raises(ValidationError, match="duplicate id"):
        Profile.model_validate(data)


def test_ids_must_be_kebab_case():
    with pytest.raises(ValidationError):
        Profile.model_validate(_minimal(achievements=[{"id": "Bad_ID", "text": "x"}]))


def test_lint_flags_unlisted_tool_and_missing_metric():
    data = _minimal(
        achievements=[
            {
                "id": "ach-1",
                "text": "Sped up the build",
                "facts": {"tools": ["Bazel"], "metrics": ["3x"]},
            },
            {"id": "ach-2", "text": "Organised a hackathon"},
        ],
        skills={"tools": ["Git"]},
    )
    warnings = lint(Profile.model_validate(data))
    assert any("'Bazel' is not listed in skills" in w for w in warnings)
    assert any("'3x' does not appear" in w for w in warnings)
    assert any("ach-2: no facts" in w for w in warnings)


def test_bootstrap_sends_every_resume_to_the_llm(tmp_path):
    (tmp_path / "a.txt").write_text("Resume A: Python at Acme", encoding="utf-8")
    (tmp_path / "b.md").write_text("Resume B: Go at Beta", encoding="utf-8")
    llm = FakeLLM(lambda prompt, schema: _minimal())

    profile = bootstrap_profile([tmp_path / "a.txt", tmp_path / "b.md"], llm)

    assert profile.contact.name == "X"
    assert "Python at Acme" in llm.prompts[0] and "Go at Beta" in llm.prompts[0]


def test_bootstrap_rejects_unknown_file_type(tmp_path):
    f = tmp_path / "resume.rtf"
    f.write_text("x")
    with pytest.raises(ValueError, match="unsupported"):
        bootstrap_profile([f], FakeLLM())
