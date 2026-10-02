from vouch.bootstrap import build
from vouch.config import Settings
from vouch.llm import FakeLLM, get_llm


def test_each_role_runs_on_its_own_model():
    settings = Settings(llm_model="big", extraction_model="small")
    assert settings.model_for("tailoring") == "big"
    assert settings.model_for("extraction") == "small"
    assert isinstance(get_llm(Settings(llm_provider="fake"), "extraction"), FakeLLM)


def test_deps_create_each_llm_once_and_only_when_used():
    made = []

    def factory(role):
        made.append(role)
        return FakeLLM()

    deps = build(Settings(llm_provider="fake"), sessions=object(), llm_factory=factory)
    assert made == []
    assert deps.llm("extraction") is deps.llm("extraction")
    deps.llm("tailoring")
    assert made == ["extraction", "tailoring"]
