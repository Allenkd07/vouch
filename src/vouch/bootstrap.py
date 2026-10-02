"""Composition root: the one place that reads settings and decides which implementations run.

The CLI and the web app build a `Deps` here and pass it along; nothing below them reaches for
globals. Tests build one with fakes (`build(sessions=..., llm_factory=...)`)."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from vouch.config import Role, Settings, get_settings
from vouch.db import get_sessionmaker
from vouch.discovery.config import DEFAULT_SEARCH, SearchConfig, load_search
from vouch.jobs.service import Scoring, load_scoring
from vouch.llm import LLM, get_llm
from vouch.profile.schema import DEFAULT_PROFILE, Profile, load_profile


@dataclass
class Deps:
    settings: Settings
    sessions: sessionmaker
    llm_factory: Callable[[Role], LLM]
    profile_path: Path = DEFAULT_PROFILE
    search_path: Path = DEFAULT_SEARCH
    _llms: dict[Role, LLM] = field(default_factory=dict, repr=False)

    def llm(self, role: Role) -> LLM:
        """The LLM for `role`, created on first use (so commands that never call an LLM don't
        need an API key)."""
        if role not in self._llms:
            self._llms[role] = self.llm_factory(role)
        return self._llms[role]

    # The profile and search files can be edited while the web app runs, so these read them
    # fresh on every call; callers load them once per request or command.
    def profile(self) -> Profile:
        return load_profile(self.profile_path)

    def search(self) -> SearchConfig:
        return load_search(self.search_path)

    def scoring(self) -> Scoring | None:
        return load_scoring(self.profile_path, self.search_path)


def build(
    settings: Settings | None = None,
    *,
    sessions: sessionmaker | None = None,
    llm_factory: Callable[[Role], LLM] | None = None,
    profile_path: Path = DEFAULT_PROFILE,
    search_path: Path = DEFAULT_SEARCH,
) -> Deps:
    settings = settings or get_settings()
    return Deps(
        settings=settings,
        sessions=sessions or get_sessionmaker(),
        llm_factory=llm_factory or (lambda role: get_llm(settings, role)),
        profile_path=profile_path,
        search_path=search_path,
    )
