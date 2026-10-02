"""What every job board (ATS) provides, and the registry of supported boards.

Each board lives in its own module and registers itself, so supporting a new ATS means adding
one module; discovery, `job add` and the company probe find it through `BOARDS`."""

from dataclasses import dataclass
from typing import ClassVar, Literal

import httpx
from pydantic import BaseModel, Field, model_validator

USER_AGENT = "Mozilla/5.0 (vouch; personal job search tool)"


class FetchError(RuntimeError):
    pass


@dataclass
class FetchedJob:
    source: str  # a board name, or web | manual
    url: str
    company: str | None
    title: str | None
    location: str | None
    description: str  # plain text
    external_id: str | None = None
    posted_at: str | None = None


class Company(BaseModel):
    """Where a company's job board lives (one entry of search.yaml's `companies`)."""

    name: str
    ats: str = Field(description="A registered board: greenhouse, lever, ashby, workday")
    board: str | None = Field(default=None, description="Board slug (greenhouse/lever/ashby)")
    url: str | None = Field(default=None, description="Careers site URL (workday)")
    search: str = Field(default="", description="Search text for workday's job list")

    @model_validator(mode="after")
    def _known_board_with_address(self) -> "Company":
        if self.ats not in BOARDS:
            raise ValueError(
                f"{self.name}: unknown ats {self.ats!r} (supported: {', '.join(sorted(BOARDS))})"
            )
        needs = BOARDS[self.ats].address
        if not getattr(self, needs):
            raise ValueError(f"{self.name}: {self.ats} companies need `{needs}`")
        return self


class JobBoard:
    """One ATS. Subclasses set `name` and `address` and implement the two fetch methods."""

    name: ClassVar[str]
    # Which Company field locates the board: a slug (`board`) or a careers-site URL (`url`).
    address: ClassVar[Literal["board", "url"]] = "board"
    # True when board slugs can be guessed from a company name (see discovery/probe.py).
    probeable: ClassVar[bool] = False
    # True when the list API returns postings without descriptions; `details` fills them in.
    partial_listing: ClassVar[bool] = False

    def matches_url(self, url: str) -> bool:
        """Whether a job URL belongs to this board (for `job add <url>`)."""
        raise NotImplementedError

    def fetch_one(self, url: str, client: httpx.Client) -> FetchedJob:
        raise NotImplementedError

    def list_jobs(self, company: Company, client: httpx.Client) -> list[FetchedJob]:
        raise NotImplementedError

    def details(self, job: FetchedJob, company: Company, client: httpx.Client) -> FetchedJob:
        """Complete a posting from `list_jobs` (only needed when `partial_listing`)."""
        return job


BOARDS: dict[str, JobBoard] = {}


def register(cls: type[JobBoard]) -> type[JobBoard]:
    if cls.name in BOARDS:
        raise ValueError(f"board {cls.name!r} is registered twice")
    BOARDS[cls.name] = cls()
    return cls


def board_for_url(url: str) -> JobBoard | None:
    return next((b for b in BOARDS.values() if b.matches_url(url)), None)


def new_client(timeout: float = 30) -> httpx.Client:
    return httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT})
