"""Job boards: fetch one posting from its URL, or list every posting on a company's board.

Importing this package registers every board in `BOARDS`."""

import httpx

from vouch.boards import ashby, greenhouse, lever, workday  # noqa: F401  (registers boards)
from vouch.boards.base import (
    BOARDS,
    USER_AGENT,
    Company,
    FetchedJob,
    FetchError,
    JobBoard,
    board_for_url,
    new_client,
)
from vouch.boards.html import html_to_text
from vouch.boards.web import fetch_page

__all__ = [
    "BOARDS",
    "USER_AGENT",
    "Company",
    "FetchError",
    "FetchedJob",
    "JobBoard",
    "board_for_url",
    "fetch_job",
    "list_jobs",
    "manual_job",
    "new_client",
]


def fetch_job(url: str, client: httpx.Client | None = None) -> FetchedJob:
    """One posting from its URL: the board's JSON API if we know the board, else the page."""
    client = client or new_client()
    board = board_for_url(url)
    try:
        return board.fetch_one(url, client) if board else fetch_page(url, client)
    except httpx.HTTPError as e:
        raise FetchError(f"could not fetch {url}: {e}") from e


def list_jobs(company: Company, client: httpx.Client) -> list[FetchedJob]:
    """All postings on the company's board. For boards with `partial_listing` (Workday) the
    postings have no description yet; see `JobBoard.details`."""
    jobs = BOARDS[company.ats].list_jobs(company, client)
    for job in jobs:
        job.company = company.name
    return jobs


def manual_job(text: str, url: str | None, company: str | None, title: str | None) -> FetchedJob:
    return FetchedJob(
        source="manual",
        url=url or "",
        company=company,
        title=title,
        location=None,
        description=html_to_text(text) if "<" in text and ">" in text else text.strip(),
    )
