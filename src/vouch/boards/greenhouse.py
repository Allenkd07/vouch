import html
import re
from urllib.parse import urlparse

import httpx

from vouch.boards.base import Company, FetchedJob, FetchError, JobBoard, register
from vouch.boards.html import html_to_text


@register
class Greenhouse(JobBoard):
    name = "greenhouse"
    probeable = True

    def matches_url(self, url: str) -> bool:
        return (urlparse(url).hostname or "").endswith("greenhouse.io")

    def fetch_one(self, url: str, client: httpx.Client) -> FetchedJob:
        m = re.search(r"greenhouse\.io/(?:embed/job_app\?for=)?([^/?#]+)/jobs/(\d+)", url)
        if not m:
            raise FetchError(f"not a Greenhouse job URL: {url}")
        board, job_id = m.groups()
        resp = client.get(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}")
        resp.raise_for_status()
        return parse_greenhouse(resp.json(), board)

    def list_jobs(self, company: Company, client: httpx.Client) -> list[FetchedJob]:
        resp = client.get(
            f"https://boards-api.greenhouse.io/v1/boards/{company.board}/jobs",
            params={"content": "true"},
        )
        resp.raise_for_status()
        return [parse_greenhouse(j, company.board) for j in resp.json().get("jobs", [])]


def parse_greenhouse(data: dict, board: str) -> FetchedJob:
    return FetchedJob(
        source="greenhouse",
        url=data.get("absolute_url") or "",
        company=data.get("company_name") or board,
        title=data.get("title"),
        location=(data.get("location") or {}).get("name"),
        # Greenhouse returns HTML with its tags entity-escaped.
        description=html_to_text(html.unescape(data.get("content", ""))),
        external_id=str(data.get("id")),
        posted_at=data.get("first_published") or data.get("updated_at"),
    )
