import re
from urllib.parse import urlparse

import httpx

from vouch.boards.base import Company, FetchedJob, FetchError, JobBoard, register
from vouch.boards.html import html_to_text


@register
class Ashby(JobBoard):
    name = "ashby"
    probeable = True

    def matches_url(self, url: str) -> bool:
        return urlparse(url).hostname == "jobs.ashbyhq.com"

    def fetch_one(self, url: str, client: httpx.Client) -> FetchedJob:
        m = re.search(r"jobs\.ashbyhq\.com/([^/?#]+)/([0-9a-f-]{36})", url)
        if not m:
            raise FetchError(f"not an Ashby job URL: {url}")
        org, job_id = m.groups()
        for job in self._board(org, client):
            if job.get("id") == job_id:
                return parse_ashby(job, org)
        raise FetchError(f"job {job_id} not found on Ashby board {org!r} (closed?)")

    def list_jobs(self, company: Company, client: httpx.Client) -> list[FetchedJob]:
        return [
            parse_ashby(j, company.board)
            for j in self._board(company.board, client)
            if j.get("isListed", True)
        ]

    def _board(self, org: str, client: httpx.Client) -> list[dict]:
        resp = client.get(f"https://api.ashbyhq.com/posting-api/job-board/{org}")
        resp.raise_for_status()
        return resp.json().get("jobs", [])


def parse_ashby(job: dict, org: str) -> FetchedJob:
    locations = [
        job.get("location"),
        *(s.get("location") for s in job.get("secondaryLocations") or []),
    ]
    if job.get("isRemote"):
        locations.append("Remote")
    return FetchedJob(
        source="ashby",
        url=job.get("jobUrl") or "",
        company=org,
        title=job.get("title"),
        location=", ".join(loc for loc in locations if loc) or None,
        description=job.get("descriptionPlain") or html_to_text(job.get("descriptionHtml", "")),
        external_id=job.get("id"),
        posted_at=job.get("publishedAt"),
    )
