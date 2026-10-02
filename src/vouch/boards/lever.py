import re
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx

from vouch.boards.base import Company, FetchedJob, FetchError, JobBoard, register
from vouch.boards.html import html_to_text


@register
class Lever(JobBoard):
    name = "lever"
    probeable = True

    def matches_url(self, url: str) -> bool:
        return urlparse(url).hostname == "jobs.lever.co"

    def fetch_one(self, url: str, client: httpx.Client) -> FetchedJob:
        m = re.search(r"jobs\.lever\.co/([^/?#]+)/([0-9a-f-]{36})", url)
        if not m:
            raise FetchError(f"not a Lever job URL: {url}")
        company, posting_id = m.groups()
        resp = client.get(f"https://api.lever.co/v0/postings/{company}/{posting_id}")
        resp.raise_for_status()
        return parse_lever(resp.json(), company)

    def list_jobs(self, company: Company, client: httpx.Client) -> list[FetchedJob]:
        resp = client.get(
            f"https://api.lever.co/v0/postings/{company.board}", params={"mode": "json"}
        )
        resp.raise_for_status()
        return [parse_lever(j, company.board) for j in resp.json()]


def parse_lever(data: dict, company: str) -> FetchedJob:
    parts = [data.get("descriptionPlain", "")]
    for section in data.get("lists", []):
        parts.append(f"{section.get('text', '')}\n{html_to_text(section.get('content', ''))}")
    parts.append(data.get("additionalPlain", ""))
    created = data.get("createdAt")
    return FetchedJob(
        source="lever",
        url=data.get("hostedUrl") or "",
        company=company,
        title=data.get("text"),
        location=_lever_location(data),
        description="\n\n".join(p.strip() for p in parts if p and p.strip()),
        external_id=data.get("id"),
        posted_at=datetime.fromtimestamp(created / 1000, UTC).date().isoformat()
        if created
        else None,
    )


def _lever_location(data: dict) -> str | None:
    cats = data.get("categories") or {}
    locations = cats.get("allLocations") or [cats.get("location")]
    if data.get("workplaceType") == "remote":
        locations = [*locations, "Remote"]
    return ", ".join(loc for loc in locations if loc) or None
