"""Any other job page: schema.org JobPosting JSON-LD if present, else the page's text.

Not a registered board (there's nothing to list); `fetch_job` falls back to it."""

import html
import json
import re

import httpx

from vouch.boards.base import FetchedJob, FetchError
from vouch.boards.html import html_to_text


def fetch_page(url: str, client: httpx.Client) -> FetchedJob:
    resp = client.get(url)
    resp.raise_for_status()
    job = parse_web(resp.text, str(resp.url))
    if len(job.description) < 200:
        raise FetchError(
            f"{url} has too little text (probably rendered by JavaScript). "
            "Copy the job description into a file and use --text-file."
        )
    return job


def parse_web(page: str, url: str) -> FetchedJob:
    """Prefer schema.org JobPosting JSON-LD (most career sites include it for Google Jobs)."""
    for block in re.findall(
        r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>', page, re.S | re.I
    ):
        try:
            data = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        for item in data if isinstance(data, list) else data.get("@graph", [data]):
            if isinstance(item, dict) and item.get("@type") == "JobPosting":
                return _from_json_ld(item, url)
    title = re.search(r"<title[^>]*>(.*?)</title>", page, re.S | re.I)
    return FetchedJob(
        source="web",
        url=url,
        company=None,
        title=html.unescape(title.group(1)).strip() if title else None,
        location=None,
        description=html_to_text(page),
    )


def _from_json_ld(item: dict, url: str) -> FetchedJob:
    org = item.get("hiringOrganization")
    location = item.get("jobLocation")
    if isinstance(location, list):
        location = location[0] if location else None
    address = (location or {}).get("address") if isinstance(location, dict) else None
    if isinstance(address, dict):
        address = ", ".join(
            v
            for k in ("addressLocality", "addressRegion", "addressCountry")
            if isinstance(v := address.get(k), str)
        )
    identifier = item.get("identifier")
    return FetchedJob(
        source="web",
        url=url,
        company=org.get("name") if isinstance(org, dict) else org,
        title=item.get("title"),
        location=address if isinstance(address, str) else None,
        description=html_to_text(html.unescape(item.get("description", ""))),
        external_id=str(identifier.get("value")) if isinstance(identifier, dict) else None,
        posted_at=item.get("datePosted"),
    )
