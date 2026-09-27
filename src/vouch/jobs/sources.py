"""Fetch one job posting from its URL. ATS JSON APIs first; any other page via JSON-LD or text."""

import html
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx

from vouch.jobs.html import html_to_text

USER_AGENT = "Mozilla/5.0 (vouch; personal job search tool)"


class FetchError(RuntimeError):
    pass


@dataclass
class FetchedJob:
    source: str  # workday | greenhouse | lever | ashby | web | manual
    url: str
    company: str | None
    title: str | None
    location: str | None
    description: str  # plain text
    external_id: str | None = None
    posted_at: str | None = None


def fetch_job(url: str, client: httpx.Client | None = None) -> FetchedJob:
    client = client or httpx.Client(
        timeout=30, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    )
    host = urlparse(url).hostname or ""
    try:
        if host.endswith("myworkdayjobs.com"):
            return _workday(url, client)
        if host.endswith("greenhouse.io"):
            return _greenhouse(url, client)
        if host == "jobs.lever.co":
            return _lever(url, client)
        if host == "jobs.ashbyhq.com":
            return _ashby(url, client)
        return _web(url, client)
    except httpx.HTTPError as e:
        raise FetchError(f"could not fetch {url}: {e}") from e


# --- Workday ---------------------------------------------------------------------------------

_LOCALE = re.compile(r"^[a-z]{2}-[A-Z]{2}$")


def workday_api_url(url: str) -> str:
    """https://{tenant}.wdN.myworkdayjobs.com/[locale/]{site}/job/{path}
    -> https://{host}/wday/cxs/{tenant}/{site}/job/{path}"""
    parsed = urlparse(url)
    tenant = parsed.hostname.split(".")[0]
    segments = [s for s in parsed.path.split("/") if s]
    if segments and _LOCALE.match(segments[0]):
        segments = segments[1:]
    if len(segments) < 3 or segments[1] != "job":
        raise FetchError(f"not a Workday job URL: {url}")
    site, rest = segments[0], "/".join(segments[2:])
    return f"https://{parsed.hostname}/wday/cxs/{tenant}/{site}/job/{rest}"


def parse_workday(data: dict, url: str) -> FetchedJob:
    info = data["jobPostingInfo"]
    org = (data.get("hiringOrganization") or {}).get("name") or ""
    company = re.sub(r"^\d+\s+", "", org).strip() or urlparse(url).hostname.split(".")[0]
    return FetchedJob(
        source="workday",
        url=info.get("externalUrl") or url,
        company=company,
        title=info.get("title"),
        location=info.get("location"),
        description=html_to_text(info.get("jobDescription", "")),
        external_id=info.get("jobReqId"),
        posted_at=info.get("startDate"),
    )


def _workday(url: str, client: httpx.Client) -> FetchedJob:
    resp = client.get(workday_api_url(url), headers={"Accept": "application/json"})
    resp.raise_for_status()
    return parse_workday(resp.json(), url)


# --- Greenhouse ------------------------------------------------------------------------------


def _greenhouse(url: str, client: httpx.Client) -> FetchedJob:
    m = re.search(r"greenhouse\.io/(?:embed/job_app\?for=)?([^/?#]+)/jobs/(\d+)", url)
    if not m:
        raise FetchError(f"not a Greenhouse job URL: {url}")
    board, job_id = m.groups()
    resp = client.get(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}")
    resp.raise_for_status()
    return parse_greenhouse(resp.json(), board)


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


# --- Lever -----------------------------------------------------------------------------------


def _lever(url: str, client: httpx.Client) -> FetchedJob:
    m = re.search(r"jobs\.lever\.co/([^/?#]+)/([0-9a-f-]{36})", url)
    if not m:
        raise FetchError(f"not a Lever job URL: {url}")
    company, posting_id = m.groups()
    resp = client.get(f"https://api.lever.co/v0/postings/{company}/{posting_id}")
    resp.raise_for_status()
    return parse_lever(resp.json(), company)


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


# --- Ashby -----------------------------------------------------------------------------------


def _ashby(url: str, client: httpx.Client) -> FetchedJob:
    m = re.search(r"jobs\.ashbyhq\.com/([^/?#]+)/([0-9a-f-]{36})", url)
    if not m:
        raise FetchError(f"not an Ashby job URL: {url}")
    org, job_id = m.groups()
    resp = client.get(f"https://api.ashbyhq.com/posting-api/job-board/{org}")
    resp.raise_for_status()
    for job in resp.json().get("jobs", []):
        if job.get("id") == job_id:
            return parse_ashby(job, org)
    raise FetchError(f"job {job_id} not found on Ashby board {org!r} (closed?)")


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


# --- Any other page --------------------------------------------------------------------------


def _web(url: str, client: httpx.Client) -> FetchedJob:
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


def manual_job(text: str, url: str | None, company: str | None, title: str | None) -> FetchedJob:
    return FetchedJob(
        source="manual",
        url=url or "",
        company=company,
        title=title,
        location=None,
        description=html_to_text(text) if "<" in text and ">" in text else text.strip(),
    )
