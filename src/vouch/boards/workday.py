import re
from urllib.parse import urlparse

import httpx

from vouch.boards.base import Company, FetchedJob, FetchError, JobBoard, register
from vouch.boards.html import html_to_text

WORKDAY_PAGE = 20  # Workday's maximum page size
WORKDAY_MAX = 200  # enough for a narrowed search; keeps runs polite
_LOCALE = re.compile(r"^[a-z]{2}-[A-Z]{2}$")


@register
class Workday(JobBoard):
    name = "workday"
    address = "url"
    partial_listing = True  # the list API returns titles only

    def matches_url(self, url: str) -> bool:
        return (urlparse(url).hostname or "").endswith("myworkdayjobs.com")

    def fetch_one(self, url: str, client: httpx.Client) -> FetchedJob:
        resp = client.get(workday_api_url(url), headers={"Accept": "application/json"})
        resp.raise_for_status()
        return parse_workday(resp.json(), url)

    def list_jobs(self, company: Company, client: httpx.Client) -> list[FetchedJob]:
        api, public = _workday_base(company)
        jobs: list[FetchedJob] = []
        offset, total = 0, None
        while offset < WORKDAY_MAX:
            resp = client.post(
                f"{api}/jobs",
                json={
                    "appliedFacets": {},
                    "limit": WORKDAY_PAGE,
                    "offset": offset,
                    "searchText": company.search,
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            postings = data.get("jobPostings", [])
            for p in postings:
                jobs.append(
                    FetchedJob(
                        source="workday",
                        url=f"{public}{p['externalPath']}",
                        company=company.name,
                        title=p.get("title"),
                        location=p.get("locationsText"),
                        description="",
                        external_id=p.get("externalPath"),
                    )
                )
            # Workday only reports the total on the first page (later pages say 0).
            total = data.get("total") if total is None else total
            offset += WORKDAY_PAGE
            if not postings or offset >= (total or 0):
                break
        return jobs

    def details(self, job: FetchedJob, company: Company, client: httpx.Client) -> FetchedJob:
        """Fill in the description (and canonical URL) for a posting from the list API."""
        api, _ = _workday_base(company)
        resp = client.get(f"{api}{job.external_id}", headers={"Accept": "application/json"})
        resp.raise_for_status()
        detailed = parse_workday(resp.json(), job.url)
        detailed.company = company.name
        return detailed


def _workday_base(company: Company) -> tuple[str, str]:
    """(api base, public base) for https://{tenant}.wdN.myworkdayjobs.com/[locale/]{site}."""
    parsed = urlparse(company.url)
    tenant = parsed.hostname.split(".")[0]
    site = [s for s in parsed.path.split("/") if s][-1]
    return (
        f"https://{parsed.hostname}/wday/cxs/{tenant}/{site}",
        f"https://{parsed.hostname}/{site}",
    )


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
