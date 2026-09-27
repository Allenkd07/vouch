"""List every open job on a company's public job board."""

from urllib.parse import urlparse

import httpx

from vouch.discovery.config import Company
from vouch.jobs.sources import (
    FetchedJob,
    parse_ashby,
    parse_greenhouse,
    parse_lever,
    parse_workday,
)

WORKDAY_PAGE = 20  # Workday's maximum page size
WORKDAY_MAX = 200  # enough for a narrowed search; keeps runs polite


def list_jobs(company: Company, client: httpx.Client) -> list[FetchedJob]:
    """All postings on the board. Workday postings have no description yet (see
    `workday_details`), because its list API returns titles only."""
    jobs = {
        "greenhouse": _greenhouse,
        "lever": _lever,
        "ashby": _ashby,
        "workday": _workday,
    }[company.ats](company, client)
    for job in jobs:
        job.company = company.name
    return jobs


def _greenhouse(company: Company, client: httpx.Client) -> list[FetchedJob]:
    resp = client.get(
        f"https://boards-api.greenhouse.io/v1/boards/{company.board}/jobs",
        params={"content": "true"},
    )
    resp.raise_for_status()
    return [parse_greenhouse(j, company.board) for j in resp.json().get("jobs", [])]


def _lever(company: Company, client: httpx.Client) -> list[FetchedJob]:
    resp = client.get(f"https://api.lever.co/v0/postings/{company.board}", params={"mode": "json"})
    resp.raise_for_status()
    return [parse_lever(j, company.board) for j in resp.json()]


def _ashby(company: Company, client: httpx.Client) -> list[FetchedJob]:
    resp = client.get(f"https://api.ashbyhq.com/posting-api/job-board/{company.board}")
    resp.raise_for_status()
    return [
        parse_ashby(j, company.board)
        for j in resp.json().get("jobs", [])
        if j.get("isListed", True)
    ]


def _workday_base(company: Company) -> tuple[str, str]:
    """(api base, public base) for https://{tenant}.wdN.myworkdayjobs.com/[locale/]{site}."""
    parsed = urlparse(company.url)
    tenant = parsed.hostname.split(".")[0]
    site = [s for s in parsed.path.split("/") if s][-1]
    return (
        f"https://{parsed.hostname}/wday/cxs/{tenant}/{site}",
        f"https://{parsed.hostname}/{site}",
    )


def _workday(company: Company, client: httpx.Client) -> list[FetchedJob]:
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


def workday_details(job: FetchedJob, company: Company, client: httpx.Client) -> FetchedJob:
    """Fill in the description (and canonical URL) for a Workday posting from the list API."""
    api, _ = _workday_base(company)
    resp = client.get(f"{api}{job.external_id}", headers={"Accept": "application/json"})
    resp.raise_for_status()
    detailed = parse_workday(resp.json(), job.url)
    detailed.company = company.name
    return detailed
