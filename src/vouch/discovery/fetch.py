"""Read every target company's job board and keep the postings that pass the filters (free)."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx

from vouch.boards import BOARDS, FetchedJob, list_jobs, new_client
from vouch.discovery.config import Company, SearchConfig
from vouch.discovery.filters import rejection_reason


@dataclass
class CompanyResult:
    name: str
    listed: int = 0
    kept: list[FetchedJob] = field(default_factory=list)
    error: str | None = None


def fetch_company(company: Company, config: SearchConfig, client: httpx.Client) -> CompanyResult:
    result = CompanyResult(company.name)
    try:
        jobs = list_jobs(company, client)
        result.listed = len(jobs)
        kept = [j for j in jobs if rejection_reason(j, config.filters) is None]
        board = BOARDS[company.ats]
        if board.partial_listing:  # descriptions come from a second request per posting
            kept = [board.details(j, company, client) for j in kept]
            kept = [j for j in kept if rejection_reason(j, config.filters) is None]
        result.kept = kept
    except (httpx.HTTPError, KeyError, ValueError) as e:
        result.error = f"{type(e).__name__}: {e}"
    return result


def fetch_all(config: SearchConfig, client: httpx.Client | None = None) -> list[CompanyResult]:
    client = client or new_client()
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda c: fetch_company(c, config, client), config.companies))
