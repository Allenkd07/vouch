"""Find which companies have a public job board we can read, and whether they hire in India.

Board names ("slugs") aren't published anywhere, so for each company name we try the usual
spellings on each supported board and keep the ones that answer with jobs."""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx

from vouch.boards import BOARDS, Company, list_jobs, new_client
from vouch.discovery.filters import has_phrase

INDIA = [
    "india",
    "bengaluru",
    "bangalore",
    "pune",
    "mumbai",
    "hyderabad",
    "chennai",
    "gurgaon",
    "gurugram",
    "noida",
    "delhi",
    "kochi",
    "ahmedabad",
    "kolkata",
    "jaipur",
]


@dataclass
class Found:
    name: str  # company name as given
    ats: str
    board: str
    total: int
    india: int
    sample: list[str] = field(default_factory=list)  # a few India job titles, to eyeball


def slug_variants(name: str) -> list[str]:
    """'Pine Labs' -> ['pinelabs', 'pine-labs', 'pinelabsindia', 'pinelabshq']."""
    words = re.findall(r"[a-z0-9]+", name.lower())
    if not words:
        return []
    joined, hyphened = "".join(words), "-".join(words)
    variants = [joined, hyphened, f"{joined}india", f"{joined}hq"]
    return list(dict.fromkeys(v for v in variants if v))


def read_names(text: str) -> list[str]:
    names = [line.strip() for line in text.splitlines()]
    return list(dict.fromkeys(n for n in names if n and not n.startswith("#")))


def _try(name: str, ats: str, slug: str, client: httpx.Client) -> Found | None:
    try:
        jobs = list_jobs(Company(name=name, ats=ats, board=slug), client)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None  # no such board (404) or an unexpected response
    if not jobs:
        return None
    india = [j for j in jobs if has_phrase(j.location or "", INDIA)]
    return Found(name, ats, slug, len(jobs), len(india), [j.title for j in india[:3]])


def probe(names: list[str], client: httpx.Client | None = None, workers: int = 16) -> list[Found]:
    """Every board found for each name (best first: most India openings)."""
    client = client or new_client(timeout=20)
    boards = [name for name, board in BOARDS.items() if board.probeable]
    attempts = [(n, ats, slug) for n in names for slug in slug_variants(n) for ats in boards]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda a: _try(*a, client), attempts)
    found = [r for r in results if r is not None]
    # The same board can answer to two spellings (e.g. redirects); keep one.
    unique = {(f.ats, f.board): f for f in found}
    return sorted(unique.values(), key=lambda f: (-f.india, -f.total, f.name.lower()))
