"""profile/search.yaml: which companies to watch and which jobs to keep."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from vouch.boards import Company

DEFAULT_SEARCH = Path("profile/search.yaml")


class Filters(BaseModel):
    title_include: list[str] = []
    title_exclude: list[str] = []
    locations: list[str] = []
    max_age_days: int | None = 45


class Ranking(BaseModel):
    analyze_per_run: int = Field(default=30, ge=0)
    batch_size: int = Field(default=6, ge=1, le=10)  # postings per extraction request
    accept_years_up_to: float = Field(default=5, ge=0)


class SearchConfig(BaseModel):
    companies: list[Company]
    filters: Filters = Filters()
    ranking: Ranking = Ranking()


def load_search(path: Path = DEFAULT_SEARCH) -> SearchConfig:
    return SearchConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
