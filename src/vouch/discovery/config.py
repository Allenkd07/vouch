"""profile/search.yaml: which companies to watch and which jobs to keep."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_SEARCH = Path("profile/search.yaml")


class Company(BaseModel):
    name: str
    ats: Literal["greenhouse", "lever", "ashby", "workday"]
    board: str | None = Field(default=None, description="Board slug (greenhouse/lever/ashby)")
    url: str | None = Field(default=None, description="Careers site URL (workday)")
    search: str = Field(default="", description="Search text for workday's job list")

    @model_validator(mode="after")
    def _needs_location(self) -> "Company":
        if self.ats == "workday" and not self.url:
            raise ValueError(f"{self.name}: workday companies need `url`")
        if self.ats != "workday" and not self.board:
            raise ValueError(f"{self.name}: {self.ats} companies need `board`")
        return self


class Filters(BaseModel):
    title_include: list[str] = []
    title_exclude: list[str] = []
    locations: list[str] = []
    max_age_days: int | None = 45


class Ranking(BaseModel):
    analyze_per_run: int = Field(default=5, ge=0)


class SearchConfig(BaseModel):
    companies: list[Company]
    filters: Filters = Filters()
    ranking: Ranking = Ranking()


def load_search(path: Path = DEFAULT_SEARCH) -> SearchConfig:
    return SearchConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
