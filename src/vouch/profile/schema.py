"""Master profile: the single source of truth for everything the resume may claim.

Every bullet carries the facts it is based on. Tailoring may rephrase a bullet, but any tool,
number or scope in the output must come from these facts.
"""

import re
from datetime import date
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, BeforeValidator, Field, model_validator

ID_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"

# YAML reads `2024` as an int and `2024-01-15` as a date; keep dates as the text written.
DateStr = Annotated[str, BeforeValidator(lambda v: str(v) if isinstance(v, int | date) else v)]


class Facts(BaseModel):
    tools: list[str] = Field(default_factory=list, description="Technologies used, as named")
    metrics: list[str] = Field(
        default_factory=list, description="Numbers exactly as stated, e.g. '2M records/day'"
    )
    scope: list[str] = Field(
        default_factory=list, description="Team size, users, ownership, e.g. 'led team of 3'"
    )

    def is_empty(self) -> bool:
        return not (self.tools or self.metrics or self.scope)


class Bullet(BaseModel):
    id: str = Field(pattern=ID_PATTERN)
    text: str = Field(min_length=1)
    facts: Facts = Field(default_factory=Facts)
    tags: list[str] = Field(default_factory=list, description="e.g. backend, ml, leadership")
    source: str | None = Field(
        default=None,
        description="Where this is evidenced: 'resume', 'repo:<name>', or 'self-reported'",
    )


class Experience(BaseModel):
    id: str = Field(pattern=ID_PATTERN)
    company: str
    title: str
    location: str | None = None
    description: str | None = Field(
        default=None, description="One line on the product or team, as written in the resume"
    )
    start: DateStr = Field(description="YYYY-MM")
    end: DateStr | None = Field(default=None, description="YYYY-MM, or null if current")
    bullets: list[Bullet] = Field(default_factory=list)


class Project(BaseModel):
    id: str = Field(pattern=ID_PATTERN)
    name: str
    url: str | None = None
    description: str | None = None
    dates: DateStr | None = None
    bullets: list[Bullet] = Field(default_factory=list)


class Education(BaseModel):
    id: str = Field(pattern=ID_PATTERN)
    institution: str
    degree: str
    field: str | None = None
    start: DateStr | None = None
    end: DateStr | None = None
    details: list[str] = Field(default_factory=list, description="GPA, coursework, honours")


class Skills(BaseModel):
    languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)
    cloud: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    other: list[str] = Field(default_factory=list)

    def all(self) -> list[str]:
        return [
            *self.languages,
            *self.frameworks,
            *self.databases,
            *self.cloud,
            *self.tools,
            *self.other,
        ]


class Link(BaseModel):
    label: str
    url: str


class Contact(BaseModel):
    name: str
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    links: list[Link] = Field(default_factory=list)


class Profile(BaseModel):
    contact: Contact
    summary: str | None = None
    experience: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: Skills = Field(default_factory=Skills)
    certifications: list[str] = Field(default_factory=list)
    achievements: list[Bullet] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "Profile":
        seen: set[str] = set()
        ids = [e.id for e in self.experience] + [p.id for p in self.projects]
        ids += [e.id for e in self.education] + [b.id for _, b in self.iter_bullets()]
        for i in ids:
            if i in seen:
                raise ValueError(f"duplicate id: {i!r} (ids must be unique across the profile)")
            seen.add(i)
        return self

    def iter_bullets(self):
        """Yield (section, bullet) for every bullet, e.g. ('experience:acme', Bullet)."""
        for e in self.experience:
            for b in e.bullets:
                yield f"experience:{e.id}", b
        for p in self.projects:
            for b in p.bullets:
                yield f"project:{p.id}", b
        for b in self.achievements:
            yield "achievement", b


def lint(profile: Profile) -> list[str]:
    """Non-fatal problems that make tailoring weaker or riskier."""
    warnings = []
    known = [f" {normalize_name(s)} " for s in profile.skills.all()]
    for section, b in profile.iter_bullets():
        where = f"{section} / {b.id}"
        if b.facts.is_empty():
            warnings.append(f"{where}: no facts recorded; tailoring can only reuse the text as-is")
        for tool in b.facts.tools:
            if not any(f" {normalize_name(tool)} " in k for k in known):
                warnings.append(f"{where}: tool {tool!r} is not listed in skills")
        for metric in b.facts.metrics:
            if not _metric_in_text(metric, b.text):
                warnings.append(f"{where}: metric {metric!r} does not appear in the bullet text")
    for e in profile.experience:
        if not e.bullets:
            warnings.append(f"experience:{e.id}: no bullets")
    return warnings


def normalize_name(name: str) -> str:
    """'React.js' -> 'react', 'CI / Git' -> 'ci git', so near-identical names match."""
    name = re.sub(r"\.js\b", "", name.lower())
    return " ".join(re.findall(r"[a-z0-9+#]+", name))


def _metric_in_text(metric: str, text: str) -> bool:
    numbers = re.findall(r"\d+(?:[.,]\d+)?", metric)
    return all(n in text for n in numbers) if numbers else metric.lower() in text.lower()


def load_profile(path: Path) -> Profile:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Profile.model_validate(data)


def dump_profile(profile: Profile) -> str:
    data = profile.model_dump(mode="json", exclude_none=True)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
