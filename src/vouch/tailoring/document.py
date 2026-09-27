"""The tailored resume as data: what the renderers draw and what gets stored per version."""

import re
from typing import Literal

from pydantic import BaseModel

from vouch.jobs.requirements import JobAnalysis
from vouch.profile.schema import Contact, Profile, normalize_name

_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


class DocBullet(BaseModel):
    id: str
    text: str  # what goes on the page
    original: str  # the profile's wording
    proposed: str | None = None  # the verified tailored wording (== original if rejected)
    flags: list[str] = []  # why a rewrite was rejected
    choice: Literal["tailored", "original", "custom"] = "tailored"
    included: bool = True

    @property
    def changed(self) -> bool:
        return self.text != self.original


class DocEntry(BaseModel):
    heading: str  # company / project / institution
    subheading: str | None = None  # title / degree
    location: str | None = None
    dates: str | None = None
    description: str | None = None
    bullets: list[DocBullet] = []


class SkillLine(BaseModel):
    label: str
    items: list[str]


class ResumeDoc(BaseModel):
    contact: Contact
    summary: str | None
    summary_original: str | None = None
    summary_proposed: str | None = None
    skills: list[SkillLine]
    experience: list[DocEntry]
    projects: list[DocEntry]
    education: list[DocEntry]

    def for_render(self) -> "ResumeDoc":
        """Copy without the bullets the user excluded."""
        doc = self.model_copy(deep=True)
        for e in [*doc.experience, *doc.projects]:
            e.bullets = [b for b in e.bullets if b.included]
        doc.projects = [p for p in doc.projects if p.bullets]
        return doc

    def all_text(self) -> str:
        parts = [self.summary or ""]
        parts += [f"{s.label}: {', '.join(s.items)}" for s in self.skills]
        for e in [*self.experience, *self.projects, *self.education]:
            parts += [e.heading, e.subheading or "", e.description or ""]
            parts += [b.text for b in e.bullets]
        return "\n".join(parts)


def format_dates(text: str | None) -> str | None:
    if not text:
        return text
    text = re.sub(r"\s*[-—–]{1,2}\s*(?=\d{4})", " – ", text)  # one dash style for ranges
    return re.sub(
        r"\b(\d{4})-(\d{2})\b", lambda m: f"{_MONTHS[int(m.group(2)) - 1]} {m.group(1)}", text
    )


def date_range(start: str | None, end: str | None, current: bool) -> str | None:
    if not start:
        return format_dates(end)
    return f"{format_dates(start)} – {format_dates(end) if end else 'Present' if current else ''}"


SKILL_LABELS = {
    "languages": ("Languages", 10),
    "frameworks": ("Frameworks & Libraries", 12),
    "databases": ("Databases", 8),
    "cloud": ("Cloud & Infrastructure", 12),
    "tools": ("Tools", 10),
    "other": ("Other", 8),
}


def tailor_skills(profile: Profile, analysis: JobAnalysis, used_tools: set[str]) -> list[SkillLine]:
    """Job-relevant skills first, then ones evidenced by the chosen bullets, capped per line.
    'Other' only keeps relevant items (it holds broad terms that otherwise add noise)."""
    wanted = {normalize_name(o) for r in analysis.requirements for o in [r.name, *r.options]}
    wanted |= {normalize_name(k) for k in analysis.keywords}
    used = {normalize_name(t) for t in used_tools}

    def rank(skill: str) -> int:
        n = normalize_name(skill)
        if n in wanted or any(f" {w} " in f" {n} " for w in wanted if w):
            return 2
        return 1 if n in used else 0

    lines = []
    for field, (label, cap) in SKILL_LABELS.items():
        items = getattr(profile.skills, field)
        ranked = sorted(items, key=lambda s: -rank(s))  # stable: keeps the profile's order
        if field == "other":
            ranked = [s for s in ranked if rank(s) == 2]
        if ranked[:cap]:
            lines.append(SkillLine(label=label, items=ranked[:cap]))
    return lines
