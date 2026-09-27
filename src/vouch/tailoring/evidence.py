"""Map each job requirement to the profile bullets that evidence it.

The whole profile fits in one prompt, so no retrieval is needed."""

from typing import Literal

from pydantic import BaseModel, Field

from vouch.jobs.requirements import JobAnalysis
from vouch.llm import LLM
from vouch.profile.schema import Profile


class BulletMatch(BaseModel):
    bullet_id: str
    strength: Literal["strong", "partial"] = Field(
        description="strong = directly demonstrates it; partial = related or adjacent"
    )


class RequirementEvidence(BaseModel):
    requirement: str = Field(description="Requirement name, exactly as given")
    bullets: list[BulletMatch]
    in_skills: bool = Field(description="The skills list names a matching skill")
    in_education: bool = Field(description="The education section satisfies it")
    note: str = Field(description="One short sentence on how well it is covered")


class EvidenceMap(BaseModel):
    items: list[RequirementEvidence]

    def for_requirement(self, name: str) -> RequirementEvidence | None:
        return next((i for i in self.items if i.requirement == name), None)


SYSTEM = """You match job requirements to evidence in a candidate's profile.
Be strict: a bullet is evidence only if it shows the candidate actually did or used the thing.
- strong: the bullet directly demonstrates the requirement (e.g. names the technology or activity).
- partial: clearly related work that a recruiter would accept as adjacent evidence.
- A requirement with options is met by any one option.
- Compound requirements (e.g. "CI/CD", "design and build") are strong only if every part is
  shown; if only one part is (e.g. CI but no automated deployment), mark it partial and say which
  part is missing in the note.
- Education requirements (degrees, fields of study) are met by the education section.
- Use only bullet ids that appear in the profile. List at most 5 bullets per requirement,
  strongest first. Return one item per requirement, in the given order."""


def map_evidence(profile: Profile, analysis: JobAnalysis, llm: LLM) -> EvidenceMap:
    reqs = "\n".join(
        f"- {r.name} ({'must-have' if r.must_have else 'nice-to-have'}, weight {r.weight})"
        + (f" options: {', '.join(r.options)}" if r.options else "")
        + f' | posting says: "{r.quote}"'
        for r in analysis.requirements
    )
    bullets = "\n".join(
        f"[{b.id}] ({section}) {b.text}  tools: {', '.join(b.facts.tools) or '-'}"
        for section, b in profile.iter_bullets()
    )
    education = "; ".join(
        ", ".join(x for x in (e.degree, e.field, e.institution, e.end) if x)
        for e in profile.education
    )
    prompt = (
        f"Job: {analysis.title}\n\nRequirements:\n{reqs}\n\n"
        f"Profile bullets:\n{bullets}\n\nSkills: {', '.join(profile.skills.all())}\n\n"
        f"Education: {education or '-'}"
    )
    result = llm.generate_json(prompt, EvidenceMap, system=SYSTEM)
    return _clean(result, profile, analysis)


def _clean(result: EvidenceMap, profile: Profile, analysis: JobAnalysis) -> EvidenceMap:
    """Drop invented bullet ids and requirements; add empty entries for any that were skipped."""
    valid_ids = {b.id for _, b in profile.iter_bullets()}
    names = [r.name for r in analysis.requirements]
    by_name = {i.requirement: i for i in result.items if i.requirement in names}
    items = []
    for name in names:
        item = by_name.get(name) or RequirementEvidence(
            requirement=name, bullets=[], in_skills=False, in_education=False, note="not assessed"
        )
        seen: set[str] = set()
        item.bullets = [
            m
            for m in item.bullets
            if m.bullet_id in valid_ids and not (m.bullet_id in seen or seen.add(m.bullet_id))
        ]
        items.append(item)
    return EvidenceMap(items=items)
