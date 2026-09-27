"""Choose which bullets go on the resume: cover must-haves first, then fill by relevance."""

from dataclasses import dataclass, field

from vouch.jobs.requirements import JobAnalysis, quote_in_text
from vouch.profile.schema import Bullet, Profile
from vouch.tailoring.evidence import EvidenceMap

STRENGTH = {"strong": 1.0, "partial": 0.5}


@dataclass
class Budget:
    """Caps per section; the renderer trims further until the page limit holds."""

    first_role: int = 9
    other_roles: int = 7
    per_project: int = 2
    min_per_role: int = 2
    total: int = 20


@dataclass
class Selection:
    # section key ("experience:<id>" / "project:<id>") -> bullet ids, best first
    sections: dict[str, list[str]]
    scores: dict[str, float]
    # requirement name -> bullet ids that support it (only bullets that were selected)
    supports: dict[str, list[str]] = field(default_factory=dict)

    def bullet_ids(self) -> list[str]:
        return [i for ids in self.sections.values() for i in ids]


def score_bullets(
    profile: Profile, analysis: JobAnalysis, evidence: EvidenceMap
) -> dict[str, float]:
    reqs = {r.name: r for r in analysis.requirements}
    scores = {b.id: 0.0 for _, b in profile.iter_bullets()}
    for item in evidence.items:
        req = reqs[item.requirement]
        importance = req.weight * (1.5 if req.must_have else 1.0)
        for m in item.bullets:
            scores[m.bullet_id] += importance * STRENGTH[m.strength]
    for _, b in profile.iter_bullets():
        haystack = f"{b.text} {' '.join(b.facts.tools)}"
        scores[b.id] += 0.3 * sum(quote_in_text(kw, haystack) for kw in analysis.keywords)
    return scores


def select_bullets(
    profile: Profile, analysis: JobAnalysis, evidence: EvidenceMap, budget: Budget | None = None
) -> Selection:
    budget = budget or Budget()
    scores = score_bullets(profile, analysis, evidence)
    section_of: dict[str, str] = {}
    bullets: dict[str, Bullet] = {}
    for section, b in profile.iter_bullets():
        section_of[b.id], bullets[b.id] = section, b

    caps = {}
    for i, e in enumerate(profile.experience):
        caps[f"experience:{e.id}"] = budget.first_role if i == 0 else budget.other_roles
    for p in profile.projects:
        caps[f"project:{p.id}"] = budget.per_project
    chosen: dict[str, list[str]] = {key: [] for key in caps}

    def take(bullet_id: str) -> bool:
        key = section_of[bullet_id]
        if key not in chosen or bullet_id in chosen[key] or len(chosen[key]) >= caps[key]:
            return False
        if sum(len(v) for v in chosen.values()) >= budget.total:
            return False
        chosen[key].append(bullet_id)
        return True

    # 1. Every role gets its minimum, best bullets first (a role with no bullets looks odd).
    for e in profile.experience:
        for b in sorted(e.bullets, key=lambda b: -scores[b.id])[: budget.min_per_role]:
            take(b.id)
    # 2. Cover each requirement with its best evidence, most important requirements first.
    reqs = sorted(analysis.requirements, key=lambda r: (not r.must_have, -r.weight))
    for req in reqs:
        item = evidence.for_requirement(req.name)
        matches = sorted(item.bullets, key=lambda m: -STRENGTH[m.strength]) if item else []
        if matches and not any(m.bullet_id in sum(chosen.values(), []) for m in matches):
            for m in matches:
                if take(m.bullet_id):
                    break
    # 3. Fill remaining space by score (zero-score bullets never make the cut).
    for bullet_id in sorted(scores, key=lambda i: -scores[i]):
        if scores[bullet_id] > 0:
            take(bullet_id)

    sections = {key: sorted(ids, key=lambda i: -scores[i]) for key, ids in chosen.items() if ids}
    selected = {i for ids in sections.values() for i in ids}
    supports = {
        item.requirement: [m.bullet_id for m in item.bullets if m.bullet_id in selected]
        for item in evidence.items
    }
    return Selection(sections=sections, scores=scores, supports=supports)


def drop_weakest(selection: Selection, profile: Profile, analysis: JobAnalysis) -> str | None:
    """Remove one bullet to save space: the lowest-scoring one whose removal keeps every
    requirement's coverage and every role's minimum; if none, the lowest-scoring overall."""
    must = {r.name for r in analysis.requirements if r.must_have}

    def removable(bullet_id: str, keep_coverage: bool) -> bool:
        key = next(k for k, ids in selection.sections.items() if bullet_id in ids)
        if key.startswith("experience:") and len(selection.sections[key]) <= 1:
            return False
        if keep_coverage:
            for req, ids in selection.supports.items():
                if req in must and ids == [bullet_id]:
                    return False
        return True

    ranked = sorted(selection.bullet_ids(), key=lambda i: selection.scores[i])
    for keep_coverage in (True, False):
        for bullet_id in ranked:
            if removable(bullet_id, keep_coverage):
                key = next(k for k, ids in selection.sections.items() if bullet_id in ids)
                selection.sections[key].remove(bullet_id)
                if not selection.sections[key]:
                    del selection.sections[key]
                for ids in selection.supports.values():
                    if bullet_id in ids:
                        ids.remove(bullet_id)
                return bullet_id
    return None
