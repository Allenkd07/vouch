"""Estimate how well a job fits the profile, without an LLM.

Technical requirements that name concrete technologies are matched against the profile's skills
and the tools recorded in its bullets. Everything else (soft skills, practices, or broad ones like
"database fundamentals") can't be checked this way and is left out of the score rather than
guessed; the full evidence mapping happens when tailoring."""

from datetime import UTC, datetime

from pydantic import BaseModel

from vouch.jobs.requirements import JobAnalysis, Requirement
from vouch.profile.schema import Profile, normalize_name

TECHNICAL = {"language", "framework", "database", "cloud", "tool"}
SENIOR_LEVELS = {"staff", "principal", "manager"}
# Requirement weight (1-3 each) at which a coverage figure gets its full say in the score; below
# it, coverage counts for less and similarity for more, so one or two checkable requirements
# can't decide a job on their own.
FULL_EVIDENCE = {"must": 6.0, "nice": 3.0}


class Fit(BaseModel):
    score: float  # 0-100
    must_coverage: float | None  # share of technical must-haves matched (None: none to check)
    nice_coverage: float | None
    similarity: float | None  # embedding cosine similarity, profile vs job
    seniority_penalty: float
    matched: list[str]
    missing_must: list[str]
    missing_nice: list[str]
    unchecked: list[str]  # requirements this estimate can't judge
    checked: int = 0  # requirements the score is based on
    total: int = 0
    notes: list[str]


def profile_vocabulary(profile: Profile) -> set[str]:
    terms = set(profile.skills.all())
    terms |= {t for _, b in profile.iter_bullets() for t in b.facts.tools}
    return {normalize_name(t) for t in terms if t}


def _same_kind_skills(profile: Profile, category: str) -> list[str]:
    """Profile skills of a requirement's category, for "X or equivalent" requirements."""
    skills = profile.skills
    return {
        "language": skills.languages,
        "framework": skills.frameworks,
        "database": skills.databases,
        "cloud": skills.cloud,
        "tool": skills.tools,
    }.get(category, [])


def _matches(req: Requirement, vocab: set[str], profile: Profile) -> bool:
    if req.or_equivalent and _same_kind_skills(profile, req.category):
        return True
    for option in req.options:
        n = normalize_name(option)
        if n and any(n == v or f" {n} " in f" {v} " or f" {v} " in f" {n} " for v in vocab):
            return True
    return False


def years_of_experience(profile: Profile, now: datetime | None = None) -> float:
    """From the earliest role's start to now (overlaps and gaps ignored)."""
    starts = []
    for e in profile.experience:
        try:
            parts = [int(p) for p in e.start.split("-")[:2]]
        except ValueError:
            continue
        starts.append(datetime(parts[0], parts[1] if len(parts) > 1 else 1, 1, tzinfo=UTC))
    if not starts:
        return 0.0
    return max(0.0, ((now or datetime.now(UTC)) - min(starts)).days / 365.25)


def _scale_similarity(similarity: float | None) -> float | None:
    """Cosine similarity between a profile and job texts usually lands in ~0.55-0.85."""
    if similarity is None:
        return None
    return min(1.0, max(0.0, (similarity - 0.55) / 0.30))


def score_fit(
    profile: Profile,
    analysis: JobAnalysis,
    similarity: float | None = None,
    now: datetime | None = None,
    accept_years_up_to: float = 5,
) -> Fit:
    vocab = profile_vocabulary(profile)
    technical = [
        r
        for r in analysis.requirements
        if r.category in TECHNICAL and (r.options or r.or_equivalent)
    ]
    unchecked = [r.name for r in analysis.requirements if r not in technical]
    must = [r for r in technical if r.must_have]
    nice = [r for r in technical if not r.must_have]
    matched = [r.name for r in technical if _matches(r, vocab, profile)]
    missing_must = [r.name for r in must if r.name not in matched]
    missing_nice = [r.name for r in nice if r.name not in matched]

    def coverage(reqs: list[Requirement]) -> float | None:
        if not reqs:
            return None
        total = sum(r.weight for r in reqs)
        return sum(r.weight for r in reqs if r.name in matched) / total

    def evidence(reqs: list[Requirement], kind: str) -> float:
        return min(1.0, sum(r.weight for r in reqs) / FULL_EVIDENCE[kind])

    must_cov, nice_cov = coverage(must), coverage(nice)
    sim = _scale_similarity(similarity)

    # Weighted average over the parts we actually know, each coverage scaled by its evidence.
    parts = [
        (0.55 * evidence(must, "must"), must_cov),
        (0.15 * evidence(nice, "nice"), nice_cov),
        (0.30, sim),
    ]
    known = [(w, v) for w, v in parts if v is not None]
    base = 100 * sum(w * v for w, v in known) / sum(w for w, _ in known) if known else 0.0

    notes, penalty = [], 0.0
    years = years_of_experience(profile, now)
    wanted = analysis.years_experience_min
    if wanted and wanted > accept_years_up_to:
        penalty += min(40.0, 12.0 * (wanted - accept_years_up_to))
        notes.append(f"asks for {wanted}+ years; you have about {years:.1f}")
    if analysis.seniority in SENIOR_LEVELS:
        penalty += 30.0
        notes.append(f"{analysis.seniority}-level role")
    return Fit(
        score=round(max(0.0, base - penalty), 1),
        must_coverage=must_cov,
        nice_coverage=nice_cov,
        similarity=similarity,
        seniority_penalty=penalty,
        matched=matched,
        missing_must=missing_must,
        missing_nice=missing_nice,
        unchecked=unchecked,
        checked=len(technical),
        total=len(analysis.requirements),
        notes=notes,
    )
