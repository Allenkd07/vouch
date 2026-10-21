"""Independent faithfulness judge for the tailoring eval.

Tailoring already rejects rewrites its own judge flags, so asking that judge about the final
resume would always pass. This one runs on a different, stronger model (the `judge` role), splits
each text into claims and checks every claim against the source, to count what the pipeline's
honesty check let through. Calibrate it on evals/traps.yaml before trusting it."""

from pydantic import BaseModel, Field

from vouch.llm import LLM
from vouch.profile.schema import Bullet, Profile
from vouch.tailoring.document import ResumeDoc
from vouch.tailoring.pipeline import bullet_facts, labelled, labelled_summary


class Claim(BaseModel):
    claim: str = Field(description="One claim the text makes, in a few words")
    supported: bool
    why: str = Field(description="If unsupported: what the source lacks; else empty")
    misleading: bool = Field(
        default=True,
        description="If unsupported: would a recruiter believe something false about what the "
        "candidate did, used, led or achieved? False for minor wording; true if supported",
    )


class Audit(BaseModel):
    id: str
    claims: list[Claim]

    @property
    def unsupported(self) -> list[Claim]:
        """What counts as fabricated: unsupported and misleading."""
        return [c for c in self.claims if not c.supported and c.misleading]

    @property
    def minor(self) -> list[Claim]:
        """Unsupported wording a recruiter would read the same way; shown, not counted."""
        return [c for c in self.claims if not c.supported and not c.misleading]


class Audits(BaseModel):
    audits: list[Audit]


SYSTEM = """You are an independent auditor checking a tailored resume for fabrication. Each item
has a SOURCE (what the candidate actually recorded) and a TEXT (what the resume now says). The
SOURCE is the candidate's own wording, then the facts they recorded for that work (tools used,
numbers, scope): a recorded fact counts as stated for that work.

For each item, list every claim the TEXT makes: each technology, number, scale, scope, action,
responsibility, outcome or result, role or seniority, and any characterisation of the work (e.g.
"production-grade", "scalable", "end-to-end", "led", "architected"). For each claim decide
whether the SOURCE states it or it follows directly from what the SOURCE states.

Unsupported: anything new; a stronger verb than the source supports ("built" -> "architected",
"helped" -> "led", "worked on" -> "owned"); a result or improvement the source doesn't state;
a broader term than the source (calling a service "microservices", CI "CI/CD"); relabeling one
thing as another (authorization as "authentication"). A technology named only in a skills list
(labelled as such) supports knowing it, not having used it for the work described.
Supported: rewording, reordering, shortening, singular/plural, the job's exact name for the
same thing, and naming what the source's work plainly is (a FastAPI backend serves APIs).

Judge strictly, but only by the SOURCE: don't reject a claim the SOURCE does state. Then rate
each unsupported claim. Misleading: a recruiter would believe something false about what the
candidate did, used, led or achieved: a tool, number, scale, result, responsibility or
seniority the source doesn't back, or a tool tied to work it wasn't recorded for. Minor: a
general word of praise that adds no checkable fact ("robust", "modern"), a small wording
shift a recruiter would read the same way, or the general name for the kind of work the
source describes, with the same tool and no new task ("orchestrated quality checks with
Airflow" -> "workflow automation with Airflow"; but "ETL pipelines with Airflow" names
different work, so misleading). When unsure, misleading. Return one audit per id,
in the given order."""


def audit(items: list[tuple[str, str, str]], llm: LLM) -> dict[str, Audit]:
    """items: (id, source, text), checked in one request. Missing ids come back as one
    unsupported claim, so a skipped item can't pass as faithful."""
    if not items:
        return {}
    prompt = "\n\n".join(f"id: {i}\nSOURCE: {source}\nTEXT: {text}" for i, source, text in items)
    by_id = {a.id: a for a in llm.generate_json(prompt, Audits, system=SYSTEM).audits}
    missing = Claim(claim="(not audited)", supported=False, why="the judge returned no audit")
    return {i: by_id.get(i) or Audit(id=i, claims=[missing]) for i, _, _ in items}


def bullet_source(b: Bullet) -> str:
    return labelled(b.text, bullet_facts(b))


def summary_source(profile: Profile, chosen: list[Bullet]) -> str:
    """What a summary was written from, labelled (the pipeline's summary_source, run together,
    is what its honesty check gets)."""
    return labelled_summary(
        profile.summary, [(b.text, bullet_facts(b)) for b in chosen], profile.skills.all()
    )


def resume_items(
    profile: Profile, doc: ResumeDoc, dropped: list[str] = ()
) -> list[tuple[str, str, str]]:
    """The text tailoring changed on the page, with the source each change may draw on: the
    same material the pipeline's honesty check gets, labelled so a recorded tool isn't taken
    for a skills list. Unchanged bullets are the profile's own wording and need no audit."""
    bullets = {b.id: b for _, b in profile.iter_bullets()}
    page = [b for e in doc.experience + doc.projects for b in e.bullets if b.included]
    items = [(b.id, bullet_source(bullets[b.id]), b.text) for b in page if b.changed]
    if doc.summary and doc.summary != profile.summary:
        # The summary was written from every chosen bullet, including ones later cut for space.
        chosen = [bullets[b.id] for b in page] + [bullets[i] for i in dropped if i in bullets]
        items.append(("summary", summary_source(profile, chosen), doc.summary))
    return items
