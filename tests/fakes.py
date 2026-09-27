"""Shared fakes for tailoring tests: a job analysis, an evidence map, and a scripted LLM."""

from vouch.jobs.requirements import JobAnalysis, Requirement
from vouch.llm import FakeLLM
from vouch.tailoring.evidence import BulletMatch, EvidenceMap, RequirementEvidence
from vouch.tailoring.rewrite import Rewrites, RewrittenBullet
from vouch.tailoring.verify import Verdict, Verdicts


def _req(name, must=True, weight=2, options=()):
    return Requirement(
        name=name,
        category="tool",
        options=list(options),
        must_have=must,
        weight=weight,
        quote=name,
    )


def make_analysis() -> JobAnalysis:
    return JobAnalysis(
        title="Backend Engineer",
        company="Acme",
        seniority="mid",
        years_experience_min=2,
        years_experience_max=None,
        location=None,
        work_mode="remote",
        summary="Build payment services.",
        responsibilities=[],
        requirements=[
            _req("Python", weight=3, options=["Python"]),
            _req("Task queues", options=["Celery"]),
            _req("Mentoring", weight=1),
            _req("Kubernetes", must=False, options=["Kubernetes"]),
        ],
        keywords=["Python", "PostgreSQL", "Redis"],
    )


def make_evidence():
    def item(req, *matches):
        return RequirementEvidence(
            requirement=req,
            bullets=[BulletMatch(bullet_id=i, strength=s) for i, s in matches],
            in_skills=False,
            in_education=False,
            note="",
        )

    return EvidenceMap(
        items=[
            item("Python", ("finlytics-1", "strong"), ("docqa-1", "partial")),
            item("Task queues", ("finlytics-2", "strong")),
            item("Mentoring", ("finlytics-3", "strong")),
            item("Kubernetes"),
        ]
    )


class ScriptedLLM(FakeLLM):
    """Evidence -> rewrite (one bullet adds a tool, summary overclaims) -> judge -> retry."""

    def __init__(self):
        super().__init__(self._respond)
        self.rewrite_calls = 0

    def _respond(self, prompt, schema):
        if schema is EvidenceMap:
            return make_evidence()
        if schema is Rewrites:
            self.rewrite_calls += 1
            if self.rewrite_calls == 1:
                return Rewrites(
                    summary="Engineer who scaled Kubernetes platforms.",
                    bullets=[
                        RewrittenBullet(
                            id="finlytics-1",
                            text="Built a Python and PostgreSQL reconciliation service on "
                            "Kubernetes processing 2M records/day",
                        ),
                        RewrittenBullet(
                            id="finlytics-2",
                            text="Cut report generation time by 60% using Celery and Redis",
                        ),
                        RewrittenBullet(id="finlytics-3", text="Led the payments team"),
                    ],
                )
            # Retry: fixes finlytics-1, summary still unsupported by the judge.
            assert "Kubernetes" in prompt and "previous attempt was rejected" in prompt
            return Rewrites(
                summary="Backend engineer building Python payment services.",
                bullets=[
                    RewrittenBullet(
                        id="finlytics-1",
                        text="Built a Python and PostgreSQL reconciliation service "
                        "processing 2M records/day",
                    ),
                    RewrittenBullet(id="finlytics-3", text="Led the payments team"),
                ],
            )
        if schema is Verdicts:
            verdicts = []
            for block in prompt.split("\n\n"):
                item_id = block.split("\n")[0].removeprefix("id: ")
                bad = "Led the payments team" in block or (
                    item_id == "summary" and self.rewrite_calls > 1
                )
                verdicts.append(Verdict(id=item_id, supported=not bad, issue="overclaim" * bad))
            return Verdicts(verdicts=verdicts)
        raise AssertionError(schema)
