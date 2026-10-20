from pathlib import Path

from fakes import ScriptedLLM, make_analysis, make_evidence

from vouch.config import Settings
from vouch.evals.faithfulness import Audit, Audits, Claim, audit, resume_items
from vouch.evals.tailoring import load_drafts, measure, rescore, run_eval, sample_summaries
from vouch.evals.traps import check_independent, load_traps
from vouch.llm import FakeLLM, LLMError
from vouch.profile.schema import load_profile
from vouch.tailoring.evidence import EvidenceMap
from vouch.tailoring.pipeline import tailor
from vouch.tailoring.rewrite import Rewrites
from vouch.tailoring.verify import Verdict, Verdicts

EXAMPLE = Path(__file__).parent.parent / "profile" / "profile.example.yaml"
TRAPS = Path(__file__).parent.parent / "evals" / "traps.yaml"


def flag_if(*phrases):
    """A judge that marks a text unsupported when it contains any of `phrases`."""

    def respond(prompt, schema):
        assert schema is Audits
        audits = []
        for block in prompt.split("\n\n"):
            item_id = block.split("\n")[0].removeprefix("id: ")
            text = block.split("TEXT: ", 1)[1]
            bad = [p for p in phrases if p in text]
            claims = [Claim(claim=p, supported=False, why="not in source") for p in bad]
            audits.append(
                Audit(id=item_id, claims=claims or [Claim(claim="x", supported=True, why="")])
            )
        return Audits(audits=audits)

    return FakeLLM(respond)


def test_judge_role_has_its_own_provider_and_model():
    s = Settings(llm_provider="gemini", judge_provider="vertex", judge_model="pro")
    assert (s.provider_for("judge"), s.model_for("judge")) == ("vertex", "pro")
    assert s.provider_for("tailoring") == "gemini"
    assert Settings(llm_provider="fake", judge_provider=None).provider_for("judge") == "fake"


def test_resume_items_are_the_changed_text_with_its_source():
    profile = load_profile(EXAMPLE)
    result = tailor(profile, make_analysis(), ScriptedLLM())
    items = {i: (source, text) for i, source, text in resume_items(profile, result.doc)}
    changed = {b.id for e in result.doc.experience for b in e.bullets if b.changed}
    assert set(items) == changed and "finlytics-1" in changed  # summary was rejected: unchanged
    source, text = items["finlytics-1"]
    # Bullet text, then its facts, labelled so the judge doesn't take them for a skills list.
    assert source.endswith("\nFacts recorded for this work: Python, PostgreSQL, 2M records/day")
    assert text == next(
        b.text for e in result.doc.experience for b in e.bullets if b.id == "finlytics-1"
    )


def test_audit_counts_a_skipped_item_as_unsupported():
    llm = FakeLLM(lambda prompt, schema: Audits(audits=[]))
    result = audit([("a", "source", "text")], llm)
    assert result["a"].unsupported[0].claim == "(not audited)"


def test_run_records_what_the_judge_flags():
    profile, analysis = load_profile(EXAMPLE), make_analysis()
    run = run_eval([("a", analysis)], profile, ScriptedLLM(), judge=flag_if("PostgreSQL"))
    score = run.scores[0]
    assert score.audited >= 1 and run.audited == score.audited
    assert [(i, [c.claim for c in claims]) for i, _, claims in score.fabrications] == [
        ("finlytics-1", ["PostgreSQL"])
    ]


def test_independent_judge_calibration_on_traps():
    traps = load_traps(TRAPS)
    outcomes = check_independent(traps, flag_if("Python and Kafka"))
    assert {o.case.id for o in outcomes if o.rejected_by} == {"adds-known-tool"}
    assert len(outcomes) == len(traps.traps) + len(traps.controls)


def test_measure_reads_coverage_rejections_and_keywords():
    profile, analysis = load_profile(EXAMPLE), make_analysis()
    result = tailor(profile, analysis, ScriptedLLM())
    score = measure(analysis, result, url="u")

    # Python, Task queues and Mentoring are the must-haves, all backed on the page.
    assert score.must_coverage == 1.0 == score.must_possible and score.left_off == []
    on_page = sum(len(e.bullets) for e in result.doc.experience + result.doc.projects)
    # The judge rejects finlytics-3 and the summary on both attempts; docqa-1 gets no rewrite.
    assert (score.rejected, score.not_returned, score.attempted) == (2, 1, on_page + 1)
    assert score.keywords_total == 3
    assert score.keywords_before == len(result.report.keywords_before)
    assert score.keywords_after == len(result.report.keywords_after)
    assert score.pages == 1 == score.max_pages


def test_must_have_with_evidence_left_off_the_page():
    profile, analysis = load_profile(EXAMPLE), make_analysis()
    result = tailor(profile, analysis, ScriptedLLM())
    for c in result.report.coverage:
        if c.requirement == "Task queues":  # pretend selection dropped its only bullet
            c.status, c.bullet_ids = "gap", []
    score = measure(analysis, result)
    assert score.left_off == ["Task queues"]
    assert score.must_coverage == 4 / 6 and score.must_possible == 1.0  # weights 3 + 1 of 6


def test_run_keeps_measured_jobs_when_the_llm_fails():
    profile, analysis = load_profile(EXAMPLE), make_analysis()

    class QuotaAfterOne(ScriptedLLM):
        jobs_started = 0

        def generate_json(self, prompt, schema, *, system=None):
            if "match job requirements" in (system or ""):  # evidence mapping starts a job
                self.jobs_started += 1
                if self.jobs_started > 1:
                    raise LLMError("quota exceeded")
            return super().generate_json(prompt, schema, system=system)

    llm = QuotaAfterOne()
    run = run_eval([("a", analysis), ("gone", None), ("b", analysis)], profile, llm)
    assert [s.url for s in run.scores] == ["a"]
    assert run.missing == ["gone"] and run.stopped == "quota exceeded"


def test_summary_drafts_are_checked_then_audited():
    drafts = iter(
        [
            "Backend engineer building payment services.",  # honest
            "Backend engineer who led payment teams.",  # the check catches it
            "Backend engineer owning payment platforms.",  # the check misses it
            None,
        ]
    )

    def respond(prompt, schema):
        if schema is EvidenceMap:
            return make_evidence()
        if schema is Rewrites:
            return Rewrites(summary=next(drafts), bullets=[])
        if schema is Verdicts:  # the pipeline's check: rejects only the "led" draft
            ids = [b.split("\n")[0].removeprefix("id: ") for b in prompt.split("\n\n")]
            bad = "who led payment teams" in prompt
            return Verdicts(verdicts=[Verdict(id=i, supported=not bad, issue="led") for i in ids])
        raise AssertionError(schema)

    judge = flag_if("led payment teams", "owning")
    profile, analysis = load_profile(EXAMPLE), make_analysis()
    run = sample_summaries([("a", analysis)], profile, FakeLLM(respond), judge, samples=4)
    marks = [(d.rejected is not None, bool(d.unsupported), d.escaped) for d in run.drafts()]
    assert marks == [(False, False, False), (True, True, False), (False, True, True)]
    assert run.jobs[0].drafts[3].text is None  # no summary returned: not audited or counted
    assert "Resume bullets" in run.jobs[0].source  # labelled, as the judge saw it


def test_minor_wording_is_shown_but_not_counted():
    claims = [
        Claim(claim="robust", supported=False, why="praise", misleading=False),
        Claim(claim="with Airflow", supported=False, why="other work"),
        Claim(claim="Python", supported=True, why=""),
    ]
    result = Audit(id="a", claims=claims)
    assert [c.claim for c in result.unsupported] == ["with Airflow"]
    assert [c.claim for c in result.minor] == ["robust"]


def test_saved_drafts_can_be_judged_again(tmp_path):
    drafts = iter(["Backend engineer owning payment platforms.", "Builds payment services."])

    def respond(prompt, schema):
        if schema is EvidenceMap:
            return make_evidence()
        if schema is Rewrites:
            return Rewrites(summary=next(drafts), bullets=[])
        ids = [b.split("\n")[0].removeprefix("id: ") for b in prompt.split("\n\n")]
        return Verdicts(verdicts=[Verdict(id=i, supported=True, issue="") for i in ids])

    profile, analysis = load_profile(EXAMPLE), make_analysis()
    run = sample_summaries([("a", analysis)], profile, FakeLLM(respond), flag_if(), samples=2)
    saved = tmp_path / "drafts.jsonl"
    saved.write_text("".join(j.model_dump_json() + "\n" for j in run.jobs), encoding="utf-8")
    assert not run.drafts()[0].unsupported

    again = rescore(load_drafts(saved), flag_if("owning"))
    assert [d.escaped for d in again.drafts()] == [True, False]

    # A later run can reuse the saved selection: same bullets, no evidence mapping.
    drafts = iter(["Builds payment services."])
    earlier = {j.url: j for j in load_drafts(saved)}

    def no_evidence(prompt, schema):
        assert schema is not EvidenceMap
        return respond(prompt, schema)

    after = sample_summaries(
        [("a", analysis)], profile, FakeLLM(no_evidence), flag_if(), samples=1, selections=earlier
    )
    assert after.jobs[0].bullet_ids == run.jobs[0].bullet_ids != []
