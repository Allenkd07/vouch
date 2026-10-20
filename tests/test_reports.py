from datetime import UTC, datetime, timedelta

from vouch.evals.faithfulness import Claim
from vouch.evals.reports import Metric, Report, compare, latest, save, summary_metrics
from vouch.evals.tailoring import SummaryDraft, SummaryJob, SummaryRun


def report(scope="12 jobs", *, coverage=0.91, flagged=3, when=0, prompts="aaaa1111"):
    return Report(
        eval="tailoring",
        created=datetime(2026, 10, 20, tzinfo=UTC) + timedelta(hours=when),
        scope=scope,
        config={"model": "flash", "prompts": prompts, "commit": f"c{when}"},
        metrics={
            "must_coverage": Metric(label="must-have coverage", value=coverage, percent=True),
            "judge_flagged": Metric(label="judge-flagged texts", value=flagged),
        },
        cost_usd=2.06,
    )


def test_latest_is_the_newest_report_with_the_same_scope(tmp_path):
    save(report(when=0), tmp_path)
    save(report(when=2, coverage=0.92), tmp_path)
    save(report("2 jobs", when=3, coverage=0.5), tmp_path)  # a quick test run: not comparable
    path, found = latest("tailoring", "12 jobs", tmp_path)
    assert found.metrics["must_coverage"].value == 0.92
    assert path.name == "tailoring-2026-10-20-020000.json"
    assert latest("tailoring", "6 jobs", tmp_path) is None
    assert latest("summaries", "12 jobs", tmp_path) is None


def test_saved_counts_stay_whole_numbers(tmp_path):
    path = save(report(), tmp_path)
    loaded = Report.model_validate_json(path.read_text(encoding="utf-8"))
    assert (
        loaded.metrics["judge_flagged"].value == 3 and loaded.metrics["judge_flagged"].show() == "3"
    )
    assert loaded.metrics["must_coverage"].show() == "91%"


def test_compare_shows_changed_settings_and_each_metric():
    lines = compare(report(coverage=0.92, flagged=1, prompts="bbbb2222"), report())
    assert lines[0] == "  prompts: aaaa1111 -> bbbb2222"  # the commit always differs: not shown
    assert lines[1].split() == ["must-have", "coverage", "91%", "->", "92%", "(+1", "pts)"]
    assert lines[2].split() == ["judge-flagged", "texts", "3", "->", "1", "(-2)"]
    assert compare(report(), report())[0].split() == ["must-have", "coverage", "91%", "->", "91%"]


def test_summary_metrics_count_escapes_and_false_rejections():
    bad = [Claim(claim="x", supported=False, why="not in source")]
    drafts = [
        SummaryDraft(text="a", rejected="adds 'Kafka'", unsupported=bad),  # caught
        SummaryDraft(text="b", rejected=None, unsupported=bad),  # let through
        SummaryDraft(text="c", rejected="adds 'REST APIs'"),  # rejected but clean
        SummaryDraft(text="d", rejected=None),
        SummaryDraft(text=None, rejected=None),  # none returned: not a draft
    ]
    run = SummaryRun(jobs=[SummaryJob(url="u", title="t", company="c", drafts=drafts)])
    m = {k: v.value for k, v in summary_metrics(run).items()}
    assert m == {
        "jobs": 1,
        "drafts": 4,
        "flagged": 0.5,
        "caught": 1,
        "escaped": 1,
        "false_rejections": 1,
    }
