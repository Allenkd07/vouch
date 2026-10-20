"""Saved eval reports (evals/reports/), so each run is compared with the last one like it.

Reports hold aggregate numbers and the configuration that produced them, never resume text or
job names (those show where I'm applying), so they can be committed as the project's eval
history."""

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from vouch.evals.ranking import Report as RankingReport
from vouch.evals.tailoring import Run, SummaryRun

DEFAULT_DIR = Path("evals/reports")


class Metric(BaseModel):
    label: str
    value: int | float | None  # int first, so counts stay counts
    percent: bool = False  # a 0-1 share, shown as a percentage

    def show(self) -> str:
        if self.value is None:
            return "-"
        if self.percent:
            return f"{self.value:.0%}"
        return str(self.value) if isinstance(self.value, int) else f"{self.value:.2f}"

    def change_from(self, old: "Metric") -> str:
        if self.value is None or old.value is None or self.value == old.value:
            return ""
        delta = self.value - old.value
        if self.percent:
            return f"{delta * 100:+.0f} pts"
        return f"{delta:+d}" if isinstance(delta, int) else f"{delta:+.2f}"


class Report(BaseModel):
    eval: str  # "tailoring", "summaries", "ranking"
    created: datetime
    scope: str  # what was measured; only reports with the same scope are compared
    config: dict[str, str]  # models, prompt version, git commit
    metrics: dict[str, Metric]
    cost_usd: float | None = None


def prompt_version() -> str:
    """Short hash of the prompts that shape tailoring and judging, so a report shows whether
    they changed between two runs."""
    from vouch.evals import faithfulness
    from vouch.tailoring import evidence, rewrite, verify

    text = "\n".join([evidence.SYSTEM, rewrite.SYSTEM, verify.JUDGE_SYSTEM, faithfulness.SYSTEM])
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def git_commit() -> str:
    """Current commit, with "+changes" when the working tree differs from it."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    return commit + ("+changes" if git("status", "--porcelain", "--untracked-files=no") else "")


def _metric(label: str, value: float | None, percent: bool = False) -> Metric:
    return Metric(label=label, value=value, percent=percent)


def tailoring_metrics(run: Run) -> dict[str, Metric]:
    s = run.scores
    found = sum(len(x.fabrications) for x in s)
    return {
        "jobs": _metric("jobs", len(s)),
        "must_coverage": _metric("must-have coverage", run.mean("must_coverage"), True),
        "must_possible": _metric("profile could reach", run.mean("must_possible"), True),
        "rejected": _metric("rewrites rejected", run.rejection_rate, True),
        "skipped": _metric("rewrites skipped", sum(x.not_returned for x in s)),
        "keywords_gained": _metric(
            "keywords gained", sum(x.keywords_after - x.keywords_before for x in s)
        ),
        "keywords_total": _metric("keywords in postings", sum(x.keywords_total for x in s)),
        "over_pages": _metric("over the page limit", sum(x.pages > x.max_pages for x in s)),
        "audited": _metric("texts audited by the judge", run.audited),
        "judge_flagged": _metric("judge-flagged texts", found if run.audited is not None else None),
    }


def summary_metrics(run: SummaryRun) -> dict[str, Metric]:
    drafts = run.drafts()
    bad = [d for d in drafts if d.unsupported]
    escaped = [d for d in drafts if d.escaped]
    false_alarms = [d for d in drafts if d.rejected and not d.unsupported]
    return {
        "jobs": _metric("jobs", len(run.jobs)),
        "drafts": _metric("drafts", len(drafts)),
        "flagged": _metric(
            "judge-flagged drafts", len(bad) / len(drafts) if drafts else None, percent=True
        ),
        "caught": _metric("caught by the check", len(bad) - len(escaped)),
        "escaped": _metric("let through", len(escaped)),
        "false_rejections": _metric("rejected but judged clean", len(false_alarms)),
    }


def ranking_metrics(report: RankingReport) -> dict[str, Metric]:
    k = report.k
    metrics = {"labelled": _metric("labelled jobs", sum(report.counts.values()))}
    for name, m in report.metrics.items():
        metrics[f"{name}_precision"] = _metric(f"{name} precision@{k}", m["precision"])
        metrics[f"{name}_ndcg"] = _metric(f"{name} NDCG@{k}", m["ndcg"])
    return metrics


def save(report: Report, directory: Path = DEFAULT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report.eval}-{report.created:%Y-%m-%d-%H%M%S}.json"
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def latest(eval: str, scope: str, directory: Path = DEFAULT_DIR) -> tuple[Path, Report] | None:
    """The most recent saved report of `eval` with the same scope."""
    found = []
    for path in directory.glob(f"{eval}-*.json"):
        report = Report.model_validate_json(path.read_text(encoding="utf-8"))
        if report.scope == scope:
            found.append((report.created, path, report))
    if not found:
        return None
    _, path, report = max(found, key=lambda f: f[0])
    return path, report


def compare(new: Report, old: Report) -> list[str]:
    """One line per changed setting, then each metric as old -> new."""
    lines = [
        f"  {key}: {old.config.get(key, '-')} -> {value}"
        for key, value in new.config.items()
        if old.config.get(key) != value and key != "commit"
    ]
    width = max((len(m.label) for m in new.metrics.values()), default=0)
    for key, m in new.metrics.items():
        if (before := old.metrics.get(key)) is None:
            continue
        change = m.change_from(before)
        line = f"  {m.label:<{width}}  {before.show():>5} -> {m.show():<5}"
        lines.append(f"{line}  ({change})" if change else line.rstrip())
    return lines


def new_report(
    eval: str,
    scope: str,
    metrics: dict[str, Metric],
    config: dict[str, str],
    cost_usd: float | None = None,
) -> Report:
    return Report(
        eval=eval,
        created=datetime.now(UTC),
        scope=scope,
        config=config,
        metrics=metrics,
        cost_usd=round(cost_usd, 2) if cost_usd is not None else None,
    )
