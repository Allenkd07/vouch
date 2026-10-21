"""`vouch eval all`: runs every step even when one breaks, and sums them up."""

import typer
from typer.testing import CliRunner

from vouch import cli


def test_a_broken_step_does_not_stop_the_rest(monkeypatch):
    ran = []

    def traps(free=False, independent=False):
        ran.append(("traps", free, independent))
        if independent:
            raise typer.Exit(1)  # a missed trap
        return "Traps caught: 22/22"

    def ranking():
        ran.append("ranking")
        raise RuntimeError("database not available\ndetails")

    def expensive(**_):
        raise AssertionError("only with --full")

    monkeypatch.setattr(cli, "eval_traps", traps)
    monkeypatch.setattr(cli, "eval_ranking", ranking)
    monkeypatch.setattr(cli, "eval_tailoring", expensive)
    monkeypatch.setattr(cli, "eval_summaries", expensive)

    result = CliRunner().invoke(cli.app, ["eval", "all"])

    assert ran == [
        ("traps", True, False),
        ("traps", False, False),
        ("traps", False, True),
        "ranking",
    ]
    summary = result.output.split("=== Summary ===")[1].splitlines()[1:]
    assert [line.split()[0] for line in summary] == ["ok", "ok", "FAIL", "error"]
    assert "Traps caught: 22/22" in summary[0]
    assert summary[3].endswith("database not available")
    assert result.exit_code == 1


def test_full_asks_before_spending(monkeypatch):
    monkeypatch.setattr(cli, "eval_traps", lambda **_: "")
    monkeypatch.setattr(cli, "eval_ranking", lambda: "")
    ran = []
    monkeypatch.setattr(cli, "eval_tailoring", lambda: ran.append("tailoring"))
    monkeypatch.setattr(cli, "eval_summaries", lambda samples: ran.append(("summaries", samples)))

    declined = CliRunner().invoke(cli.app, ["eval", "all", "--full"], input="n\n")
    assert declined.exit_code == 1 and ran == []

    accepted = CliRunner().invoke(cli.app, ["eval", "all", "--full", "--yes"])
    assert accepted.exit_code == 0, accepted.output
    assert ran == ["tailoring", ("summaries", 2)]
