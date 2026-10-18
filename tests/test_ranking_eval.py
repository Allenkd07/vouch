from datetime import UTC, datetime

import pytest

from vouch.evals.labels import Label, load_labels, save_label
from vouch.evals.ranking import Item, evaluate, ndcg_at, precision_at


def _item(n, grade, score=None, similarity=None, reasons=()):
    label = Label(grade=grade, reasons=list(reasons), labelled_at=datetime.now(UTC))
    return Item(f"https://x/{n}", f"Job {n}", "Acme", label, score, similarity)


def test_metrics_reward_putting_good_jobs_first():
    best = [_item(1, "good"), _item(2, "okay"), _item(3, "bad")]
    assert precision_at(best, 2) == 0.5
    assert ndcg_at(best, 3) == 1.0
    # bad, okay, good: (1/log2(3) + 2/log2(4)) / (2 + 1/log2(3))
    assert ndcg_at(list(reversed(best)), 3) == pytest.approx(0.6199, abs=1e-4)


def test_fit_ranker_falls_back_to_similarity_and_reports_disagreements():
    items = [
        _item(1, "bad", score=90, similarity=0.2, reasons=["seniority"]),
        _item(2, "good", score=85, similarity=0.9),
        _item(3, "okay", score=None, similarity=0.8),
        _item(4, "good", score=None, similarity=0.1),
        _item(5, "skip", score=99),  # left out of the metrics
    ]
    report = evaluate(items, k=2)
    assert sum(report.counts.values()) == 4
    # fit order: 1 (90), 2 (85), then unscored by similarity: 3, 4
    assert report.metrics["fit"]["precision"] == 0.5
    assert report.metrics["similarity"]["precision"] == 0.5  # order 2, 3, 1, 4
    assert [i.url for _, i in report.top_bad] == ["https://x/1"]
    assert [(n, i.url) for n, i in report.missed_good] == [(4, "https://x/4")]
    assert report.bad_reasons == {"seniority": 1}


def test_labels_round_trip_and_replace(tmp_path):
    path = tmp_path / "labels.yaml"
    assert load_labels(path) == {}
    save_label("https://x/1", "bad", reasons=["stack", "made-up"], note=" Go only ", path=path)
    save_label("https://x/2", "good", title="SDE", company="Acme", path=path)
    save_label("https://x/1", "okay", path=path)  # relabel

    labels = load_labels(path)
    assert {u: lb.grade for u, lb in labels.items()} == {
        "https://x/1": "okay",
        "https://x/2": "good",
    }
    assert labels["https://x/2"].company == "Acme"
    with pytest.raises(ValueError):
        save_label("https://x/3", "great", path=path)


def test_unknown_reasons_are_dropped(tmp_path):
    label = save_label("https://x/1", "bad", reasons=["stack", "made-up"], path=tmp_path / "l.yaml")
    assert label.reasons == ["stack"]
