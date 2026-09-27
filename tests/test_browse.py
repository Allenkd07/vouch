from datetime import UTC, datetime
from types import SimpleNamespace

from vouch.web.browse import Params, Row, browse, job_cities, query_string

NOW = datetime(2026, 9, 27, tzinfo=UTC)


def _row(title, company, location, score=None, similarity=None, status=None, age=None):
    job = SimpleNamespace(
        id=hash(title + company),
        title=title,
        company=company,
        location=location,
        fetched_at=NOW,
    )
    match = (
        SimpleNamespace(score=score, similarity=similarity, fit=None)
        if (score is not None or similarity is not None)
        else None
    )
    app = SimpleNamespace(status=status) if status else None
    return Row(job, match, app, job_cities(location), age, 0, "XX")


ROWS = [
    _row("SDE II", "JioStar", "Bengaluru- We Work", score=80, similarity=0.68, status="applied"),
    _row("Backend Engineer", "Sarvam AI", "Bangalore", similarity=0.71, age=3),
    _row("Software Engineer 3", "MongoDB", "Gurgaon, India", score=55, similarity=0.69, age=10),
    _row("Data Engineer", "FamPay", "Pune", similarity=0.60, age=1),
]


def titles(b):
    return [r.job.title for r in b.rows]


def test_cities_are_normalised():
    assert job_cities("Bengaluru- We Work") == ["Bengaluru"]
    assert job_cities("bangalore") == ["Bengaluru"]
    assert job_cities("Gurgaon, India") == ["Gurugram"]
    assert job_cities("Hyderabad, Telangana, India; Remote") == ["Hyderabad", "Remote"]
    assert job_cities("Zurich") == ["Zurich"]  # unknown places shown as written
    assert job_cities(None) == []


def test_sort_by_fit_puts_unscored_last_ordered_by_similarity():
    assert titles(browse(ROWS, Params(sort="fit"))) == [
        "SDE II",
        "Software Engineer 3",
        "Backend Engineer",
        "Data Engineer",
    ]
    assert titles(browse(ROWS, Params(sort="newest")))[:2] == ["Data Engineer", "Backend Engineer"]


def test_filters_combine():
    b = browse(ROWS, Params(city="Bengaluru", fit="unscored"))
    assert titles(b) == ["Backend Engineer"]
    assert titles(browse(ROWS, Params(q="engineer mongo"))) == ["Software Engineer 3"]
    assert titles(browse(ROWS, Params(fit="60"))) == ["SDE II"]
    assert titles(browse(ROWS, Params(status="none", sort="company"))) == [
        "Data Engineer",
        "Software Engineer 3",
        "Backend Engineer",
    ]


def test_facet_counts_ignore_their_own_filter():
    # With city=Bengaluru selected, the city counts still show every city (so you can switch),
    # while company counts only cover Bengaluru jobs.
    b = browse(ROWS, Params(city="Bengaluru"))
    assert dict(b.cities) == {"Bengaluru": 2, "Gurugram": 1, "Pune": 1}
    assert dict(b.companies) == {"JioStar": 1, "Sarvam AI": 1}
    assert b.statuses == {"applied": 1, "none": 1}


def test_query_string_drops_one_filter():
    p = Params(q="sde", city="Pune", sort="fit")
    assert query_string(p, city="") == "q=sde&sort=fit"
