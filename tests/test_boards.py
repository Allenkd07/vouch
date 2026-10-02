import httpx
import pytest
from pydantic import ValidationError

from vouch.boards import BOARDS, Company, board_for_url, fetch_job
from vouch.discovery.config import SearchConfig


def test_every_supported_board_is_registered():
    assert set(BOARDS) == {"greenhouse", "lever", "ashby", "workday"}
    assert all(name == board.name for name, board in BOARDS.items())


@pytest.mark.parametrize(
    ("url", "board"),
    [
        ("https://job-boards.greenhouse.io/acme/jobs/42", "greenhouse"),
        ("https://boards.greenhouse.io/embed/job_app?for=acme/jobs/42", "greenhouse"),
        ("https://jobs.lever.co/acme/0b9f8a1e-1234-4c3d-9e8f-123456789abc", "lever"),
        ("https://jobs.ashbyhq.com/acme/0b9f8a1e-1234-4c3d-9e8f-123456789abc", "ashby"),
        ("https://jiostar.wd102.myworkdayjobs.com/jiostar/job/Bengaluru/SDE_JR1", "workday"),
    ],
)
def test_board_for_url(url, board):
    assert board_for_url(url).name == board


def test_unknown_sites_have_no_board():
    assert board_for_url("https://careers.example.com/jobs/1") is None


def test_company_needs_a_registered_board_and_its_address():
    with pytest.raises(ValidationError, match="unknown ats 'taleo'"):
        Company(name="Acme", ats="taleo", board="acme")
    with pytest.raises(ValidationError, match="need `url`"):
        Company(name="Acme", ats="workday", board="acme")
    with pytest.raises(ValidationError, match="need `board`"):
        Company(name="Acme", ats="lever", url="https://jobs.lever.co/acme")
    config = SearchConfig.model_validate(
        {"companies": [{"name": "Acme", "ats": "ashby", "board": "acme"}]}
    )
    assert config.companies[0].ats == "ashby"


def test_fetch_job_uses_the_board_api():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "boards-api.greenhouse.io"
        assert request.url.path == "/v1/boards/acme/jobs/42"
        return httpx.Response(
            200, json={"id": 42, "title": "SDE", "content": "&lt;p&gt;Go&lt;/p&gt;"}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    job = fetch_job("https://job-boards.greenhouse.io/acme/jobs/42", client)
    assert (job.source, job.title, job.description) == ("greenhouse", "SDE", "Go")
