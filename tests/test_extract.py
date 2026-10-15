from types import SimpleNamespace

import pytest

from job_extractor.extract import ExtractionError, check_salary, extract, load_prompt
from job_extractor.schema import JobPosting
from job_extractor.settings import Settings

POSTING = JobPosting(
    role_type="ml_engineer",
    seniority="mid",
    work_mode="remote",
    german="nice_to_have",
    skills=["python"],
    clouds=["aws"],
    industry="insurance",
)


class FakeClient:
    """Stands in for anthropic.Anthropic and records the request."""

    def __init__(self, response):
        self.kwargs = None

        def parse(**kwargs):
            self.kwargs = kwargs
            return response

        self.messages = SimpleNamespace(parse=parse)


def fake_response(stop_reason="end_turn", parsed_output=POSTING):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=None,
        parsed_output=parsed_output,
        model="claude-test",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


def test_prompt_exists():
    assert "<posting>" in load_prompt("v1")


def test_extract_returns_parsed_posting():
    client = FakeClient(fake_response())
    result = extract("ML Engineer\nBerlin\n\nBody", client, Settings(anthropic_api_key=None))

    assert result.posting == POSTING
    assert result.model == "claude-test"
    assert result.prompt_version == "v1"
    assert (result.input_tokens, result.output_tokens) == (100, 20)
    assert result.latency_s >= 0
    assert result.api_attempts == 1
    assert not result.repaired
    assert client.kwargs["output_format"] is JobPosting
    assert client.kwargs["system"][0]["text"] == load_prompt("v1")
    assert client.kwargs["messages"][0]["content"].startswith("<posting>\nML Engineer")


def test_extract_flags_repaired_salary():
    posting = POSTING.model_copy(update={"salary_min_eur": 80000})
    client = FakeClient(fake_response(parsed_output=posting))
    result = extract("No salary here.", client, Settings(anthropic_api_key=None))

    assert result.posting.salary_min_eur is None
    assert result.repaired


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_extract_raises_on_bad_stop(stop_reason):
    client = FakeClient(fake_response(stop_reason=stop_reason, parsed_output=None))
    with pytest.raises(ExtractionError):
        extract("x", client, Settings(anthropic_api_key=None))


@pytest.mark.parametrize(
    ("text", "salary", "expected"),
    [
        ("Austria: €90,000 - €168,000 gross annually", (90000, 168000), (90000, 168000)),
        ("100.000 € Jahresgehalt.", (100000, 100000), (100000, 100000)),
        ("100k Jahresgehalt", (100000, 100000), (100000, 100000)),
        ("Salary: €6,000 per month", (72000, 72000), (72000, 72000)),
        ("Competitive salary and stock options.", (80000, 80000), (None, None)),
        ("From €70,000, depending on experience.", (70000, 90000), (70000, None)),
        ("No salary here.", (None, None), (None, None)),
    ],
)
def test_check_salary(text, salary, expected):
    posting = POSTING.model_copy(update={"salary_min_eur": salary[0], "salary_max_eur": salary[1]})
    result = check_salary(text, posting)
    assert (result.salary_min_eur, result.salary_max_eur) == expected
