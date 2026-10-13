from types import SimpleNamespace

import pytest

from job_extractor.extract import ExtractionError, extract, load_prompt
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

        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=parse))


def fake_response(stop_reason="end_turn", parsed_output=POSTING):
    return SimpleNamespace(stop_reason=stop_reason, stop_details=None, parsed_output=parsed_output)


def test_prompt_exists():
    assert "<posting>" in load_prompt("v1")


def test_extract_returns_parsed_posting():
    client = FakeClient(fake_response())
    result = extract("ML Engineer\nBerlin\n\nBody", client, Settings(anthropic_api_key=None))

    assert result == POSTING
    assert client.kwargs["output_format"] is JobPosting
    assert client.kwargs["system"][0]["text"] == load_prompt("v1")
    assert client.kwargs["messages"][0]["content"].startswith("<posting>\nML Engineer")


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_extract_raises_on_bad_stop(stop_reason):
    client = FakeClient(fake_response(stop_reason=stop_reason, parsed_output=None))
    with pytest.raises(ExtractionError):
        extract("x", client, Settings(anthropic_api_key=None))
