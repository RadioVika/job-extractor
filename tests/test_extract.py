from types import SimpleNamespace

import anthropic
import httpx
import pytest
from pydantic import ValidationError
from tenacity import wait_none

from job_extractor.extract import (
    ExtractionError,
    _call,
    check_salary,
    extract,
    load_prompt,
    load_split,
)
from job_extractor.schema import JobPosting
from job_extractor.settings import Settings

POSTING = JobPosting(
    role_type="ml_engineer",
    seniority="mid",
    work_mode="remote",
    german="nice_to_have",
    skills=["python"],
    clouds=["aws"],
    data_types=["tabular"],
    industry="insurance",
)


class FakeClient:
    """Stands in for anthropic.Anthropic and records the requests.

    Returns the given responses in order; an exception in the list is raised instead.
    """

    def __init__(self, *responses):
        self.calls = []
        queue = list(responses)

        def parse(**kwargs):
            self.calls.append(kwargs)
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(item, Exception):
                raise item
            return item

        self.messages = SimpleNamespace(parse=parse)

    @property
    def kwargs(self):
        return self.calls[-1]


def fake_response(stop_reason="end_turn", parsed_output=POSTING):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=None,
        parsed_output=parsed_output,
        model="claude-test",
        content=[SimpleNamespace(type="text", text="{}")],
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


def validation_error():
    try:
        JobPosting.model_validate({})
    except ValidationError as e:
        return e


BAD_RANGE = POSTING.model_copy(update={"salary_min_eur": 90000, "salary_max_eur": 70000})
TEXT = "ML Engineer\nBerlin\n\n€70,000 - €90,000"


@pytest.mark.parametrize("first", [validation_error(), fake_response(parsed_output=BAD_RANGE)])
def test_extract_repairs_invalid_output(first):
    client = FakeClient(first, fake_response())
    result = extract(TEXT, client, Settings(anthropic_api_key=None))

    assert result.posting == POSTING
    assert result.repaired
    assert result.api_attempts == 2
    assert len(client.calls) == 2
    repair_prompt = client.calls[1]["messages"][0]["content"]
    assert repair_prompt.startswith("<posting>\nML Engineer")
    assert "A previous attempt failed validation:" in repair_prompt
    # tokens are only known for calls that returned a response
    expected = 100 if isinstance(first, Exception) else 200
    assert result.input_tokens == expected


def test_extract_gives_up_after_one_repair():
    client = FakeClient(fake_response(parsed_output=BAD_RANGE))
    with pytest.raises(ExtractionError, match="after repair"):
        extract(TEXT, client, Settings(anthropic_api_key=None))
    assert len(client.calls) == 2


def test_extract_does_not_repair_refusal():
    client = FakeClient(fake_response(stop_reason="refusal", parsed_output=None))
    with pytest.raises(ExtractionError, match="refused"):
        extract("x", client, Settings(anthropic_api_key=None))
    assert len(client.calls) == 1


def api_error(cls, status):
    request = httpx.Request("POST", "https://x")
    return cls("x", response=httpx.Response(status, request=request), body=None)


def test_call_retries_rate_limit():
    client = FakeClient(api_error(anthropic.RateLimitError, 429), fake_response())
    messages = [{"role": "user", "content": "x"}]
    response, attempts, _ = _call(
        client, Settings(anthropic_api_key=None), messages, wait=wait_none()
    )

    assert response.parsed_output == POSTING
    assert attempts == 2
    assert len(client.calls) == 2


def test_extract_does_not_retry_auth_error():
    client = FakeClient(api_error(anthropic.AuthenticationError, 401))
    with pytest.raises(anthropic.AuthenticationError):
        extract("x", client, Settings(anthropic_api_key=None))
    assert len(client.calls) == 1


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


def test_load_split(tmp_path):
    splits = tmp_path / "splits.json"
    splits.write_text('{"seed": 1, "dev": ["p001", "p002"], "test": ["p003"]}')
    assert load_split("dev", splits) == ["p001", "p002"]
    assert load_split("test", splits) == ["p003"]
    with pytest.raises(ValueError):
        load_split("seed", splits)
