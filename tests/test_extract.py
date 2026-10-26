import pytest
from pydantic import ValidationError
from tenacity import wait_none

from job_extractor.extract import (
    ExtractionError,
    _call,
    extract,
    load_prompt,
    load_settings,
    load_split,
)
from job_extractor.llm import LLMError, LLMResponse
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
    """Stands in for an LLMClient and records the requests.

    Returns the given responses in order; an exception in the list is raised instead.
    """

    provider = "fake"
    model = "fake-model"

    def __init__(self, *responses):
        self.calls = []
        self.queue = list(responses)

    def parse(self, system, user, schema):
        self.calls.append({"system": system, "user": user, "schema": schema})
        item = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def kwargs(self):
        return self.calls[-1]


def fake_response(stop="ok", parsed_output=POSTING):
    return LLMResponse(
        parsed=parsed_output,
        model="fake-model",
        input_tokens=100,
        output_tokens=20,
        stop=stop,
        detail=None,
        text="{}",
    )


def test_prompt_exists():
    assert "<posting>" in load_prompt("v1")


def test_extract_returns_parsed_posting():
    client = FakeClient(fake_response())
    result = extract("ML Engineer\nBerlin\n\nBody", client, Settings())

    assert result.posting == POSTING
    assert result.provider == "fake"
    assert result.model == "fake-model"
    assert result.prompt_version == "v1"
    assert (result.input_tokens, result.output_tokens) == (100, 20)
    assert result.latency_s >= 0
    assert result.api_attempts == 1
    assert not result.repaired
    assert client.kwargs["schema"] is JobPosting
    assert client.kwargs["system"] == load_prompt("v1")
    assert client.kwargs["user"].startswith("<posting>\nML Engineer")


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_extract_raises_on_bad_stop(stop):
    client = FakeClient(fake_response(stop=stop, parsed_output=None))
    with pytest.raises(ExtractionError):
        extract("x", client, Settings())


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
    result = extract(TEXT, client, Settings())

    assert result.posting == POSTING
    assert result.repaired
    assert result.api_attempts == 2
    assert len(client.calls) == 2
    repair_prompt = client.calls[1]["user"]
    assert repair_prompt.startswith("<posting>\nML Engineer")
    assert "A previous attempt failed validation:" in repair_prompt
    # tokens are only known for calls that returned a response
    expected = 100 if isinstance(first, Exception) else 200
    assert result.input_tokens == expected


def test_extract_gives_up_after_one_repair():
    client = FakeClient(fake_response(parsed_output=BAD_RANGE))
    with pytest.raises(ExtractionError, match="after repair"):
        extract(TEXT, client, Settings())
    assert len(client.calls) == 2


def test_extract_does_not_repair_refusal():
    client = FakeClient(fake_response(stop="refusal", parsed_output=None))
    with pytest.raises(ExtractionError, match="refused"):
        extract("x", client, Settings())
    assert len(client.calls) == 1


def test_call_retries_transient_error():
    client = FakeClient(LLMError("rate limited", transient=True), fake_response())
    response, attempts, _ = _call(client, Settings(), "x", wait=wait_none())

    assert response.parsed == POSTING
    assert attempts == 2
    assert len(client.calls) == 2


def test_extract_does_not_retry_permanent_error():
    client = FakeClient(LLMError("bad key", transient=False))
    with pytest.raises(LLMError):
        extract("x", client, Settings())
    assert len(client.calls) == 1


def test_load_split(tmp_path):
    splits = tmp_path / "splits.json"
    splits.write_text('{"seed": 1, "dev": ["p001", "p002"], "test": ["p003"]}')
    assert load_split("dev", splits) == ["p001", "p002"]
    assert load_split("test", splits) == ["p003"]
    with pytest.raises(ValueError):
        load_split("seed", splits)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Run with no .env file and only the given environment variables."""
    monkeypatch.chdir(tmp_path)
    for name in ("PROVIDER", "MODEL"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_load_settings_defaults(env):
    settings = load_settings()
    assert (settings.provider, settings.model) == ("anthropic", None)


def test_load_settings_reads_config(env):
    env.setenv("PROVIDER", "openai")
    env.setenv("MODEL", "gpt-custom")
    settings = load_settings()
    assert (settings.provider, settings.model) == ("openai", "gpt-custom")


def test_load_settings_cli_overrides_config(env):
    env.setenv("PROVIDER", "openai")
    env.setenv("MODEL", "gpt-custom")
    assert load_settings(model="gpt-other").model == "gpt-other"
    # Switching provider drops the configured model, which belongs to the other provider.
    settings = load_settings(provider="anthropic")
    assert (settings.provider, settings.model) == ("anthropic", None)
    settings = load_settings(provider="anthropic", model="claude-x")
    assert (settings.provider, settings.model) == ("anthropic", "claude-x")
    # Same provider as configured: the configured model stays.
    assert load_settings(provider="openai").model == "gpt-custom"
