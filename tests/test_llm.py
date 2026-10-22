from types import SimpleNamespace

import anthropic
import httpx
import httpx2
import openai
import pytest

from job_extractor.llm import (
    AnthropicClient,
    LLMError,
    OpenAIClient,
    make_client,
    resolve_model,
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
)


class FakeSDK:
    """Stands in for an SDK client: `path` is the method's attribute path, e.g. messages.parse."""

    def __init__(self, path, result):
        self.calls = []

        def method(**kwargs):
            self.calls.append(kwargs)
            if isinstance(result, Exception):
                raise result
            return result

        group, name = path.split(".")
        setattr(self, group, SimpleNamespace(**{name: method}))

    @property
    def kwargs(self):
        return self.calls[-1]


def test_resolve_model():
    assert resolve_model(Settings(provider="anthropic", model=None)) == "claude-opus-5-5"
    assert resolve_model(Settings(provider="openai", model=None)) == "gpt-6-luna"
    assert resolve_model(Settings(provider="openai", model="gpt-x")) == "gpt-x"


@pytest.mark.parametrize(
    ("provider", "cls"), [("anthropic", AnthropicClient), ("openai", OpenAIClient)]
)
def test_make_client(provider, cls):
    client = make_client(Settings(provider=provider, model=None, effort="high"))
    assert isinstance(client, cls)
    assert client.provider == provider
    assert client.effort == "high"


# Anthropic


def anthropic_response(stop_reason="end_turn", parsed_output=POSTING, stop_details=None):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=stop_details,
        parsed_output=parsed_output,
        model="claude-test",
        content=[SimpleNamespace(type="text", text="{}")],
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


def test_anthropic_parse():
    sdk = FakeSDK("messages.parse", anthropic_response())
    response = AnthropicClient(None, "claude-x", "low", sdk=sdk).parse("sys", "user", JobPosting)

    assert response.parsed == POSTING
    assert (response.model, response.stop, response.text) == ("claude-test", "ok", "{}")
    assert (response.input_tokens, response.output_tokens) == (100, 20)
    assert sdk.kwargs["model"] == "claude-x"
    assert sdk.kwargs["system"][0]["text"] == "sys"
    assert sdk.kwargs["messages"] == [{"role": "user", "content": "user"}]
    assert sdk.kwargs["output_format"] is JobPosting
    assert sdk.kwargs["output_config"] == {"effort": "low"}


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [("refusal", "refusal"), ("max_tokens", "max_tokens"), ("pause_turn", "other")],
)
def test_anthropic_stop(stop_reason, expected):
    sdk = FakeSDK("messages.parse", anthropic_response(stop_reason, parsed_output=None))
    response = AnthropicClient(None, "m", "low", sdk=sdk).parse("s", "u", JobPosting)
    assert response.stop == expected


def anthropic_error(cls, status):
    request = httpx.Request("POST", "https://x")
    return cls("x", response=httpx.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (anthropic_error(anthropic.RateLimitError, 429), True),
        (anthropic_error(anthropic.InternalServerError, 500), True),
        (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), True),
        (anthropic_error(anthropic.AuthenticationError, 401), False),
    ],
)
def test_anthropic_wraps_errors(error, transient):
    client = AnthropicClient(None, "m", "low", sdk=FakeSDK("messages.parse", error))
    with pytest.raises(LLMError) as info:
        client.parse("s", "u", JobPosting)
    assert info.value.transient is transient
    assert info.value.__cause__ is error


# OpenAI


def openai_response(content=None, incomplete_reason=None):
    if content is None:
        content = [SimpleNamespace(type="output_text", text="{}")]
    return SimpleNamespace(
        output=[SimpleNamespace(type="message", content=content)],
        output_parsed=POSTING if incomplete_reason is None else None,
        output_text="{}",
        incomplete_details=SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None,
        model="gpt-test",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


def test_openai_parse():
    sdk = FakeSDK("responses.parse", openai_response())
    response = OpenAIClient(None, "gpt-x", "low", sdk=sdk).parse("sys", "user", JobPosting)

    assert response.parsed == POSTING
    assert (response.model, response.stop, response.text) == ("gpt-test", "ok", "{}")
    assert (response.input_tokens, response.output_tokens) == (100, 20)
    assert sdk.kwargs["model"] == "gpt-x"
    assert sdk.kwargs["instructions"] == "sys"
    assert sdk.kwargs["input"] == [{"role": "user", "content": "user"}]
    assert sdk.kwargs["text_format"] is JobPosting
    assert sdk.kwargs["reasoning"] == {"effort": "low"}


def test_openai_refusal():
    refusal = [SimpleNamespace(type="refusal", refusal="I can't help with that.")]
    sdk = FakeSDK("responses.parse", openai_response(content=refusal))
    response = OpenAIClient(None, "m", "low", sdk=sdk).parse("s", "u", JobPosting)
    assert (response.stop, response.detail) == ("refusal", "I can't help with that.")


@pytest.mark.parametrize(
    ("reason", "expected"), [("max_output_tokens", "max_tokens"), ("content_filter", "other")]
)
def test_openai_incomplete(reason, expected):
    sdk = FakeSDK("responses.parse", openai_response(incomplete_reason=reason))
    response = OpenAIClient(None, "m", "low", sdk=sdk).parse("s", "u", JobPosting)
    assert (response.stop, response.detail, response.parsed) == (expected, reason, None)


def openai_error(cls, status):
    request = httpx2.Request("POST", "https://x")
    return cls("x", response=httpx2.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (openai_error(openai.RateLimitError, 429), True),
        (openai_error(openai.InternalServerError, 503), True),
        (openai.APIConnectionError(request=httpx2.Request("POST", "https://x")), True),
        (openai_error(openai.AuthenticationError, 401), False),
    ],
)
def test_openai_wraps_errors(error, transient):
    client = OpenAIClient(None, "m", "low", sdk=FakeSDK("responses.parse", error))
    with pytest.raises(LLMError) as info:
        client.parse("s", "u", JobPosting)
    assert info.value.transient is transient
    assert info.value.__cause__ is error
