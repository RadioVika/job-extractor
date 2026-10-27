from types import SimpleNamespace

import anthropic
import httpx
import httpx2
import openai
import pytest
from pydantic import ValidationError

from job_extractor.llm import (
    AnthropicClient,
    LlamaClient,
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
    assert resolve_model(Settings(provider="anthropic", model=None)) == "claude-haiku-5-5"
    assert resolve_model(Settings(provider="openai", model=None)) == "gpt-6-luna"
    assert resolve_model(Settings(provider="llama", model=None)) == "bonsai-2"
    assert resolve_model(Settings(provider="openai", model="gpt-x")) == "gpt-x"


@pytest.mark.parametrize(
    ("provider", "cls"),
    [
        ("anthropic", AnthropicClient),
        ("openai", OpenAIClient),
        ("llama", LlamaClient),
    ],
)
def test_make_client(provider, cls):
    keys = {f"{name}_api_key": "test-key" for name in ("anthropic", "openai")}
    client = make_client(Settings(provider=provider, model=None, effort="high", **keys))
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


# Llama (local llama-server)


def llama_response(finish_reason="stop", content=None, refusal=None):
    message = SimpleNamespace(
        content=POSTING.model_dump_json() if content is None else content, refusal=refusal
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=message)],
        model="bonsai-test",
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


def llama_client(result):
    sdk = SimpleNamespace(chat=FakeSDK("completions.create", result))
    return LlamaClient("http://x/v1", "bonsai-x", "low", sdk=sdk), sdk.chat


def test_make_client_points_llama_at_local_port():
    client = make_client(Settings(provider="llama", model=None, llama_port=9999))
    assert str(client.sdk.base_url) == "http://127.0.0.1:9999/v1/"
    assert client.model == "bonsai-2"


def test_llama_parse():
    client, sdk = llama_client(llama_response())
    response = client.parse("sys", "user", JobPosting)

    assert response.parsed == POSTING
    assert (response.model, response.stop) == ("bonsai-test", "ok")
    assert (response.input_tokens, response.output_tokens) == (100, 20)
    assert sdk.kwargs["model"] == "bonsai-x"
    assert sdk.kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]
    json_schema = sdk.kwargs["response_format"]["json_schema"]
    assert json_schema["schema"] == JobPosting.model_json_schema()
    assert sdk.kwargs["reasoning_effort"] == "low"


@pytest.mark.parametrize(
    ("effort", "expected"),
    [("low", "low"), ("medium", "medium"), ("high", "xhigh"), ("xhigh", "xhigh"), ("max", "xhigh")],
)
def test_llama_reasoning_effort(effort, expected):
    sdk = SimpleNamespace(chat=FakeSDK("completions.create", llama_response()))
    LlamaClient("http://x/v1", "m", effort, sdk=sdk).parse("s", "u", JobPosting)
    assert sdk.chat.kwargs["reasoning_effort"] == expected


@pytest.mark.parametrize("reasoning", ["none", "low", "medium", "xhigh"])
def test_llama_reasoning_overrides_effort(reasoning):
    sdk = SimpleNamespace(chat=FakeSDK("completions.create", llama_response()))
    LlamaClient("http://x/v1", "m", "max", reasoning=reasoning, sdk=sdk).parse("s", "u", JobPosting)
    assert sdk.chat.kwargs["reasoning_effort"] == reasoning


def test_make_client_passes_llama_reasoning():
    settings = Settings(provider="llama", model=None, effort="medium", llama_reasoning="none")
    assert make_client(settings).reasoning == "none"
    settings = Settings(provider="llama", model=None, effort="max", llama_reasoning=None)
    assert make_client(settings).reasoning == "xhigh"


def test_llama_invalid_output_raises_validation_error():
    client, _ = llama_client(llama_response(content='{"role_type": "x"}'))
    with pytest.raises(ValidationError):
        client.parse("s", "u", JobPosting)


@pytest.mark.parametrize(
    ("finish_reason", "expected"), [("length", "max_tokens"), ("content_filter", "other")]
)
def test_llama_stop(finish_reason, expected):
    client, _ = llama_client(llama_response(finish_reason, content='{"trunc'))
    response = client.parse("s", "u", JobPosting)
    assert (response.stop, response.detail, response.parsed) == (expected, finish_reason, None)


def test_llama_refusal():
    client, _ = llama_client(llama_response(content="", refusal="No."))
    response = client.parse("s", "u", JobPosting)
    assert (response.stop, response.detail) == ("refusal", "No.")


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (openai_error(openai.InternalServerError, 503), True),
        (openai.APIConnectionError(request=httpx2.Request("POST", "https://x")), True),
        (openai_error(openai.BadRequestError, 400), False),
    ],
)
def test_llama_wraps_errors(error, transient):
    client, _ = llama_client(error)
    with pytest.raises(LLMError) as info:
        client.parse("s", "u", JobPosting)
    assert info.value.transient is transient
    assert info.value.__cause__ is error
