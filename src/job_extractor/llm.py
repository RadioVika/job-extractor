"""A provider-neutral interface for structured LLM calls, with one adapter per SDK.

The rest of the app talks to LLMClient and never imports a provider SDK. To add a provider,
write an adapter with a parse() method, add it to make_client, and give it a default model.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import anthropic
import openai
from pydantic import BaseModel

from job_extractor.settings import LlamaReasoning, Provider, Settings

DEFAULT_MODELS: dict[Provider, str] = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-6-luna",
    # Served by a local llama-server under this alias (see llama_server.py).
    "llama": "bonsai-2",
}
MAX_TOKENS = 16000


@dataclass(frozen=True)
class LLMResponse:
    parsed: BaseModel | None
    model: str
    input_tokens: int
    output_tokens: int
    # "other" covers any stop the extractor doesn't treat specially.
    stop: Literal["ok", "refusal", "max_tokens", "other"]
    detail: str | None  # provider's stop reason or refusal text, for error messages
    text: str  # raw response text, for debug logging


class LLMError(Exception):
    """An API error from any provider; the SDK's exception is chained as __cause__."""

    def __init__(self, message: str, transient: bool):
        super().__init__(message)
        # Rate limits, network errors and 5xx are worth retrying; other 4xx are not.
        self.transient = transient


class LLMClient(Protocol):
    provider: str
    model: str

    def parse(self, system: str, user: str, schema: type[BaseModel]) -> LLMResponse:
        """Make one request (no retries) and parse the output into schema.

        Raises LLMError on API errors and pydantic.ValidationError if the output doesn't
        match schema.
        """
        ...


def resolve_model(settings: Settings) -> str:
    return settings.model or DEFAULT_MODELS[settings.provider]


def make_client(settings: Settings) -> LLMClient:
    model = resolve_model(settings)
    if settings.provider == "anthropic":
        return AnthropicClient(settings.anthropic_api_key, model, settings.effort)
    if settings.provider == "openai":
        return OpenAIClient(settings.openai_api_key, model, settings.effort)
    if settings.provider == "llama":
        return LlamaClient(
            f"http://127.0.0.1:{settings.llama_port}/v1",
            model,
            settings.effort,
            reasoning=settings.llama_reasoning,
        )
    raise ValueError(f"unknown provider {settings.provider!r}")


def _secret(key) -> str | None:
    # None lets the SDK resolve credentials itself (environment variable, login profile...).
    return key.get_secret_value() if key else None


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, api_key, model: str, effort: str, sdk: anthropic.Anthropic | None = None):
        self.model = model
        self.effort = effort
        # Retries are handled by tenacity in extract._call, so the SDK's own are disabled.
        self.sdk = sdk or anthropic.Anthropic(api_key=_secret(api_key), max_retries=0)

    def parse(self, system: str, user: str, schema: type[BaseModel]) -> LLMResponse:
        try:
            resp = self.sdk.messages.parse(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=[{"type": "text", "text": system}],
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config={"effort": self.effort},
            )
        except anthropic.APIError as e:
            raise LLMError(f"anthropic: {e}", _transient(e, anthropic)) from e

        stop = {"end_turn": "ok", "refusal": "refusal", "max_tokens": "max_tokens"}
        detail = resp.stop_details if resp.stop_reason == "refusal" else resp.stop_reason
        return LLMResponse(
            parsed=resp.parsed_output,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop=stop.get(resp.stop_reason, "other"),
            detail=None if detail is None else str(detail),
            text="".join(block.text for block in resp.content if block.type == "text"),
        )


class OpenAIClient:
    provider = "openai"

    def __init__(self, api_key, model: str, effort: str, sdk: openai.OpenAI | None = None):
        self.model = model
        self.effort = effort
        # Retries are handled by tenacity in extract._call, so the SDK's own are disabled.
        self.sdk = sdk or openai.OpenAI(api_key=_secret(api_key), max_retries=0)

    def parse(self, system: str, user: str, schema: type[BaseModel]) -> LLMResponse:
        try:
            resp = self.sdk.responses.parse(
                model=self.model,
                max_output_tokens=MAX_TOKENS,
                instructions=system,
                input=[{"role": "user", "content": user}],
                text_format=schema,
                reasoning={"effort": self.effort},
            )
        except openai.APIError as e:
            raise LLMError(f"openai: {e}", _transient(e, openai)) from e

        refusals = [
            content.refusal
            for item in resp.output
            if item.type == "message"
            for content in item.content
            if content.type == "refusal"
        ]
        reason = resp.incomplete_details.reason if resp.incomplete_details else None
        if refusals:
            stop, detail = "refusal", " ".join(refusals)
        elif reason == "max_output_tokens":
            stop, detail = "max_tokens", reason
        elif reason:
            stop, detail = "other", reason
        else:
            stop, detail = "ok", None
        return LLMResponse(
            parsed=resp.output_parsed,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop=stop,
            detail=detail,
            text=resp.output_text,
        )


class LlamaClient:
    """A local llama-server, through its OpenAI-compatible Chat Completions endpoint."""

    provider = "llama"

    # How EFFORT maps to a reasoning level when LLAMA_REASONING isn't set. Bonsai 2 rejects
    # "high", so the top three efforts share xhigh.
    REASONING_EFFORTS: dict[str, LlamaReasoning] = {
        "low": "low",
        "medium": "medium",
        "high": "xhigh",
        "xhigh": "xhigh",
        "max": "xhigh",
    }

    def __init__(
        self,
        base_url: str,
        model: str,
        effort: str,
        reasoning: LlamaReasoning | None = None,
        sdk: openai.OpenAI | None = None,
    ):
        self.model = model
        self.effort = effort
        self.reasoning = reasoning or self.REASONING_EFFORTS[effort]
        # llama-server needs no key, but the SDK insists on one. Retries are handled by
        # tenacity in extract._call, so the SDK's own are disabled.
        self.sdk = sdk or openai.OpenAI(base_url=base_url, api_key="local", max_retries=0)

    def parse(self, system: str, user: str, schema: type[BaseModel]) -> LLMResponse:
        # create rather than parse: the SDK's parse raises on a length stop instead of
        # returning it, and llama-server turns the JSON schema into a grammar either way.
        try:
            resp = self.sdk.chat.completions.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
                },
                # Reasoning counts against max_tokens and dominates run time; "none" works only
                # while the server runs with --reasoning auto (its default).
                reasoning_effort=self.reasoning,
            )
        except openai.APIError as e:
            raise LLMError(f"llama: {e}", _transient(e, openai)) from e

        choice = resp.choices[0]
        message = choice.message
        text = message.content or ""
        reason = choice.finish_reason
        if message.refusal:
            stop, detail = "refusal", message.refusal
        elif reason == "length":
            stop, detail = "max_tokens", reason
        elif reason == "stop":
            stop, detail = "ok", None
        else:
            stop, detail = "other", reason
        return LLMResponse(
            # Validated here so bad output raises ValidationError like the other adapters.
            parsed=schema.model_validate_json(text) if stop == "ok" else None,
            model=resp.model or self.model,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            stop=stop,
            detail=detail,
            text=text,
        )


def _transient(e: Exception, sdk) -> bool:
    """The Anthropic and OpenAI SDKs share an exception hierarchy, so one check covers both."""
    if isinstance(e, sdk.RateLimitError | sdk.APIConnectionError):
        return True
    return isinstance(e, sdk.APIStatusError) and e.status_code >= 500
