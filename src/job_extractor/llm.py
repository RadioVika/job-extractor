"""A provider-neutral interface for structured LLM calls, with one adapter per SDK.

The rest of the app talks to LLMClient and never imports a provider SDK. To add a provider,
write an adapter with a parse() method, add it to make_client, and give it a default model.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import anthropic
import httpx
import openai
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel

from job_extractor.settings import Provider, Settings

DEFAULT_MODELS: dict[Provider, str] = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-6-luna",
    "gemini": "gemini-4-argon",
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
    if settings.provider == "gemini":
        return GeminiClient(settings.google_api_key, model, settings.effort)
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


class GeminiClient:
    provider = "gemini"

    # Gemini's thinking levels stop at HIGH, so the top three efforts share it.
    THINKING_LEVELS = {
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
        "xhigh": "HIGH",
        "max": "HIGH",
    }

    def __init__(self, api_key, model: str, effort: str, sdk: genai.Client | None = None):
        self.model = model
        self.effort = effort
        # The SDK makes a single attempt unless retry options are set, so tenacity in
        # extract._call stays in charge of retries.
        self.sdk = sdk or genai.Client(api_key=_secret(api_key))

    def parse(self, system: str, user: str, schema: type[BaseModel]) -> LLMResponse:
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=MAX_TOKENS,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=genai_types.ThinkingConfig(
                thinking_level=self.THINKING_LEVELS[self.effort]
            ),
        )
        try:
            resp = self.sdk.models.generate_content(model=self.model, contents=user, config=config)
        except genai_errors.APIError as e:
            raise LLMError(f"gemini: {e}", e.code == 429 or e.code >= 500) from e
        except httpx.TransportError as e:  # the SDK lets network errors through unwrapped
            raise LLMError(f"gemini: {e!r}", transient=True) from e

        usage = resp.usage_metadata
        blocked = resp.prompt_feedback.block_reason if resp.prompt_feedback else None
        candidate = resp.candidates[0] if resp.candidates else None
        reason = candidate.finish_reason if candidate else None
        text = resp.text or ""
        if blocked:
            stop, detail = "refusal", f"prompt blocked: {blocked.value}"
        elif reason == genai_types.FinishReason.MAX_TOKENS:
            stop, detail = "max_tokens", reason.value
        elif reason in (genai_types.FinishReason.STOP, None):
            stop, detail = "ok", None
        elif reason in _GEMINI_REFUSALS:
            stop, detail = "refusal", reason.value
        else:
            stop, detail = "other", reason.value
        return LLMResponse(
            # The SDK's own resp.parsed swallows validation errors, so validate here to raise
            # ValidationError like the other adapters.
            parsed=schema.model_validate_json(text) if stop == "ok" else None,
            model=resp.model_version or self.model,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            # Thinking tokens are billed as output, as they are for the other providers.
            output_tokens=(
                (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
                if usage
                else 0
            ),
            stop=stop,
            detail=detail,
            text=text,
        )


_GEMINI_REFUSALS = {
    genai_types.FinishReason.SAFETY,
    genai_types.FinishReason.BLOCKLIST,
    genai_types.FinishReason.PROHIBITED_CONTENT,
    genai_types.FinishReason.SPII,
    genai_types.FinishReason.RECITATION,
}


def _transient(e: Exception, sdk) -> bool:
    """The Anthropic and OpenAI SDKs share an exception hierarchy, so one check covers both."""
    if isinstance(e, sdk.RateLimitError | sdk.APIConnectionError):
        return True
    return isinstance(e, sdk.APIStatusError) and e.status_code >= 500
