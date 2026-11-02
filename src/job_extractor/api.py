"""HTTP API around the extractor: uv run uvicorn job_extractor.api:app"""

import logging
from typing import Annotated

import anthropic
import openai
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints

from job_extractor.extract import ExtractionError, extract, load_settings
from job_extractor.llm import LLMClient, LLMError, make_client
from job_extractor.schema import JobPosting
from job_extractor.settings import Provider

MAX_TEXT_CHARS = 20_000

logger = logging.getLogger(__name__)

app = FastAPI(title="job-extractor")


class ExtractRequest(BaseModel):
    text: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS)
    ]
    provider: Provider = "anthropic"


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ExtractResponse(BaseModel):
    job_posting: JobPosting
    usage: Usage
    latency_ms: float = Field(description="Time spent in model calls, including retries")


def get_provider(req: ExtractRequest) -> LLMClient:
    """The LLM client for the requested provider; tests replace it via dependency_overrides."""
    try:
        return make_client(load_settings(req.provider))
    except Exception as e:
        # The OpenAI SDK raises here when no API key can be found.
        logger.exception("could not create %s client", req.provider)
        raise HTTPException(502, f"{req.provider} request failed: client setup failed") from e


def _reason(e: Exception) -> str:
    """A short, safe description of a provider failure (no response bodies or keys)."""
    cause = e.__cause__
    if isinstance(cause, anthropic.APITimeoutError | openai.APITimeoutError):
        return "timeout"
    if isinstance(cause, anthropic.APIConnectionError | openai.APIConnectionError):
        return "connection error"
    if isinstance(cause, anthropic.APIStatusError | openai.APIStatusError):
        return f"HTTP {cause.status_code}"
    if isinstance(e, ExtractionError):
        # Our own messages (refusal, truncation, invalid output); validation details can be long.
        return str(e).splitlines()[0][:200]
    return type(e).__name__


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract")
def extract_posting(
    req: ExtractRequest, client: Annotated[LLMClient, Depends(get_provider)]
) -> ExtractResponse:
    # A sync endpoint: FastAPI runs it in a worker thread, so the blocking SDK call is fine.
    try:
        result = extract(req.text, client, load_settings(req.provider))
    except (ExtractionError, LLMError) as e:
        logger.exception("extraction with %s failed", req.provider)
        raise HTTPException(502, f"{req.provider} request failed: {_reason(e)}") from e
    return ExtractResponse(
        job_posting=result.posting,
        usage=Usage(input_tokens=result.input_tokens, output_tokens=result.output_tokens),
        latency_ms=result.latency_s * 1000,
    )
