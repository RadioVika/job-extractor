"""Extract a structured JobPosting from a posting's plain text with LLM."""

import asyncio
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import anthropic
from pydantic import ValidationError
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from job_extractor.schema import JobPosting
from job_extractor.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "prompts"
LABELS = ROOT / "data" / "labels"
RAW = ROOT / "data" / "raw"
CONCURRENCY = 5

logger = logging.getLogger(__name__)

# Amounts as written in postings: "90,000", "100.000", "100 000", "85000", "100k", "85.5k".
# Thousands groups need exactly three digits, so "8.50" is read as 8 and 50, not 850.
_NUMBER = re.compile(
    r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s?[kK]\b"
    r"|(?<!\d)(\d{1,3}(?:[.,\s  ]\d{3})+|\d+)(?!\d)"
)


class ExtractionError(RuntimeError):
    pass


class InvalidOutput(ExtractionError):
    """The output parsed, but failed a content check; worth one repair attempt."""


@dataclass(frozen=True)
class ExtractionResult:
    posting: JobPosting
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    api_attempts: int
    repaired: bool


def _amounts_in(text: str) -> set[int]:
    amounts = set()
    for k_value, plain in _NUMBER.findall(text):
        if k_value:
            amounts.add(round(float(k_value.replace(",", ".")) * 1000))
        else:
            amounts.add(int(re.sub(r"\D", "", plain)))
    return amounts


def check_salary(posting: str, result: JobPosting) -> JobPosting:
    """Null out salary fields whose value can't be found in the posting text.

    A value counts as found if it appears as written, or as a monthly figure (value / 12),
    since the prompt asks the model to annualise monthly salaries.
    """
    amounts = _amounts_in(posting)
    update = {}
    for field in ("salary_min_eur", "salary_max_eur"):
        value = getattr(result, field)
        if value is None:
            continue
        if value in amounts or (value % 12 == 0 and value // 12 in amounts):
            continue
        logger.warning("%s=%d not found in posting text, dropping it", field, value)
        update[field] = None
    return result.model_copy(update=update) if update else result


def check_content(result: JobPosting) -> None:
    """Raise InvalidOutput if the posting is internally inconsistent."""
    lo, hi = result.salary_min_eur, result.salary_max_eur
    if lo is not None and hi is not None and lo > hi:
        raise InvalidOutput(f"salary_min_eur ({lo}) is greater than salary_max_eur ({hi})")


@cache
def load_prompt(version: str) -> str:
    prompt = (PROMPTS / f"{version}.md").read_text(encoding="utf-8")
    # Cached, so this is logged once per version rather than on every call.
    logger.debug("system prompt %s:\n%s", version, prompt)
    return prompt


def make_client(settings: Settings) -> anthropic.Anthropic:
    key = settings.anthropic_api_key
    # Retries are handled by tenacity in _call, so the SDK's own retries are disabled.
    return anthropic.Anthropic(api_key=key.get_secret_value() if key else None, max_retries=0)


_DEFAULT_WAIT = wait_random_exponential(max=30)


def _transient(e: BaseException) -> bool:
    """Rate limits, network errors and 5xx are worth retrying; other 4xx are not."""
    if isinstance(e, anthropic.RateLimitError | anthropic.APIConnectionError):
        return True
    return isinstance(e, anthropic.APIStatusError) and e.status_code >= 500


def _response_text(resp) -> str:
    return "".join(block.text for block in resp.content if block.type == "text")


def _call(
    client: anthropic.Anthropic,
    settings: Settings,
    messages: list[dict],
    wait=_DEFAULT_WAIT,
    posting_id: str = "-",
):
    """Call the model with retries; return (response, attempts, latency in seconds)."""

    def log_retry(state: RetryCallState) -> None:
        logger.info(
            "posting=%s retry: attempt %d failed with %r, sleeping %.1fs",
            posting_id,
            state.attempt_number,
            state.outcome.exception(),
            state.next_action.sleep,
        )

    for message in messages:
        logger.debug("posting=%s %s prompt:\n%s", posting_id, message["role"], message["content"])
    t0 = time.perf_counter()
    for attempt in Retrying(
        retry=retry_if_exception(_transient),
        wait=wait,
        stop=stop_after_attempt(5),
        before_sleep=log_retry,
        reraise=True,
    ):
        with attempt:
            resp = client.messages.parse(
                model=settings.model,
                max_tokens=16000,
                system=[
                    {
                        "type": "text",
                        "text": load_prompt(settings.prompt_version),
                    }
                ],
                messages=messages,
                output_format=JobPosting,
                output_config={"effort": settings.effort},
            )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("posting=%s response text:\n%s", posting_id, _response_text(resp))
    return resp, attempt.retry_state.attempt_number, time.perf_counter() - t0


@dataclass
class _Tally:
    """Usage summed over the first call and the repair call."""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    api_attempts: int = 0


def _attempt(
    client: anthropic.Anthropic,
    settings: Settings,
    content: str,
    tally: _Tally,
    posting_id: str = "-",
) -> JobPosting:
    """One model call plus checks; raises ValidationError or InvalidOutput on bad output."""
    try:
        response, api_attempts, latency_s = _call(
            client=client,
            settings=settings,
            messages=[{"role": "user", "content": content}],
            posting_id=posting_id,
        )
    except ValidationError as e:
        tally.api_attempts += 1
        logger.info(
            "posting=%s parse failure: output did not match schema (%d errors)",
            posting_id,
            e.error_count(),
        )
        raise
    tally.model = response.model
    tally.input_tokens += response.usage.input_tokens
    tally.output_tokens += response.usage.output_tokens
    tally.api_attempts += api_attempts
    logger.info(
        "posting=%s model=%s prompt=%s latency=%.2fs input_tokens=%d output_tokens=%d "
        "api_attempts=%d stop_reason=%s",
        posting_id,
        response.model,
        settings.prompt_version,
        latency_s,
        response.usage.input_tokens,
        response.usage.output_tokens,
        api_attempts,
        response.stop_reason,
    )

    if response.stop_reason == "refusal":
        raise ExtractionError(f"model refused: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise ExtractionError("output truncated at max_tokens")
    if response.parsed_output is None:
        logger.info("posting=%s parse failure: no parsed output", posting_id)
        raise ExtractionError(f"no parsed output (stop_reason={response.stop_reason})")

    try:
        check_content(response.parsed_output)
    except InvalidOutput as e:
        logger.info("posting=%s parse failure: %s", posting_id, e)
        raise
    return response.parsed_output


def extract(
    posting: str,
    client: anthropic.Anthropic | None = None,
    settings: Settings | None = None,
    posting_id: str = "-",
) -> ExtractionResult:
    """Parse one posting (title line, location line, blank line, body) into a JobPosting.

    If the output fails validation, the model gets one more try with the errors attached.
    posting_id only labels log lines.
    """
    settings = settings or Settings()
    client = client or make_client(settings)

    content = f"<posting>\n{posting.strip()}\n</posting>"
    tally = _Tally()
    repaired = False
    start = time.perf_counter()
    try:
        parsed = _attempt(client, settings, content, tally, posting_id)
    except (ValidationError, InvalidOutput) as e:
        logger.warning("posting=%s output failed validation, retrying once: %s", posting_id, e)
        repaired = True
        feedback = (
            f"{content}\n\nA previous attempt failed validation: {e}. "
            "Return output that fixes these."
        )
        try:
            parsed = _attempt(client, settings, feedback, tally, posting_id)
        except (ValidationError, InvalidOutput) as e2:
            raise ExtractionError(f"output still invalid after repair attempt: {e2}") from e2
    latency_s = time.perf_counter() - start

    checked = check_salary(posting, parsed)
    logger.info(
        "posting=%s done: model=%s prompt=%s latency=%.2fs input_tokens=%d output_tokens=%d "
        "api_attempts=%d repaired=%s",
        posting_id,
        tally.model,
        settings.prompt_version,
        latency_s,
        tally.input_tokens,
        tally.output_tokens,
        tally.api_attempts,
        repaired or checked != parsed,
    )
    return ExtractionResult(
        posting=checked,
        model=tally.model,
        prompt_version=settings.prompt_version,
        input_tokens=tally.input_tokens,
        output_tokens=tally.output_tokens,
        latency_s=latency_s,
        api_attempts=tally.api_attempts,
        repaired=repaired or checked != parsed,
    )


def _p95(values: list[float]) -> float:
    """Nearest-rank 95th percentile."""
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


async def _extract_file(
    raw: Path, client: anthropic.Anthropic, settings: Settings, sem: asyncio.Semaphore
) -> ExtractionResult:
    async with sem:
        # extract is synchronous; run it in a worker thread so up to CONCURRENCY calls overlap.
        text = await asyncio.to_thread(raw.read_text, encoding="utf-8")
        return await asyncio.to_thread(extract, text, client, settings, raw.stem)


async def _extract_all(
    raws: list[Path], client: anthropic.Anthropic, settings: Settings
) -> list[ExtractionResult | BaseException]:
    sem = asyncio.Semaphore(CONCURRENCY)
    return await asyncio.gather(
        *(_extract_file(raw, client, settings, sem) for raw in raws), return_exceptions=True
    )


def main(labels_dir: Path = LABELS, raw_dir: Path = RAW) -> None:
    """Extract every posting that has a label in labels_dir and print run statistics."""
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    client = make_client(settings)
    paths = sorted(labels_dir.glob("*.json"))
    raws = [raw_dir / f"{label.stem}.txt" for label in paths]

    outcomes = asyncio.run(_extract_all(raws, client, settings))

    results: list[ExtractionResult] = []
    failures = 0
    for label, outcome in zip(paths, outcomes, strict=True):
        if isinstance(outcome, OSError | ExtractionError | anthropic.APIError):
            failures += 1
            print(f"{label.stem}: FAILED {type(outcome).__name__}: {outcome}", file=sys.stderr)
            continue
        if isinstance(outcome, BaseException):
            raise outcome
        results.append(outcome)
        print(f"{label.stem}: {outcome.posting.model_dump_json()}")

    repaired = sum(r.repaired for r in results)
    tokens = sum(r.input_tokens + r.output_tokens for r in results)
    print(f"\nfiles:        {len(paths)}")
    print(f"failures:     {failures}")
    if paths:
        print(f"repair rate:  {repaired / len(paths):.1%} ({repaired}/{len(paths)})")
    print(f"total tokens: {tokens}")
    if results:
        print(f"p95 latency:  {_p95([r.latency_s for r in results]):.2f}s")


if __name__ == "__main__":
    # uv run python -m job_extractor.extract
    main()
