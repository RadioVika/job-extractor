"""Extract a structured JobPosting from a posting's plain text with LLM."""

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
    return (PROMPTS / f"{version}.md").read_text(encoding="utf-8")


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


def _call(
    client: anthropic.Anthropic,
    settings: Settings,
    messages: list[dict],
    wait=_DEFAULT_WAIT,
):
    """Call the model with retries; return (response, attempts, latency in seconds)."""
    t0 = time.perf_counter()
    for attempt in Retrying(
        retry=retry_if_exception(_transient),
        wait=wait,
        stop=stop_after_attempt(5),
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

    return resp, attempt.retry_state.attempt_number, time.perf_counter() - t0


@dataclass
class _Tally:
    """Usage summed over the first call and the repair call."""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    api_attempts: int = 0


def _attempt(
    client: anthropic.Anthropic, settings: Settings, content: str, tally: _Tally
) -> JobPosting:
    """One model call plus checks; raises ValidationError or InvalidOutput on bad output."""
    try:
        response, api_attempts, _ = _call(
            client=client, settings=settings, messages=[{"role": "user", "content": content}]
        )
    except ValidationError:
        tally.api_attempts += 1
        raise
    tally.model = response.model
    tally.input_tokens += response.usage.input_tokens
    tally.output_tokens += response.usage.output_tokens
    tally.api_attempts += api_attempts

    if response.stop_reason == "refusal":
        raise ExtractionError(f"model refused: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise ExtractionError("output truncated at max_tokens")
    if response.parsed_output is None:
        raise ExtractionError(f"no parsed output (stop_reason={response.stop_reason})")

    check_content(response.parsed_output)
    return response.parsed_output


def extract(
    posting: str, client: anthropic.Anthropic | None = None, settings: Settings | None = None
) -> ExtractionResult:
    """Parse one posting (title line, location line, blank line, body) into a JobPosting.

    If the output fails validation, the model gets one more try with the errors attached.
    """
    settings = settings or Settings()
    client = client or make_client(settings)

    content = f"<posting>\n{posting.strip()}\n</posting>"
    tally = _Tally()
    repaired = False
    start = time.perf_counter()
    try:
        parsed = _attempt(client, settings, content, tally)
    except (ValidationError, InvalidOutput) as e:
        logger.warning("output failed validation, retrying once: %s", e)
        repaired = True
        feedback = (
            f"{content}\n\nA previous attempt failed validation: {e}. "
            "Return output that fixes these."
        )
        try:
            parsed = _attempt(client, settings, feedback, tally)
        except (ValidationError, InvalidOutput) as e2:
            raise ExtractionError(f"output still invalid after repair attempt: {e2}") from e2
    latency_s = time.perf_counter() - start

    checked = check_salary(posting, parsed)
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


def main(labels_dir: Path = LABELS, raw_dir: Path = RAW) -> None:
    """Extract every posting that has a label in labels_dir and print run statistics."""
    settings = Settings()
    client = make_client(settings)
    paths = sorted(labels_dir.glob("*.json"))

    results: list[ExtractionResult] = []
    failures = 0
    for label in paths:
        raw = raw_dir / f"{label.stem}.txt"
        try:
            result = extract(raw.read_text(encoding="utf-8"), client, settings)
        except (OSError, ExtractionError, anthropic.APIError) as e:
            failures += 1
            print(f"{label.stem}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            continue
        results.append(result)
        print(f"{label.stem}: {result.posting.model_dump_json()}")

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
