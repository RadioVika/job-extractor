"""Extract a structured JobPosting from a posting's plain text with LLM."""

import argparse
import asyncio
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import get_args

from pydantic import ValidationError
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from job_extractor.llm import LLMClient, LLMError, LLMResponse, make_client
from job_extractor.schema import JobPosting
from job_extractor.settings import Provider, Settings

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "prompts"
LABELS = ROOT / "data" / "labels"
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits.json"
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
    provider: str
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


_DEFAULT_WAIT = wait_random_exponential(max=30)


def _transient(e: BaseException) -> bool:
    return isinstance(e, LLMError) and e.transient


def _call(
    client: LLMClient,
    settings: Settings,
    content: str,
    wait=_DEFAULT_WAIT,
    posting_id: str = "-",
) -> tuple[LLMResponse, int, float]:
    """Call the model with retries; return (response, attempts, latency in seconds)."""

    def log_retry(state: RetryCallState) -> None:
        logger.info(
            "posting=%s retry: attempt %d failed with %r, sleeping %.1fs",
            posting_id,
            state.attempt_number,
            state.outcome.exception(),
            state.next_action.sleep,
        )

    logger.debug("posting=%s user prompt:\n%s", posting_id, content)
    t0 = time.perf_counter()
    for attempt in Retrying(
        retry=retry_if_exception(_transient),
        wait=wait,
        stop=stop_after_attempt(5),
        before_sleep=log_retry,
        reraise=True,
    ):
        with attempt:
            resp = client.parse(load_prompt(settings.prompt_version), content, JobPosting)

    logger.debug("posting=%s response text:\n%s", posting_id, resp.text)
    return resp, attempt.retry_state.attempt_number, time.perf_counter() - t0


@dataclass
class _Tally:
    """Usage summed over the first call and the repair call."""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    api_attempts: int = 0


def _attempt(
    client: LLMClient,
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
            content=content,
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
    tally.input_tokens += response.input_tokens
    tally.output_tokens += response.output_tokens
    tally.api_attempts += api_attempts
    logger.info(
        "posting=%s provider=%s model=%s prompt=%s latency=%.2fs input_tokens=%d "
        "output_tokens=%d api_attempts=%d stop=%s",
        posting_id,
        client.provider,
        response.model,
        settings.prompt_version,
        latency_s,
        response.input_tokens,
        response.output_tokens,
        api_attempts,
        response.detail or response.stop,
    )

    if response.stop == "refusal":
        raise ExtractionError(f"model refused: {response.detail}")
    if response.stop == "max_tokens":
        raise ExtractionError("output truncated at max_tokens")
    if response.parsed is None:
        logger.info("posting=%s parse failure: no parsed output", posting_id)
        raise ExtractionError(f"no parsed output (stop={response.detail or response.stop})")

    try:
        check_content(response.parsed)
    except InvalidOutput as e:
        logger.info("posting=%s parse failure: %s", posting_id, e)
        raise
    return response.parsed


def extract(
    posting: str,
    client: LLMClient | None = None,
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
        "posting=%s done: provider=%s model=%s prompt=%s latency=%.2fs input_tokens=%d "
        "output_tokens=%d api_attempts=%d repaired=%s",
        posting_id,
        client.provider,
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
        provider=client.provider,
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
    raw: Path, client: LLMClient, settings: Settings, sem: asyncio.Semaphore
) -> ExtractionResult:
    async with sem:
        # extract is synchronous; run it in a worker thread so up to CONCURRENCY calls overlap.
        text = await asyncio.to_thread(raw.read_text, encoding="utf-8")
        return await asyncio.to_thread(extract, text, client, settings, raw.stem)


async def _extract_all(
    raws: list[Path], client: LLMClient, settings: Settings
) -> list[ExtractionResult | BaseException]:
    sem = asyncio.Semaphore(CONCURRENCY)
    return await asyncio.gather(
        *(_extract_file(raw, client, settings, sem) for raw in raws), return_exceptions=True
    )


def load_split(split: str, splits_file: Path = SPLITS) -> list[str]:
    """Return the posting IDs in one split ("dev" or "test") of splits_file."""
    splits = json.loads(splits_file.read_text(encoding="utf-8"))
    if split not in splits or split == "seed":
        raise ValueError(f"unknown split {split!r}; expected one of: dev, test")
    return splits[split]


def load_settings(provider: str | None = None, model: str | None = None) -> Settings:
    """Settings from .env and the environment, with CLI values (if given) taking precedence.

    Switching provider without naming a model uses that provider's default model, so a MODEL
    from .env meant for one provider is never sent to another.
    """
    settings = Settings()
    update = {}
    if provider and provider != settings.provider:
        update |= {"provider": provider, "model": None}
    if model:
        update["model"] = model
    return Settings(**(settings.model_dump() | update)) if update else settings


def main(
    split: str,
    provider: str | None = None,
    model: str | None = None,
    labels_dir: Path = LABELS,
    raw_dir: Path = RAW,
    splits_file: Path = SPLITS,
) -> None:
    """Extract every labeled posting in the given split and print run statistics."""
    settings = load_settings(provider, model)
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    client = make_client(settings)
    paths = [labels_dir / f"{posting_id}.json" for posting_id in load_split(split, splits_file)]
    raws = [raw_dir / f"{label.stem}.txt" for label in paths]

    outcomes = asyncio.run(_extract_all(raws, client, settings))

    results: list[ExtractionResult] = []
    failures = 0
    for label, outcome in zip(paths, outcomes, strict=True):
        if isinstance(outcome, OSError | ExtractionError | LLMError):
            failures += 1
            print(f"{label.stem}: FAILED {type(outcome).__name__}: {outcome}", file=sys.stderr)
            continue
        if isinstance(outcome, BaseException):
            raise outcome
        results.append(outcome)
        print(f"{label.stem}: {outcome.posting.model_dump_json()}")

    repaired = sum(r.repaired for r in results)
    tokens = sum(r.input_tokens + r.output_tokens for r in results)
    print(f"\nprovider:     {client.provider}")
    print(f"model:        {client.model}")
    print(f"split:        {split}")
    print(f"files:        {len(paths)}")
    print(f"failures:     {failures}")
    if paths:
        print(f"repair rate:  {repaired / len(paths):.1%} ({repaired}/{len(paths)})")
    print(f"total tokens: {tokens}")
    if results:
        print(f"p95 latency:  {_p95([r.latency_s for r in results]):.2f}s")


if __name__ == "__main__":
    # uv run python -m job_extractor.extract dev --provider openai
    parser = argparse.ArgumentParser(description="Run the extractor on one split of the labels.")
    parser.add_argument("split", choices=["dev", "test"], help="which split from data/splits.json")
    parser.add_argument(
        "--provider",
        choices=list(get_args(Provider)),
        help="LLM provider (default: PROVIDER from .env, else anthropic)",
    )
    parser.add_argument(
        "--model", help="model name (default: MODEL from .env, else the provider's default)"
    )
    args = parser.parse_args()
    main(args.split, provider=args.provider, model=args.model)
