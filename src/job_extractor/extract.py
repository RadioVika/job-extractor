"""Extract a structured JobPosting from a posting's plain text with LLM."""

import logging
import re
import sys
from functools import cache
from pathlib import Path

import anthropic

from job_extractor.schema import JobPosting
from job_extractor.settings import Settings

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"

logger = logging.getLogger(__name__)

# Amounts as written in postings: "90,000", "100.000", "100 000", "85000", "100k", "85.5k".
# Thousands groups need exactly three digits, so "8.50" is read as 8 and 50, not 850.
_NUMBER = re.compile(
    r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s?[kK]\b"
    r"|(?<!\d)(\d{1,3}(?:[.,\s  ]\d{3})+|\d+)(?!\d)"
)


class ExtractionError(RuntimeError):
    pass


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


@cache
def load_prompt(version: str) -> str:
    return (PROMPTS / f"{version}.md").read_text(encoding="utf-8")


def make_client(settings: Settings) -> anthropic.Anthropic:
    key = settings.anthropic_api_key
    return anthropic.Anthropic(api_key=key.get_secret_value() if key else None)


def extract(
    posting: str, client: anthropic.Anthropic | None = None, settings: Settings | None = None
) -> JobPosting:
    """Parse one posting (title line, location line, blank line, body) into a JobPosting."""
    settings = settings or Settings()
    client = client or make_client(settings)

    response = client.messages.parse(
        model=settings.model,
        max_tokens=16000,
        system=[
            {
                "type": "text",
                "text": load_prompt(settings.prompt_version),
            }
        ],
        messages=[{"role": "user", "content": f"<posting>\n{posting.strip()}\n</posting>"}],
        output_format=JobPosting,
        output_config={"effort": settings.effort},
    )

    if response.stop_reason == "refusal":
        raise ExtractionError(f"model refused: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise ExtractionError("output truncated at max_tokens")
    if response.parsed_output is None:
        raise ExtractionError(f"no parsed output (stop_reason={response.stop_reason})")
    return check_salary(posting, response.parsed_output)


if __name__ == "__main__":
    # uv run python -m job_extractor.extract data/raw/p001.txt [...]
    settings = Settings()
    client = make_client(settings)
    for path in sys.argv[1:]:
        result = extract(Path(path).read_text(encoding="utf-8"), client, settings)
        print(f"{path}: {result.model_dump_json()}")
