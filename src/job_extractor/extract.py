"""Extract a structured JobPosting from a posting's plain text with Claude."""

import sys
from functools import cache
from pathlib import Path

import anthropic

from job_extractor.schema import JobPosting
from job_extractor.settings import Settings

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"


class ExtractionError(RuntimeError):
    pass


@cache
def load_prompt(version: str) -> str:
    return (PROMPTS / f"{version}.md").read_text(encoding="utf-8")


def make_client(settings: Settings) -> anthropic.Anthropic:
    key = settings.anthropic_api_key
    return anthropic.Anthropic(api_key=key.get_secret_value() if key else None)


def extract(
    posting: str,
    client: anthropic.Anthropic | None = None,
    settings: Settings | None = None,
) -> JobPosting:
    """Parse one posting (title line, location line, blank line, body) into a JobPosting."""
    settings = settings or Settings()
    client = client or make_client(settings)

    response = client.beta.messages.parse(
        model=settings.model,
        max_tokens=16000,
        # The system prompt is identical for every posting, so cache it.
        system=[
            {
                "type": "text",
                "text": load_prompt(settings.prompt_version),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": f"<posting>\n{posting.strip()}\n</posting>"}],
        output_format=JobPosting,
        output_config={"effort": settings.effort},
        # If a safety classifier declines, the API retries on a fallback model in the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    if response.stop_reason == "refusal":
        raise ExtractionError(f"model refused: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise ExtractionError("output truncated at max_tokens")
    if response.parsed_output is None:
        raise ExtractionError(f"no parsed output (stop_reason={response.stop_reason})")
    return response.parsed_output


if __name__ == "__main__":
    # uv run python -m job_extractor.extract data/raw/p001.txt [...]
    settings = Settings()
    client = make_client(settings)
    for path in sys.argv[1:]:
        result = extract(Path(path).read_text(encoding="utf-8"), client, settings)
        print(f"{path}: {result.model_dump_json()}")
