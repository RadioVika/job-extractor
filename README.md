# job-extractor
Job hunt assistant: pulls job postings and extracts structured data from them (role type,
seniority, skills, salary...) with an LLM.

## Setup

```sh
uv sync
```

Put API keys in a `.env` file at the repo root (git-ignored). You only need the key for the
provider you use:

```sh
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...

# Optional
PROVIDER=anthropic        # anthropic (default), openai or gemini
MODEL=                    # empty = the provider's default model
EFFORT=medium             # low, medium, high, xhigh, max
LOG_LEVEL=INFO            # DEBUG also logs prompts and raw responses
```

## LLM providers

| Provider    | Default model     | Key                 |
|-------------|-------------------|---------------------|
| `anthropic` | `claude-opus-5-5` | `ANTHROPIC_API_KEY` |
| `openai`    | `gpt-6-luna`      | `OPENAI_API_KEY`    |
| `gemini`    | `gemini-4-argon`  | `GOOGLE_API_KEY`    |

The provider and model come from (highest priority first):

1. the `--provider` / `--model` command-line flags
2. `PROVIDER` / `MODEL` in `.env` or the environment
3. the defaults: `anthropic` and that provider's default model

If `--provider` picks a different provider than the one in `.env`, the `MODEL` from `.env` is
ignored and that provider's default model is used, unless you also pass `--model`.

`EFFORT` maps to each provider's reasoning setting. Gemini's thinking levels stop at `HIGH`,
so `high`, `xhigh` and `max` all mean `HIGH` there. `gemini-4-argon` has no free-tier quota;
on a free Google key use a Flash model, e.g. `--model gemini-3.8-flash`.

## Running extraction

Runs the extractor on every posting in one split of `data/splits.json` and prints the parsed
postings plus run statistics (failures, repair rate, tokens, p95 latency):

```sh
uv run python -m job_extractor.extract dev                        # default provider, antropic
uv run python -m job_extractor.extract test --provider openai      # OpenAI, gpt-6-luna
uv run python -m job_extractor.extract test --provider gemini --model gemini-3.8-flash
uv run python -m job_extractor.extract test --provider anthropic --model claude-sonnet-5-5
```

### Adding a provider

All provider code lives in `src/job_extractor/llm.py`; the rest of the app only uses the
`LLMClient` interface. To add one:

1. Write an adapter class with a `parse(system, user, schema) -> LLMResponse` method that
   calls the official SDK with its retries turned off, and wraps the SDK's API errors in
   `LLMError(transient=...)`.
2. Add the provider name to `Provider` in `settings.py` (plus an API key field), an entry in
   `DEFAULT_MODELS`, and a branch in `make_client`.

## Development

```sh
uv run pytest
uv run ruff check --fix . && uv run ruff format .
```

# Testing Dataset
Raw data is available per request
