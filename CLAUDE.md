# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Job hunt assistant: collects job postings and extracts structured data from them with an LLM (Anthropic, OpenAI, Gemini, or a local model through llama-server). Python 3.12, managed with `uv` (src layout, `uv_build` backend).

## Commands

```sh
uv sync                                  # install deps, including the dev group
uv run pytest                            # all tests
uv run pytest tests/test_smoke.py::test_package_imports   # one test
uv run ruff check --fix . && uv run ruff format .         # lint + format
uv run pre-commit run --all-files        # all hooks (only checks git-tracked files)
uv run job-extractor                     # console entry point -> job_extractor:main
uv run python scripts/fetch_postings.py <greenhouse-board> [...]   # e.g. getyourguide
```

Ruff: line length 100, rules `E F I B UP` (see `pyproject.toml`). Pre-commit runs the generic `pre-commit-hooks` checks plus `ruff-check --fix` and `ruff-format`.

## Architecture

Extraction (`src/job_extractor/`):
- `extract.py` turns a posting's plain text into a `JobPosting` (`schema.py`): one structured-output LLM call with tenacity retries on transient errors, one repair attempt if the output fails schema or content checks, then `check_salary` drops salary values not found in the text. Run it on a split: `uv run python -m job_extractor.extract dev [--provider openai|gemini|llama] [--model NAME]`; `main` runs 5 postings concurrently (1 for llama).
- `llm.py` is the only module that imports a provider SDK. It defines the `LLMClient` protocol (`parse(system, user, schema) -> LLMResponse`), adapters for Anthropic (`messages.parse`), OpenAI (`responses.parse`) and Gemini (`google-genai` `models.generate_content` with a response schema; the adapter validates the JSON itself because the SDK's `parsed` swallows validation errors), llama (the `openai` SDK's `chat.completions.create` with a `json_schema` response format against a local llama-server; `parse` isn't used because it raises on a length stop; `reasoning_effort` is `LLAMA_REASONING` (none|low|medium|xhigh) if set, else `EFFORT` with high/xhigh/max mapped to `xhigh` because Bonsai 2 rejects `high`; default model `bonsai-2`), `DEFAULT_MODELS`, and `make_client`. Adapters disable SDK retries (tenacity is pinned `>=9.1` because google-genai requires `<9.2`) and wrap SDK errors in `LLMError(transient=...)`; stop reasons map to `ok | refusal | max_tokens | other`.
- `llama_server.py`: `extract.main` wraps llama runs in `llama_server(settings)`, which reuses a server already healthy on `LLAMA_PORT`, or starts `LLAMA_SERVER_BIN` with `LLAMA_MODEL_PATH` (aliased to the model name), polls `/health`, and terminates it afterwards. Bonsai 2's ternary GGUF needs PrismML's llama.cpp fork, not stock llama.cpp or Ollama.
- `settings.py` is `pydantic-settings` reading the git-ignored `.env`: `PROVIDER` (anthropic default), `MODEL` (None = provider default), API keys, `LLAMA_*` server settings, `EFFORT`, `PROMPT_VERSION`, `LOG_LEVEL`. CLI flags override it via `extract.load_settings`; switching provider on the CLI drops the `.env` model.
- System prompts are `prompts/<version>.md`. `api.py` (FastAPI) is still an empty stub; `__init__.py` has a placeholder `main()`.

Data ingestion currently lives outside the package, in `scripts/fetch_postings.py`:
- It pulls jobs from the public Greenhouse boards API, keeps titles that match an AI/DS keyword regex, and caps each company at 10 postings.
- Each posting's HTML is converted to plain text and written to `data/raw/pNNN.txt` (first line is the title, then the location, a blank line, then the body). `data/raw/` is git-ignored.
- One metadata record per posting is appended to `data/postings.jsonl` (`id, url, company, title, date, source`). This file is tracked. IDs are assigned sequentially from the current line count, and URLs already in the index are skipped, so re-running is idempotent per URL.
- Paths are relative, so run the script from the repo root. `beautifulsoup4` is only in the `dev` dependency group.

Labels are hand-made in `data/labels/pNNN.json` (`scripts/label_ui.py`, guide in `docs/LABELING_GUIDE.md`); `scripts/split_labels.py` writes the dev/test split to `data/splits.json`.
