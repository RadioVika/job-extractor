# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Job hunt assistant: collects job postings and (planned) extracts structured data from them with an LLM. Python 3.12, managed with `uv` (src layout, `uv_build` backend).

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

The pipeline is still early: `src/job_extractor/{api,extract,schema,settings}.py` exist as empty stubs, and `__init__.py` only defines a placeholder `main()`. The dependencies show the intended design: FastAPI/uvicorn for `api.py`, the Anthropic and OpenAI SDKs for LLM extraction in `extract.py`, Pydantic models in `schema.py`, `pydantic-settings` for configuration in `settings.py` (a git-ignored `.env` sits at the repo root), and tenacity for retries.

Data ingestion currently lives outside the package, in `scripts/fetch_postings.py`:
- It pulls jobs from the public Greenhouse boards API, keeps titles that match an AI/DS keyword regex, and caps each company at 10 postings.
- Each posting's HTML is converted to plain text and written to `data/raw/pNNN.txt` (first line is the title, then the location, a blank line, then the body). `data/raw/` is git-ignored.
- One metadata record per posting is appended to `data/postings.jsonl` (`id, url, company, title, date, source`). This file is tracked. IDs are assigned sequentially from the current line count, and URLs already in the index are skipped, so re-running is idempotent per URL.
- Paths are relative, so run the script from the repo root. `beautifulsoup4` is only in the `dev` dependency group.

`prompts/` exists but is empty, presumably for the extraction prompts.
