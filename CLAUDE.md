# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Job hunt assistant: collects job postings and extracts structured data from them with an LLM (Anthropic, OpenAI, or a local model through llama-server). Python 3.12, managed with `uv` (src layout, `uv_build` backend).

## Commands

```sh
uv sync                                  # install deps, including the dev group
uv run pytest                            # all tests
uv run pytest tests/test_smoke.py::test_package_imports   # one test
uv run ruff check --fix . && uv run ruff format .         # lint + format
uv run pre-commit run --all-files        # all hooks (only checks git-tracked files)
uv run job-extractor                     # console entry point -> job_extractor:main
uv run python scripts/fetch_postings.py <greenhouse-board> [...]   # e.g. getyourguide
uv run python -m evaluation.run dev [--provider ...] [--model ...]   # extract + score a split
uv run python -m evaluation.run dev --predictions data/cache/<run>.jsonl   # re-score, no LLM
```

Ruff: line length 100, rules `E F I B UP` (see `pyproject.toml`). Pre-commit runs the generic `pre-commit-hooks` checks plus `ruff-check --fix` and `ruff-format`.

## Architecture

Extraction (`src/job_extractor/`):
- `extract.py` turns a posting's plain text into a `JobPosting` (`schema.py`): one structured-output LLM call with tenacity retries on transient errors, one repair attempt if the output fails schema or content checks. Salary values are not checked against the text here; evaluation reports that. Run it on a split: `uv run python -m job_extractor.extract dev [--provider openai|llama] [--model NAME]`; `main` runs 5 postings concurrently (1 for llama).
- `llm.py` is the only module that imports a provider SDK. It defines the `LLMClient` protocol (`parse(system, user, schema) -> LLMResponse`), adapters for Anthropic (`messages.parse`), OpenAI (`responses.parse`) and llama (the `openai` SDK's `chat.completions.create` with a `json_schema` response format against a local llama-server; `parse` isn't used because it raises on a length stop; `reasoning_effort` is `LLAMA_REASONING` (none|low|medium|xhigh) if set, else `EFFORT` with high/xhigh/max mapped to `xhigh` because Bonsai 2 rejects `high`; default model `bonsai-2`), `DEFAULT_MODELS`, and `make_client`. Adapters disable SDK retries and wrap SDK errors in `LLMError(transient=...)`; stop reasons map to `ok | refusal | max_tokens | other`.
- `llama_server.py`: `extract.main` wraps llama runs in `llama_server(settings)`, which reuses a server already healthy on `LLAMA_PORT`, or starts `LLAMA_SERVER_BIN` with `LLAMA_MODEL_PATH` (aliased to the model name), polls `/health`, and terminates it afterwards. Bonsai 2's ternary GGUF needs PrismML's llama.cpp fork, not stock llama.cpp or Ollama.
- `settings.py` is `pydantic-settings` reading the git-ignored `.env`: `PROVIDER` (anthropic default), `MODEL` (None = provider default), API keys, `LLAMA_*` server settings, `EFFORT`, `PROMPT_VERSION`, `LOG_LEVEL`. CLI flags override it via `extract.load_settings`; switching provider on the CLI drops the `.env` model.
- System prompts are `prompts/<version>.md`. `api.py` (FastAPI) is still an empty stub; `__init__.py` has a placeholder `main()`.

Data ingestion currently lives outside the package, in `scripts/fetch_postings.py`:
- It pulls jobs from the public Greenhouse boards API, keeps titles that match an AI/DS keyword regex, and caps each company at 10 postings.
- Each posting's HTML is converted to plain text and written to `data/raw/pNNN.txt` (first line is the title, then the location, a blank line, then the body). `data/raw/` is git-ignored.
- One metadata record per posting is appended to `data/postings.jsonl` (`id, url, company, title, date, source`). This file is tracked. IDs are assigned sequentially from the current line count, and URLs already in the index are skipped, so re-running is idempotent per URL.
- Paths are relative, so run the script from the repo root. `beautifulsoup4` is only in the `dev` dependency group.

Evaluation (`src/evaluation/`, a second package; `pyproject.toml` lists both in `module-name`):
- `run.py` extracts a split with `extract._extract_all`, writes one JSON line per posting (prediction or error) to the git-ignored `data/cache/<time>_<split>_<provider>_<model>_<prompt>.jsonl`, then scores it, prints a one-row Markdown table for the README, and writes `<same name>_metrics.json` next to it: a `summary` (the table's metrics, in the same order) followed by `errors` keyed by posting ID (extraction error, accuracy misses, missing/extra skills, skills and salary not in the text). Failed extractions are left out of the quality scores but count against the schema-valid rate.
- `metrics.score_run` scores the run records: schema-valid rate (postings that ended with a valid output / all postings), repair-retry rate, nearest-rank p50/p95 latency over the valid outputs, and USD per 1000 postings (mean token cost of a valid output × 1000; failed extractions don't record tokens).
- `prices.py` reads `prices.yaml` (repo root): entries of `model`, `date`, `price: {input, output}` in USD per 1M tokens; the latest entry not dated in the future wins. A test requires a price for every `DEFAULT_MODELS` entry.
- `metrics.py`: `seniority`, `role_type`, `work_mode` and `german` are scored by exact match (accuracy + misses; `ACCURACY_FIELDS`). Skills: micro and macro precision/recall, and two "invented" rates over predicted skills: not in the label (= 1 - micro precision) and not in the posting text.
- `salary.py`: `ungrounded_salary` finds predicted salary values with no matching amount in the text (as written, or monthly = value / 12); `score_salary` reports the share of predicted values not in the text.
- `grounding.py`: one broad regex per `Skill`, from the wording table in `docs/LABELING_GUIDE.md`, used for the not-in-text check. Add a pattern when adding a `Skill` (a test enforces it).

Labels are hand-made in `data/labels/pNNN.json` (`scripts/label_ui.py`, guide in `docs/LABELING_GUIDE.md`); `scripts/split_labels.py` writes the dev/test split to `data/splits.json`.

Error analysis: `uv run python scripts/error_ui.py` (http://127.0.0.1:8001) shows a run from `data/cache` (newest with records by default) as posting text | gold labels | prediction with per-posting metrics. Metrics are recomputed live against the current labels with the `evaluation` functions; gold edits go through `label_ui.apply_change`, so both UIs write labels the same way.
