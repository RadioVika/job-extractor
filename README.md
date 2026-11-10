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

# Optional
PROVIDER=anthropic        # anthropic (default), openai or llama
MODEL=                    # empty = the provider's default model
EFFORT=medium             # low, medium, high, xhigh, max
LOG_LEVEL=INFO            # DEBUG also logs prompts and raw responses

# Local model (provider llama)
LLAMA_MODEL_PATH=         # path to the Bonsai 2 GGUF file
LLAMA_SERVER_BIN=llama-server   # PrismML's llama-server build
LLAMA_PORT=8080
LLAMA_CTX_SIZE=32768             # 65536 needs more than 16 GB of RAM
LLAMA_REASONING=                 # none, low, medium or xhigh; empty = follow EFFORT
```

## LLM providers

| Provider    | Default model     | Key                 |
|-------------|-------------------|---------------------|
| `anthropic` | `claude-opus-5-5` | `ANTHROPIC_API_KEY` |
| `openai`    | `gpt-6-luna`      | `OPENAI_API_KEY`    |
| `llama`     | `bonsai-2`        | none (local)        |

The provider and model come from (highest priority first):

1. the `--provider` / `--model` command-line flags
2. `PROVIDER` / `MODEL` in `.env` or the environment
3. the defaults: `anthropic` and that provider's default model

If `--provider` picks a different provider than the one in `.env`, the `MODEL` from `.env` is
ignored and that provider's default model is used, unless you also pass `--model`.

`EFFORT` maps to each provider's reasoning setting. For the local `llama` provider, `EFFORT` is sent as `reasoning_effort`. Bonsai 2 only accepts `low`, `medium`
and `xhigh`, so `high`, `xhigh` and `max` all mean `xhigh`. Its reasoning counts against the
output limit, and at `xhigh` it more often runs out before answering, so `medium` (the default)
is the recommended setting. `LLAMA_REASONING` overrides that mapping for the local model only;
`none` turns reasoning off, which is most of the run time (see below).

### Local model (llama)

The `llama` provider runs PrismML's Bonsai 2 27B (ternary `PQ2_0` pack, ~7.2 GB) on your
machine through `llama-server`. The ternary format doesn't load in stock llama.cpp or Ollama,
so you need PrismML's llama.cpp fork (`prism` branch) or its prebuilt binaries; see
[PrismML's llama.cpp guide](https://docs.prismml.com/run/llamacpp) and
[Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo) for the binary and the GGUF. Use the
`PQ2_0` GGUF, not the F16 file that Hugging Face's "Use this model" commands point at.

Set `LLAMA_MODEL_PATH` (and `LLAMA_SERVER_BIN` if `llama-server` isn't on your `PATH`), then:

```sh
uv run python -m job_extractor.extract dev --provider llama
```

The extractor starts `llama-server` on `LLAMA_PORT`, waits up to 5 minutes for the model to
load, runs the postings one at a time, and stops the server at the end. If a server is already
answering on that port, it is used as is and left running, which saves the load time across
runs.

On a 16 GB Mac (M1 Pro), Bonsai 2 27B takes about 4 minutes per posting with reasoning at
`medium`, about 80% of it spent generating reasoning tokens. With `LLAMA_REASONING=none` it
takes about 35-60 seconds, but check the results against the labels before relying on it:

```sh
LLAMA_REASONING=none uv run python -m job_extractor.extract dev --provider llama
```

If every request fails with `500 Compute error.`, the GPU is out of memory: lower
`LLAMA_CTX_SIZE` and close other memory-heavy apps.

## Running extraction

Runs the extractor on every posting in one split of `data/splits.json` and prints the parsed
postings plus run statistics (failures, repair rate, tokens, p95 latency):

```sh
uv run python -m job_extractor.extract dev                        # default provider, antropic
uv run python -m job_extractor.extract test --provider openai      # OpenAI, gpt-6-luna
uv run python -m job_extractor.extract test --provider anthropic --model claude-sonnet-5-5
```

## Running the API

```sh
uv run uvicorn job_extractor.api:app --reload    # http://127.0.0.1:8000, docs at /docs
```

`POST /extract` takes the posting text (up to 20,000 characters) and an optional `provider`
(`anthropic` by default, `openai`, or `llama` if a llama-server is already running on
`LLAMA_PORT`). The model and API keys come from `.env` as above.

```sh
curl -s -X POST http://127.0.0.1:8000/extract \
  -H 'Content-Type: application/json' \
  -d '{"text": "Senior ML Engineer\nBerlin (hybrid)\n\nRanking models in Python and PyTorch on GCP. Salary 80,000-95,000 EUR.", "provider": "anthropic"}'
# {"job_posting": {"role_type": "ml_engineer", "seniority": "senior", ...},
#  "usage": {"input_tokens": 4732, "output_tokens": 112}, "latency_ms": 2122.5}
```

Invalid input returns 422. If the provider fails (API, network or timeout error, refusal, or
output that is still invalid after the repair attempt), the API returns 502 with
`{"detail": "<provider> request failed: <reason>"}`. `GET /health` returns `{"status": "ok"}`.

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

## Evaluation

### Dataset

The evaluation set is 150 AI/data job postings (mostly Germany-based companies, pulled from
public job boards), each labeled by hand against the `JobPosting` schema. The labeling rules
for every field are in [docs/LABELING_GUIDE.md](docs/LABELING_GUIDE.md).

- `data/postings.jsonl` (tracked): one metadata record per posting (`id`, `url`, `company`,
  `title`, `date`, `source`).
- `data/labels/pNNN.json`: the gold labels.
- `data/splits.json`: a fixed split with a fixed seed: `dev` (100 postings), `test` (50) and
  `dev_short` (20 postings sampled from `dev`, for quick runs).
- `data/raw/pNNN.txt` (git-ignored): the posting texts. They are not published with the repo;
  they are available on request.

### Running an evaluation

```sh
uv run python -m evaluation.run dev                         # default provider and model
uv run python -m evaluation.run dev_short --provider openai
uv run python -m evaluation.run dev --predictions data/cache/<run>.jsonl   # re-score, no LLM calls
```

Each run writes its predictions to `data/cache/<time>_<split>_<provider>_<model>_<prompt>.jsonl`,
prints a one-row Markdown summary table, and writes `<same name>_metrics.json` with the summary
and the per-posting errors. It reports:

- Accuracy for `seniority`, `role_type`, `work_mode` and `german` (exact match).
- Skills: precision and recall, and the share of predicted skills not mentioned in the posting
  text ("invented").
- Salary: the share of predicted values with no matching amount in the text.
- Schema-valid rate, repair-retry rate, p50/p95 latency, and cost per 1000 postings (from
  `prices.yaml`).

### Labeling UI

```sh
uv run python scripts/label_ui.py        # http://127.0.0.1:8000
```

Shows the posting text next to the schema fields; clicking a value saves it to
`data/labels/pNNN.json`. Typing a new tag into a field's "+ new" box adds it to that field's enum
in `schema.py`. It uses port 8000, like the API, so don't run both at once.

### Error analysis UI

```sh
uv run python scripts/error_ui.py        # http://127.0.0.1:8001
```

Opens a run from `data/cache` (the newest by default; the header picks another) and shows each
posting in three panels: the text, the gold labels, and the model's prediction with its
per-posting metrics. Metrics are recomputed live against the current labels, and the gold labels
can be fixed in place (saved the same way as in the labeling UI), so label fixes show up in the
scores right away.
