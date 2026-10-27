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

# Metrics

| Model                      | Skills precision | Skills recall | Skills invented | Seniority accuracy | Role type accuracy | Work mode accuracy | German accuracy | Schema-valid rate | Repair-retry rate | p50 latency | p95 latency | Cost / 1000 postings |
| -------------------------- | ---------------- | ------------- | --------------- | ------------------ | ------------------ | ------------------ | --------------- | ----------------- | ----------------- | ----------- | ----------- | -------------------- |
| openai/gpt-6-luna          |            74.6% |         89.9% |            6.3% |              78.0% |              86.0% |              83.0% |           90.0% |            100.0% |              0.0% |      4.82 s |      7.84 s |                $0.53 |
| anthropic/claude-haiku-5-5 |            77.0% |         88.3% |            3.1% |              78.0% |              85.0% |              81.0% |           90.0% |            100.0% |              1.0% |      5.32 s |      7.34 s |                $1.10 |
| llama/bonsai-2             |            66.1% |         58.7% |           17.3% |              66.0% |              80.0% |              70.0% |           85.0% |            100.0% |              0.0% |      9.06 s |     12.43 s |                $0.00 |
