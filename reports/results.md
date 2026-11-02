# Evaluation results

One section per `python -m evaluation.run`, newest last. Written by `src/evaluation/run.py`.

## 2026-11-02 10:56 · dev · openai/gpt-6-luna

- Date: 2026-11-02 10:56
- Git commit: 0d84185 (uncommitted changes)
- Prompt version: v2
- Provider: openai
- Model: gpt-6-luna
- Dataset: dev
- Postings: 100
- Price date (prices.yaml): 2026-10-27
- Predictions: `20261102-105612_dev_openai_gpt-6-luna_v2.jsonl`

| Model             | Skills precision | Skills recall | Skills invented | Seniority accuracy | Role type accuracy | Work mode accuracy | German accuracy | Schema-valid rate | Repair-retry rate | p50 latency | p95 latency | Cost / 1000 postings |
| ----------------- | ---------------: | ------------: | --------------: | -----------------: | -----------------: | -----------------: | --------------: | ----------------: | ----------------: | ----------: | ----------: | -------------------: |
| openai/gpt-6-luna |            79.5% |         97.0% |            2.3% |              79.0% |              86.0% |              86.0% |           90.0% |            100.0% |              0.0% |      4.92 s |      8.08 s |                $0.54 |

## 2026-11-02 10:58 · dev · anthropic/claude-haiku-5-5

- Date: 2026-11-02 10:58
- Git commit: 0d84185 (uncommitted changes)
- Prompt version: v2
- Provider: anthropic
- Model: claude-haiku-5-5
- Dataset: dev
- Postings: 100
- Price date (prices.yaml): 2026-10-27
- Predictions: `20261102-105842_dev_anthropic_claude-haiku-5-5_v2.jsonl`

| Model                      | Skills precision | Skills recall | Skills invented | Seniority accuracy | Role type accuracy | Work mode accuracy | German accuracy | Schema-valid rate | Repair-retry rate | p50 latency | p95 latency | Cost / 1000 postings |
| -------------------------- | ---------------: | ------------: | --------------: | -----------------: | -----------------: | -----------------: | --------------: | ----------------: | ----------------: | ----------: | ----------: | -------------------: |
| anthropic/claude-haiku-5-5 |            80.5% |         94.5% |            2.6% |              78.0% |              89.0% |              80.0% |           90.0% |            100.0% |              0.0% |      4.79 s |      6.42 s |                $1.10 |
