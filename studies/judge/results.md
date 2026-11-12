# Judge validation results

One section per `python -m evaluation.judge_run`, newest last. Written by
`src/evaluation/judge_run.py`. Item-level tables: positive class is "prediction is wrong";
Error P/R are the judge's error-detection precision and recall against manual gold.

## 2026-11-12 17:13 · judge claude-opus-5-5 (high) · judge_v1

- Git commit: 7fdf18b (uncommitted changes)

Report from manual gold vs from judge verdicts:

| Metric                  | openai/gpt-6-luna gold |  judge |  Δ pp | anthropic/claude-haiku-5-5 gold |  judge |  Δ pp | Same ranking |
| ----------------------- | ---------------------: | -----: | ----: | ------------------------------: | -----: | ----: | -----------: |
| seniority_accuracy      |                  90.0% | 100.0% | +10.0 |                           90.0% | 100.0% | +10.0 |          yes |
| role_type_accuracy      |                  80.0% |  90.0% | +10.0 |                           90.0% |  95.0% |  +5.0 |          yes |
| work_mode_accuracy      |                  90.0% |  95.0% |  +5.0 |                           90.0% |  95.0% |  +5.0 |          yes |
| german_accuracy         |                 100.0% | 100.0% |  +0.0 |                          100.0% | 100.0% |  +0.0 |          yes |
| salary_min_eur_accuracy |                 100.0% | 100.0% |  +0.0 |                          100.0% | 100.0% |  +0.0 |          yes |
| salary_max_eur_accuracy |                 100.0% | 100.0% |  +0.0 |                          100.0% | 100.0% |  +0.0 |          yes |
| skills_precision        |                  93.3% |  99.4% |  +6.1 |                           95.7% | 100.0% |  +4.3 |          yes |
| skills_recall           |                  97.5% |  97.6% |  +0.2 |                           98.7% |  97.0% |  -1.7 |           NO |
| clouds_precision        |                  92.3% | 100.0% |  +7.7 |                           92.3% | 100.0% |  +7.7 |          yes |
| clouds_recall           |                 100.0% | 100.0% |  +0.0 |                          100.0% | 100.0% |  +0.0 |          yes |

### openai/gpt-6-luna

- Predictions: `20261112-093440_dev_short_openai_gpt-6-luna_v2.jsonl`
- Verdicts: `20261112-093440_dev_short_openai_gpt-6-luna_v2__claude-opus-5-5_judge_v1.jsonl`
- Scored postings: 20 (failed extractions: 0, failed verdicts: 0)
- Judge: claude-opus-5-5, $57.61 per 1000 postings, latency p50 10.24 s, p95 12.39 s, repair retries 0

| Field            | Items | Judge: wrong | Gold: wrong |    κ | Error P | Error R |  Agree |
| ---------------- | ----: | -----------: | ----------: | ---: | ------: | ------: | -----: |
| seniority        |    20 |            0 |           2 | 0.00 |     n/a |    0.0% |  90.0% |
| role_type        |    20 |            2 |           4 | 0.62 |  100.0% |   50.0% |  90.0% |
| work_mode        |    20 |            1 |           2 | 0.64 |  100.0% |   50.0% |  95.0% |
| german           |    20 |            0 |           0 |  n/a |     n/a |     n/a | 100.0% |
| salary_min_eur   |    20 |            0 |           0 |  n/a |     n/a |     n/a | 100.0% |
| salary_max_eur   |    20 |            0 |           0 |  n/a |     n/a |     n/a | 100.0% |
| skills           |   165 |            1 |          11 | 0.16 |  100.0% |    9.1% |  93.9% |
| clouds           |    13 |            0 |           1 | 0.00 |     n/a |    0.0% |  92.3% |
| skills (missing) |       |            4 |           4 |      |   75.0% |   75.0% |        |
| clouds (missing) |       |            0 |           0 |      |     n/a |     n/a |        |

### anthropic/claude-haiku-5-5

- Predictions: `20261112-093525_dev_short_anthropic_claude-haiku-5-5_v2.jsonl`
- Verdicts: `20261112-093525_dev_short_anthropic_claude-haiku-5-5_v2__claude-opus-5-5_judge_v1.jsonl`
- Scored postings: 20 (failed extractions: 0, failed verdicts: 0)
- Judge: claude-opus-5-5, $57.80 per 1000 postings, latency p50 9.71 s, p95 13.03 s, repair retries 0

| Field            | Items | Judge: wrong | Gold: wrong |    κ | Error P | Error R |  Agree |
| ---------------- | ----: | -----------: | ----------: | ---: | ------: | ------: | -----: |
| seniority        |    20 |            0 |           2 | 0.00 |     n/a |    0.0% |  90.0% |
| role_type        |    20 |            1 |           2 | 0.64 |  100.0% |   50.0% |  95.0% |
| work_mode        |    20 |            1 |           2 | 0.64 |  100.0% |   50.0% |  95.0% |
| german           |    20 |            0 |           0 |  n/a |     n/a |     n/a | 100.0% |
| salary_min_eur   |    20 |            0 |           0 |  n/a |     n/a |     n/a | 100.0% |
| salary_max_eur   |    20 |            0 |           0 |  n/a |     n/a |     n/a | 100.0% |
| skills           |   163 |            0 |           7 | 0.00 |     n/a |    0.0% |  95.7% |
| clouds           |    13 |            0 |           1 | 0.00 |     n/a |    0.0% |  92.3% |
| skills (missing) |       |            5 |           2 |      |   20.0% |   50.0% |        |
| clouds (missing) |       |            0 |           0 |      |     n/a |     n/a |        |
