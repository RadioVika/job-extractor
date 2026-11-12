# Can an LLM replace manual gold labels?

## Question

Can the manual labeling process be replaced by a model? Completely or per field?

## Approaches

1. **Field-by-field validation (judge).** The judge sees the posting text and a prediction and
   answers `correct: yes/no` for each field. For `skills` and `clouds` it answers per predicted
   item and lists the items that are missing. It never sees the gold label.
2. **Gold regeneration.** An expensive model labels the postings from scratch with the labeling
   guide, without seeing any prediction. Its labels are then used as gold.

## Data

Only a small subset of postings, because every disagreement has to be re-checked by hand. All
steps use the same subset.

## Steps

1. **Rerun the extractor with two providers** (e.g. Anthropic and OpenAI) on the subset, using
   `evaluation.run`.
2. **Error analysis and gold update.** Go through the errors in `scripts/error_ui.py` and fix the
   gold labels that are wrong. Do this before any judge output exists, so the fixes aren't
   influenced by the model. These updated labels are the reference for every step below.
3. **Judge validation.** Ask the judge to validate both runs from step 1.
   - Compare its verdicts with manual gold (`pred == gold`) per field: Cohen's κ and
     error-detection precision/recall (the positive class is "prediction is wrong").
   - Compute the standard report metrics (field accuracy, skills precision/recall, salary) from
     the judge verdicts, and put them next to the same metrics from manual gold.
4. **Gold regeneration.** Generate labels with the expensive model and compare them with manual
   gold per field: accuracy and κ for enum fields, precision/recall for `skills` and `clouds`,
   exact match for salary. Re-check every disagreement by hand: count separately the cases where
   the model was wrong and the cases where manual gold was wrong.
5. **Metrics on LLM gold.** Score both runs from step 1 against the LLM-generated gold with the
   standard `score_run` report, and compare with the report on manual gold.

## Metrics

- **Item level** (steps 3 and 4): per field, κ against manual gold, plus error-detection
  precision/recall for the judge. Raw agreement is shown only for context, since most predictions
  are correct.
- **Report level** (steps 3 and 5): per field, the difference in percentage points between the
  report from the model and the report from manual gold, and whether both rank the two providers
  the same way.
- **Cost**: USD per 1000 postings for the judge and the label generator (`evaluation/prices.py`).

## How to answer the question

Manual labeling can be replaced for a field if both hold:
- the model agrees with manual gold on that field (high κ, and most disagreements in step 4
  turn out to be manual-gold errors on re-check), and
- the report computed from the model (step 3 or 5) is close to the manual one and ranks the
  providers the same way.

## Running step 3

```sh
uv run python -m evaluation.judge_run \
    data/cache/20261112-093440_dev_short_openai_gpt-6-luna_v2.jsonl \
    data/cache/20261112-093525_dev_short_anthropic_claude-haiku-5-5_v2.jsonl
uv run python -m evaluation.judge_run <runs...> --rescore   # re-score cached verdicts, no LLM
```

- Judge: `claude-opus-5-5` at `high` effort by default (`--provider`, `--model`, `--effort`).
  Its system prompt is `prompts/judge_v1.md` + `docs/LABELING_GUIDE.md` + the `JobPosting` JSON
  schema; the user message is the posting text and the prediction. It never sees gold.
- Verdicts: `data/cache/judge/<run>__<judge-model>_judge_v1.jsonl`, one line per posting, with a
  reason for every verdict. Postings already judged are skipped on the next run, so a partial
  run resumes. Metrics and per-posting disagreements go to `<same name>_metrics.json`.
- Results (report from gold vs judge with Δ pp and the ranking check, plus per-field κ and
  error-detection precision/recall) are printed and appended to `results.md`.
- Code: `src/evaluation/judge.py` (prompt, verdict schema, calls), `judge_metrics.py` (pure
  scoring), `judge_run.py` (CLI and report).
