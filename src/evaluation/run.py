"""Run the extractor on a split, save the predictions, and score them against the labels.

uv run python -m evaluation.run dev [--provider openai] [--model NAME]
uv run python -m evaluation.run dev --predictions data/cache/<run>.jsonl   # re-score, no LLM
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import get_args

from evaluation.metrics import (
    ACCURACY_FIELDS,
    AccuracyReport,
    Example,
    RunReport,
    SalaryReport,
    SkillsReport,
    score_accuracy,
    score_run,
    score_salary,
    score_skills,
)
from evaluation.prices import price_for
from job_extractor.extract import (
    CONCURRENCY,
    LABELS,
    LLAMA_CONCURRENCY,
    RAW,
    ROOT,
    SPLITS,
    ExtractionError,
    _extract_all,
    load_settings,
    load_split,
)
from job_extractor.llama_server import llama_server
from job_extractor.llm import LLMError, make_client
from job_extractor.schema import JobPosting
from job_extractor.settings import Provider

CACHE = ROOT / "data" / "cache"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-")


def run_predictions(
    split: str,
    provider: str | None = None,
    model: str | None = None,
    raw_dir: Path = RAW,
    splits_file: Path = SPLITS,
    cache_dir: Path = CACHE,
) -> Path:
    """Extract every posting in the split and write one JSON line per posting to cache_dir."""
    settings = load_settings(provider, model)
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    client = make_client(settings)
    ids = load_split(split, splits_file)
    raws = [raw_dir / f"{posting_id}.txt" for posting_id in ids]

    local = settings.provider == "llama"
    with llama_server(settings) if local else nullcontext():
        concurrency = LLAMA_CONCURRENCY if local else CONCURRENCY
        outcomes = asyncio.run(_extract_all(raws, client, settings, concurrency))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{split}_{client.provider}_{_slug(client.model)}_{settings.prompt_version}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{name}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for posting_id, outcome in zip(ids, outcomes, strict=True):
            record: dict = {"id": posting_id}
            if isinstance(outcome, OSError | ExtractionError | LLMError):
                record["error"] = f"{type(outcome).__name__}: {outcome}"
                print(f"{posting_id}: FAILED {record['error']}", file=sys.stderr)
            elif isinstance(outcome, BaseException):
                raise outcome
            else:
                record |= {
                    "posting": outcome.posting.model_dump(mode="json"),
                    "provider": outcome.provider,
                    "model": outcome.model,
                    "prompt_version": outcome.prompt_version,
                    "input_tokens": outcome.input_tokens,
                    "output_tokens": outcome.output_tokens,
                    "latency_s": round(outcome.latency_s, 3),
                    "repaired": outcome.repaired,
                }
            f.write(json.dumps(record) + "\n")
    return out


def load_records(predictions: Path) -> list[dict]:
    lines = predictions.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_examples(
    predictions: Path, labels_dir: Path = LABELS, raw_dir: Path = RAW
) -> tuple[list[Example], dict[str, str]]:
    """Pair each saved prediction with its label and text; also return failed ID -> error."""
    examples, failed = [], {}
    for record in load_records(predictions):
        posting_id = record["id"]
        if "posting" not in record:
            failed[posting_id] = record.get("error", "")
            continue
        examples.append(
            Example(
                posting_id=posting_id,
                gold=JobPosting.model_validate_json(
                    (labels_dir / f"{posting_id}.json").read_text(encoding="utf-8")
                ),
                pred=JobPosting.model_validate(record["posting"]),
                text=(raw_dir / f"{posting_id}.txt").read_text(encoding="utf-8"),
            )
        )
    return examples, failed


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _secs(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} s"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.2f}"


def report(
    run: RunReport, accuracy: list[AccuracyReport], skills: SkillsReport, salary: SalaryReport
) -> str:
    lines = [
        f"scored postings:   {skills.n} (failed extractions: {len(run.failed)})",
        f"schema-valid:      {_pct(run.schema_valid_rate)} ({run.n - len(run.failed)}/{run.n})",
        f"repair retries:    {_pct(run.repair_retry_rate)} ({run.repaired}/{run.n})",
        f"latency:           p50 {_secs(run.latency_p50)}, p95 {_secs(run.latency_p95)}",
        f"cost:              {_usd(run.cost_per_1000)} per 1000 postings",
        "",
        "accuracy",
    ]
    lines += [f"  {r.field + ':':<17}{_pct(r.accuracy)} ({r.correct}/{r.n})" for r in accuracy]
    lines += [
        "",
        f"skills             ({skills.predicted} predicted, {skills.tp + skills.fn} in labels)",
        f"  precision:       {_pct(skills.micro_precision)} micro, "
        f"{_pct(skills.macro_precision)} macro",
        f"  recall:          {_pct(skills.micro_recall)} micro, {_pct(skills.macro_recall)} macro",
        f"  invented, not in label: {_pct(skills.invented_not_in_label)} "
        f"({skills.not_in_label}/{skills.predicted})",
        f"  invented, not in text:  {_pct(skills.invented_not_in_text)} "
        f"({skills.not_in_text}/{skills.predicted})",
        "",
        f"salary             ({salary.predicted} values predicted)",
        f"  invented, not in text:  {_pct(salary.invented_not_in_text)} "
        f"({salary.not_in_text}/{salary.predicted})",
    ]
    return "\n".join(lines)


def summary(run: RunReport, accuracy: list[AccuracyReport], skills: SkillsReport) -> dict:
    """The headline numbers of a run, in README table order. Skill scores are micro averages;
    "invented" is the share of predicted skills the posting text never mentions."""
    return {
        "skills_precision": skills.micro_precision,
        "skills_recall": skills.micro_recall,
        "skills_invented": skills.invented_not_in_text,
        **{f"{r.field}_accuracy": r.accuracy for r in accuracy},
        "schema_valid_rate": run.schema_valid_rate,
        "repair_retry_rate": run.repair_retry_rate,
        "latency_p50_s": run.latency_p50,
        "latency_p95_s": run.latency_p95,
        "cost_per_1000_usd": run.cost_per_1000,
    }


TABLE_COLUMNS = {
    "skills_precision": "Skills precision",
    "skills_recall": "Skills recall",
    "skills_invented": "Skills invented",
    "seniority_accuracy": "Seniority accuracy",
    "role_type_accuracy": "Role type accuracy",
    "work_mode_accuracy": "Work mode accuracy",
    "german_accuracy": "German accuracy",
    "schema_valid_rate": "Schema-valid rate",
    "repair_retry_rate": "Repair-retry rate",
    "latency_p50_s": "p50 latency",
    "latency_p95_s": "p95 latency",
    "cost_per_1000_usd": "Cost / 1000 postings",
}

_FORMATS = {"latency_p50_s": _secs, "latency_p95_s": _secs, "cost_per_1000_usd": _usd}


def markdown_table(model: str, values: dict) -> str:
    """A one-row Markdown table of the summary, ready to paste into the README.

    Columns are padded to equal width so the table also reads well in the terminal.
    """
    headers = ["Model", *TABLE_COLUMNS.values()]
    cells = [model] + [_FORMATS.get(key, _pct)(values[key]) for key in TABLE_COLUMNS]
    widths = [max(len(h), len(c), 3) for h, c in zip(headers, cells, strict=True)]
    # Model is left-aligned, the numbers right-aligned (":" on the right of the rule).
    rule = ["-" * widths[0]] + ["-" * (w - 1) + ":" for w in widths[1:]]
    header_row = [h.ljust(w) for h, w in zip(headers, widths, strict=True)]
    value_row = [cells[0].ljust(widths[0])] + [
        c.rjust(w) for c, w in zip(cells[1:], widths[1:], strict=True)
    ]
    return "\n".join("| " + " | ".join(row) + " |" for row in (header_row, rule, value_row))


def posting_errors(
    run: RunReport, accuracy: list[AccuracyReport], skills: SkillsReport, salary: SalaryReport
) -> dict[str, dict]:
    """Everything that went wrong, by posting ID; postings with no errors are left out."""
    errors: dict[str, dict] = {}
    for posting_id, error in run.failed.items():
        errors.setdefault(posting_id, {})["extraction_error"] = error
    for r in accuracy:
        for m in r.misses:
            errors.setdefault(m.posting_id, {})[r.field] = {"gold": m.gold, "pred": m.pred}
    per_posting = {
        "skills_missing": skills.missing,
        "skills_extra": skills.extra,
        "skills_not_in_text": skills.ungrounded,
        "salary_not_in_text": salary.ungrounded,
    }
    for key, by_id in per_posting.items():
        for posting_id, values in by_id.items():
            errors.setdefault(posting_id, {})[key] = [str(v) for v in values]
    return dict(sorted(errors.items()))


def metrics_json(
    predictions: Path,
    run: RunReport,
    accuracy: list[AccuracyReport],
    skills: SkillsReport,
    salary: SalaryReport,
) -> dict:
    return {
        "predictions": predictions.name,
        "postings": run.n,
        "scored": skills.n,
        "summary": summary(run, accuracy, skills),
        "errors": posting_errors(run, accuracy, skills, salary),
    }


def _first_ok(records: list[dict]) -> dict | None:
    return next((r for r in records if "posting" in r), None)


def _model_name(records: list[dict], predictions: Path) -> str:
    """provider/model from the first successful record, else from the file name."""
    first = _first_ok(records)
    return predictions.stem if first is None else f"{first['provider']}/{first['model']}"


def main(
    split: str,
    provider: str | None = None,
    model: str | None = None,
    predictions: Path | None = None,
) -> None:
    if predictions is None:
        predictions = run_predictions(split, provider, model)
        print(f"predictions: {predictions}")
    records = load_records(predictions)
    examples, _ = load_examples(predictions)
    first = _first_ok(records)
    price = None if first is None else price_for(first["model"])
    if first is not None and price is None:
        print(f"no price for {first['model']} in prices.yaml; cost is n/a", file=sys.stderr)
    run = score_run(records, price)
    accuracy = [score_accuracy(examples, field) for field in ACCURACY_FIELDS]
    skills, salary = score_skills(examples), score_salary(examples)
    print(report(run, accuracy, skills, salary))

    metrics = metrics_json(predictions, run, accuracy, skills, salary)
    print("\n" + markdown_table(_model_name(records, predictions), metrics["summary"]))

    out = predictions.with_name(f"{predictions.stem}_metrics.json")
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nmetrics: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the extractor on one split.")
    parser.add_argument(
        "split", choices=["dev", "dev_short", "test"], help="which split from data/splits.json"
    )
    parser.add_argument("--provider", choices=list(get_args(Provider)), help="LLM provider")
    parser.add_argument("--model", help="model name (default: the provider's default)")
    parser.add_argument(
        "--predictions", type=Path, help="score an existing predictions file instead of extracting"
    )
    args = parser.parse_args()
    main(args.split, provider=args.provider, model=args.model, predictions=args.predictions)
