"""Run the extractor on a split, save the predictions, and score seniority and skills.

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
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import get_args

from evaluation.metrics import (
    ACCURACY_FIELDS,
    AccuracyReport,
    Example,
    SalaryReport,
    SkillsReport,
    score_accuracy,
    score_salary,
    score_skills,
)
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


def load_examples(
    predictions: Path, labels_dir: Path = LABELS, raw_dir: Path = RAW
) -> tuple[list[Example], list[str]]:
    """Pair each saved prediction with its label and text; also return the failed IDs."""
    examples, failed = [], []
    for line in predictions.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        posting_id = record["id"]
        if "posting" not in record:
            failed.append(posting_id)
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


def report(
    accuracy: list[AccuracyReport], skills: SkillsReport, salary: SalaryReport, failed: list[str]
) -> str:
    lines = [f"scored postings:   {skills.n} (failed extractions: {len(failed)})", "", "accuracy"]
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


def metrics_json(
    accuracy: list[AccuracyReport], skills: SkillsReport, salary: SalaryReport, failed: list[str]
) -> dict:
    return {
        "failed": failed,
        "accuracy": {
            r.field: {
                "n": r.n,
                "correct": r.correct,
                "accuracy": r.accuracy,
                "misses": [asdict(m) for m in r.misses],
            }
            for r in accuracy
        },
        "skills": {
            "n": skills.n,
            "predicted": skills.predicted,
            "tp": skills.tp,
            "fp": skills.fp,
            "fn": skills.fn,
            "micro_precision": skills.micro_precision,
            "micro_recall": skills.micro_recall,
            "macro_precision": skills.macro_precision,
            "macro_recall": skills.macro_recall,
            "invented_not_in_label": skills.invented_not_in_label,
            "invented_not_in_text": skills.invented_not_in_text,
            "ungrounded": {k: [s.value for s in v] for k, v in skills.ungrounded.items()},
        },
        "salary": {
            "n": salary.n,
            "predicted": salary.predicted,
            "not_in_text": salary.not_in_text,
            "invented_not_in_text": salary.invented_not_in_text,
            "ungrounded": salary.ungrounded,
        },
    }


def main(
    split: str,
    provider: str | None = None,
    model: str | None = None,
    predictions: Path | None = None,
) -> None:
    if predictions is None:
        predictions = run_predictions(split, provider, model)
        print(f"predictions: {predictions}")
    examples, failed = load_examples(predictions)
    accuracy = [score_accuracy(examples, field) for field in ACCURACY_FIELDS]
    skills, salary = score_skills(examples), score_salary(examples)
    print(report(accuracy, skills, salary, failed))

    out = predictions.with_name(f"{predictions.stem}_metrics.json")
    out.write_text(json.dumps(metrics_json(accuracy, skills, salary, failed), indent=2) + "\n")
    print(f"\nmetrics: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the extractor on one split.")
    parser.add_argument("split", choices=["dev", "test"], help="which split from data/splits.json")
    parser.add_argument("--provider", choices=list(get_args(Provider)), help="LLM provider")
    parser.add_argument("--model", help="model name (default: the provider's default)")
    parser.add_argument(
        "--predictions", type=Path, help="score an existing predictions file instead of extracting"
    )
    args = parser.parse_args()
    main(args.split, provider=args.provider, model=args.model, predictions=args.predictions)
