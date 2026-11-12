"""Judge extraction runs with an LLM and compare its verdicts with manual gold labels.

uv run python -m evaluation.judge_run data/cache/<run1>.jsonl [data/cache/<run2>.jsonl ...]
uv run python -m evaluation.judge_run <runs...> --rescore   # reuse cached verdicts, no LLM

Defaults to claude-opus-5-5 at high effort; see studies/judge/README.md for the experiment.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import get_args

from evaluation.judge import (
    JUDGE_CACHE,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    PROMPT_VERSION,
    JudgeVerdict,
    load_verdicts,
    run_judge,
    verdicts_path,
)
from evaluation.judge_metrics import (
    FIELDS,
    LIST_FIELDS,
    Agreement,
    Judged,
    MissingAgreement,
    field_agreement,
    item_agreement,
    report_metrics,
    same_ranking,
)
from evaluation.metrics import _percentile
from evaluation.prices import price_for
from evaluation.run import _git_commit, _model_name, _pct, _secs, _usd, load_examples, load_records
from job_extractor.extract import LABELS, RAW, ROOT, load_settings
from job_extractor.llm import make_client, resolve_model
from job_extractor.settings import Provider, Settings

RESULTS = ROOT / "studies" / "judge" / "results.md"
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def judge_settings(provider: str | None, model: str | None, effort: str) -> Settings:
    """The judge's settings: Opus 5.5 unless another provider or model is named."""
    provider = provider or JUDGE_PROVIDER
    if model is None and provider == JUDGE_PROVIDER:
        model = JUDGE_MODEL
    settings = load_settings(provider, model)
    return Settings(**(settings.model_dump() | {"effort": effort}))


def load_judged(
    predictions: Path, verdicts: Path, labels_dir: Path = LABELS, raw_dir: Path = RAW
) -> tuple[list[Judged], dict[str, str], dict[str, str]]:
    """Postings with both an extraction and a verdict; also failed extractions and failed
    verdicts (ID -> error). Both kinds of failure are left out of every score."""
    examples, failed_extractions = load_examples(predictions, labels_dir, raw_dir)
    records = load_verdicts(verdicts)
    judged, failed_verdicts = [], {}
    for ex in examples:
        record = records.get(ex.posting_id)
        if record is None or "verdict" not in record:
            failed_verdicts[ex.posting_id] = (record or {}).get("error", "not judged")
            continue
        verdict = JudgeVerdict.model_validate(record["verdict"])
        judged.append(Judged(ex.posting_id, ex.gold, ex.pred, verdict))
    return judged, failed_extractions, failed_verdicts


def judge_cost(verdicts: Path) -> dict:
    """Judge cost per 1000 postings and latency, over successful verdicts."""
    ok = [r for r in load_verdicts(verdicts).values() if "verdict" in r]
    model = ok[0]["model"] if ok else None
    price = None if model is None else price_for(model)
    costs = [] if price is None else [price.cost(r["input_tokens"], r["output_tokens"]) for r in ok]
    latencies = [r["latency_s"] for r in ok]
    return {
        "model": model,
        "repaired": sum(bool(r.get("repaired")) for r in ok),
        "cost_per_1000_usd": sum(costs) / len(costs) * 1000 if costs else None,
        "latency_p50_s": _percentile(latencies, 50),
        "latency_p95_s": _percentile(latencies, 95),
    }


def _agreement_json(a: Agreement) -> dict:
    return {
        "n": a.n,
        "judge_wrong": a.tp + a.fp,
        "gold_wrong": a.tp + a.fn,
        "tp": a.tp,
        "fp": a.fp,
        "fn": a.fn,
        "tn": a.tn,
        "kappa": a.kappa,
        "error_precision": a.error_precision,
        "error_recall": a.error_recall,
        "agreement": a.agreement,
    }


def _missing_json(m: MissingAgreement) -> dict:
    return {
        "judge_missing": m.judge_missing,
        "gold_missing": m.gold_missing,
        "both": m.both,
        "precision": m.precision,
        "recall": m.recall,
    }


def score(
    predictions: Path, verdicts: Path, labels_dir: Path = LABELS, raw_dir: Path = RAW
) -> dict:
    """Item-level agreement, the report from gold and from the judge, judge cost."""
    judged, failed_extractions, failed_verdicts = load_judged(
        predictions, verdicts, labels_dir, raw_dir
    )
    agreements = [field_agreement(judged, f) for f in FIELDS]
    missing = []
    for field in LIST_FIELDS:
        items, miss = item_agreement(judged, field)
        agreements.append(items)
        missing.append(miss)

    disagreements: dict[str, dict[str, list[str]]] = {}
    for a in agreements:
        for posting_id, notes in (a.disagreements or {}).items():
            disagreements.setdefault(posting_id, {})[a.field] = notes
    return {
        "predictions": predictions.name,
        "verdicts": verdicts.name,
        "run": _model_name(load_records(predictions), predictions),
        "scored": len(judged),
        "failed_extractions": failed_extractions,
        "failed_verdicts": failed_verdicts,
        "judge": judge_cost(verdicts),
        "agreement": {a.field: _agreement_json(a) for a in agreements},
        "missing": {m.field: _missing_json(m) for m in missing},
        "report": {
            "gold": report_metrics(judged, "gold"),
            "judge": report_metrics(judged, "judge"),
        },
        "disagreements": dict(sorted(disagreements.items())),
    }


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _pp(gold: float | None, judge: float | None) -> str:
    return "n/a" if gold is None or judge is None else f"{(judge - gold) * 100:+.1f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """A Markdown table, padded so it also reads well in the terminal."""
    widths = [max(len(h), *(len(r[i]) for r in rows), 3) for i, h in enumerate(headers)]
    rule = ["-" * widths[0]] + ["-" * (w - 1) + ":" for w in widths[1:]]

    def line(cells: list[str]) -> str:
        padded = [cells[0].ljust(widths[0])]
        padded += [c.rjust(w) for c, w in zip(cells[1:], widths[1:], strict=True)]
        return "| " + " | ".join(padded) + " |"

    return "\n".join([line(headers), "| " + " | ".join(rule) + " |", *map(line, rows)])


def agreement_table(metrics: dict) -> str:
    headers = ["Field", "Items", "Judge: wrong", "Gold: wrong", "κ", "Error P", "Error R", "Agree"]
    rows = [
        [
            field,
            str(a["n"]),
            str(a["judge_wrong"]),
            str(a["gold_wrong"]),
            _num(a["kappa"]),
            _pct(a["error_precision"]),
            _pct(a["error_recall"]),
            _pct(a["agreement"]),
        ]
        for field, a in metrics["agreement"].items()
    ]
    rows += [
        [
            f"{field} (missing)",
            "",
            str(m["judge_missing"]),
            str(m["gold_missing"]),
            "",
            _pct(m["precision"]),
            _pct(m["recall"]),
            "",
        ]
        for field, m in metrics["missing"].items()
    ]
    return _table(headers, rows)


def report_table(runs: list[dict]) -> str:
    """Gold vs judge report per run; with two or more runs, whether the ranking matches."""
    headers = ["Metric"]
    for m in runs:
        headers += [f"{m['run']} gold", "judge", "Δ pp"]
    if len(runs) > 1:
        headers.append("Same ranking")
    rows = []
    for key in runs[0]["report"]["gold"]:
        row = [key]
        for m in runs:
            gold, judge = m["report"]["gold"][key], m["report"]["judge"][key]
            row += [_pct(gold), _pct(judge), _pp(gold, judge)]
        if len(runs) > 1:
            same = same_ranking(
                {m["run"]: m["report"]["gold"][key] for m in runs},
                {m["run"]: m["report"]["judge"][key] for m in runs},
            )
            row.append("n/a" if same is None else "yes" if same else "NO")
        rows.append(row)
    return _table(headers, rows)


def run_section(metrics: dict) -> str:
    j = metrics["judge"]
    return "\n".join(
        [
            f"### {metrics['run']}",
            "",
            f"- Predictions: `{metrics['predictions']}`",
            f"- Verdicts: `{metrics['verdicts']}`",
            f"- Scored postings: {metrics['scored']} (failed extractions: "
            f"{len(metrics['failed_extractions'])}, failed verdicts: "
            f"{len(metrics['failed_verdicts'])})",
            f"- Judge: {j['model']}, {_usd(j['cost_per_1000_usd'])} per 1000 postings, "
            f"latency p50 {_secs(j['latency_p50_s'])}, p95 {_secs(j['latency_p95_s'])}, "
            f"repair retries {j['repaired']}",
            "",
            agreement_table(metrics),
        ]
    )


def results_section(runs: list[dict], settings: Settings, now: datetime | None = None) -> str:
    when = (now or datetime.now()).strftime("%Y-%m-%d %H:%M")
    return "\n".join(
        [
            f"## {when} · judge {resolve_model(settings)} ({settings.effort}) · {PROMPT_VERSION}",
            "",
            f"- Git commit: {_git_commit()}",
            "",
            "Report from manual gold vs from judge verdicts:",
            "",
            report_table(runs),
            "",
            *(run_section(m) + "\n" for m in runs),
        ]
    )


RESULTS_HEADER = """# Judge validation results

One section per `python -m evaluation.judge_run`, newest last. Written by
`src/evaluation/judge_run.py`. Item-level tables: positive class is "prediction is wrong";
Error P/R are the judge's error-detection precision and recall against manual gold.
"""


def append_results(section: str, path: Path = RESULTS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(RESULTS_HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + section)


def main(
    runs: list[Path],
    provider: str | None = None,
    model: str | None = None,
    effort: str = "high",
    rescore: bool = False,
) -> None:
    settings = judge_settings(provider, model, effort)
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    model_name = resolve_model(settings)
    all_metrics = []
    for predictions in runs:
        if rescore:
            verdicts = verdicts_path(predictions, model_name, PROMPT_VERSION, JUDGE_CACHE)
            if not verdicts.exists():
                sys.exit(f"no cached verdicts for {predictions.name}: {verdicts}")
        else:
            verdicts = run_judge(predictions, make_client(settings))
        print(f"verdicts: {verdicts}")
        metrics = score(predictions, verdicts)
        out = verdicts.with_name(f"{verdicts.stem}_metrics.json")
        out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"metrics: {out}")
        all_metrics.append(metrics)

    section = results_section(all_metrics, settings)
    print("\n" + section)
    append_results(section)
    print(f"results: {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate extraction runs with an LLM judge.")
    parser.add_argument("runs", nargs="+", type=Path, help="predictions files from data/cache")
    parser.add_argument("--provider", choices=list(get_args(Provider)), help="judge provider")
    parser.add_argument("--model", help=f"judge model (default: {JUDGE_MODEL})")
    parser.add_argument("--effort", choices=EFFORTS, default="high", help="judge effort")
    parser.add_argument(
        "--rescore", action="store_true", help="score cached verdicts without calling the judge"
    )
    args = parser.parse_args()
    main(args.runs, args.provider, args.model, args.effort, args.rescore)
