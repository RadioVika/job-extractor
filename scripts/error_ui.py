"""Local web UI for error analysis: one run's predictions next to the posting text and gold labels.

Three panels per posting: data/raw/pNNN.txt, the gold labels from data/labels/pNNN.json (editable,
saved exactly as scripts/label_ui.py saves them), and the model's output from a run in data/cache/
with per-posting metrics. The newest run with records opens by default; the header picks another.
Metrics are computed live against the current labels, so they follow label fixes; the header also
shows the run's _metrics.json snapshot from scoring time when there is one. Run from the repo root:

  uv run python scripts/error_ui.py        # then open http://127.0.0.1:8001
"""

import json
from pathlib import Path
from typing import Any

import label_ui
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from evaluation.grounding import SKILL_PATTERNS, is_grounded
from evaluation.metrics import (
    ACCURACY_FIELDS,
    Example,
    _per_posting,
    score_accuracy,
    score_run,
    score_skills,
)
from evaluation.prices import Price, price_for
from evaluation.run import TABLE_COLUMNS, load_records, summary
from evaluation.salary import SALARY_FIELDS, ungrounded_salary
from job_extractor.schema import JobPosting

CACHE = Path("data/cache")
LIST_FIELDS = ("skills", "clouds", "data_types")


def run_files() -> list[Path]:
    """Run files, newest first: names start with a sortable timestamp."""
    return sorted(CACHE.glob("*.jsonl"), reverse=True)


def run_path(run: str) -> Path:
    # Only names of existing files are accepted, so a request can't reach other paths.
    path = next((p for p in run_files() if p.stem == run), None)
    if path is None:
        raise HTTPException(404, f"unknown run {run}")
    return path


def run_info(path: Path) -> dict[str, Any]:
    """Name parts of <time>_<split>_<provider>_<model>_<prompt>; the split may contain "_"."""
    parts = path.stem.split("_")
    info: dict[str, Any] = {"name": path.stem}
    if len(parts) >= 5:
        info |= {
            "time": parts[0],
            "split": "_".join(parts[1:-3]),
            "provider": parts[-3],
            "model": parts[-2],
            "prompt": parts[-1],
        }
    info["records"] = len(load_records(path))
    info["has_metrics"] = metrics_path(path).exists()
    return info


def metrics_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_metrics.json")


def gold_for(posting_id: str) -> JobPosting | None:
    """The gold label, or None if it is missing, incomplete or the raw text is gone."""
    if not (label_ui.RAW / f"{posting_id}.txt").exists():
        return None
    state = label_ui.posting_state(posting_id)
    if not state["labels"] or state["errors"]:
        return None
    return JobPosting.model_validate(state["labels"])


def raw_text(posting_id: str) -> str:
    path = label_ui.RAW / f"{posting_id}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def run_price(records: list[dict]) -> Price | None:
    first = next((r for r in records if "posting" in r), None)
    return None if first is None else price_for(first["model"])


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def posting_metrics(
    record: dict, gold: JobPosting | None, text: str, price: Price | None
) -> dict[str, Any]:
    """Per-posting scores with the same definitions as evaluation.metrics.

    Scores against the label are left out while the label is missing or invalid; `issues` counts
    everything wrong (field misses, extra / missing list values, salary off or not in the text).
    """
    if "posting" not in record:
        return {"error": record.get("error", ""), "issues": None}
    pred = JobPosting.model_validate(record["posting"])
    skills_not_in_text = sorted(s for s in pred.skills if not is_grounded(s, text))
    salary_not_in_text = ungrounded_salary(text, pred)
    salary_predicted = sum(getattr(pred, f) is not None for f in SALARY_FIELDS)
    tokens = (record.get("input_tokens"), record.get("output_tokens"))
    result: dict[str, Any] = {
        "error": None,
        "stats": {
            "latency_s": record.get("latency_s"),
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "repaired": record.get("repaired", False),
            "cost_usd": None if price is None or None in tokens else price.cost(*tokens),
        },
        "skills_not_in_text": skills_not_in_text,
        "skills_invented_not_in_text": _ratio(len(skills_not_in_text), len(pred.skills)),
        "salary_not_in_text": salary_not_in_text,
        "salary_invented_not_in_text": _ratio(len(salary_not_in_text), salary_predicted),
        "gold_valid": gold is not None,
        "issues": None,
    }
    if gold is None:
        return result

    accuracy = {f: getattr(pred, f) == getattr(gold, f) for f in ACCURACY_FIELDS}
    lists = {}
    for f in LIST_FIELDS:
        p, g = set(getattr(pred, f)), set(getattr(gold, f))
        precision, recall = _per_posting(p, g)
        lists[f] = {
            "matched": sorted(p & g),
            "extra": sorted(p - g),
            "missing": sorted(g - p),
            "precision": precision,
            "recall": recall,
        }
    salary_match = {f: getattr(pred, f) == getattr(gold, f) for f in SALARY_FIELDS}
    result |= {
        "accuracy": accuracy,
        "accuracy_correct": sum(accuracy.values()),
        "lists": lists,
        "skills_invented_not_in_label": _ratio(len(lists["skills"]["extra"]), len(pred.skills)),
        "salary_match": salary_match,
        "industry_match": (pred.industry or "").strip().lower()
        == (gold.industry or "").strip().lower(),
        "issues": sum(not ok for ok in accuracy.values())
        + sum(len(v["extra"]) + len(v["missing"]) for v in lists.values())
        + sum(not ok for ok in salary_match.values())
        + len(salary_not_in_text),
    }
    return result


def live_summary(records: list[dict], price: Price | None) -> tuple[dict[str, Any], int]:
    """The run's headline metrics against the labels as they are now, and how many were scored.

    Postings whose label is missing or invalid are left out of the quality scores.
    """
    examples = []
    for r in records:
        gold = gold_for(r["id"]) if "posting" in r else None
        if gold is not None:
            pred = JobPosting.model_validate(r["posting"])
            examples.append(Example(r["id"], gold, pred, raw_text(r["id"])))
    accuracy = [score_accuracy(examples, f) for f in ACCURACY_FIELDS]
    return summary(score_run(records, price), accuracy, score_skills(examples)), len(examples)


def find_record(records: list[dict], posting_id: str) -> dict:
    record = next((r for r in records if r["id"] == posting_id), None)
    if record is None:
        raise HTTPException(404, f"{posting_id} is not in this run")
    return record


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "fields": label_ui.FIELDS,
        "columns": TABLE_COLUMNS,
        "patterns": {skill.value: p.pattern for skill, p in SKILL_PATTERNS.items()},
    }


@app.get("/api/runs")
def runs() -> list[dict[str, Any]]:
    return [run_info(p) for p in run_files()]


@app.get("/api/runs/{run}")
def run(run: str) -> dict[str, Any]:
    path = run_path(run)
    records = load_records(path)
    price = run_price(records)
    index = {r["id"]: r for r in label_ui.read_index()}
    postings = []
    for r in records:
        m = posting_metrics(r, gold_for(r["id"]), raw_text(r["id"]), price)
        skills = m.get("lists", {}).get("skills", {})
        postings.append(
            {
                "id": r["id"],
                "company": index.get(r["id"], {}).get("company", ""),
                "title": index.get(r["id"], {}).get("title", ""),
                "error": m["error"],
                "gold_valid": m.get("gold_valid", gold_for(r["id"]) is not None),
                "issues": m["issues"],
                "accuracy_correct": m.get("accuracy_correct"),
                "skills_precision": skills.get("precision"),
                "skills_recall": skills.get("recall"),
            }
        )
    live, scored = live_summary(records, price)
    snapshot = None
    if metrics_path(path).exists():
        snapshot = json.loads(metrics_path(path).read_text(encoding="utf-8")).get("summary")
    return {
        **run_info(path),
        "price_known": price is not None,
        "live": live,
        "scored": scored,
        "snapshot": snapshot,
        "postings": postings,
    }


def posting_detail(run: str, posting_id: str) -> dict[str, Any]:
    records = load_records(run_path(run))
    record = find_record(records, posting_id)
    label_ui.label_path(posting_id)  # validates the id and that the raw text exists
    text = raw_text(posting_id)
    meta = next((r for r in label_ui.read_index() if r["id"] == posting_id), {})
    return {
        "id": posting_id,
        "company": meta.get("company", ""),
        "url": meta.get("url", ""),
        "text": text,
        **label_ui.posting_state(posting_id),
        "prediction": record.get("posting"),
        "metrics": posting_metrics(record, gold_for(posting_id), text, run_price(records)),
    }


@app.get("/api/runs/{run}/postings/{posting_id}")
def posting(run: str, posting_id: str) -> dict[str, Any]:
    return posting_detail(run, posting_id)


class LabelChange(BaseModel):
    field: str
    value: Any = None


@app.post("/api/runs/{run}/postings/{posting_id}/labels")
def change_label(run: str, posting_id: str, change: LabelChange) -> dict[str, Any]:
    find_record(load_records(run_path(run)), posting_id)
    label_ui.apply_change(posting_id, change.field, change.value)
    return posting_detail(run, posting_id)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Error analysis</title>
<style>
  :root { --bg:#f4f3ef; --panel:#fff; --text:#1f1e1c; --muted:#78766f; --faint:#a8a69e;
          --line:#e7e5de; --chip:#f2f0ea; --chip-hover:#e9e6dd; --on:#2f6f4f; --on-text:#fff;
          --on-soft:#e2efe7; --warn:#b4532a; --warn-soft:#fbefe7; --mid:#9a6b00;
          --mid-soft:#fbf0d4; --mark:#fcecaa; --pred:#3b6fb6; --pred-soft:#e4edf8;
          --shadow:0 1px 2px rgb(0 0 0 / .04), 0 2px 8px rgb(0 0 0 / .04); }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#171716; --panel:#1f1f1d; --text:#e9e8e4; --muted:#9d9b94; --faint:#6c6a64;
            --line:#33322f; --chip:#2a2927; --chip-hover:#353431; --on:#5bb085;
            --on-text:#0c1a12; --on-soft:#1f3329; --warn:#e5866a; --warn-soft:#3a241c;
            --mid:#e2b93b; --mid-soft:#3d3412; --mark:#584a12; --pred:#7aa7e6;
            --pred-soft:#1c2a3d; --shadow:none; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif;
         background:var(--bg); color:var(--text); height:100vh; display:flex;
         flex-direction:column; -webkit-font-smoothing:antialiased; }
  ::-webkit-scrollbar { width:10px; }
  ::-webkit-scrollbar-thumb { background:var(--line); border-radius:999px;
                              border:3px solid transparent; background-clip:content-box; }
  select, input, button { font:inherit; color:inherit; background:var(--chip);
                          border:1px solid transparent; border-radius:8px; padding:6px 10px; }
  select:focus, input:focus { outline:none; border-color:var(--on);
                              box-shadow:0 0 0 3px var(--on-soft); }
  button { cursor:pointer; }
  button:hover { background:var(--chip-hover); }
  kbd { font:11px ui-monospace, monospace; padding:1px 5px; border-radius:4px;
        border:1px solid var(--line); border-bottom-width:2px; background:var(--panel); }
  .num { font-variant-numeric:tabular-nums; }

  /* Header: run picker and live run metrics */
  header { display:flex; gap:10px; align-items:center; padding:10px 16px;
           background:var(--panel); border-bottom:1px solid var(--line); }
  .brand { font-weight:700; letter-spacing:-.01em; white-space:nowrap; }
  .brand span { color:var(--on); }
  #run { flex:1; max-width:560px; min-width:0; }
  header .note { color:var(--faint); font-size:12px; margin-left:auto; white-space:nowrap; }
  #summary { display:flex; flex-wrap:wrap; gap:6px 18px; padding:8px 16px;
             background:var(--panel); border-bottom:1px solid var(--line); font-size:12px; }
  #summary .m { display:flex; flex-direction:column; }
  #summary .k { color:var(--muted); font-size:11px; }
  #summary .v { font-weight:650; font-size:14px; }
  #summary .was { color:var(--faint); font-weight:400; font-size:11px; margin-left:4px; }

  main { flex:1; display:grid; min-height:0;
         grid-template-columns: 250px minmax(0, 1.15fr) minmax(0, 1fr) minmax(0, 1fr); }
  main > section { overflow:auto; border-right:1px solid var(--line); }
  h3.panel { position:sticky; top:0; z-index:1; margin:0; padding:8px 16px; font-size:11px;
             text-transform:uppercase; letter-spacing:.08em; color:var(--muted);
             background:var(--bg); border-bottom:1px solid var(--line); }

  /* Posting list */
  #list .controls { display:flex; flex-direction:column; gap:6px; padding:10px 12px;
                    border-bottom:1px solid var(--line); font-size:12px; }
  #list .controls label { display:flex; gap:6px; align-items:center; color:var(--muted); }
  #list .controls select { padding:4px 8px; font-size:12px; }
  .item { display:block; width:100%; text-align:left; border-radius:0; background:none;
          padding:8px 12px; border-bottom:1px solid var(--line); }
  .item.active { background:var(--on-soft); box-shadow:inset 3px 0 0 var(--on); }
  .item .top { display:flex; gap:6px; align-items:center; }
  .item .id { font:600 12px ui-monospace, monospace; }
  .item .sub { color:var(--muted); font-size:12px; overflow:hidden; white-space:nowrap;
               text-overflow:ellipsis; }
  .badge { margin-left:auto; border-radius:999px; padding:0 7px; font-size:11px;
           font-weight:650; background:var(--chip); color:var(--muted); }
  .badge.good { background:var(--on-soft); color:var(--on); }
  .badge.mid { background:var(--mid-soft); color:var(--mid); }
  .badge.bad { background:var(--warn-soft); color:var(--warn); }

  /* Posting text */
  #text .inner { padding:20px 24px 64px; }
  #text .meta { font-size:12px; color:var(--muted); text-transform:uppercase;
                letter-spacing:.06em; font-weight:600; }
  #text .meta a { color:var(--faint); text-transform:none; letter-spacing:0; margin-left:6px; }
  #text h1 { font-size:21px; line-height:1.25; margin:6px 0 6px; }
  #text .loc { color:var(--muted); padding-bottom:16px; margin-bottom:18px;
               border-bottom:1px solid var(--line); }
  #body { white-space:pre-wrap; font-size:14.5px; line-height:1.7; }
  mark { background:var(--mark); color:inherit; border-radius:3px; padding:0 1px; }
  mark.hover { background:var(--on); color:var(--on-text); }

  /* Cards shared by the gold and prediction panels */
  .pad { padding:12px 14px 64px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:12px;
          padding:10px 12px 12px; margin-bottom:10px; box-shadow:var(--shadow); }
  .card h2 { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted);
             margin:0 0 8px; display:flex; gap:8px; align-items:center; font-weight:700; }
  .card h2 .right { margin-left:auto; text-transform:none; letter-spacing:0; font-weight:500; }
  .card.bad { border-left:3px solid var(--warn); }
  .chips { display:flex; flex-wrap:wrap; gap:5px; }
  .chip { border-radius:999px; padding:3px 10px; font-size:12.5px; line-height:1.4;
          background:var(--chip); border:1px solid transparent; }
  button.chip:hover { background:var(--chip-hover); }
  .chip.on { background:var(--on); color:var(--on-text); font-weight:600; }
  .chip.on::before { content:"✓ "; }
  .chip.pred { border:1px dashed var(--pred); }
  .chip.on.pred { border-style:solid; }
  .chip.extra { background:var(--warn-soft); color:var(--warn); font-weight:600; }
  .chip.extra::before { content:"+ "; }
  .chip.missing { background:transparent; border:1px dashed var(--warn); color:var(--warn); }
  .chip.missing::before { content:"− "; }
  .chip.match { background:var(--on-soft); color:var(--on); font-weight:600; }
  .chip .flag { margin-left:4px; }
  .card input { width:100%; max-width:240px; }
  #status { padding:8px 12px; border-radius:10px; margin-bottom:10px; font-size:13px; }
  #status.ok { background:var(--on-soft); color:var(--on); font-weight:600; }
  #status.bad { background:var(--warn-soft); color:var(--warn); }
  #status.empty { background:var(--chip); color:var(--muted); }
  .legend { color:var(--faint); font-size:11.5px; margin:0 0 10px; display:flex; gap:10px;
            flex-wrap:wrap; }
  .legend .chip { font-size:11px; padding:0 7px; }

  /* Prediction metrics */
  .tiles { display:grid; grid-template-columns:repeat(auto-fill, minmax(118px, 1fr)); gap:6px; }
  .tile { border-radius:10px; padding:6px 10px; background:var(--chip); }
  .tile .k { font-size:11px; color:var(--muted); }
  .tile .v { font-size:17px; font-weight:700; }
  .tile .d { font-size:11px; color:var(--muted); }
  .tile.good { background:var(--on-soft); } .tile.good .v { color:var(--on); }
  .tile.mid { background:var(--mid-soft); } .tile.mid .v { color:var(--mid); }
  .tile.bad { background:var(--warn-soft); } .tile.bad .v { color:var(--warn); }
  .stats { display:flex; flex-wrap:wrap; gap:4px 14px; margin-top:8px; font-size:12px;
           color:var(--muted); }
  .stats b { color:var(--text); font-weight:600; }
  .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .ok-mark { color:var(--on); font-weight:700; }
  .bad-mark { color:var(--warn); font-weight:700; }
  .gold-was { color:var(--muted); font-size:12px; }
  .use { margin-left:auto; font-size:11.5px; padding:2px 8px; background:var(--pred-soft);
         color:var(--pred); }
  .use:hover { background:var(--pred-soft); filter:brightness(.96); }
  .errbox { background:var(--warn-soft); color:var(--warn); border-radius:10px; padding:10px 12px;
            font:12px/1.5 ui-monospace, monospace; white-space:pre-wrap; word-break:break-word; }
  .empty-note { color:var(--muted); padding:20px; }
</style>
</head>
<body>
<header>
  <div class="brand">Job<span>Errors</span></div>
  <select id="run" title="Run from data/cache"></select>
  <span class="note"><kbd>Alt</kbd> <kbd>↑</kbd> <kbd>↓</kbd> switch posting ·
    hover a skill to find it in the text</span>
</header>
<div id="summary"></div>
<main>
  <section id="list">
    <h3 class="panel">Postings</h3>
    <div class="controls">
      <label>Sort <select id="sort">
        <option value="id">by ID</option>
        <option value="issues">most issues first</option>
        <option value="skills_recall">lowest skills recall</option>
        <option value="skills_precision">lowest skills precision</option>
        <option value="accuracy_correct">fewest fields correct</option>
      </select></label>
      <label><input type="checkbox" id="only-issues"> only postings with issues</label>
    </div>
    <div id="items"></div>
  </section>
  <section id="text"><h3 class="panel">Posting</h3><div class="inner"></div></section>
  <section id="gold"><h3 class="panel">Gold labels</h3><div class="pad">
    <div id="status"></div>
    <div class="legend"><span><span class="chip on">gold</span></span>
      <span><span class="chip pred">predicted</span></span></div>
    <div id="form"></div></div></section>
  <section id="pred"><h3 class="panel">LLM response</h3>
    <div class="pad" id="pred-body"></div></section>
</main>
<script>
const params = new URLSearchParams(location.search);
let meta = {}, runInfo = null, current = null;

const api = async (url, opts) => {
  const res = await fetch(url, opts);
  if (!res.ok) { alert((await res.json()).detail ?? res.statusText); throw new Error(url); }
  return res.json();
};
const el = (tag, props = {}, children = []) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
};
const enc = encodeURIComponent;

// Formatting and colouring of metric values.
const pct = v => v == null ? "—" : `${Math.round(v * 100)}%`;
const FORMATS = {
  latency_p50_s: v => v == null ? "—" : `${v.toFixed(2)} s`,
  latency_p95_s: v => v == null ? "—" : `${v.toFixed(2)} s`,
  cost_per_1000_usd: v => v == null ? "—" : `$${v.toFixed(2)}`,
};
const fmt = (key, v) => (FORMATS[key] ?? pct)(v);
// Higher is better unless the metric counts something bad.
const LOWER_IS_BETTER = new Set(["skills_invented", "repair_retry_rate", "invented"]);
function grade(v, lowerIsBetter = false) {
  if (v == null) return "";
  const good = lowerIsBetter ? 1 - v : v;
  return good >= 0.9 ? "good" : good >= 0.6 ? "mid" : "bad";
}

function setUrl() {
  const q = new URLSearchParams({ run: runInfo.name });
  if (current) q.set("id", current.id);
  history.replaceState(null, "", `?${q}`);
}

// ---------- runs and the posting list ----------

async function loadRuns() {
  const runs = await api("/api/runs");
  const picker = document.getElementById("run");
  picker.replaceChildren(...runs.map(r => el("option", {
    value: r.name,
    textContent: r.split
      ? `${r.time.replace(/^(\d{4})(\d\d)(\d\d)-(\d\d)(\d\d)\d\d$/, "$1-$2-$3 $4:$5")} · ${r.split}`
        + ` · ${r.provider}/${r.model} · ${r.prompt}`
        + ` · ${r.records ? r.records + " postings" : "empty"}`
      : r.name,
    disabled: !r.records,
  })));
  return runs;
}

async function openRun(name, postingId) {
  runInfo = await api(`/api/runs/${enc(name)}`);
  document.getElementById("run").value = name;
  renderSummary();
  renderList();
  const ids = runInfo.postings.map(p => p.id);
  await openPosting(ids.includes(postingId) ? postingId : visible()[0]?.id ?? ids[0]);
}

async function refreshRun() {
  runInfo = await api(`/api/runs/${enc(runInfo.name)}`);
  renderSummary();
  renderList();
}

function renderSummary() {
  const s = runInfo.live, snap = runInfo.snapshot;
  const box = document.getElementById("summary");
  box.replaceChildren(...Object.entries(meta.columns).map(([key, label]) => {
    const v = el("span", { className: "v num", textContent: fmt(key, s[key]) });
    if (snap && fmt(key, snap[key]) !== fmt(key, s[key])) {
      v.append(el("span", { className: "was", textContent: `was ${fmt(key, snap[key])}`,
                             title: "Value in the run's _metrics.json, from scoring time" }));
    }
    return el("div", { className: "m" }, [el("span", { className: "k", textContent: label }), v]);
  }));
  box.append(el("div", { className: "m" }, [
    el("span", { className: "k", textContent: "Scored now" }),
    el("span", { className: "v num", textContent: `${runInfo.scored} / ${runInfo.records}`,
                 title: "Postings with a valid prediction and a valid gold label" }),
  ]));
}

const SORTS = {
  id: null,
  issues: p => -(p.issues ?? Infinity),
  skills_recall: p => p.skills_recall ?? Infinity,
  skills_precision: p => p.skills_precision ?? Infinity,
  accuracy_correct: p => p.accuracy_correct ?? Infinity,
};

function visible() {
  let list = runInfo.postings;
  if (document.getElementById("only-issues").checked) {
    list = list.filter(p => p.error || p.issues || !p.gold_valid);
  }
  const key = SORTS[document.getElementById("sort").value];
  return key ? [...list].sort((a, b) => key(a) - key(b) || a.id.localeCompare(b.id)) : list;
}

function badgeFor(p) {
  if (p.error) return el("span", { className: "badge bad", textContent: "ERR",
                                   title: "Extraction failed" });
  if (!p.gold_valid) return el("span", { className: "badge", textContent: "no label",
                                         title: "Gold label missing or invalid" });
  const cls = p.issues === 0 ? "good" : p.issues <= 3 ? "mid" : "bad";
  return el("span", { className: `badge ${cls}`, textContent: p.issues === 0 ? "✓" : p.issues,
                      title: `${p.issues} issues` });
}

function renderList() {
  const items = document.getElementById("items");
  items.replaceChildren(...visible().map(p => el("button", {
    className: "item" + (current?.id === p.id ? " active" : ""),
    onclick: () => openPosting(p.id),
  }, [
    el("div", { className: "top" }, [el("span", { className: "id", textContent: p.id }),
                                     badgeFor(p)]),
    el("div", { className: "sub", textContent: `${p.company} · ${p.title}`,
                title: `${p.company} · ${p.title}` }),
  ])));
  items.querySelector(".active")?.scrollIntoView({ block: "nearest" });
}

function step(delta) {
  const list = visible();
  const i = list.findIndex(p => p.id === current?.id) + delta;
  if (i >= 0 && i < list.length) openPosting(list[i].id);
}

// ---------- one posting ----------

async function openPosting(id) {
  if (!id) return;
  current = await api(`/api/runs/${enc(runInfo.name)}/postings/${id}`);
  setUrl();
  renderText();
  renderGold();
  renderPrediction();
  renderList();
  document.getElementById("text").scrollTop = 0;
}

async function change(field, value) {
  current = await api(`/api/runs/${enc(runInfo.name)}/postings/${current.id}/labels`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ field, value }),
  });
  renderGold();
  renderPrediction();
  refreshRun();
}

function renderText() {
  const [title, location, ...rest] = current.text.split("\n");
  const metaLine = el("div", { className: "meta" }, [current.company]);
  metaLine.append(current.url
    ? el("a", { href: current.url, target: "_blank", textContent: current.id })
    : el("a", { textContent: current.id }));
  document.querySelector("#text .inner").replaceChildren(
    metaLine,
    el("h1", { textContent: title }),
    el("div", { className: "loc", textContent: location || "Location not given" }),
    el("div", { id: "body", textContent: rest.join("\n").trim() }),
  );
  highlight(null);
}

function renderGold() {
  const labels = current.labels;
  const pred = current.prediction ?? {};
  const status = document.getElementById("status");
  const fresh = !Object.keys(labels).length;
  if (fresh) {
    status.className = "empty";
    status.textContent = "Not labeled yet — click a label to create the file.";
  } else if (current.errors.length) {
    status.className = "bad";
    status.textContent = "Invalid, left out of the scores: " + current.errors.join("; ");
  } else {
    status.className = "ok";
    status.textContent = `✓ data/labels/${current.id}.json is valid`;
  }
  document.getElementById("form").replaceChildren(...Object.entries(meta.fields).map(
    ([name, spec]) => {
      const value = labels[name];
      const bad = !fresh && current.errors.some(e => e.startsWith(name + ":"));
      const heading = el("h2", { textContent: name.replaceAll("_", " ") });
      let body;
      if (spec.kind === "enum" || spec.kind === "multi") {
        const chosen = spec.kind === "multi" ? (value ?? []) : [value];
        const predicted = spec.kind === "multi" ? (pred[name] ?? []) : [pred[name]];
        if (spec.kind === "multi") {
          heading.append(el("span", { className: "right", textContent: `${chosen.length}` }));
        }
        body = el("div", { className: "chips" }, spec.options.map(opt => el("button", {
          className: "chip" + (chosen.includes(opt) ? " on" : "")
            + (predicted.includes(opt) ? " pred" : ""),
          title: predicted.includes(opt) ? "Predicted by the model" : "",
          textContent: opt,
          onclick: () => change(name, opt),
          onmouseenter: () => highlight(opt),
          onmouseleave: () => highlight(null),
        })));
      } else {
        body = el("input", {
          type: spec.kind === "int" ? "number" : "text",
          value: value ?? "",
          placeholder: spec.kind === "int" ? "empty = null" : "",
          onchange: e => change(name, e.target.value),
        });
      }
      return el("div", { className: "card" + (bad ? " bad" : "") }, [heading, body]);
    }));
}

function tile(label, value, detail, cls) {
  return el("div", { className: `tile ${cls ?? ""}` }, [
    el("div", { className: "k", textContent: label }),
    el("div", { className: "v num", textContent: value }),
    el("div", { className: "d num", textContent: detail ?? "" }),
  ]);
}

const useButton = (field, value) => el("button", {
  className: "use", textContent: "use as gold",
  title: "Replace the gold value with the prediction",
  onclick: () => change(field, value ?? ""),
});

const markFor = ok => el("span", { className: ok ? "ok-mark" : "bad-mark",
                                   textContent: ok ? "✓" : "✗" });

function renderPrediction() {
  const box = document.getElementById("pred-body");
  const m = current.metrics, pred = current.prediction;
  if (m.error) {
    box.replaceChildren(el("div", { className: "card bad" }, [
      el("h2", { textContent: "Extraction failed" }),
      el("div", { className: "errbox", textContent: m.error }),
    ]));
    return;
  }
  const scored = m.gold_valid;
  const nSkills = pred.skills.length;
  const notInLabel = scored ? m.lists.skills.extra.length : null;
  const sk = scored ? m.lists.skills : null;
  const pr = (l) => `P ${pct(l.precision)} · R ${pct(l.recall)}`;
  const tiles = [];
  if (scored) {
    const fieldsTotal = Object.keys(m.accuracy).length;
    tiles.push(
      tile("Fields correct", `${m.accuracy_correct}/${fieldsTotal}`,
           Object.entries(m.accuracy).filter(([, ok]) => !ok).map(([f]) => f).join(", ")
             || "all match", grade(m.accuracy_correct / fieldsTotal)),
      tile("Skills precision", pct(sk.precision),
           `${sk.matched.length}/${sk.matched.length + sk.extra.length} predicted`,
           grade(sk.precision)),
      tile("Skills recall", pct(sk.recall),
           `${sk.matched.length}/${sk.matched.length + sk.missing.length} gold`,
           grade(sk.recall)),
      tile("Invented: not in label", pct(m.skills_invented_not_in_label),
           `${notInLabel}/${nSkills} skills`, grade(m.skills_invented_not_in_label, true)),
    );
  }
  tiles.push(tile("Invented: not in text", pct(m.skills_invented_not_in_text),
                  `${m.skills_not_in_text.length}/${nSkills} skills`,
                  grade(m.skills_invented_not_in_text, true)));
  if (scored) {
    for (const f of ["clouds", "data_types"]) {
      const l = m.lists[f];
      tiles.push(tile(`${f.replace("_", " ")} P / R`, `${pct(l.precision)} / ${pct(l.recall)}`,
                      `${l.extra.length} extra · ${l.missing.length} missing`,
                      grade(Math.min(l.precision ?? 1, l.recall ?? 1))));
    }
  }
  const salaryN = ["salary_min_eur", "salary_max_eur"].filter(f => pred[f] != null).length;
  tiles.push(tile("Salary not in text", pct(m.salary_invented_not_in_text),
                  salaryN ? `${m.salary_not_in_text.length}/${salaryN} values` : "none predicted",
                  grade(m.salary_invented_not_in_text, true)));

  const s = m.stats;
  const stats = el("div", { className: "stats num" }, [
    el("span", {}, ["latency ", el("b", { textContent: s.latency_s == null ? "—"
                                                       : `${s.latency_s.toFixed(2)} s` })]),
    el("span", {}, ["tokens ", el("b", { textContent: `${s.input_tokens ?? "—"} in` }),
                    " / ", el("b", { textContent: `${s.output_tokens ?? "—"} out` })]),
    el("span", {}, ["cost ", el("b", { textContent: s.cost_usd == null ? "no price"
                                                    : `$${s.cost_usd.toFixed(4)}` })]),
    el("span", {}, ["repair retry ", el("b", { textContent: s.repaired ? "yes" : "no" })]),
  ]);
  const cards = [el("div", { className: "card" }, [
    el("h2", { textContent: "Metrics" }, []),
    el("div", { className: "tiles" }, tiles),
    stats,
  ])];
  if (!scored) {
    cards.push(el("div", { className: "card bad" }, [el("h2", {
      textContent: "Gold label missing or invalid — only text checks are shown" })]));
  }

  const gold = current.labels;
  for (const [name, spec] of Object.entries(meta.fields)) {
    const heading = el("h2", { textContent: name.replaceAll("_", " ") });
    let body;
    if (spec.kind === "multi") {
      const l = scored ? m.lists[name]
        : { matched: [], extra: pred[name], missing: [] };
      const notInText = name === "skills" ? new Set(m.skills_not_in_text) : new Set();
      const chip = (v, cls, title) => {
        const c = el("span", { className: `chip ${cls}`, textContent: v, title,
                               onmouseenter: () => highlight(v),
                               onmouseleave: () => highlight(null) });
        if (notInText.has(v)) {
          c.append(el("span", { className: "flag", textContent: "⚠",
                                title: "Nothing in the posting text maps to this skill" }));
        }
        return c;
      };
      body = el("div", { className: "chips" }, [
        ...l.matched.map(v => chip(v, "match", "Predicted and in gold")),
        ...l.extra.map(v => chip(v, scored ? "extra" : "", "Predicted, not in gold")),
        ...l.missing.map(v => chip(v, "missing", "In gold, not predicted")),
      ]);
      if (!body.children.length) body.append(el("span", { className: "gold-was",
                                                          textContent: "none" }));
      if (scored) {
        heading.append(el("span", { className: "right num", textContent: pr(l) }));
        if (l.extra.length || l.missing.length) {
          cards.push(el("div", { className: "card" }, [
            el("div", { className: "row" }, [heading, useButton(name, pred[name])]), body]));
          continue;
        }
      }
    } else {
      const p = pred[name], g = gold[name];
      let ok = null;
      if (scored) {
        ok = name in m.accuracy ? m.accuracy[name]
          : name in m.salary_match ? m.salary_match[name]
          : name === "industry" ? m.industry_match : null;
      }
      const parts = [el("span", { className: "chip" + (p == null ? "" : " pred"),
                                  textContent: p ?? "null" })];
      if (ok !== null) parts.push(markFor(ok));
      if (ok === false) parts.push(el("span", { className: "gold-was",
                                                textContent: `gold: ${g ?? "null"}` }));
      if (m.salary_not_in_text.includes(name)) {
        parts.push(el("span", { className: "bad-mark", textContent: "⚠ not in text",
                                title: "No matching amount (or monthly amount) in the text" }));
      }
      if (ok === false) parts.push(useButton(name, p));
      body = el("div", { className: "row" }, parts);
    }
    cards.push(el("div", { className: "card" }, [heading, body]));
  }
  box.replaceChildren(...cards);
}

// ---------- highlighting in the posting text ----------

const escapeRe = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
// Skills use the grounding regex from evaluation/grounding.py; other tags match their words.
const patternFor = opt => meta.patterns[opt]
  ?? `\\b${opt.split("_").map(escapeRe).join("[\\s_/-]*")}s?\\b`;

function markBody(re, className) {
  const body = document.getElementById("body");
  if (!body) return;
  const text = body.textContent;
  const parts = [];
  let last = 0;
  for (const m of re ? text.matchAll(re) : []) {
    if (!m[0]) continue;
    parts.push(text.slice(last, m.index), el("mark", { className, textContent: m[0] }));
    last = m.index + m[0].length;
  }
  parts.push(text.slice(last));
  body.replaceChildren(...parts);
}

// By default the gold and predicted skills are marked; hovering a tag marks just that tag.
function highlight(option) {
  if (option) {
    markBody(new RegExp(patternFor(option), "gi"), "hover");
    document.querySelector("#body mark")?.scrollIntoView({ block: "center", behavior: "smooth" });
    return;
  }
  const skills = new Set([...(current.labels.skills ?? []), ...(current.prediction?.skills ?? [])]);
  const sources = [...skills].map(patternFor).map(p => `(?:${p})`);
  markBody(sources.length ? new RegExp(sources.join("|"), "gi") : null, "");
}

// ---------- wiring ----------

document.getElementById("run").onchange = e => openRun(e.target.value, current?.id);
document.getElementById("sort").onchange = renderList;
document.getElementById("only-issues").onchange = renderList;
document.addEventListener("keydown", e => {
  if (!e.altKey) return;
  if (e.key === "ArrowUp" || e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
  if (e.key === "ArrowDown" || e.key === "ArrowRight") { e.preventDefault(); step(1); }
});

(async () => {
  meta = await api("/api/meta");
  const runs = await loadRuns();
  const usable = runs.filter(r => r.records);
  if (!usable.length) {
    document.querySelector("main").replaceChildren(el("div", { className: "empty-note",
      textContent: "No runs with records in data/cache — run python -m evaluation.run first." }));
    return;
  }
  const wanted = usable.find(r => r.name === params.get("run")) ?? usable[0];
  openRun(wanted.name, params.get("id"));
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
