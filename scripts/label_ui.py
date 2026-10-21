"""Local web UI for labeling postings: data/raw/pNNN.txt on the left, schema labels on the right.

Clicking a label writes it to data/labels/pNNN.json. Enum fields hold one value (clicking sets
it), list fields toggle values in and out. Typing a new tag into a field's "+ new" box adds it as a
member of that field's enum in src/job_extractor/schema.py and applies it. Run from the repo root:

  uv run python scripts/label_ui.py        # then open http://127.0.0.1:8000
"""

import ast
import importlib
import json
import keyword
import re
import types
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError

import job_extractor.schema as schema
from job_extractor.schema import JobPosting

RAW = Path("data/raw")
LABELS = Path("data/labels")
INDEX = Path("data/postings.jsonl")
POSTING_ID = re.compile(r"^p\d+$")
TAG = re.compile(r"^[a-z][a-z0-9_]*$")


def describe_field(annotation: Any) -> dict[str, Any]:
    """Map a JobPosting annotation to a UI widget: enum, multi, int or text."""
    origin = get_origin(annotation)
    if origin is list:
        (item,) = get_args(annotation)
        return {"kind": "multi", "enum": item.__name__, "options": sorted(m.value for m in item)}
    if origin in (Union, types.UnionType):
        (inner,) = [a for a in get_args(annotation) if a is not type(None)]
        return describe_field(inner)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {
            "kind": "enum",
            "enum": annotation.__name__,
            "options": sorted(m.value for m in annotation),
        }
    if annotation is int:
        return {"kind": "int"}
    return {"kind": "text"}


def build_fields() -> dict[str, dict[str, Any]]:
    return {name: describe_field(f.annotation) for name, f in JobPosting.model_fields.items()}


FIELDS = build_fields()


def add_enum_member(class_name: str, value: str) -> None:
    """Append `value = "value"` to the enum class in schema.py, then reload the schema."""
    global JobPosting, FIELDS
    path = Path(schema.__file__)
    source = path.read_text(encoding="utf-8")
    (node,) = [
        n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == class_name
    ]
    lines = source.splitlines(keepends=True)
    lines.insert(node.end_lineno, f'    {value} = "{value}"\n')
    path.write_text("".join(lines), encoding="utf-8")
    importlib.reload(schema)
    JobPosting = schema.JobPosting
    FIELDS = build_fields()


def label_path(posting_id: str) -> Path:
    if not POSTING_ID.match(posting_id) or not (RAW / f"{posting_id}.txt").exists():
        raise HTTPException(404, f"unknown posting {posting_id}")
    return LABELS / f"{posting_id}.json"


def read_labels(posting_id: str) -> dict[str, Any]:
    path = label_path(posting_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_labels(posting_id: str, labels: dict[str, Any]) -> None:
    # Schema field order, one field per line, lists kept inline (matches the hand-written files).
    lines = [
        f"  {json.dumps(name)}: {json.dumps(labels[name], ensure_ascii=False)}"
        for name in FIELDS
        if name in labels
    ]
    LABELS.mkdir(parents=True, exist_ok=True)
    label_path(posting_id).write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")


def validation_errors(labels: dict[str, Any]) -> list[str]:
    try:
        JobPosting.model_validate(labels)
    except ValidationError as e:
        return [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
    return []


def posting_state(posting_id: str) -> dict[str, Any]:
    labels = read_labels(posting_id)
    return {"labels": labels, "errors": validation_errors(labels)}


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


@app.get("/api/fields")
def fields() -> dict[str, Any]:
    return FIELDS


@app.get("/api/postings")
def postings() -> list[dict[str, Any]]:
    meta = {record["id"]: record for record in read_index()}
    result = []
    for path in sorted(RAW.glob("p*.txt")):
        pid = path.stem
        labels_file = LABELS / f"{pid}.json"
        status = "none"
        if labels_file.exists():
            labels = json.loads(labels_file.read_text(encoding="utf-8"))
            status = "invalid" if validation_errors(labels) else "done"
        record = meta.get(pid, {})
        result.append(
            {
                "id": pid,
                "title": record.get("title", ""),
                "company": record.get("company", ""),
                "status": status,
            }
        )
    return result


class NewPosting(BaseModel):
    company: str
    title: str
    location: str = ""
    url: str = ""
    text: str


def read_index() -> list[dict[str, Any]]:
    if not INDEX.exists():
        return []
    lines = INDEX.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@app.post("/api/postings")
def add_posting(new: NewPosting) -> dict[str, Any]:
    """Save pasted posting text as the next pNNN, the same way scripts/fetch_postings.py does."""
    company, title, body = new.company.strip().lower(), new.title.strip(), new.text.strip()
    if not (company and title and body):
        raise HTTPException(400, "company, title and text are required")
    existing = read_index()
    url = new.url.strip()
    if url and any(p["url"] == url for p in existing):
        raise HTTPException(400, f"{url} is already in {INDEX}")

    # Count raw files too, so an id never collides with a text file missing from the index.
    used = [int(p["id"][1:]) for p in existing] + [int(p.stem[1:]) for p in RAW.glob("p*.txt")]
    pid = f"p{max(used, default=0) + 1:03d}"
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{pid}.txt").write_text(f"{title}\n{new.location.strip()}\n\n{body}", encoding="utf-8")

    record = {
        "id": pid,
        "url": url,
        "company": company,
        "title": title,
        "date": date.today().isoformat(),
        "source": "manual",
    }
    needs_newline = INDEX.exists() and not INDEX.read_text(encoding="utf-8").endswith("\n")
    with INDEX.open("a", encoding="utf-8") as index:
        index.write(("\n" if needs_newline else "") + json.dumps(record, ensure_ascii=False) + "\n")
    return record


@app.get("/api/postings/{posting_id}")
def posting(posting_id: str) -> dict[str, Any]:
    label_path(posting_id)  # validates the id
    text = (RAW / f"{posting_id}.txt").read_text(encoding="utf-8")
    return {"id": posting_id, "text": text} | posting_state(posting_id)


class NewOption(BaseModel):
    value: str


@app.post("/api/fields/{field}/options")
def add_option(field: str, option: NewOption) -> dict[str, Any]:
    spec = FIELDS.get(field)
    if spec is None or "enum" not in spec:
        raise HTTPException(400, f"{field} has no fixed tags")
    value = re.sub(r"[\s\-]+", "_", option.value.strip().lower())
    if not TAG.match(value) or keyword.iskeyword(value):
        raise HTTPException(400, f"{value!r} is not a valid tag (use snake_case)")
    if value not in spec["options"]:
        add_enum_member(spec["enum"], value)
    return {"fields": FIELDS, "value": value}


class LabelChange(BaseModel):
    field: str
    value: Any = None


@app.post("/api/postings/{posting_id}/labels")
def change_label(posting_id: str, change: LabelChange) -> dict[str, Any]:
    spec = FIELDS.get(change.field)
    if spec is None:
        raise HTTPException(400, f"unknown field {change.field}")
    labels = read_labels(posting_id)
    value = change.value
    if spec["kind"] in ("enum", "multi") and value not in spec["options"]:
        raise HTTPException(400, f"{value!r} is not a valid {change.field}")

    if spec["kind"] == "multi":
        current = labels.get(change.field) or []
        labels[change.field] = (
            [v for v in current if v != value] if value in current else [*current, value]
        )
    elif spec["kind"] == "enum":
        labels[change.field] = value
    elif spec["kind"] == "int":
        labels[change.field] = None if value in (None, "") else int(value)
    else:
        labels[change.field] = str(value or "").strip()

    # A fresh file gets the optional / list fields too, so it only needs the required picks.
    for name, field_spec in FIELDS.items():
        if name not in labels:
            if field_spec["kind"] == "multi":
                labels[name] = []
            elif not JobPosting.model_fields[name].is_required():
                labels[name] = JobPosting.model_fields[name].default
    write_labels(posting_id, labels)
    return posting_state(posting_id)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Label postings</title>
<style>
  :root { --bg:#f4f3ef; --panel:#fff; --text:#1f1e1c; --muted:#78766f; --faint:#a8a69e;
          --line:#e7e5de; --chip:#f2f0ea; --chip-hover:#e9e6dd; --on:#2f6f4f; --on-text:#fff;
          --on-soft:#e2efe7; --warn:#b4532a; --warn-soft:#fbefe7; --mark:#fcecaa;
          --seen:#d9a400; --seen-soft:#fbeeb8; --seen-hover:#f6e29a;
          --shadow:0 1px 2px rgb(0 0 0 / .04), 0 2px 8px rgb(0 0 0 / .04); }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#171716; --panel:#1f1f1d; --text:#e9e8e4; --muted:#9d9b94; --faint:#6c6a64;
            --line:#33322f; --chip:#2a2927; --chip-hover:#353431; --on:#5bb085;
            --on-text:#0c1a12; --on-soft:#1f3329; --warn:#e5866a; --warn-soft:#3a241c;
            --mark:#584a12; --seen:#e2b93b; --seen-soft:#3d3412;
            --seen-hover:#4a3f15; --shadow:none; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif;
         background:var(--bg); color:var(--text); height:100vh; display:flex;
         flex-direction:column; -webkit-font-smoothing:antialiased; }
  ::-webkit-scrollbar { width:10px; }
  ::-webkit-scrollbar-thumb { background:var(--line); border-radius:999px;
                              border:3px solid transparent; background-clip:content-box; }

  /* Header */
  header { display:flex; gap:8px; align-items:center; padding:10px 20px;
           border-bottom:1px solid var(--line); background:var(--panel); }
  .brand { font-weight:700; letter-spacing:-.01em; margin-right:8px; white-space:nowrap; }
  .brand span { color:var(--on); }
  header select { flex:1; max-width:520px; min-width:0; }
  select, input, button { font:inherit; color:inherit; background:var(--chip);
                          border:1px solid transparent; border-radius:8px; padding:6px 10px;
                          transition:background .12s, border-color .12s, color .12s; }
  select:focus, input:focus, textarea:focus { outline:none; border-color:var(--on);
                                              box-shadow:0 0 0 3px var(--on-soft); }
  button { cursor:pointer; }
  button:hover { background:var(--chip-hover); }
  .icon { width:34px; padding:6px 0; }
  .primary { background:var(--on); color:var(--on-text); font-weight:600; }
  .primary:hover { background:var(--on); filter:brightness(1.08); }
  #progress { display:flex; align-items:center; gap:8px; margin-left:auto; color:var(--muted);
              font-size:12px; white-space:nowrap; }
  #progress .bar { width:120px; height:6px; border-radius:999px; background:var(--chip);
                   overflow:hidden; }
  #progress .fill { height:100%; background:var(--on); border-radius:999px;
                    transition:width .3s; }
  kbd { font:11px ui-monospace, monospace; padding:1px 5px; border-radius:4px;
        border:1px solid var(--line); border-bottom-width:2px; background:var(--panel); }
  .hint { color:var(--faint); font-size:12px; padding:6px 20px; display:flex; gap:16px;
          border-bottom:1px solid var(--line); background:var(--panel); }
  .hint .dot { display:inline-block; width:7px; height:7px; border-radius:50%;
               background:var(--seen); margin-right:4px; vertical-align:1px; }

  main { flex:1; display:grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); min-height:0; }

  /* Posting text */
  #text { overflow:auto; background:var(--panel); border-right:1px solid var(--line); }
  #text > .inner { padding:24px 28px 64px; }
  #text .meta { font-size:12px; color:var(--muted); text-transform:uppercase;
                letter-spacing:.06em; font-weight:600; }
  #text .meta .id { color:var(--faint); font-family:ui-monospace, monospace;
                    text-transform:none; letter-spacing:0; margin-left:6px; }
  #text h1 { font-size:24px; line-height:1.25; letter-spacing:-.015em; margin:6px 0 8px; }
  #text .loc { color:var(--muted); padding-bottom:20px; margin-bottom:24px;
               border-bottom:1px solid var(--line); }
  #body { white-space:pre-wrap; font-size:15px; line-height:1.7; }

  /* Label panel */
  #labels { overflow:auto; padding:20px 24px 64px; }
  #labels > * { max-width:760px; margin-left:auto; margin-right:auto; }
  .field { background:var(--panel); border:1px solid var(--line); border-radius:12px;
           padding:12px 14px 14px; margin-bottom:10px; box-shadow:var(--shadow); }
  .field.missing { border-left:3px solid var(--warn); }
  .field h2 { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted);
              margin:0 0 10px; display:flex; gap:8px; align-items:center; font-weight:700; }
  .field h2 .count { background:var(--chip); color:var(--muted); border-radius:999px;
                     padding:0 7px; letter-spacing:0; font-size:11px; }
  .field h2 .count.some { background:var(--on-soft); color:var(--on); }
  .field.missing h2 .req { color:var(--warn); text-transform:none; letter-spacing:0;
                           font-weight:500; margin-left:auto; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; }
  .chip { border-radius:999px; padding:4px 12px; font-size:13px; line-height:1.4; }
  .chip:hover { background:var(--chip-hover); }
  .chip.seen { background:var(--seen-soft); box-shadow:inset 0 0 0 1px var(--seen);
              font-weight:600; }
  .chip.seen:hover { background:var(--seen-hover); }
  .chip.seen::after { content:""; display:inline-block; width:6px; height:6px;
                      border-radius:50%; background:var(--seen); margin-left:6px;
                      vertical-align:1px; }
  .chip.on { background:var(--on); color:var(--on-text); font-weight:600; box-shadow:none; }
  .chip.on:hover { background:var(--on); filter:brightness(1.08); }
  .chip.on::before { content:"✓ "; }
  .chip.on.seen::after { background:var(--on-text); opacity:.7; }
  .chip.new { width:96px; background:transparent; border:1px dashed var(--line);
              color:var(--muted); }
  .chip.new:focus { width:160px; border-style:solid; }
  .field input:not(.chip) { width:100%; max-width:280px; background:var(--chip); }

  #status { padding:10px 14px; border-radius:10px; margin-bottom:14px; font-size:13px;
            display:flex; gap:8px; align-items:flex-start; }
  #status.ok { background:var(--on-soft); color:var(--on); font-weight:600; }
  #status.bad { background:var(--warn-soft); color:var(--warn); }
  #status.empty { background:var(--chip); color:var(--muted); }

  mark { background:var(--mark); color:inherit; border-radius:3px; padding:0 1px; }
  mark.hover { background:var(--on); color:var(--on-text); }

  /* New posting dialog */
  dialog { background:var(--panel); color:var(--text); border:1px solid var(--line);
           border-radius:14px; width:min(720px, 92vw); padding:24px;
           box-shadow:0 20px 60px rgb(0 0 0 / .25); }
  dialog::backdrop { background:rgb(0 0 0 / .35); backdrop-filter:blur(2px); }
  dialog h3 { margin:0 0 12px; font-size:17px; }
  dialog form { display:grid; gap:12px; }
  dialog .row { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  dialog label { display:grid; gap:4px; font-size:12px; font-weight:600; color:var(--muted); }
  dialog input, dialog textarea { font-weight:400; color:var(--text); }
  dialog textarea { font:inherit; font-weight:400; color:inherit; background:var(--chip);
                    min-height:40vh; border:1px solid transparent; border-radius:8px;
                    padding:8px 10px; resize:vertical; }
  dialog menu { display:flex; justify-content:flex-end; gap:8px; margin:4px 0 0; padding:0; }
</style>
</head>
<body>
<header>
  <div class="brand">Job<span>Labeler</span></div>
  <button id="prev" class="icon" title="Previous (Alt+←)">←</button>
  <select id="picker"></select>
  <button id="next" class="icon" title="Next (Alt+→)">→</button>
  <button id="add" class="primary" title="Paste the text of a new posting">+ New posting</button>
  <div id="progress"><span id="progress-text"></span>
    <div class="bar"><div class="fill" id="progress-fill"></div></div></div>
</header>
<div class="hint">
  <span><kbd>Alt</kbd> <kbd>←</kbd> <kbd>→</kbd> switch posting</span>
  <span>Click a label to add / remove it · hover to find it in the text</span>
  <span><span class="dot"></span>mentioned in the posting</span>
</div>
<dialog id="new-posting">
  <h3>Add a posting</h3>
  <form method="dialog">
    <div class="row">
      <label>Company * <input name="company" required placeholder="e.g. getyourguide"></label>
      <label>Title * <input name="title" required></label>
    </div>
    <div class="row">
      <label>Location <input name="location"></label>
      <label>URL <input name="url" type="url"></label>
    </div>
    <label>Posting text * <textarea name="text" required></textarea></label>
    <menu>
      <button value="cancel" formnovalidate>Cancel</button>
      <button value="save" class="primary">Save</button>
    </menu>
  </form>
</dialog>
<main>
  <section id="text"></section>
  <section id="labels"><div id="status"></div><div id="form"></div></section>
</main>
<script>
const STATUS_ICON = { none: "○", invalid: "◐", done: "●" };
let fields = {}, postings = [], current = null;

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

async function loadPostings() {
  postings = await api("/api/postings");
  const picker = document.getElementById("picker");
  const selected = picker.value;
  picker.replaceChildren(...postings.map(p => el("option", {
    value: p.id,
    textContent: `${STATUS_ICON[p.status]} ${p.id} · ${p.company} · ${p.title}`,
  })));
  if (selected) picker.value = selected;
  const done = postings.filter(p => p.status === "done").length;
  document.getElementById("progress-text").textContent = `${done} / ${postings.length} labeled`;
  document.getElementById("progress-fill").style.width =
    `${postings.length ? (100 * done) / postings.length : 0}%`;
}

async function open(id) {
  current = await api(`/api/postings/${id}`);
  document.getElementById("picker").value = id;
  history.replaceState(null, "", `#${id}`);
  const [title, location, ...rest] = current.text.split("\\n");
  const company = postings.find(p => p.id === id)?.company ?? "";
  document.getElementById("text").replaceChildren(el("div", { className: "inner" }, [
    el("div", { className: "meta" }, [company, el("span", { className: "id", textContent: id })]),
    el("h1", { textContent: title }),
    el("div", { className: "loc", textContent: location || "Location not given" }),
    el("div", { id: "body", textContent: rest.join("\\n").trim() }),
  ]));
  document.getElementById("text").scrollTop = 0;
  render();
}

async function change(field, value) {
  Object.assign(current, await api(`/api/postings/${current.id}/labels`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ field, value }),
  }));
  render();
  loadPostings();
}

async function addOption(field, value) {
  const res = await api(`/api/fields/${field}/options`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value }),
  });
  fields = res.fields;
  const chosen = current.labels[field];
  // Apply the new tag, unless it already existed and is already on.
  if (!(Array.isArray(chosen) ? chosen.includes(res.value) : chosen === res.value)) {
    await change(field, res.value);
  } else {
    render();
  }
}

function render() {
  const labels = current.labels;
  const status = document.getElementById("status");
  const fresh = !Object.keys(labels).length;
  if (fresh) {
    status.className = "empty";
    status.textContent = "Not labeled yet — click a label to create the file.";
  } else if (current.errors.length) {
    status.className = "bad";
    status.textContent = "Incomplete: " + current.errors.join("; ");
  } else {
    status.className = "ok";
    status.textContent = `✓ data/labels/${current.id}.json is valid`;
  }

  document.getElementById("form").replaceChildren(...Object.entries(fields).map(([name, spec]) => {
    const value = labels[name];
    const missing = !fresh
      && (!(name in labels) || current.errors.some(e => e.startsWith(name + ":")));
    const heading = el("h2", { textContent: name.replaceAll("_", " ") });
    if (missing) heading.append(el("span", { className: "req", textContent: "needs a value" }));
    let body;
    if (spec.kind === "enum" || spec.kind === "multi") {
      const chosen = spec.kind === "multi" ? (value ?? []) : [value];
      if (spec.kind === "multi") heading.insertBefore(el("span", {
        className: "count" + (chosen.length ? " some" : ""),
        textContent: chosen.length,
      }), heading.childNodes[1] ?? null);
      body = el("div", { className: "chips" }, spec.options.map(opt => el("button", {
        className: "chip" + (chosen.includes(opt) ? " on" : "") + (mentioned(opt) ? " seen" : ""),
        title: mentioned(opt) ? "Mentioned in the posting" : "",
        textContent: opt,
        onclick: () => change(name, opt),
        onmouseenter: () => highlight(opt),
        onmouseleave: () => highlight(null),
      })));
      body.append(el("input", {
        className: "chip new",
        placeholder: "+ new",
        title: `Add a new ${name} tag to schema.py (Enter)`,
        onkeydown: e => {
          if (e.key === "Enter" && e.target.value.trim()) addOption(name, e.target.value);
        },
      }));
    } else {
      body = el("input", {
        type: spec.kind === "int" ? "number" : "text",
        value: value ?? "",
        placeholder: spec.kind === "int" ? "empty = null" : "",
        onchange: e => change(name, e.target.value),
      });
    }
    return el("div", { className: "field" + (missing ? " missing" : "") }, [heading, body]);
  }));
  highlight(null);
}

// Catch-all values that would match ordinary prose ("other", "not") rather than a real mention.
const GENERIC = new Set(["other", "unspecified", "not_mentioned"]);

const escapeRe = s => s.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&");

// Regex source matching the whole tag: "scikit_learn" matches "scikit-learn",
// "ci_cd" matches "CI/CD", "llm" matches "LLMs".
const tagPattern = option =>
  `\\\\b${option.split("_").map(escapeRe).join("[\\\\s_/-]*")}s?\\\\b`;

// Whether the whole tag appears in the posting.
function mentioned(option) {
  return !GENERIC.has(option) && new RegExp(tagPattern(option), "i").test(current.text);
}

// One regex for every tag of every field, longest first so "machine_learning" beats "machine".
function allTagsRegex() {
  const options = [...new Set(Object.values(fields).flatMap(spec => spec.options ?? []))]
    .filter(opt => !GENERIC.has(opt))
    .sort((a, b) => b.length - a.length);
  return options.length ? new RegExp(options.map(tagPattern).join("|"), "gi") : null;
}

// Wrap every match of `re` in the posting body in <mark class=className>.
function markBody(re, className) {
  const body = document.getElementById("body");
  const text = body.textContent;
  const parts = [];
  let last = 0;
  for (const m of re ? text.matchAll(re) : []) {
    parts.push(text.slice(last, m.index), el("mark", { className, textContent: m[0] }));
    last = m.index + m[0].length;
  }
  parts.push(text.slice(last));
  body.replaceChildren(...parts);
}

// By default every tag mention is marked; hovering a label marks just its words instead.
function highlight(option) {
  if (!option) { markBody(allTagsRegex(), ""); return; }
  const words = option.split("_").filter(w => w.length > 1).map(escapeRe);
  markBody(new RegExp(`\\\\b(${words.join("|")})`, "gi"), "hover");
  document.querySelector("#body mark")?.scrollIntoView({ block: "center", behavior: "smooth" });
}

function step(delta) {
  const i = postings.findIndex(p => p.id === current.id) + delta;
  if (i >= 0 && i < postings.length) open(postings[i].id);
}

const dialog = document.getElementById("new-posting");
document.getElementById("add").onclick = () => dialog.showModal();
dialog.onclose = async () => {
  if (dialog.returnValue !== "save") return;
  const form = dialog.querySelector("form");
  const record = await api("/api/postings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(Object.fromEntries(new FormData(form))),
  });
  form.reset();
  await loadPostings();
  open(record.id);
};

document.getElementById("picker").onchange = e => open(e.target.value);
document.getElementById("prev").onclick = () => step(-1);
document.getElementById("next").onclick = () => step(1);
document.addEventListener("keydown", e => {
  if (e.altKey && e.key === "ArrowLeft") step(-1);
  if (e.altKey && e.key === "ArrowRight") step(1);
});

(async () => {
  fields = await api("/api/fields");
  await loadPostings();
  const fromHash = location.hash.slice(1);
  const firstTodo = postings.find(p => p.status !== "done");
  open(postings.some(p => p.id === fromHash) ? fromHash : (firstTodo ?? postings[0]).id);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
