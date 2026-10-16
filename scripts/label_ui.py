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
    meta = {}
    if INDEX.exists():
        for line in INDEX.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                meta[record["id"]] = record
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
  :root { --bg:#f7f7f5; --panel:#fff; --text:#1d1d1b; --muted:#6b6b66; --line:#e2e1dc;
          --chip:#efeeea; --on:#2f6f4f; --on-text:#fff; --warn:#a4462b; --mark:#fff3b0; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#191918; --panel:#222220; --text:#e8e7e3; --muted:#9a9993; --line:#3a3936;
            --chip:#2e2d2b; --on:#4f9e74; --on-text:#0e1a13; --warn:#e07a5c; --mark:#5a4d12; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 system-ui, sans-serif; background:var(--bg); color:var(--text);
         height:100vh; display:flex; flex-direction:column; }
  header { display:flex; gap:8px; align-items:center; padding:8px 16px;
           border-bottom:1px solid var(--line); background:var(--panel); }
  header select { flex:1; max-width:560px; }
  select, input, button { font:inherit; color:inherit; background:var(--chip);
                          border:1px solid var(--line); border-radius:6px; padding:4px 8px; }
  button { cursor:pointer; }
  .hint { color:var(--muted); font-size:12px; margin-left:auto; }
  main { flex:1; display:grid; grid-template-columns: 1fr 1fr; min-height:0; }
  #text { overflow:auto; padding:16px 24px; white-space:pre-wrap; background:var(--panel);
          border-right:1px solid var(--line); }
  #text h1 { font-size:18px; margin:0; white-space:normal; }
  #text .loc { color:var(--muted); margin-bottom:12px; }
  #labels { overflow:auto; padding:16px 24px; }
  .field { margin-bottom:16px; }
  .field h2 { font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
              margin:0 0 6px; display:flex; gap:8px; }
  .field.missing h2 { color:var(--warn); }
  .chips { display:flex; flex-wrap:wrap; gap:6px; }
  .chip { border-radius:999px; padding:3px 10px; }
  .chip:hover { border-color:var(--on); }
  .chip.new { width:90px; background:transparent; border-style:dashed; }
  .chip.seen { background:var(--mark); }
  .chip.on { background:var(--on); color:var(--on-text); border-color:var(--on); font-weight:600; }
  #status { padding:8px 12px; border-radius:6px; margin-bottom:16px; font-size:13px; }
  #status.ok { background:var(--chip); }
  #status.bad { color:var(--warn); border:1px solid var(--warn); }
  mark { background:var(--mark); color:inherit; }
</style>
</head>
<body>
<header>
  <button id="prev" title="Previous (Alt+←)">←</button>
  <select id="picker"></select>
  <button id="next" title="Next (Alt+→)">→</button>
  <span class="hint">Alt+← / Alt+→ to switch posting · click a label to add / remove it</span>
</header>
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
}

async function open(id) {
  current = await api(`/api/postings/${id}`);
  document.getElementById("picker").value = id;
  history.replaceState(null, "", `#${id}`);
  const [title, location, ...rest] = current.text.split("\\n");
  document.getElementById("text").replaceChildren(
    el("h1", { textContent: title }),
    el("div", { className: "loc", textContent: location }),
    el("div", { id: "body", textContent: rest.join("\\n").trim() }),
  );
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
  if (!Object.keys(labels).length) {
    status.className = "bad";
    status.textContent = "No label file yet — click a label to create it.";
  } else if (current.errors.length) {
    status.className = "bad";
    status.textContent = "Incomplete: " + current.errors.join("; ");
  } else {
    status.className = "ok";
    status.textContent = `✓ data/labels/${current.id}.json is valid`;
  }

  document.getElementById("form").replaceChildren(...Object.entries(fields).map(([name, spec]) => {
    const value = labels[name];
    const missing = !(name in labels) || current.errors.some(e => e.startsWith(name + ":"));
    const heading = el("h2", { textContent: name.replaceAll("_", " ") });
    let body;
    if (spec.kind === "enum" || spec.kind === "multi") {
      const chosen = spec.kind === "multi" ? (value ?? []) : [value];
      if (spec.kind === "multi") heading.append(el("span", { textContent: `(${chosen.length})` }));
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
}

// Catch-all values that would match ordinary prose ("other", "not") rather than a real mention.
const GENERIC = new Set(["other", "unspecified", "not_mentioned"]);

// Whether the whole tag appears in the posting: "scikit_learn" matches "scikit-learn",
// "ci_cd" matches "CI/CD", "llm" matches "LLMs".
function mentioned(option) {
  if (GENERIC.has(option)) return false;
  const words = option.split("_").map(w => w.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&"));
  return new RegExp(`\\\\b${words.join("[\\\\s_/-]*")}s?\\\\b`, "i").test(current.text);
}

// Hovering a label marks where its words appear in the posting text.
function highlight(option) {
  const body = document.getElementById("body");
  const text = body.textContent;
  if (!option) { body.textContent = text; return; }
  const words = option.split("_").filter(w => w.length > 1)
    .map(w => w.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&"));
  const re = new RegExp(`\\\\b(${words.join("|")})`, "gi");
  const parts = [];
  let last = 0;
  for (const m of text.matchAll(re)) {
    parts.push(text.slice(last, m.index), el("mark", { textContent: m[0] }));
    last = m.index + m[0].length;
  }
  parts.push(text.slice(last));
  body.replaceChildren(...parts);
  body.querySelector("mark")?.scrollIntoView({ block: "center", behavior: "smooth" });
}

function step(delta) {
  const i = postings.findIndex(p => p.id === current.id) + delta;
  if (i >= 0 && i < postings.length) open(postings[i].id);
}

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
