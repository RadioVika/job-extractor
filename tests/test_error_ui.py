import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import error_ui  # noqa: E402
import label_ui  # noqa: E402

GOLD = {
    "role_type": "data_scientist",
    "seniority": "senior",
    "work_mode": "hybrid",
    "german": "not_mentioned",
    "skills": ["python", "sql", "spark"],
    "clouds": ["aws"],
    "data_types": [],
    "industry": "travel",
    "salary_min_eur": 70000,
    "salary_max_eur": None,
}
TEXT = "Data Scientist\nBerlin\n\nWe use Python and SQL on AWS. Salary from 70,000 EUR.\n"
RUN = "20261027-120000_dev_short_anthropic_test-model_v1"


@pytest.fixture
def client(tmp_path, monkeypatch):
    raw, labels, cache = tmp_path / "raw", tmp_path / "labels", tmp_path / "cache"
    for d in (raw, labels, cache):
        d.mkdir()
    for pid in ("p001", "p002"):
        (raw / f"{pid}.txt").write_text(TEXT, encoding="utf-8")
    (labels / "p001.json").write_text(json.dumps(GOLD), encoding="utf-8")
    pred = GOLD | {"seniority": "mid", "skills": ["python", "sql", "pytorch"]}
    records = [
        {
            "id": "p001",
            "posting": pred,
            "provider": "anthropic",
            "model": "test-model",
            "prompt_version": "v1",
            "input_tokens": 100,
            "output_tokens": 10,
            "latency_s": 1.5,
            "repaired": False,
        },
        {"id": "p002", "error": "LLMError: boom"},
    ]
    (cache / f"{RUN}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    (cache / "20261026-090000_dev_openai_older_v1.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(label_ui, "RAW", raw)
    monkeypatch.setattr(label_ui, "LABELS", labels)
    monkeypatch.setattr(label_ui, "INDEX", tmp_path / "postings.jsonl")
    monkeypatch.setattr(error_ui, "CACHE", cache)
    return TestClient(error_ui.app)


def test_runs_newest_first_with_parsed_names(client):
    runs = client.get("/api/runs").json()
    assert [r["name"] for r in runs] == [RUN, "20261026-090000_dev_openai_older_v1"]
    assert runs[0]["split"] == "dev_short"
    assert (runs[0]["provider"], runs[0]["model"], runs[0]["prompt"]) == (
        "anthropic",
        "test-model",
        "v1",
    )
    assert [r["records"] for r in runs] == [2, 0]


def test_unknown_run_or_posting_is_404(client):
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/..%2Fraw").status_code == 404
    assert client.get(f"/api/runs/{RUN}/postings/p999").status_code == 404


def test_posting_metrics(client):
    m = client.get(f"/api/runs/{RUN}/postings/p001").json()["metrics"]
    assert m["accuracy"] == {
        "seniority": False,
        "role_type": True,
        "work_mode": True,
        "german": True,
    }
    skills = m["lists"]["skills"]
    assert (skills["matched"], skills["extra"], skills["missing"]) == (
        ["python", "sql"],
        ["pytorch"],
        ["spark"],
    )
    assert skills["precision"] == pytest.approx(2 / 3)
    assert skills["recall"] == pytest.approx(2 / 3)
    assert m["skills_invented_not_in_label"] == pytest.approx(1 / 3)
    assert m["skills_not_in_text"] == ["pytorch"]
    assert m["lists"]["data_types"]["precision"] == 1.0  # both empty is a perfect score
    assert m["salary_not_in_text"] == []
    assert m["stats"]["latency_s"] == 1.5
    assert m["issues"] == 3  # seniority, one extra skill, one missing skill


def test_run_lists_failures_and_scores_live(client):
    run = client.get(f"/api/runs/{RUN}").json()
    by_id = {p["id"]: p for p in run["postings"]}
    assert by_id["p002"]["error"] == "LLMError: boom"
    assert by_id["p001"]["issues"] == 3
    assert run["scored"] == 1
    assert run["live"]["schema_valid_rate"] == 0.5
    assert run["live"]["seniority_accuracy"] == 0.0
    assert run["snapshot"] is None


def test_label_edit_updates_file_and_metrics(client):
    url = f"/api/runs/{RUN}/postings/p001/labels"
    detail = client.post(url, json={"field": "seniority", "value": "mid"}).json()
    assert detail["metrics"]["accuracy"]["seniority"] is True
    # A list value replaces the whole field: "use as gold" for skills.
    detail = client.post(url, json={"field": "skills", "value": ["python", "sql", "pytorch"]})
    assert detail.json()["metrics"]["issues"] == 0
    saved = json.loads((label_ui.LABELS / "p001.json").read_text(encoding="utf-8"))
    assert saved["seniority"] == "mid"
    assert saved["skills"] == ["python", "sql", "pytorch"]
    assert client.get(f"/api/runs/{RUN}").json()["live"]["skills_precision"] == 1.0


def test_label_edit_rejects_unknown_values(client):
    url = f"/api/runs/{RUN}/postings/p001/labels"
    response = client.post(url, json={"field": "skills", "value": ["python", "cobol"]})
    assert response.status_code == 400
