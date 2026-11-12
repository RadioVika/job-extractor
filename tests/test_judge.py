import json

import pytest
from tenacity import wait_none

from evaluation.judge import (
    InvalidVerdict,
    JudgeError,
    JudgeVerdict,
    check_coverage,
    judge,
    run_judge,
    system_prompt,
    user_prompt,
)
from evaluation.judge_metrics import (
    Agreement,
    Judged,
    field_agreement,
    item_agreement,
    report_metrics,
    same_ranking,
)
from evaluation.judge_run import load_judged, report_table, score
from evaluation.prices import price_for
from job_extractor.llm import LLMResponse
from job_extractor.schema import JobPosting


def posting(skills=(), clouds=(), **fields):
    defaults = {
        "role_type": "ml_engineer",
        "seniority": "mid",
        "work_mode": "remote",
        "german": "not_mentioned",
    }
    return JobPosting(skills=list(skills), clouds=list(clouds), **(defaults | fields))


def verdict(pred, wrong=(), wrong_items=(), missing_skills=(), missing_clouds=()):
    """A verdict on pred: fields in `wrong` and items in `wrong_items` are judged incorrect."""
    field = {"reason": "r", "correct": True}
    fields = ("role_type", "seniority", "work_mode", "german", "salary_min_eur", "salary_max_eur")
    return JudgeVerdict(
        **{f: field | {"correct": f not in wrong} for f in fields},
        skills=[{"skill": s, "reason": "r", "correct": s not in wrong_items} for s in pred.skills],
        missing_skills=[{"skill": s, "reason": "r"} for s in missing_skills],
        clouds=[{"cloud": c, "reason": "r", "correct": c not in wrong_items} for c in pred.clouds],
        missing_clouds=[{"cloud": c, "reason": "r"} for c in missing_clouds],
    )


class FakeClient:
    provider = "fake"
    model = "fake-judge"

    def __init__(self, *responses):
        self.calls = []
        self.queue = list(responses)

    def parse(self, system, user, schema):
        self.calls.append({"system": system, "user": user, "schema": schema})
        item = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if isinstance(item, Exception):
            raise item
        return item


def response(parsed, stop="ok"):
    return LLMResponse(
        parsed=parsed,
        model="fake-judge",
        input_tokens=1000,
        output_tokens=200,
        stop=stop,
        detail=None,
        text="{}",
    )


def test_system_prompt_has_guide_and_schema():
    system = system_prompt()
    assert "<labeling_guide>" in system and "# Labeling guide" in system
    assert "<schema>" in system and '"salary_min_eur"' in system


def test_user_prompt_has_text_and_prediction_only():
    pred = posting(["python"])
    user = user_prompt("Senior ML Engineer\nBerlin\n\nPython", pred)
    assert "<posting>\nSenior ML Engineer" in user
    assert json.loads(user.split("<prediction>\n")[1].split("\n</prediction>")[0]) == (
        pred.model_dump(mode="json")
    )


def test_check_coverage_accepts_full_verdict():
    pred = posting(["python", "sql"], ["aws"])
    check_coverage(verdict(pred, missing_skills=["spark"]), pred)


@pytest.mark.parametrize(
    "bad",
    [
        lambda pred: verdict(posting(["python"])),  # sql left out
        lambda pred: verdict(posting(["python", "sql", "spark"])),  # spark not predicted
        lambda pred: verdict(pred, missing_skills=["sql"]),  # missing lists a predicted item
    ],
)
def test_check_coverage_rejects(bad):
    pred = posting(["python", "sql"])
    with pytest.raises(InvalidVerdict):
        check_coverage(bad(pred), pred)


def test_judge_repairs_once_then_succeeds():
    pred = posting(["python", "sql"])
    client = FakeClient(response(verdict(posting(["python"]))), response(verdict(pred)))
    result = judge("text", pred, client, "system", wait=wait_none())
    assert result.repaired and len(client.calls) == 2
    assert "previous verdict was invalid" in client.calls[1]["user"]
    assert result.input_tokens == 2000


def test_judge_refusal_raises():
    with pytest.raises(JudgeError, match="refused"):
        judge("t", posting(), FakeClient(response(None, stop="refusal")), "s", wait=wait_none())


def test_kappa_and_error_detection():
    a = Agreement("f")
    for judge_wrong, gold_wrong, count in [(1, 1, 2), (1, 0, 1), (0, 1, 1), (0, 0, 6)]:
        for _ in range(count):
            a.add(bool(judge_wrong), bool(gold_wrong))
    # observed 0.8; both say wrong 30% of the time -> chance 0.3*0.3 + 0.7*0.7 = 0.58
    assert a.kappa == pytest.approx((0.8 - 0.58) / 0.42)
    assert a.error_precision == pytest.approx(2 / 3)
    assert a.error_recall == pytest.approx(2 / 3)
    assert a.agreement == pytest.approx(0.8)


def test_kappa_undefined_when_everything_is_right():
    a = Agreement("f", tn=5)
    assert a.kappa is None and a.agreement == 1


def judged_examples():
    gold1 = posting(["python", "sql"], ["aws"], seniority="senior")
    pred1 = posting(["python", "spark"], ["aws"], seniority="mid")
    # judge: seniority wrong (agrees), spark wrong (agrees), sql missing (agrees)
    v1 = verdict(pred1, wrong=["seniority"], wrong_items=["spark"], missing_skills=["sql"])
    gold2 = posting(["python"], work_mode="hybrid")
    pred2 = posting(["python", "docker"], work_mode="remote")
    # judge: work_mode correct (disagrees), docker correct (disagrees), nothing missing
    v2 = verdict(pred2)
    return [Judged("p1", gold1, pred1, v1), Judged("p2", gold2, pred2, v2)]


def test_field_agreement():
    judged = judged_examples()
    seniority = field_agreement(judged, "seniority")
    assert (seniority.tp, seniority.fp, seniority.fn, seniority.tn) == (1, 0, 0, 1)
    work_mode = field_agreement(judged, "work_mode")
    assert (work_mode.tp, work_mode.fn) == (0, 1)
    assert "p2" in work_mode.disagreements


def test_item_agreement():
    items, missing = item_agreement(judged_examples(), "skills")
    # p1: python ok/ok, spark wrong/wrong; p2: python ok/ok, docker ok/wrong
    assert (items.tp, items.fp, items.fn, items.tn) == (1, 0, 1, 2)
    assert (missing.judge_missing, missing.gold_missing, missing.both) == (1, 1, 1)


def test_report_metrics_gold_and_judge():
    judged = judged_examples()
    gold, judge_report = report_metrics(judged, "gold"), report_metrics(judged, "judge")
    assert gold["seniority_accuracy"] == 0.5 and judge_report["seniority_accuracy"] == 0.5
    assert gold["work_mode_accuracy"] == 0.5 and judge_report["work_mode_accuracy"] == 1.0
    # gold: 2 of 4 predicted skills right, 1 missing; judge: 3 of 4 right, 1 missing
    assert gold["skills_precision"] == 0.5 and gold["skills_recall"] == pytest.approx(2 / 3)
    assert judge_report["skills_precision"] == 0.75 and judge_report["skills_recall"] == 0.75
    assert gold["clouds_precision"] == 1.0 and judge_report["clouds_recall"] == 1.0


@pytest.mark.parametrize(
    ("gold", "judge_values", "expected"),
    [
        ({"a": 0.9, "b": 0.8}, {"a": 0.95, "b": 0.7}, True),
        ({"a": 0.9, "b": 0.8}, {"a": 0.7, "b": 0.8}, False),
        ({"a": 0.9, "b": 0.9}, {"a": 0.9, "b": 0.8}, False),
        ({"a": 0.9}, {"a": 0.9}, None),
    ],
)
def test_same_ranking(gold, judge_values, expected):
    assert same_ranking(gold, judge_values) is expected


def test_opus_has_a_price():
    assert price_for("claude-opus-5-5") is not None


def write_run(tmp_path):
    raw, labels, cache = tmp_path / "raw", tmp_path / "labels", tmp_path / "cache"
    for d in (raw, labels, cache):
        d.mkdir()
    gold, pred = posting(["python"], seniority="senior"), posting(["python", "sql"])
    (raw / "p001.txt").write_text("Senior ML Engineer\nBerlin\n\nPython")
    (labels / "p001.json").write_text(gold.model_dump_json())
    (labels / "p002.json").write_text(gold.model_dump_json())
    record = {
        "id": "p001",
        "posting": pred.model_dump(mode="json"),
        "provider": "fake",
        "model": "fake-model",
        "prompt_version": "v2",
    }
    predictions = cache / "run.jsonl"
    lines = [json.dumps(record), json.dumps({"id": "p002", "error": "ExtractionError: boom"})]
    predictions.write_text("\n".join(lines) + "\n")
    return predictions, pred, raw, labels


def test_run_judge_end_to_end(tmp_path):
    predictions, pred, raw, labels = write_run(tmp_path)
    client = FakeClient(response(verdict(pred, wrong=["seniority"], wrong_items=["sql"])))
    out = run_judge(predictions, client, raw_dir=raw, cache_dir=tmp_path / "judge")
    assert out.name == "run__fake-judge_judge_v1.jsonl"
    assert len(client.calls) == 1  # the failed extraction isn't judged
    assert '"seniority": "senior"' not in client.calls[0]["user"]  # no gold label

    # A second run reuses the cached verdict.
    run_judge(predictions, client, raw_dir=raw, cache_dir=tmp_path / "judge")
    assert len(client.calls) == 1

    judged, failed_extractions, failed_verdicts = load_judged(predictions, out, labels, raw)
    assert [j.posting_id for j in judged] == ["p001"]
    assert failed_extractions == {"p002": "ExtractionError: boom"} and failed_verdicts == {}


def test_score_and_report_table(tmp_path):
    predictions, pred, raw, labels = write_run(tmp_path)
    client = FakeClient(response(verdict(pred, wrong=["seniority"], wrong_items=["sql"])))
    out = run_judge(predictions, client, raw_dir=raw, cache_dir=tmp_path / "judge")
    metrics = score(predictions, out, labels, raw)
    assert metrics["agreement"]["seniority"]["kappa"] is None  # one posting, both say wrong
    assert metrics["agreement"]["seniority"]["error_recall"] == 1.0
    assert metrics["report"]["judge"]["skills_precision"] == 0.5
    assert metrics["report"]["gold"]["skills_precision"] == 0.5
    table = report_table([metrics, metrics | {"run": "other"}])
    assert "Same ranking" in table and "fake/fake-model gold" in table
