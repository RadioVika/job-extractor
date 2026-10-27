import json

import pytest

from evaluation.grounding import SKILL_PATTERNS, is_grounded
from evaluation.metrics import (
    ACCURACY_FIELDS,
    Example,
    score_accuracy,
    score_run,
    score_salary,
    score_skills,
)
from evaluation.prices import Price, price_for
from evaluation.run import TABLE_COLUMNS, load_examples, markdown_table, metrics_json
from evaluation.salary import SALARY_FIELDS, ungrounded_salary
from job_extractor.schema import JobPosting, Skill


def posting(skills=(), **fields):
    defaults = {
        "role_type": "ml_engineer",
        "seniority": "mid",
        "work_mode": "remote",
        "german": "not_mentioned",
        "clouds": [],
        "data_types": [],
    }
    return JobPosting(skills=list(skills), **(defaults | fields))


def example(posting_id="p1", gold=(), pred=(), text=""):
    return Example(posting_id=posting_id, gold=posting(gold), pred=posting(pred), text=text)


def test_every_skill_has_a_pattern():
    assert set(SKILL_PATTERNS) == set(Skill)


@pytest.mark.parametrize(
    ("skill", "text", "expected"),
    [
        ("python", "Strong Python skills", True),
        ("spark", "Experience with PySpark", True),
        ("kubernetes", "We run on K8s", True),
        ("llm", "Erfahrung mit generativer KI", True),
        ("sql", "Experience with NoSQL stores", False),
        ("rl", "You will control the world", False),
        ("pytorch", "TensorFlow only", False),
    ],
)
def test_is_grounded(skill, text, expected):
    assert is_grounded(Skill(skill), text) is expected


@pytest.mark.parametrize(
    ("field", "gold", "wrong"),
    [
        ("seniority", "senior", "lead"),
        ("role_type", "data_scientist", "ml_engineer"),
        ("work_mode", "hybrid", "remote"),
        ("german", "required", "nice_to_have"),
    ],
)
def test_accuracy_and_misses(field, gold, wrong):
    examples = [
        Example("p1", posting(**{field: gold}), posting(**{field: gold}), ""),
        Example("p2", posting(**{field: gold}), posting(**{field: wrong}), ""),
    ]
    report = score_accuracy(examples, field)
    assert report.field == field
    assert report.accuracy == 0.5
    assert [(m.posting_id, m.gold, m.pred) for m in report.misses] == [("p2", gold, wrong)]


def test_accuracy_fields():
    assert ACCURACY_FIELDS == ("seniority", "role_type", "work_mode", "german")


def test_accuracy_empty():
    assert score_accuracy([], "seniority").accuracy is None


def test_skills_perfect_match():
    text = "Python and SQL"
    report = score_skills([example(gold=["python", "sql"], pred=["sql", "python"], text=text)])
    assert report.micro_precision == report.micro_recall == 1.0
    assert report.invented_not_in_label == report.invented_not_in_text == 0.0


def test_skills_micro_vs_macro():
    report = score_skills(
        [
            # 1 of 1 predicted is right, 1 of 2 gold is found
            example("p1", gold=["python", "sql"], pred=["python"], text="Python, SQL"),
            # 1 of 3 predicted is right, 1 of 1 gold is found
            example("p2", gold=["docker"], pred=["docker", "kafka", "pytorch"], text="Docker"),
        ]
    )
    assert (report.tp, report.fp, report.fn) == (2, 2, 1)
    assert report.micro_precision == 2 / 4
    assert report.micro_recall == 2 / 3
    assert report.macro_precision == pytest.approx((1 + 1 / 3) / 2)
    assert report.macro_recall == pytest.approx((1 / 2 + 1) / 2)


def test_skills_two_kinds_of_invented():
    # mlops is mentioned but not in the label; pytorch is in neither.
    report = score_skills(
        [example(gold=["python"], pred=["python", "mlops", "pytorch"], text="Python in production")]
    )
    assert report.not_in_label == 2
    assert report.not_in_text == 1
    assert report.invented_not_in_label == pytest.approx(2 / 3)
    assert report.invented_not_in_text == pytest.approx(1 / 3)
    assert report.ungrounded == {"p1": [Skill.pytorch]}
    assert report.extra == {"p1": [Skill.mlops, Skill.pytorch]}
    assert report.missing == {}


def test_skills_empty_sets():
    report = score_skills(
        [
            example("p1"),  # both empty: perfect, counted in both macro averages
            example("p2", gold=["python"], text="Python"),  # nothing predicted: no precision
        ]
    )
    assert report.macro_precision == 1.0
    assert report.macro_recall == 0.5
    assert report.micro_precision is None
    assert report.invented_not_in_text is None


@pytest.mark.parametrize(
    ("text", "salary", "missing"),
    [
        ("Austria: €90,000 - €168,000 gross annually", (90000, 168000), []),
        ("100.000 € Jahresgehalt.", (100000, 100000), []),
        ("100 000 € Jahresgehalt.", (100000, 100000), []),
        ("100k Jahresgehalt", (100000, 100000), []),
        ("Salary: €6,000 per month", (72000, 72000), []),
        ("Competitive salary and stock options.", (80000, 80000), SALARY_FIELDS),
        ("From €70,000, depending on experience.", (70000, 90000), ["salary_max_eur"]),
        ("No salary here.", (None, None), []),
    ],
)
def test_ungrounded_salary(text, salary, missing):
    pred = posting(salary_min_eur=salary[0], salary_max_eur=salary[1])
    assert ungrounded_salary(text, pred) == list(missing)


def test_score_salary():
    def ex(posting_id, text, lo, hi):
        pred = posting(salary_min_eur=lo, salary_max_eur=hi)
        return Example(posting_id, posting(), pred, text)

    report = score_salary(
        [
            ex("p1", "€70,000 - €90,000", 70000, 90000),
            ex("p2", "From €70,000", 70000, 95000),
            ex("p3", "No salary here.", None, None),
        ]
    )
    assert (report.predicted, report.not_in_text) == (4, 1)
    assert report.invented_not_in_text == 0.25
    assert report.ungrounded == {"p2": ["salary_max_eur"]}
    assert score_salary([ex("p1", "", None, None)]).invented_not_in_text is None


def test_load_examples(tmp_path):
    labels, raw = tmp_path / "labels", tmp_path / "raw"
    labels.mkdir()
    raw.mkdir()
    (labels / "p1.json").write_text(posting(["python"], seniority="senior").model_dump_json())
    (raw / "p1.txt").write_text("Python")
    predictions = tmp_path / "run.jsonl"
    predictions.write_text(
        json.dumps({"id": "p1", "posting": posting(["python"]).model_dump(mode="json")})
        + "\n"
        + json.dumps({"id": "p2", "error": "ExtractionError: boom"})
        + "\n"
    )
    examples, failed = load_examples(predictions, labels, raw)
    assert failed == {"p2": "ExtractionError: boom"}
    assert [ex.posting_id for ex in examples] == ["p1"]
    assert examples[0].gold.seniority == "senior"
    assert examples[0].text == "Python"


def test_score_run():
    records = [
        {"id": "p1", "posting": {}, "latency_s": 1.0, "repaired": False},
        {"id": "p2", "posting": {}, "latency_s": 3.0, "repaired": True},
        {"id": "p3", "posting": {}, "latency_s": 2.0, "repaired": False},
        {"id": "p4", "error": "ExtractionError: boom"},
    ]
    run = score_run(records)
    assert run.failed == {"p4": "ExtractionError: boom"}
    assert run.schema_valid_rate == 0.75
    assert run.repair_retry_rate == 0.25
    assert (run.latency_p50, run.latency_p95) == (2.0, 3.0)
    empty = score_run([])
    assert empty.schema_valid_rate is None and empty.latency_p95 is None
    assert run.cost_per_1000 is None  # no price given


def test_cost_per_1000():
    records = [
        {"id": "p1", "posting": {}, "input_tokens": 3000, "output_tokens": 400},
        {"id": "p2", "posting": {}, "input_tokens": 5000, "output_tokens": 600},
        {"id": "p3", "error": "ExtractionError: boom"},
    ]
    run = score_run(records, Price(input=1.0, output=10.0))
    # Mean posting: 4000 in + 500 out = $0.004 + $0.005; failed extractions are left out.
    assert run.cost_per_1000 == pytest.approx(9.0)


def test_price_for(tmp_path):
    from datetime import date

    prices = tmp_path / "prices.yaml"
    prices.write_text(
        "- {model: m, date: 2026-01-01, price: {input: 1, output: 2}}\n"
        "- {model: m, date: 2026-06-01, price: {input: 3, output: 4}}\n"
    )
    assert price_for("m", prices, date(2026, 3, 1)) == Price(1.0, 2.0)
    assert price_for("m", prices, date(2026, 7, 1)) == Price(3.0, 4.0)
    assert price_for("m", prices, date(2025, 1, 1)) is None
    assert price_for("other", prices) is None


def test_prices_cover_default_models():
    from job_extractor.llm import DEFAULT_MODELS

    for model in DEFAULT_MODELS.values():
        assert price_for(model) is not None, model


def test_metrics_json_and_table(tmp_path):
    examples = [
        example("p1", gold=["python"], pred=["python"], text="Python"),
        example("p2", gold=["python", "sql"], pred=["python", "pytorch"], text="Python, SQL"),
    ]
    records = [
        {"id": "p1", "posting": {}, "latency_s": 1.0, "repaired": False},
        {"id": "p2", "posting": {}, "latency_s": 2.0, "repaired": False},
        {"id": "p3", "error": "ExtractionError: boom"},
    ]
    accuracy = [score_accuracy(examples, f) for f in ACCURACY_FIELDS]
    metrics = metrics_json(
        tmp_path / "run.jsonl",
        score_run(records),
        accuracy,
        score_skills(examples),
        score_salary(examples),
    )
    assert list(metrics["summary"]) == list(TABLE_COLUMNS)
    assert metrics["summary"]["skills_precision"] == 2 / 3
    assert metrics["errors"] == {
        "p2": {
            "skills_missing": ["sql"],
            "skills_extra": ["pytorch"],
            "skills_not_in_text": ["pytorch"],
        },
        "p3": {"extraction_error": "ExtractionError: boom"},
    }
    table = markdown_table("openai/gpt", metrics["summary"]).splitlines()
    assert len(table) == 3
    assert len({len(line) for line in table}) == 1  # columns line up
    cells = [c.strip() for c in table[2].strip("|").split("|")]
    assert cells[:4] == ["openai/gpt", "66.7%", "66.7%", "33.3%"]
    assert cells[-3:] == ["1.00 s", "2.00 s", "n/a"]
