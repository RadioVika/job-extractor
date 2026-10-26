import json

import pytest

from evaluation.grounding import SKILL_PATTERNS, is_grounded
from evaluation.metrics import ACCURACY_FIELDS, Example, score_accuracy, score_salary, score_skills
from evaluation.run import load_examples
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
    assert failed == ["p2"]
    assert [ex.posting_id for ex in examples] == ["p1"]
    assert examples[0].gold.seniority == "senior"
    assert examples[0].text == "Python"
