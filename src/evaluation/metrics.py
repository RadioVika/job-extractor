"""Score extracted postings against the hand-made labels. Pure functions, no I/O."""

import math
from collections import Counter
from dataclasses import dataclass, field

from evaluation.grounding import is_grounded
from evaluation.prices import Price
from evaluation.salary import SALARY_FIELDS, ungrounded_salary
from job_extractor.schema import JobPosting, Skill


@dataclass(frozen=True)
class Example:
    """One scored posting: the gold label, the model's output and the posting text."""

    posting_id: str
    gold: JobPosting
    pred: JobPosting
    text: str


# Single-value fields scored by exact match.
ACCURACY_FIELDS = ("seniority", "role_type", "work_mode", "german")


@dataclass(frozen=True)
class Miss:
    posting_id: str
    gold: str
    pred: str


@dataclass
class AccuracyReport:
    field: str
    n: int
    correct: int
    misses: list[Miss]

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.n if self.n else None


@dataclass
class SalaryReport:
    n: int
    predicted: int  # non-null salary values predicted, summed over postings
    not_in_text: int  # of those, values with no matching amount in the posting text
    # posting_id -> salary fields whose value the posting text doesn't contain
    ungrounded: dict[str, list[str]] = field(default_factory=dict)

    @property
    def invented_not_in_text(self) -> float | None:
        return _ratio(self.not_in_text, self.predicted)


@dataclass
class SkillsReport:
    n: int
    predicted: int  # skills predicted, summed over postings
    tp: int
    fp: int
    fn: int
    # Invented, type 1: predicted but not in the gold label (the same count as fp).
    not_in_label: int
    # Invented, type 2: predicted but nothing in the posting text maps to it.
    not_in_text: int
    macro_precision: float | None
    macro_recall: float | None
    # posting_id -> skills predicted that the posting text never mentions
    ungrounded: dict[str, list[Skill]] = field(default_factory=dict)
    # posting_id -> skills predicted but not in the label (false positives)
    extra: dict[str, list[Skill]] = field(default_factory=dict)
    # posting_id -> skills in the label but not predicted (false negatives)
    missing: dict[str, list[Skill]] = field(default_factory=dict)

    @property
    def micro_precision(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def micro_recall(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def invented_not_in_label(self) -> float | None:
        """Share of predicted skills that the label doesn't have (equals 1 - micro precision)."""
        return _ratio(self.not_in_label, self.predicted)

    @property
    def invented_not_in_text(self) -> float | None:
        """Share of predicted skills with no supporting wording in the posting text."""
        return _ratio(self.not_in_text, self.predicted)


@dataclass
class RunReport:
    """Reliability and speed of one extraction run, over every posting in the split."""

    n: int
    # posting_id -> error, for postings that ended without a valid JobPosting
    failed: dict[str, str]
    repaired: int  # valid outputs that needed the repair retry
    latencies: list[float]  # seconds per valid output, repair retry included
    # USD per valid output, repair retry included; empty if the model has no price
    costs: list[float] = field(default_factory=list)

    @property
    def schema_valid_rate(self) -> float | None:
        """Share of postings that ended with an output passing the schema and content checks."""
        return _ratio(self.n - len(self.failed), self.n)

    @property
    def repair_retry_rate(self) -> float | None:
        return _ratio(self.repaired, self.n)

    @property
    def latency_p50(self) -> float | None:
        return _percentile(self.latencies, 50)

    @property
    def latency_p95(self) -> float | None:
        return _percentile(self.latencies, 95)

    @property
    def cost_per_1000(self) -> float | None:
        """USD for 1000 postings at the mean cost of a valid output."""
        mean = _mean(self.costs)
        return None if mean is None else mean * 1000


def _percentile(values: list[float], pct: int) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(pct / 100 * len(ordered)), 1) - 1]


def score_run(records: list[dict], price: Price | None = None) -> RunReport:
    """Summarise the saved records of a run (one per posting: a prediction or an error).

    Costs need the model's price; failed extractions don't record tokens, so they're left out.
    """
    ok = [r for r in records if "posting" in r]
    return RunReport(
        n=len(records),
        failed={r["id"]: r["error"] for r in records if "posting" not in r},
        repaired=sum(bool(r.get("repaired")) for r in ok),
        latencies=[r["latency_s"] for r in ok if r.get("latency_s") is not None],
        costs=[]
        if price is None
        else [price.cost(r["input_tokens"], r["output_tokens"]) for r in ok],
    )


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score_accuracy(examples: list[Example], field: str) -> AccuracyReport:
    """Exact match of one enum field per posting: right (1) or wrong (0)."""
    misses = [
        Miss(ex.posting_id, getattr(ex.gold, field).value, getattr(ex.pred, field).value)
        for ex in examples
        if getattr(ex.pred, field) != getattr(ex.gold, field)
    ]
    return AccuracyReport(
        field=field, n=len(examples), correct=len(examples) - len(misses), misses=misses
    )


def _per_posting(pred: set[Skill], gold: set[Skill]) -> tuple[float | None, float | None]:
    """Precision and recall for one posting.

    Both empty is a perfect score. Otherwise an empty denominator leaves the value undefined
    (None) and the posting is skipped in that macro average.
    """
    if not pred and not gold:
        return 1.0, 1.0
    hits = len(pred & gold)
    return _ratio(hits, len(pred)), _ratio(hits, len(gold))


def score_skills(examples: list[Example]) -> SkillsReport:
    ungrounded: dict[str, list[Skill]] = {}
    extra: dict[str, list[Skill]] = {}
    missing_from_pred: dict[str, list[Skill]] = {}
    totals: Counter[str] = Counter()
    precisions: list[float] = []
    recalls: list[float] = []

    for ex in examples:
        pred, gold = set(ex.pred.skills), set(ex.gold.skills)
        missing = sorted(s for s in pred if not is_grounded(s, ex.text))
        if missing:
            ungrounded[ex.posting_id] = missing
        if pred - gold:
            extra[ex.posting_id] = sorted(pred - gold)
        if gold - pred:
            missing_from_pred[ex.posting_id] = sorted(gold - pred)

        totals["predicted"] += len(pred)
        totals["tp"] += len(pred & gold)
        totals["fp"] += len(pred - gold)
        totals["fn"] += len(gold - pred)
        totals["not_in_text"] += len(missing)

        precision, recall = _per_posting(pred, gold)
        if precision is not None:
            precisions.append(precision)
        if recall is not None:
            recalls.append(recall)

    return SkillsReport(
        n=len(examples),
        predicted=totals["predicted"],
        tp=totals["tp"],
        fp=totals["fp"],
        fn=totals["fn"],
        not_in_label=totals["fp"],
        not_in_text=totals["not_in_text"],
        macro_precision=_mean(precisions),
        macro_recall=_mean(recalls),
        ungrounded=ungrounded,
        extra=extra,
        missing=missing_from_pred,
    )


def score_salary(examples: list[Example]) -> SalaryReport:
    """Count predicted salary values that can't be found in the posting text."""
    ungrounded: dict[str, list[str]] = {}
    predicted = 0
    for ex in examples:
        predicted += sum(getattr(ex.pred, f) is not None for f in SALARY_FIELDS)
        missing = ungrounded_salary(ex.text, ex.pred)
        if missing:
            ungrounded[ex.posting_id] = missing
    return SalaryReport(
        n=len(examples),
        predicted=predicted,
        not_in_text=sum(len(v) for v in ungrounded.values()),
        ungrounded=ungrounded,
    )
