"""Compare judge verdicts with manual gold labels. Pure functions, no I/O.

Gold says a field is wrong when pred != gold; the judge says so when its verdict is not correct.
Both sources are turned into the same report (field accuracy, skills and clouds
precision/recall), so the judge's report can be put next to the one from manual gold.
"""

from dataclasses import dataclass
from itertools import combinations

from evaluation.judge import JudgeVerdict
from evaluation.metrics import ACCURACY_FIELDS
from evaluation.salary import SALARY_FIELDS
from job_extractor.schema import JobPosting

# Fields with one verdict each, scored by exact match against gold.
FIELDS = (*ACCURACY_FIELDS, *SALARY_FIELDS)
# Fields with one verdict per predicted item, plus a list of missing items.
LIST_FIELDS = ("skills", "clouds")


@dataclass(frozen=True)
class Judged:
    """One posting with a successful extraction and a successful verdict."""

    posting_id: str
    gold: JobPosting
    pred: JobPosting
    verdict: JudgeVerdict


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def judged_correct_items(j: Judged, field: str) -> set:
    return j.verdict.correct_skills() if field == "skills" else j.verdict.correct_clouds()


def judged_missing_items(j: Judged, field: str) -> set:
    if field == "skills":
        return {m.skill for m in j.verdict.missing_skills}
    return {m.cloud for m in j.verdict.missing_clouds}


@dataclass
class Agreement:
    """Judge vs gold on "is this prediction wrong?"; the positive class is "wrong"."""

    field: str
    tp: int = 0  # both say wrong
    fp: int = 0  # judge says wrong, gold says right
    fn: int = 0  # judge says right, gold says wrong
    tn: int = 0  # both say right
    # posting_id -> what disagreed, for re-checking by hand
    disagreements: dict[str, list[str]] | None = None

    def add(self, judge_wrong: bool, gold_wrong: bool) -> None:
        if judge_wrong and gold_wrong:
            self.tp += 1
        elif judge_wrong:
            self.fp += 1
        elif gold_wrong:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def agreement(self) -> float | None:
        return _ratio(self.tp + self.tn, self.n)

    @property
    def error_precision(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def error_recall(self) -> float | None:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def kappa(self) -> float | None:
        """Cohen's κ; None when chance agreement is 1 (both raters gave one label throughout)."""
        if not self.n:
            return None
        observed = (self.tp + self.tn) / self.n
        judge_wrong = (self.tp + self.fp) / self.n
        gold_wrong = (self.tp + self.fn) / self.n
        chance = judge_wrong * gold_wrong + (1 - judge_wrong) * (1 - gold_wrong)
        if chance == 1:
            return None
        return (observed - chance) / (1 - chance)


@dataclass
class MissingAgreement:
    """Items the judge says are missing vs items in gold but not predicted."""

    field: str
    judge_missing: int = 0
    gold_missing: int = 0
    both: int = 0

    @property
    def precision(self) -> float | None:
        return _ratio(self.both, self.judge_missing)

    @property
    def recall(self) -> float | None:
        return _ratio(self.both, self.gold_missing)


def _note(disagreements: dict[str, list[str]], posting_id: str, text: str) -> None:
    disagreements.setdefault(posting_id, []).append(text)


def field_agreement(judged: list[Judged], field: str) -> Agreement:
    """One item per posting: the field's verdict vs pred == gold."""
    result = Agreement(field, disagreements={})
    for j in judged:
        judge_wrong = not getattr(j.verdict, field).correct
        gold_wrong = getattr(j.pred, field) != getattr(j.gold, field)
        result.add(judge_wrong, gold_wrong)
        if judge_wrong != gold_wrong:
            pred, gold = getattr(j.pred, field), getattr(j.gold, field)
            side = "judge says wrong" if judge_wrong else "judge says correct"
            _note(result.disagreements, j.posting_id, f"{side}: pred={pred} gold={gold}")
    return result


def item_agreement(judged: list[Judged], field: str) -> tuple[Agreement, MissingAgreement]:
    """One item per predicted skill (or cloud): its verdict vs being in gold; plus the
    judge's missing list against gold - pred."""
    items = Agreement(field, disagreements={})
    missing = MissingAgreement(field)
    for j in judged:
        pred, gold = set(getattr(j.pred, field)), set(getattr(j.gold, field))
        correct = judged_correct_items(j, field)
        for item in sorted(pred):
            judge_wrong, gold_wrong = item not in correct, item not in gold
            items.add(judge_wrong, gold_wrong)
            if judge_wrong != gold_wrong:
                side = "judge says wrong" if judge_wrong else "judge says correct"
                in_gold = "in gold" if not gold_wrong else "not in gold"
                _note(items.disagreements, j.posting_id, f"{side}: {item} ({in_gold})")
        judge_missing, gold_missing = judged_missing_items(j, field), gold - pred
        missing.judge_missing += len(judge_missing)
        missing.gold_missing += len(gold_missing)
        missing.both += len(judge_missing & gold_missing)
        for item in sorted(judge_missing ^ gold_missing):
            who = "judge only" if item in judge_missing else "gold only"
            _note(items.disagreements, j.posting_id, f"missing {item}: {who}")
    return items, missing


def report_metrics(judged: list[Judged], source: str) -> dict[str, float | None]:
    """The standard report, from gold labels (source="gold") or judge verdicts ("judge").

    Accuracy is the share of postings whose field is correct. For skills and clouds,
    precision is correct predicted items / predicted items, and recall is
    correct / (correct + missing), with "missing" being gold - pred or the judge's list.
    """
    if source not in ("gold", "judge"):
        raise ValueError(f"unknown source {source!r}")
    metrics: dict[str, float | None] = {}
    for field in FIELDS:
        if source == "gold":
            correct = sum(getattr(j.pred, field) == getattr(j.gold, field) for j in judged)
        else:
            correct = sum(getattr(j.verdict, field).correct for j in judged)
        metrics[f"{field}_accuracy"] = _ratio(correct, len(judged))
    for field in LIST_FIELDS:
        predicted = correct = missing = 0
        for j in judged:
            pred = set(getattr(j.pred, field))
            if source == "gold":
                gold = set(getattr(j.gold, field))
                hits, misses = pred & gold, gold - pred
            else:
                hits, misses = judged_correct_items(j, field) & pred, judged_missing_items(j, field)
            predicted += len(pred)
            correct += len(hits)
            missing += len(misses)
        metrics[f"{field}_precision"] = _ratio(correct, predicted)
        metrics[f"{field}_recall"] = _ratio(correct, correct + missing)
    return metrics


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def same_ranking(gold: dict[str, float | None], judge: dict[str, float | None]) -> bool | None:
    """Whether gold and the judge order every pair of runs the same way on one metric.

    Both map run name -> metric value. A tie on one side and not the other counts as a
    different order. None if fewer than two runs have a value in both.
    """
    runs = [r for r in gold if gold[r] is not None and judge.get(r) is not None]
    if len(runs) < 2:
        return None
    return all(
        _sign(gold[a] - gold[b]) == _sign(judge[a] - judge[b]) for a, b in combinations(runs, 2)
    )
