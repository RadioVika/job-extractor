"""Ask an LLM judge whether each field of an extracted posting is correct.

The judge sees the posting text, the labeling guide, the schema and the prediction, never the
gold label. Verdicts are cached as one JSON line per posting next to the runs in data/cache.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from evaluation.run import CACHE, _slug, load_records
from job_extractor.extract import RAW, ROOT
from job_extractor.llm import LLMClient, LLMError
from job_extractor.schema import Cloud, JobPosting, Skill

PROMPTS = ROOT / "studies" / "judge" / "prompts"
GUIDE = ROOT / "docs" / "LABELING_GUIDE.md"
JUDGE_CACHE = CACHE / "judge"
PROMPT_VERSION = "judge_v1"
JUDGE_PROVIDER = "anthropic"
JUDGE_MODEL = "claude-opus-5-5"
CONCURRENCY = 5

logger = logging.getLogger(__name__)


class FieldVerdict(BaseModel):
    reason: str
    correct: bool


class SkillVerdict(BaseModel):
    skill: Skill
    reason: str
    correct: bool


class CloudVerdict(BaseModel):
    cloud: Cloud
    reason: str
    correct: bool


class MissingSkill(BaseModel):
    skill: Skill
    reason: str


class MissingCloud(BaseModel):
    cloud: Cloud
    reason: str


class JudgeVerdict(BaseModel):
    role_type: FieldVerdict
    seniority: FieldVerdict
    work_mode: FieldVerdict
    german: FieldVerdict
    salary_min_eur: FieldVerdict
    salary_max_eur: FieldVerdict
    skills: list[SkillVerdict]
    missing_skills: list[MissingSkill]
    clouds: list[CloudVerdict]
    missing_clouds: list[MissingCloud]

    def correct_skills(self) -> set[Skill]:
        return {v.skill for v in self.skills if v.correct}

    def correct_clouds(self) -> set[Cloud]:
        return {v.cloud for v in self.clouds if v.correct}


class JudgeError(RuntimeError):
    pass


class InvalidVerdict(JudgeError):
    """The verdict parsed, but doesn't cover the predicted items; worth one repair attempt."""


def system_prompt(version: str = PROMPT_VERSION, guide: Path = GUIDE) -> str:
    """Judge instructions, then the labeling guide and the JobPosting schema."""
    instructions = (PROMPTS / f"{version}.md").read_text(encoding="utf-8").strip()
    schema = json.dumps(JobPosting.model_json_schema(), indent=2)
    return (
        f"{instructions}\n\n<labeling_guide>\n{guide.read_text(encoding='utf-8').strip()}\n"
        f"</labeling_guide>\n\n<schema>\n{schema}\n</schema>"
    )


def user_prompt(text: str, pred: JobPosting) -> str:
    return (
        f"<posting>\n{text.strip()}\n</posting>\n\n"
        f"<prediction>\n{pred.model_dump_json(indent=2)}\n</prediction>"
    )


def check_coverage(verdict: JudgeVerdict, pred: JobPosting) -> None:
    """The judge must rate every predicted item once, and list only unpredicted ones as missing."""
    lists = {
        "skills": ([v.skill for v in verdict.skills], [m.skill for m in verdict.missing_skills]),
        "clouds": ([v.cloud for v in verdict.clouds], [m.cloud for m in verdict.missing_clouds]),
    }
    problems = []
    for field, (judged, missing) in lists.items():
        predicted = getattr(pred, field)
        if sorted(judged) != sorted(set(predicted)):
            problems.append(
                f"{field}: judged {sorted(map(str, judged))}, "
                f"predicted {sorted(map(str, set(predicted)))}"
            )
        if set(missing) & set(predicted):
            overlap = sorted(map(str, set(missing) & set(predicted)))
            problems.append(f"missing_{field} lists predicted items {overlap}")
    if problems:
        raise InvalidVerdict("; ".join(problems))


@dataclass(frozen=True)
class JudgeResult:
    verdict: JudgeVerdict
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    repaired: bool


def _transient(e: BaseException) -> bool:
    return isinstance(e, LLMError) and e.transient


_DEFAULT_WAIT = wait_random_exponential(max=30)


def _attempt(
    client: LLMClient, system: str, user: str, pred: JobPosting, posting_id: str, tally: dict, wait
) -> JudgeVerdict:
    """One judge call with retries on transient errors, plus the coverage check."""

    def log_retry(state: RetryCallState) -> None:
        logger.info(
            "posting=%s judge retry: attempt %d failed with %r",
            posting_id,
            state.attempt_number,
            state.outcome.exception(),
        )

    for attempt in Retrying(
        retry=retry_if_exception(_transient),
        wait=wait,
        stop=stop_after_attempt(5),
        before_sleep=log_retry,
        reraise=True,
    ):
        with attempt:
            resp = client.parse(system, user, JudgeVerdict)
    tally["model"] = resp.model
    tally["input_tokens"] += resp.input_tokens
    tally["output_tokens"] += resp.output_tokens
    logger.info(
        "posting=%s judge model=%s input_tokens=%d output_tokens=%d stop=%s",
        posting_id,
        resp.model,
        resp.input_tokens,
        resp.output_tokens,
        resp.detail or resp.stop,
    )
    if resp.stop == "refusal":
        raise JudgeError(f"judge refused: {resp.detail}")
    if resp.stop == "max_tokens":
        raise JudgeError("judge output truncated at max_tokens")
    if resp.parsed is None:
        raise JudgeError(f"no parsed output (stop={resp.detail or resp.stop})")
    check_coverage(resp.parsed, pred)
    return resp.parsed


def judge(
    text: str,
    pred: JobPosting,
    client: LLMClient,
    system: str,
    posting_id: str = "-",
    wait=_DEFAULT_WAIT,
) -> JudgeResult:
    """Judge one prediction; if the verdict is invalid, retry once with the errors attached."""
    user = user_prompt(text, pred)
    tally = {"model": client.model, "input_tokens": 0, "output_tokens": 0}
    repaired = False
    start = time.perf_counter()
    try:
        verdict = _attempt(client, system, user, pred, posting_id, tally, wait)
    except (ValidationError, InvalidVerdict) as e:
        logger.warning("posting=%s judge verdict invalid, retrying once: %s", posting_id, e)
        repaired = True
        feedback = f"{user}\n\nA previous verdict was invalid: {e}. Return one that fixes this."
        try:
            verdict = _attempt(client, system, feedback, pred, posting_id, tally, wait)
        except (ValidationError, InvalidVerdict) as e2:
            raise JudgeError(f"verdict still invalid after repair attempt: {e2}") from e2
    return JudgeResult(
        verdict=verdict,
        model=tally["model"],
        input_tokens=tally["input_tokens"],
        output_tokens=tally["output_tokens"],
        latency_s=time.perf_counter() - start,
        repaired=repaired,
    )


async def _judge_all(
    items: list[tuple[str, str, JobPosting]],
    client: LLMClient,
    system: str,
    concurrency: int = CONCURRENCY,
) -> list[JudgeResult | BaseException]:
    sem = asyncio.Semaphore(concurrency)

    async def one(posting_id: str, text: str, pred: JobPosting) -> JudgeResult:
        async with sem:
            # judge is synchronous; run it in a worker thread so calls overlap.
            return await asyncio.to_thread(judge, text, pred, client, system, posting_id)

    return await asyncio.gather(*(one(*item) for item in items), return_exceptions=True)


def verdicts_path(
    predictions: Path, model: str, version: str = PROMPT_VERSION, cache_dir: Path = JUDGE_CACHE
) -> Path:
    return cache_dir / f"{predictions.stem}__{_slug(model)}_{version}.jsonl"


def load_verdicts(path: Path) -> dict[str, dict]:
    """Posting ID -> verdict record (a verdict or an error); empty if the file doesn't exist."""
    if not path.exists():
        return {}
    return {r["id"]: r for r in load_records(path)}


def run_judge(
    predictions: Path,
    client: LLMClient,
    version: str = PROMPT_VERSION,
    raw_dir: Path = RAW,
    cache_dir: Path = JUDGE_CACHE,
    concurrency: int = CONCURRENCY,
) -> Path:
    """Judge every successful extraction in a run and write one JSON line per posting.

    Postings that already have a verdict in the cache file are not judged again, so an
    interrupted or partly failed run can be resumed by running it again.
    """
    out = verdicts_path(predictions, client.model, version, cache_dir)
    done = {k: r for k, r in load_verdicts(out).items() if "verdict" in r}
    todo = [
        (r["id"], (raw_dir / f"{r['id']}.txt").read_text(encoding="utf-8"), r["posting"])
        for r in load_records(predictions)
        if "posting" in r and r["id"] not in done
    ]
    items = [(pid, text, JobPosting.model_validate(p)) for pid, text, p in todo]
    if items:
        outcomes = asyncio.run(_judge_all(items, client, system_prompt(version), concurrency))
    else:
        outcomes = []

    records = dict(done)
    for (posting_id, _, _), outcome in zip(items, outcomes, strict=True):
        record: dict = {"id": posting_id}
        if isinstance(outcome, JudgeError | LLMError):
            record["error"] = f"{type(outcome).__name__}: {outcome}"
            logger.error("posting=%s judge FAILED %s", posting_id, record["error"])
        elif isinstance(outcome, BaseException):
            raise outcome
        else:
            record |= {
                "verdict": outcome.verdict.model_dump(mode="json"),
                "model": outcome.model,
                "prompt_version": version,
                "input_tokens": outcome.input_tokens,
                "output_tokens": outcome.output_tokens,
                "latency_s": round(outcome.latency_s, 3),
                "repaired": outcome.repaired,
            }
        records[posting_id] = record

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for posting_id in sorted(records):
            f.write(json.dumps(records[posting_id]) + "\n")
    return out
