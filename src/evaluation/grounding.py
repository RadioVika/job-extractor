"""Check whether a skill is mentioned in a posting's text at all.

The patterns follow the wording table in docs/LABELING_GUIDE.md. They are deliberately broad:
a skill counts as ungrounded only when nothing in the text could plausibly support it, so a
hit here says "mentioned", not "the hire uses it hands-on".
"""

import re

from job_extractor.schema import Skill

_PATTERNS: dict[Skill, str] = {
    Skill.python: r"\bpython|\bpyspark\b",
    Skill.sql: r"\bsql\b|postgres|mysql|\bt-sql\b",
    Skill.typescript: r"\btypescript\b|\bts\b",
    Skill.spark: r"\bspark\b|\bpyspark\b",
    Skill.pandas: r"\bpandas\b",
    Skill.polars: r"\bpolars\b",
    Skill.scikit_learn: r"scikit|sklearn",
    Skill.pytorch: r"pytorch|\btorch\b",
    Skill.tensorflow: r"tensorflow|\bkeras\b",
    Skill.nlp: r"\bnlp\b|natural language|text classification|\bner\b|named entity",
    Skill.asr: r"\basr\b|speech",
    Skill.search: r"search|retrieval|query understanding",
    Skill.ranking: r"rank|recommend|personali[sz]",
    Skill.hugging_face: r"hugging\s?face|transformers|\bhf\b",
    Skill.rl: r"\brl\b|reinforcement learning|bandit|\brlhf\b",
    Skill.llm: (
        r"\bllms?\b|gen\s?ai|generative ai|foundation models?|large language models?"
        r"|openai|anthropic|\bgpt|claude|gemini|language models?|generative[rnms]? ki"
    ),
    Skill.rag: r"\brag\b|retrieval[- ]augmented|retrieval",
    Skill.agents: r"\bagent|tool[- ]use|tool[- ]calling|function calling",
    Skill.prompt_engineering: r"prompt",
    Skill.fine_tuning: r"fine[- ]?tun|\blora\b|\brlhf\b|instruction[- ]tun|post[- ]training",
    Skill.langchain: r"langchain|langgraph",
    Skill.vector_db: r"vector|pinecone|qdrant|pgvector|weaviate|chroma|milvus|embedding",
    Skill.llm_evaluation: r"\bevals?\b|evaluat|llm[- ]as[- ]a?[- ]?judge|golden data",
    Skill.guardrails: r"guardrail|safety|output validation",
    Skill.tracing: r"tracing|langsmith|langfuse",
    Skill.observability: r"observability|monitor",
    Skill.fastapi: r"fastapi",
    Skill.docker: r"docker|container",
    Skill.kubernetes: r"kubernetes|\bk8s\b|\beks\b|\bgke\b|\baks\b",
    Skill.ci_cd: r"\bci\b|\bcd\b|ci/cd|github actions|gitlab|continuous (integration|delivery)",
    Skill.terraform: r"terraform|infrastructure[- ]as[- ]code|\biac\b",
    Skill.mlops: r"mlops|deploy|serving|production|monitor",
    Skill.mlflow: r"mlflow",
    Skill.airflow: r"airflow",
    Skill.databricks: r"databricks",
    Skill.etl: r"\betl\b|\belt\b|pipeline",
    Skill.ab_testing: r"a/b|\bab test|experiment",
    Skill.statistics: r"statistic|hypothesis|bayesian|probabilit",
    Skill.causal_inference: (
        r"causal|uplift|quasi[- ]experiment|diff[- ]in[- ]diff|synthetic control"
    ),
    Skill.snowflake: r"snowflake",
    Skill.numpy: r"numpy",
    Skill.xgboost: r"xgboost|gradient[- ]boost",
    Skill.bigquery: r"bigquery|big query",
    Skill.nosql: r"nosql|mongo|dynamo|cassandra|redis",
    Skill.lightgbm: r"lightgbm|gradient[- ]boost",
    Skill.kafka: r"kafka",
    Skill.crewai: r"crew\s?ai",
    Skill.forecasting: r"forecast|time[- ]series|demand prediction",
}

SKILL_PATTERNS: dict[Skill, re.Pattern[str]] = {
    skill: re.compile(pattern, re.IGNORECASE) for skill, pattern in _PATTERNS.items()
}


def is_grounded(skill: Skill, text: str) -> bool:
    """True if the posting text contains any wording that maps to this skill."""
    return SKILL_PATTERNS[skill].search(text) is not None
