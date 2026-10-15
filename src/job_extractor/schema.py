from enum import StrEnum

from pydantic import BaseModel, Field


class Seniority(StrEnum):
    junior = "junior"
    mid = "mid"
    senior = "senior"
    lead = "lead"
    staff = "staff"
    unspecified = "unspecified"


class WorkMode(StrEnum):
    onsite = "onsite"
    hybrid = "hybrid"
    remote = "remote"
    unspecified = "unspecified"


class GermanLevel(StrEnum):
    required = "required"
    nice_to_have = "nice_to_have"
    not_mentioned = "not_mentioned"


class RoleType(StrEnum):
    ai_engineer = "ai_engineer"
    ml_engineer = "ml_engineer"
    data_scientist = "data_scientist"
    data_engineer = "data_engineer"
    data_analyst = "data_analyst"
    other = "other"


class Cloud(StrEnum):
    aws = "aws"
    gcp = "gcp"
    azure = "azure"


class Skill(StrEnum):
    # languages and data
    python = "python"
    sql = "sql"
    typescript = "typescript"
    spark = "spark"
    pandas = "pandas"
    polars = "polars"
    # classical ML and deep learning
    scikit_learn = "scikit_learn"
    pytorch = "pytorch"
    tensorflow = "tensorflow"
    # LLM work
    llm = "llm"  # "LLMs", "GenAI", "foundation models" all map here
    rag = "rag"
    agents = "agents"
    prompt_engineering = "prompt_engineering"
    fine_tuning = "fine_tuning"
    langchain = "langchain"  # LangChain and LangGraph
    vector_db = "vector_db"  # Pinecone, Qdrant, pgvector, Weaviate...
    llm_evaluation = "llm_evaluation"
    guardrails = "guardrails"
    tracing = "tracing"
    observability = "observability"
    # engineering and ops
    fastapi = "fastapi"
    docker = "docker"
    kubernetes = "kubernetes"
    ci_cd = "ci_cd"
    terraform = "terraform"
    mlops = "mlops"  # model serving, monitoring
    mlflow = "mlflow"
    airflow = "airflow"
    databricks = "databricks"
    # analysis
    ab_testing = "ab_testing"
    statistics = "statistics"
    causal_inference = "causal_inference"
    # data bases
    snowflake = "snowflake"


class DataType(StrEnum):
    images = "images"
    tabular = "tabular"
    prices = "prices"
    text = "text"


class JobPosting(BaseModel):
    role_type: RoleType  # ai_engineer, ml_engineer, data_scientist, data_engineer, other
    seniority: Seniority
    work_mode: WorkMode
    german: GermanLevel
    skills: list[Skill] = Field(description="Only skills from the fixed vocabulary")
    clouds: list[Cloud]  # aws, gcp, azure
    data_types: list[DataType]
    industry: str
    salary_min_eur: int | None = None
    salary_max_eur: int | None = None
