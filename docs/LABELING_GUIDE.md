# Labeling guide

How to hand-label job postings against the `JobPosting` schema in `src/job_extractor/schema.py`. The labels are the ground truth for the extractor, so consistency matters more than any single judgement call: when in doubt, follow the rule here, and if a rule is wrong, change it here and relabel.

## Files

- Input: `data/raw/pNNN.txt`. First line is the title follwing by posting body. `data/postings.jsonl` has the URL and company for each ID.
- Output: `data/labels/pNNN.json`, same ID as the raw file, one JSON object per file.

You can check a label against the schema before saving:

```sh
uv run python -c "from pathlib import Path; from job_extractor.schema import JobPosting; JobPosting.model_validate_json(Path('data/labels/p001.json').read_text())"
```

## General rules

1. **Label what the posting says, not what you assume.** No inferring a cloud from the company, or Python from "data scientist".
2. **Read the whole posting**, including benefits and the footer. Work mode and language often only show up there (for example `#LI-Hybrid`).
3. **When the posting is silent, use the "nothing stated" value**: `unspecified`, `not_mentioned`, `[]` or `null`. Never guess.

## Fields

### `role_type`

What the job mainly is. Use the title first; read the body only when the title is ambiguous (for example "AI/ML Engineer").

| Value | Use for |
|---|---|
| `ai_engineer` | Building products on top of LLMs and foundation models: AI Engineer, Applied AI Engineer, LLM Engineer. |
| `ml_engineer` | Training, serving and operating models: ML Engineer, MLOps Engineer. |
| `data_scientist` | Modelling, analysis and experimentation: Data Scientist, Applied Scientist, Research Scientist. |
| `data_engineer` | Pipelines and data platforms: Data Engineer, Analytics Engineer. |
| `data_analyst` | Reporting, dashboards and business analysis: Data Analyst, BI Analyst. |
| `other` | Everything else, including **people managers** (Data Science Manager, Head of ML), product managers and AI Process Owners. |

A "Data Scientist" who builds LLM features is still `data_scientist`: the title wins unless it's clearly misleading.

### `seniority`

Take it from the title.

| Value | Title signals |
|---|---|
| `junior` | Junior, Graduate, Entry level, Associate, level I |
| `mid` | Medior, Intermediate, level II |
| `senior` | Senior, level III. Also manager titles that say Senior ("Senior Data Science Manager" → `senior`). |
| `lead` | Lead, Tech Lead, Head of |
| `staff` | Staff, Principal |
| `unspecified` | The title has no level and the body gives no years of experience. |

If the title has no level, use the required years of experience: under 2 → `junior`, 2–4 → `mid`, 5+ → `senior`.

### `work_mode`

| Value | Use when |
|---|---|
| `onsite` | Office only, or on-site stated explicitly. |
| `hybrid` | Some office days are expected ("three days in the office"), "hybrid", or `#LI-Hybrid`. |
| `remote` | Fully remote, including "remote within Germany/EU" and "(Remote)" in the title. |
| `unspecified` | Nothing about where the work happens. |

Perks like "work from anywhere 30 days a year" don't make a hybrid job `remote`.

### `german`

| Value | Use when |
|---|---|
| `required` | Fluent German, a minimum level (B2, C1…), "German-speaking" in the title, or the whole posting is written in German. |
| `nice_to_have` | German is "a plus", "beneficial" or "an advantage". |
| `not_mentioned` | German isn't mentioned. "Excellent English" alone is `not_mentioned`. |

### `skills`

Only values from the `Skill` vocabulary. Tag a skill when the hire is expected to **use it hands-on**: it appears in the requirements, the nice-to-haves, or a responsibility the person does themselves.

Don't tag:
- skills of other people on the team ("working closely with MLOps Engineers" is not `mlops`)
- things a manager only oversees or sets standards for
- the company's product description

Mapping from posting wording to vocabulary:

| Skill | Posting says |
|---|---|
| `python`, `sql`, `typescript`, `spark`, `pandas`, `polars` | The language or library by name. PySpark → `spark`. |
| `scikit_learn`, `pytorch`, `tensorflow` | The library by name. Keras → `tensorflow`. |
| `llm` | LLMs, GenAI, foundation models, OpenAI/Anthropic APIs. |
| `rag` | RAG, retrieval-augmented generation, retrieval over documents. |
| `agents` | Agents, agentic AI, tool use, multi-agent systems. |
| `prompt_engineering` | Prompt engineering or prompt design. |
| `fine_tuning` | Fine-tuning, LoRA, RLHF, instruction tuning. |
| `langchain` | LangChain or LangGraph. |
| `vector_db` | Pinecone, Qdrant, pgvector, Weaviate, Chroma, or "vector databases". |
| `llm_evaluation` | Evals, evaluator design, LLM-as-judge, golden datasets. |
| `guardrails` | Guardrails, output validation and safety filters for LLMs. |
| `tracing` | Tracing LLM calls: LangSmith, Langfuse, "tracing". |
| `observability` | Observability or monitoring of systems in production. |
| `fastapi` | FastAPI by name. |
| `docker`, `kubernetes`, `terraform`, `airflow` | The tool by name. K8s → `kubernetes`. |
| `ci_cd` | CI/CD, GitHub Actions, GitLab CI. |
| `mlops` | Deploying, serving or monitoring ML models in production; MLflow. |
| `ab_testing` | A/B tests, online experiments. |
| `statistics` | Statistics, hypothesis testing, causal inference. |
| `images` | Working with image data or computer vision. |

A tool missing from the vocabulary (Snowflake, dbt, Java…) is left out. If one keeps coming up, add it to `Skill` and relabel.

### `clouds`

Only clouds the posting names, directly or through one of their services:

- `aws`: AWS, S3, SageMaker, Bedrock, Lambda, Redshift
- `gcp`: GCP, Google Cloud, BigQuery, Vertex AI
- `azure`: Azure, Azure OpenAI

Databricks and Snowflake run on several clouds, so they don't count.

### `industry`

What the hiring company sells, as one or two lowercase words. Reuse a value that's already there before inventing a new one:

```sh
cat data/labels/*.json | grep -o '"industry": "[^"]*"' | sort | uniq -c
```

### `salary_min_eur`, `salary_max_eur`

Annual gross salary in euros, as whole numbers.

- A range: min and max. A single figure: both fields get the same value.
- A monthly figure: multiply by 12.
- Another currency, or no salary: both `null`.

## Example

`p001`, GetYourGuide, "Senior Data Science Manager, Central Data Products":

```json
{
  "role_type": "other",
  "seniority": "senior",
  "work_mode": "hybrid",
  "german": "not_mentioned",
  "skills": ["llm"],
  "clouds": [],
  "industry": "travel",
  "salary_min_eur": null,
  "salary_max_eur": null
}
```

- `other`: it's a people-manager role.
- `senior`: from "Senior" in the title.
- `hybrid`: "three days of in office collaboration" and `#LI-Hybrid`.
- `skills`: only `llm`, from "Experience applying LLMs" in the extras. LLM-as-judge monitoring and the MLOps Engineers describe what the team does, not what this person does hands-on.
