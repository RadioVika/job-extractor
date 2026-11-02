import pytest
from fastapi.testclient import TestClient

from job_extractor.api import app, get_provider
from job_extractor.llm import LLMError, LLMResponse
from job_extractor.schema import JobPosting

POSTING = JobPosting(
    role_type="ai_engineer",
    seniority="senior",
    work_mode="hybrid",
    german="not_mentioned",
    skills=["python", "llm"],
    clouds=["aws"],
)
TEXT = "Senior AI Engineer\nBerlin\n\nBuild LLM features in Python on AWS."


class FakeClient:
    """An LLMClient whose parse returns, or raises, the given item."""

    provider = "fake"
    model = "fake-model"

    def __init__(self, item):
        self.item = item
        self.calls = 0

    def parse(self, system, user, schema):
        self.calls += 1
        if callable(self.item):
            return self.item()
        if isinstance(self.item, Exception):
            raise self.item
        return self.item


def ok_response() -> LLMResponse:
    return LLMResponse(
        parsed=POSTING,
        model="fake-model",
        input_tokens=100,
        output_tokens=20,
        stop="ok",
        detail=None,
        text=POSTING.model_dump_json(),
    )


@pytest.fixture
def use_fake():
    def install(item) -> FakeClient:
        fake = FakeClient(item)
        app.dependency_overrides[get_provider] = lambda: fake
        return fake

    yield install
    app.dependency_overrides.clear()


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_extract_happy_path(use_fake):
    use_fake(ok_response())
    resp = client.post("/extract", json={"text": TEXT, "provider": "openai"})
    assert resp.status_code == 200
    body = resp.json()
    assert JobPosting.model_validate(body["job_posting"]) == POSTING
    assert body["usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert body["latency_ms"] >= 0


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": "   \n\t "},
        {"text": "x" * 20_001},
        {"text": TEXT, "provider": "gemini"},
        {"provider": "openai"},
    ],
    ids=["empty", "whitespace", "too-long", "unknown-provider", "missing-text"],
)
def test_extract_invalid_input(use_fake, payload):
    fake = use_fake(ok_response())
    resp = client.post("/extract", json=payload)
    assert resp.status_code == 422
    assert fake.calls == 0


def test_extract_provider_error(use_fake):
    use_fake(LLMError("openai: Error code: 401 - secret body", transient=False))
    resp = client.post("/extract", json={"text": TEXT, "provider": "openai"})
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert detail.startswith("openai request failed: ")
    assert "secret body" not in detail


def test_extract_invalid_output(use_fake):
    # Real adapters raise ValidationError when the output doesn't match the schema.
    fake = use_fake(lambda: JobPosting.model_validate_json('{"role_type": "astronaut"'))
    resp = client.post("/extract", json={"text": TEXT})
    assert resp.status_code == 502
    assert resp.json()["detail"] == (
        "anthropic request failed: output still invalid after repair attempt: "
        "1 validation error for JobPosting"
    )
    assert fake.calls == 2  # the first try plus one repair attempt
