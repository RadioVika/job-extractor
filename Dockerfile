# syntax=docker/dockerfile:1

# Build stage: resolve the venv with uv; uv itself doesn't ship in the final image.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /bin/

WORKDIR /app
# UV_PYTHON_DOWNLOADS=0: use the image's Python, never download one.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0

# Layer 1: dependencies only. Rebuilt when uv.lock or pyproject.toml changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

# Layer 2: the project itself. The editable install keeps extract.ROOT at /app,
# which is where prompts/ is looked up.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev
# UV_COMPILE_BYTECODE skips editable sources, and the runtime user can't write __pycache__.
RUN python -m compileall -q src


FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
RUN useradd --system --create-home --uid 1001 app

WORKDIR /app
COPY --from=builder /app /app
# Last, so editing a prompt doesn't rebuild the venv.
COPY prompts/ ./prompts/

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["uvicorn", "job_extractor.api:app", "--host", "0.0.0.0", "--port", "8000"]
