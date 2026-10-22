"""Start and stop a local llama-server for the llama provider.

extract.main wraps a llama run in llama_server(settings). If a server already answers on the
port it is used as is and left running; otherwise one is started with LLAMA_MODEL_PATH and
stopped when the run ends.
"""

import logging
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import IO

import httpx

from job_extractor.llm import resolve_model
from job_extractor.settings import Settings

STARTUP_TIMEOUT_S = 300  # loading a 27B model from a cold disk can take minutes
POLL_INTERVAL_S = 1.0
STOP_TIMEOUT_S = 10

logger = logging.getLogger(__name__)


class LlamaServerError(RuntimeError):
    pass


def is_healthy(port: int) -> bool:
    """True once the server has loaded the model (/health answers 503 while loading)."""
    try:
        return httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def server_command(settings: Settings) -> list[str]:
    return [
        settings.llama_server_bin,
        "--model",
        str(settings.llama_model_path),
        "--alias",
        resolve_model(settings),
        "--host",
        "127.0.0.1",
        "--port",
        str(settings.llama_port),
        "--ctx-size",
        str(settings.llama_ctx_size),
        # One slot gets the whole context; extract.main sends one request at a time.
        "--parallel",
        "1",
        "--jinja",
    ]


@contextmanager
def llama_server(
    settings: Settings,
    healthy: Callable[[int], bool] = is_healthy,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    timeout: float = STARTUP_TIMEOUT_S,
    poll_interval: float = POLL_INTERVAL_S,
) -> Iterator[None]:
    """Make sure a llama-server is running on settings.llama_port for the duration."""
    port = settings.llama_port
    if healthy(port):
        logger.info("llama-server already running on port %d, using it", port)
        yield
        return

    model_path = settings.llama_model_path
    if model_path is None:
        raise LlamaServerError("LLAMA_MODEL_PATH is not set; point it at the Bonsai 2 GGUF file")
    if not model_path.is_file():
        raise LlamaServerError(f"LLAMA_MODEL_PATH {model_path} is not a file")

    command = server_command(settings)
    logger.info("starting llama-server: %s", " ".join(command))
    # The server's output goes to a temp file, quoted in the error if startup fails.
    with tempfile.TemporaryFile() as log:
        try:
            proc = popen(command, stdout=log, stderr=subprocess.STDOUT)
        except OSError as e:
            raise LlamaServerError(
                f"cannot run {settings.llama_server_bin!r} (set LLAMA_SERVER_BIN): {e}"
            ) from e
        try:
            _wait_until_ready(proc, port, healthy, timeout, poll_interval, log)
            yield
        finally:
            _stop(proc)


def _wait_until_ready(
    proc: subprocess.Popen,
    port: int,
    healthy: Callable[[int], bool],
    timeout: float,
    poll_interval: float,
    log: IO[bytes],
) -> None:
    start = time.monotonic()
    while True:
        if proc.poll() is not None:
            raise LlamaServerError(
                f"llama-server exited with code {proc.returncode}:\n{_tail(log)}"
            )
        if healthy(port):
            logger.info("llama-server ready after %.1fs", time.monotonic() - start)
            return
        if time.monotonic() - start >= timeout:
            raise LlamaServerError(f"llama-server not ready after {timeout:.0f}s:\n{_tail(log)}")
        time.sleep(poll_interval)


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    logger.info("stopping llama-server")
    proc.terminate()
    try:
        proc.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _tail(log: IO[bytes], size: int = 2000) -> str:
    log.flush()
    log.seek(0)
    return log.read()[-size:].decode(errors="replace")
