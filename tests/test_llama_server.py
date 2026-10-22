import subprocess

import pytest

from job_extractor.llama_server import LlamaServerError, llama_server, server_command
from job_extractor.settings import Settings


class FakeProcess:
    """Stands in for subprocess.Popen; `exit_code` set means the process has already exited."""

    def __init__(self, exit_code=None, output=b"", ignore_terminate=False):
        self.returncode = exit_code
        self.output = output
        self.ignore_terminate = ignore_terminate
        self.command = None
        self.terminated = self.killed = False

    def __call__(self, command, stdout, stderr):
        self.command = command
        stdout.write(self.output)
        return self

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.ignore_terminate:
            self.returncode = 0

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("llama-server", timeout)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def healthy_after(checks):
    """A health check that fails `checks` times, then succeeds."""
    calls = []

    def healthy(port):
        calls.append(port)
        return len(calls) > checks

    return healthy


@pytest.fixture
def settings(tmp_path):
    model = tmp_path / "bonsai.gguf"
    model.write_bytes(b"")
    # Every llama setting is given explicitly so a developer's .env doesn't leak in.
    return Settings(
        provider="llama",
        model=None,
        llama_server_bin="llama-server",
        llama_model_path=model,
        llama_port=9999,
        llama_ctx_size=65536,
    )


def run(settings, healthy, proc, **kwargs):
    with llama_server(settings, healthy=healthy, popen=proc, poll_interval=0, **kwargs):
        return proc.terminated


def test_server_command(settings):
    command = server_command(settings)
    assert command[0] == "llama-server"
    assert command[command.index("--model") + 1] == str(settings.llama_model_path)
    assert command[command.index("--alias") + 1] == "bonsai-2"
    assert command[command.index("--port") + 1] == "9999"


def test_reuses_running_server(settings):
    proc = FakeProcess()
    run(settings, healthy_after(0), proc)
    assert proc.command is None


def test_starts_and_stops_server(settings):
    proc = FakeProcess()
    terminated_during_run = run(settings, healthy_after(3), proc)
    assert proc.command == server_command(settings)
    assert not terminated_during_run
    assert proc.terminated and not proc.killed


def test_kills_server_that_ignores_terminate(settings, monkeypatch):
    monkeypatch.setattr("job_extractor.llama_server.STOP_TIMEOUT_S", 0)
    proc = FakeProcess(ignore_terminate=True)
    run(settings, healthy_after(1), proc)
    assert proc.terminated and proc.killed


def test_stops_server_when_run_fails(settings):
    proc = FakeProcess()
    with pytest.raises(KeyboardInterrupt):
        with llama_server(settings, healthy=healthy_after(1), popen=proc, poll_interval=0):
            raise KeyboardInterrupt
    assert proc.terminated


def test_server_exits_during_startup(settings):
    proc = FakeProcess(exit_code=1, output=b"error: unknown model type")
    with pytest.raises(LlamaServerError, match="code 1:\nerror: unknown model type"):
        run(settings, healthy_after(99), proc)


def test_server_startup_timeout(settings):
    proc = FakeProcess(output=b"loading")
    with pytest.raises(LlamaServerError, match="not ready after 0s:\nloading"):
        run(settings, healthy_after(99), proc, timeout=0)
    assert proc.terminated


def test_missing_binary(settings):
    def popen(*args, **kwargs):
        raise FileNotFoundError("llama-server")

    with pytest.raises(LlamaServerError, match="LLAMA_SERVER_BIN"):
        with llama_server(settings, healthy=healthy_after(99), popen=popen):
            pass


def test_model_path_required(settings, tmp_path):
    unset = settings.model_copy(update={"llama_model_path": None})
    with pytest.raises(LlamaServerError, match="LLAMA_MODEL_PATH is not set"):
        run(unset, healthy_after(99), FakeProcess())
    missing = settings.model_copy(update={"llama_model_path": tmp_path / "missing.gguf"})
    with pytest.raises(LlamaServerError, match="is not a file"):
        run(missing, healthy_after(99), FakeProcess())
