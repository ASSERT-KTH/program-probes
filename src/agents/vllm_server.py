from __future__ import annotations

import subprocess
import os
import signal
import time
import urllib.error
import urllib.request
from contextlib import AbstractContextManager
from pathlib import Path
from typing import IO

from src.configs import ModelConfig, VllmServerConfig


def build_vllm_command(model_config: ModelConfig, server_config: VllmServerConfig) -> list[str]:
    """Build the vLLM OpenAI-compatible server command without importing vLLM."""
    served_name = server_config.served_model_name or model_config.model_id
    return [
        "vllm",
        "serve",
        model_config.model_id,
        "--host",
        server_config.host,
        "--port",
        str(server_config.port),
        "--served-model-name",
        served_name,
        *server_config.extra_args,
    ]


def wait_for_vllm(base_url: str, timeout_s: int, interval_s: float) -> None:
    """Wait until vLLM's OpenAI-compatible /models endpoint is reachable."""
    deadline = time.monotonic() + timeout_s
    url = f"{base_url.rstrip('/')}/models"
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(interval_s)

    message = f"Timed out waiting for vLLM at {url}"
    if last_error is not None:
        message = f"{message}: {last_error}"
    raise TimeoutError(message)


class VllmServer(AbstractContextManager["VllmServer"]):
    """Subprocess owner for a vLLM OpenAI-compatible server."""

    def __init__(self, model_config: ModelConfig, server_config: VllmServerConfig):
        self.model_config = model_config
        self.server_config = server_config
        self.process: subprocess.Popen | None = None
        self._log_file: IO[str] | None = None

    @property
    def base_url(self) -> str:
        return self.server_config.base_url

    @property
    def model_name(self) -> str:
        return self.server_config.served_model_name or self.model_config.model_id

    def start(self) -> "VllmServer":
        stdout: int | IO[str] = subprocess.DEVNULL
        stderr: int | IO[str] = subprocess.STDOUT
        if self.server_config.log_path is not None:
            log_path = Path(self.server_config.log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = log_path.open("a")
            stdout = self._log_file

        self.process = subprocess.Popen(
            build_vllm_command(self.model_config, self.server_config),
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )

        wait_for_vllm(
            self.base_url,
            timeout_s=self.server_config.startup_timeout_s,
            interval_s=self.server_config.healthcheck_interval_s,
        )
        return self

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> "VllmServer":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
