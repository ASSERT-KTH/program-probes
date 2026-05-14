from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


@dataclass
class ModalSandboxEnvironment:
    """mini-SWE-agent environment adapter backed by modal.Sandbox.

    The agent controller runs in the local Python process. Only bash actions are
    sent to the Modal sandbox and converted back into mini-SWE-agent observations.
    """

    app_name: str = "program-probes-agent"
    image: Any | None = None
    timeout: int = 60
    env: dict[str, str] = field(default_factory=dict)
    command_history: list[dict[str, Any]] = field(default_factory=list)
    _sandbox: Any | None = field(default=None, init=False, repr=False)

    def start(self) -> "ModalSandboxEnvironment":
        try:
            import modal
        except ImportError as exc:
            raise ImportError("modal is required for ModalSandboxEnvironment.") from exc

        app = modal.App.lookup(self.app_name, create_if_missing=True)
        kwargs: dict[str, Any] = {"app": app}
        if self.image is not None:
            kwargs["image"] = self.image
        if self.env:
            kwargs["environment"] = self.env
        self._sandbox = modal.Sandbox.create(**kwargs)
        return self

    @property
    def sandbox(self) -> Any:
        if self._sandbox is None:
            self.start()
        return self._sandbox

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]:
        command = action.get("command", "")
        if not command:
            return {"returncode": 1, "output": "No command provided."}

        print(f"[modal bash] {command}", flush=True)
        wrapped = command if not cwd else f"cd {cwd} && {command}"
        process = self.sandbox.exec("bash", "-lc", wrapped, timeout=self.timeout)
        stdout = process.stdout.read()
        stderr = process.stderr.read() if getattr(process, "stderr", None) is not None else ""
        if hasattr(process, "wait"):
            process.wait()
        returncode = getattr(process, "returncode", 0)

        output = stdout
        if stderr:
            output = f"{output}\n{stderr}" if output else stderr
        result = {"returncode": returncode, "output": output, "exception_info": ""}
        self.command_history.append(
            {
                "command": command,
                "cwd": cwd,
                "returncode": returncode,
                "output": output,
            }
        )
        self._check_finished(result)
        return result

    def _check_finished(self, output: dict[str, Any]) -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(platform.uname()._asdict(), os.environ, kwargs)

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": {
                        "app_name": self.app_name,
                        "timeout": self.timeout,
                        "env": self.env,
                    },
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def close(self) -> None:
        if self._sandbox is not None:
            self._sandbox.terminate()
            self._sandbox = None

    def __enter__(self) -> "ModalSandboxEnvironment":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
