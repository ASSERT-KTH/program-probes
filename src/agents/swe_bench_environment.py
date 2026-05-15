from __future__ import annotations

from typing import Any

from src.agents.modal_environment import ModalSandboxEnvironment


def _docker_image_url(instance_id: str) -> str:
    escaped = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{escaped}"


class SWEBenchModalEnvironment(ModalSandboxEnvironment):
    """ModalSandboxEnvironment pre-loaded with the per-instance SWE-bench Docker image.

    After each bash command the environment captures `git diff HEAD -- .` so that
    the command history records which tools actually changed program state.
    """

    def __init__(self, instance: dict, *, timeout: int = 180, app_name: str = "program-probes-swebench"):
        try:
            import modal
        except ImportError as exc:
            raise ImportError("modal is required for SWEBenchModalEnvironment.") from exc

        image_url = _docker_image_url(instance["instance_id"])
        image = modal.Image.from_registry(image_url)
        super().__init__(app_name=app_name, image=image, timeout=timeout)
        self.instance = instance

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]:
        result = super().execute(action, cwd=cwd)
        diff = self._capture_diff()
        result["diff"] = diff
        return result

    def _capture_diff(self) -> str:
        try:
            process = self.sandbox.exec("bash", "-lc", "cd /testbed && git diff HEAD -- .", timeout=30)
            diff = process.stdout.read()
            if hasattr(process, "wait"):
                process.wait()
            return diff or ""
        except Exception:
            return ""

    def get_patch(self) -> str:
        """Return the current git diff — the full patch produced by the agent."""
        return self._capture_diff()

    def evaluate(self, eval_script: str) -> bool:
        """Run the SWE-bench eval script against the current repo state.

        The eval script is written to /tmp/swe_eval.sh inside the sandbox and
        executed there.  Return True if all targeted tests pass (exit code 0).
        """
        if not eval_script:
            return False
        try:
            import shlex
            upload_cmd = f"cat > /tmp/swe_eval.sh << 'SWE_EVAL_EOF'\n{eval_script}\nSWE_EVAL_EOF"
            process = self.sandbox.exec("bash", "-lc", upload_cmd, timeout=30)
            if hasattr(process, "wait"):
                process.wait()

            process = self.sandbox.exec(
                "bash", "-lc", "bash /tmp/swe_eval.sh", timeout=self.timeout
            )
            stdout = process.stdout.read()
            stderr = process.stderr.read() if getattr(process, "stderr", None) is not None else ""
            if hasattr(process, "wait"):
                process.wait()
            returncode = getattr(process, "returncode", 1)
            print(f"[swe-eval] returncode={returncode}", flush=True)
            if stdout:
                print(f"[swe-eval stdout]\n{stdout[-2000:]}", flush=True)
            if stderr:
                print(f"[swe-eval stderr]\n{stderr[-500:]}", flush=True)
            return returncode == 0
        except Exception as exc:
            print(f"[swe-eval] exception during evaluation: {exc}", flush=True)
            return False
