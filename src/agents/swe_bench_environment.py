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
        """Run the SWE-bench eval script and return True iff the instance is fully resolved.

        Uses swebench's own grading pipeline: parses per-test PASSED/FAILED lines and
        checks that all FAIL_TO_PASS tests now pass and all PASS_TO_PASS tests still pass.
        Exit-code alone is not reliable because the eval script's last command is always
        `git checkout` (resetting test files), which exits 0 regardless of test outcome.
        """
        if not eval_script:
            return False
        import os
        import tempfile
        from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS, ResolvedStatus
        from swebench.harness.grading import get_eval_tests_report, get_logs_eval, get_resolution_status
        from swebench.harness.test_spec.test_spec import make_test_spec
        try:
            upload_cmd = f"cat > /tmp/swe_eval.sh << 'SWE_EVAL_EOF'\n{eval_script}\nSWE_EVAL_EOF"
            process = self.sandbox.exec("bash", "-lc", upload_cmd, timeout=30)
            if hasattr(process, "wait"):
                process.wait()

            # Merge stderr into stdout: pytest output goes to stderr but the
            # >>>>> Start/End Test Output markers go to stdout.
            process = self.sandbox.exec(
                "bash", "-lc", "bash /tmp/swe_eval.sh 2>&1", timeout=self.timeout
            )
            output = process.stdout.read()
            if hasattr(process, "wait"):
                process.wait()

            print(f"[swe-eval output]\n{output[-3000:]}", flush=True)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
                f.write(output)
                log_path = f.name

            try:
                test_spec = make_test_spec(self.instance)
                eval_sm, found = get_logs_eval(test_spec, log_path)
                if not found:
                    print("[swe-eval] log parser found no test results", flush=True)
                    return False
                gold = {FAIL_TO_PASS: test_spec.FAIL_TO_PASS, PASS_TO_PASS: test_spec.PASS_TO_PASS}
                report = get_eval_tests_report(eval_sm, gold)
                resolved = get_resolution_status(report) == ResolvedStatus.FULL.value
                print(f"[swe-eval] resolved={resolved}", flush=True)
                return resolved
            finally:
                os.unlink(log_path)

        except Exception as exc:
            print(f"[swe-eval] exception during evaluation: {exc}", flush=True)
            return False
