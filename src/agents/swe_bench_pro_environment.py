from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from src.agents.modal_environment import ModalSandboxEnvironment


def _get_image_uri(instance: dict) -> str:
    """Return the Docker Hub URI for a SWE-bench Pro instance.

    The tag is stored directly in the dataset as `dockerhub_tag`.
    """
    tag = instance["dockerhub_tag"]
    return f"jefzda/sweap-images:{tag}"


def _extract_env_exports(dockerfile_text: str) -> str:
    """Convert Dockerfile ENV lines to shell export statements."""
    lines = []
    for line in dockerfile_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("ENV "):
            lines.append(stripped.replace("ENV ", "export ", 1))
    return "\n".join(lines)


def _build_entry_script(instance: dict, scripts_dir: Path) -> str:
    """Build the entryscript.sh content for a Pro instance.

    Mirrors create_entryscript() from the official SWE-bench_Pro-os harness
    (https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/swe_bench_pro_eval.py).
    """
    instance_id = instance["instance_id"]
    base_commit = instance["base_commit"]
    before_repo_set_cmd = instance["before_repo_set_cmd"].strip().split("\n")[-1]
    selected_files = ",".join(ast.literal_eval(instance["selected_test_files_to_run"]))

    base_df = scripts_dir / "dockerfiles" / "base_dockerfile" / instance_id / "Dockerfile"
    inst_df = scripts_dir / "dockerfiles" / "instance_dockerfile" / instance_id / "Dockerfile"

    env_exports = ""
    for df_path in (base_df, inst_df):
        if df_path.exists():
            env_exports += _extract_env_exports(df_path.read_text()) + "\n"

    return f"""\
#!/bin/bash
set -uo pipefail
{env_exports}
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{before_repo_set_cmd}
bash /workspace/run_script.sh {selected_files} > /workspace/stdout.log 2> /workspace/stderr.log
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""


def _grade(output_json: dict, instance: dict) -> bool:
    """Return True iff all fail_to_pass and pass_to_pass tests are PASSED."""
    passed = {t["name"] for t in output_json.get("tests", []) if t["status"] == "PASSED"}
    f2p = set(ast.literal_eval(instance["fail_to_pass"]))
    p2p = set(ast.literal_eval(instance["pass_to_pass"]))
    return (f2p | p2p) <= passed


class SWEBenchProModalEnvironment(ModalSandboxEnvironment):
    """ModalSandboxEnvironment pre-loaded with a SWE-bench Pro Docker image.

    `scripts_dir` must point to a local clone of
    https://github.com/scaleapi/SWE-bench_Pro-os so that per-instance
    run_scripts/ and dockerfiles/ are available at eval time.
    """

    def __init__(
        self,
        instance: dict,
        *,
        scripts_dir: str | Path,
        timeout: int = 180,
        app_name: str = "program-probes-swebench-pro",
    ):
        try:
            import modal
        except ImportError as exc:
            raise ImportError("modal is required for SWEBenchProModalEnvironment.") from exc

        image_uri = _get_image_uri(instance)
        image = modal.Image.from_registry(image_uri)
        super().__init__(app_name=app_name, image=image, timeout=timeout)
        self.instance = instance
        self.scripts_dir = Path(scripts_dir)

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]:
        result = super().execute(action, cwd=cwd)
        diff = self._capture_diff()
        result["diff"] = diff
        return result

    def _capture_diff(self) -> str:
        try:
            process = self.sandbox.exec("bash", "-lc", "cd /app && git diff HEAD -- .", timeout=30)
            diff = process.stdout.read()
            if hasattr(process, "wait"):
                process.wait()
            return diff or ""
        except Exception:
            return ""

    def get_patch(self) -> str:
        return self._capture_diff()

    def _write_sandbox_file(self, remote_path: str, content: str) -> None:
        """Write a text file directly into the sandbox via sandbox.open()."""
        with self.sandbox.open(remote_path, "w") as f:
            f.write(content)

    def evaluate(self, patch: str = "", _eval_script: str = "") -> tuple[bool, str]:
        """Run the Pro eval harness and return (resolved, log).

        Uploads patch.diff, run_script.sh, parser.py, and entryscript.sh into
        /workspace/ inside the sandbox, then executes entryscript.sh and reads
        back /workspace/output.json to grade the result.

        Pass `patch` explicitly to avoid a redundant get_patch() RPC when the
        caller already holds the diff. If omitted, get_patch() is called here.
        `_eval_script` is accepted for interface compatibility but ignored — the
        entry script is generated from the instance fields and local run_scripts/.
        """
        instance_id = self.instance["instance_id"]
        run_scripts_dir = self.scripts_dir / "run_scripts" / instance_id

        run_script_path = run_scripts_dir / "run_script.sh"
        parser_path = run_scripts_dir / "parser.py"
        for p in (run_script_path, parser_path):
            if not p.exists():
                print(f"[swe-pro-eval] missing required file: {p}", flush=True)
                return False, f"missing {p}"

        if not patch:
            patch = self.get_patch()
        if not patch.strip():
            print("[swe-pro-eval] no patch to evaluate", flush=True)
            return False, "no patch"

        entry_script = _build_entry_script(self.instance, self.scripts_dir)

        try:
            self.sandbox.exec("bash", "-lc", "mkdir -p /workspace", timeout=30).wait()

            self._write_sandbox_file("/workspace/patch.diff", patch)
            self._write_sandbox_file("/workspace/run_script.sh", run_script_path.read_text())
            self._write_sandbox_file("/workspace/parser.py", parser_path.read_text())
            self._write_sandbox_file("/workspace/entryscript.sh", entry_script)

            process = self.sandbox.exec(
                "bash", "/workspace/entryscript.sh", timeout=self.timeout
            )
            stdout = process.stdout.read()
            stderr = process.stderr.read() if getattr(process, "stderr", None) else ""
            if hasattr(process, "wait"):
                process.wait()
            log = f"{stdout}\n{stderr}".strip()
            print(f"[swe-pro-eval output]\n{log[-3000:]}", flush=True)

            with self.sandbox.open("/workspace/output.json", "r") as f:
                output_json = json.load(f)

            resolved = _grade(output_json, self.instance)
            print(f"[swe-pro-eval] resolved={resolved}", flush=True)
            return resolved, log

        except Exception as exc:
            print(f"[swe-pro-eval] exception during evaluation: {exc}", flush=True)
            return False, str(exc)
