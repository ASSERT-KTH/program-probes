"""Label SWE-bench Pro agent trajectories by replaying cumulative diffs in Modal sandboxes.

For each trajectory, identifies edit steps (commands that changed files via their
recorded cumulative diff), then applies each diff from a clean git state inside a
per-instance Modal sandbox and evaluates the code using the Pro eval harness.

Use ``run_labeler_pro.py`` as the CLI entry point.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.swe_bench_pro_environment import _build_entry_script, _grade, _get_image_uri


_COMPILE_ERROR_PATTERNS = (
    "SyntaxError",
    "ImportError",
    "ModuleNotFoundError",
    "ERROR collecting",
    "import file mismatch",
)


def _compiles_from_log(log: Optional[str]) -> Optional[bool]:
    if log is None:
        return None
    for pattern in _COMPILE_ERROR_PATTERNS:
        if pattern in log:
            return False
    return True


def _create_pro_sandbox(instance: Dict, app_name: str, timeout: int) -> Any:
    """Create a Modal sandbox pre-loaded with the instance's Pro Docker image."""
    import modal

    app = modal.App.lookup(app_name, create_if_missing=True)
    image_uri = _get_image_uri(instance)
    image = modal.Image.from_registry(image_uri)
    return modal.Sandbox.create(app=app, image=image, timeout=timeout)


def _exec_in_sandbox(sandbox: Any, command: str, timeout: int = 180) -> Dict:
    """Execute a bash command in *sandbox* and return ``{returncode, output}``."""
    process = sandbox.exec("bash", "-lc", command, timeout=timeout)
    stdout = process.stdout.read()
    stderr = process.stderr.read() if getattr(process, "stderr", None) is not None else ""
    if hasattr(process, "wait"):
        process.wait()
    returncode = getattr(process, "returncode", 0)
    output = stdout
    if stderr:
        output = f"{stdout}\n{stderr}" if stdout else stderr
    return {"returncode": returncode, "output": output}


def _write_sandbox_file(sandbox: Any, remote_path: str, content: str) -> None:
    with sandbox.open(remote_path, "w") as f:
        f.write(content)


def _apply_patch_in_sandbox(sandbox: Any, patch: str, base_commit: str) -> bool:
    """Reset /app to base_commit and apply *patch*.  Returns True on success."""
    _exec_in_sandbox(
        sandbox,
        f"cd /app && git reset --hard {base_commit} && git checkout {base_commit}",
        timeout=30,
    )
    if not patch.strip():
        return True

    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(patch)
        patch_local = f.name

    try:
        with open(patch_local) as pf:
            content = pf.read()
        upload = f"cat > /tmp/edit.patch << 'PATCHEOF'\n{content}\nPATCHEOF"
        r = _exec_in_sandbox(sandbox, upload, timeout=30)
        if r["returncode"] != 0:
            return False
        r = _exec_in_sandbox(sandbox, "cd /app && git apply /tmp/edit.patch", timeout=30)
        return r["returncode"] == 0
    finally:
        os.unlink(patch_local)


def _run_pro_eval(
    sandbox: Any,
    instance: Dict,
    scripts_dir: Path,
    patch: str,
    eval_timeout: int = 600,
) -> tuple[Optional[bool], Optional[bool], Optional[Dict]]:
    """Apply *patch* and run the Pro eval harness.

    Returns (resolved, compiles, output_json).  All values are None on error.
    """
    base_commit = instance["base_commit"]
    instance_id = instance["instance_id"]

    run_scripts_dir = scripts_dir / "run_scripts" / instance_id
    run_script_path = run_scripts_dir / "run_script.sh"
    parser_path = run_scripts_dir / "parser.py"

    for p in (run_script_path, parser_path):
        if not p.exists():
            print(f"[pro-labeler] missing required file: {p}", flush=True)
            return None, None, None

    ok = _apply_patch_in_sandbox(sandbox, patch, base_commit)
    if not ok:
        print(f"[pro-labeler] {instance_id}: git apply failed", flush=True)
        return None, None, None

    entry_script = _build_entry_script(instance, scripts_dir)

    try:
        _exec_in_sandbox(sandbox, "mkdir -p /workspace", timeout=30)
        _write_sandbox_file(sandbox, "/workspace/patch.diff", patch)
        _write_sandbox_file(sandbox, "/workspace/run_script.sh", run_script_path.read_text())
        _write_sandbox_file(sandbox, "/workspace/parser.py", parser_path.read_text())
        _write_sandbox_file(sandbox, "/workspace/entryscript.sh", entry_script)

        process = sandbox.exec("bash", "/workspace/entryscript.sh", timeout=eval_timeout)
        stdout = process.stdout.read()
        stderr = process.stderr.read() if getattr(process, "stderr", None) else ""
        if hasattr(process, "wait"):
            process.wait()
        log = f"{stdout}\n{stderr}".strip()
        print(f"[pro-labeler] eval output (last 1000 chars):\n{log[-1000:]}", flush=True)

        with sandbox.open("/workspace/output.json", "r") as f:
            output_json = json.load(f)

        resolved = _grade(output_json, instance)
        compiles = _compiles_from_log(log)
        return resolved, compiles, output_json

    except Exception as exc:
        print(f"[pro-labeler] {instance_id}: eval exception: {exc}", flush=True)
        return None, None, None


def label_pro_trajectory(
    trajectory_path: Path,
    *,
    instance: Dict,
    scripts_dir: Path,
    output_path: Path,
    modal_app_name: str = "program-probes-labeler-pro",
    sandbox_timeout: int = 3600,
    eval_timeout: int = 600,
) -> Dict:
    """Label a single Pro trajectory by replaying cumulative diffs in a Modal sandbox.

    Mirrors the structure of swebench_labeler.label_trajectory but uses the Pro
    eval harness (entryscript.sh → output.json → _grade()) instead of the pytest
    log parser.
    """
    with open(trajectory_path) as f:
        traj = json.load(f)

    command_history = traj.get("command_history", [])
    if not command_history:
        print(f"[pro-labeler] {trajectory_path.name}: no command history", flush=True)
        result = {
            "instance_id": instance["instance_id"],
            "trajectory_path": str(trajectory_path),
            "edits": [],
        }
        _write_labels(result, output_path)
        return result

    # Identify edit steps — first command in each group that shares a cumulative diff
    edit_indices: List[int] = []
    last_diff = ""
    for i, entry in enumerate(command_history):
        d = entry.get("diff", "")
        if d and d != last_diff:
            edit_indices.append(i)
            last_diff = d

    if not edit_indices:
        print(f"[pro-labeler] {trajectory_path.name}: no edits", flush=True)
        result = {
            "instance_id": instance["instance_id"],
            "trajectory_path": str(trajectory_path),
            "edits": [],
        }
        _write_labels(result, output_path)
        return result

    print(
        f"[pro-labeler] {trajectory_path.name}: {len(edit_indices)} edits at cmds {edit_indices}",
        flush=True,
    )

    sandbox = _create_pro_sandbox(instance, modal_app_name, sandbox_timeout)

    edits: List[Dict] = []
    try:
        # Baseline: clean checkout (no patch)
        print("[pro-labeler]   clean checkout: running baseline eval ...", flush=True)
        resolved, compiles, output_json = _run_pro_eval(
            sandbox, instance, scripts_dir, patch="", eval_timeout=eval_timeout
        )
        print(
            f"[pro-labeler]   clean checkout: resolved={resolved}, compiles={compiles}",
            flush=True,
        )
        edits.append({
            "cmd_idx": -1,
            "compiles": compiles,
            "test_results": _to_test_results(resolved, output_json),
        })

        for cmd_idx in edit_indices:
            cumulative_diff = command_history[cmd_idx].get("diff", "")
            print(f"[pro-labeler]   cmd[{cmd_idx}]: running eval ...", flush=True)

            resolved, compiles, output_json = _run_pro_eval(
                sandbox, instance, scripts_dir,
                patch=cumulative_diff, eval_timeout=eval_timeout,
            )
            if resolved is None and compiles is None:
                edits.append({
                    "cmd_idx": cmd_idx,
                    "compiles": None,
                    "test_results": None,
                    "apply_error": True,
                })
                print(f"[pro-labeler]   cmd[{cmd_idx}]: eval failed", flush=True)
                continue

            edits.append({
                "cmd_idx": cmd_idx,
                "compiles": compiles,
                "test_results": _to_test_results(resolved, output_json),
            })
            print(
                f"[pro-labeler]   cmd[{cmd_idx}]: resolved={resolved}, compiles={compiles}",
                flush=True,
            )
    finally:
        sandbox.terminate()

    result = {
        "instance_id": instance["instance_id"],
        "trajectory_path": str(trajectory_path),
        "edits": edits,
    }
    _write_labels(result, output_path)
    print(f"[pro-labeler] wrote {output_path}", flush=True)
    return result


def _to_test_results(resolved: Optional[bool], output_json: Optional[Dict]) -> Optional[Dict]:
    """Convert Pro output.json to the same shape as swebench_labeler test_results."""
    if output_json is None:
        return None
    tests = output_json.get("tests", [])
    passed = [t["name"] for t in tests if t.get("status") == "PASSED"]
    failed = [t["name"] for t in tests if t.get("status") == "FAILED"]
    error = [t["name"] for t in tests if t.get("status") not in ("PASSED", "FAILED")]
    return {"passed": passed, "failed": failed, "error": error, "resolved": resolved}


def _write_labels(result: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def load_pro_instance(instance_id: str) -> Dict:
    """Load a single SWE-bench Pro instance from HuggingFace."""
    from datasets import load_dataset

    data = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    matches = [inst for inst in data if inst["instance_id"] == instance_id]
    if not matches:
        raise ValueError(f"Instance {instance_id!r} not found in SWE-bench Pro")
    return dict(matches[0])
