"""Label SWE-bench agent trajectories by replaying cumulative diffs in Modal sandboxes.

For each trajectory, identifies edit steps (commands that changed files via their
recorded cumulative diff), then applies each diff from a clean git state inside a
per-instance Modal sandbox and evaluates the code at that point.

Use ``run_labeler.py`` as the CLI entry point.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def _changed_python_files(diff: str) -> List[str]:
    """Extract unique .py file paths from a unified diff header."""
    files: List[str] = []
    for line in diff.splitlines():
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            fname = line[6:]
            if "\t" in fname:
                fname = fname.split("\t")[0]
            if fname.endswith(".py") and fname not in files:
                files.append(fname)
    return files


def _parse_test_results(eval_log: str, instance: Dict) -> Optional[Dict]:
    """Run the swebench grading pipeline on *eval_log*.

    Returns ``{"passed": [...], "failed": [...], "error": [...], "resolved": bool}``
    where *resolved* uses the standard SWE-bench resolution check (all FAIL_TO_PASS
    now pass AND all PASS_TO_PASS still pass).  Returns *None* if parsing failed.
    """
    try:
        from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS, ResolvedStatus
        from swebench.harness.grading import get_eval_tests_report, get_logs_eval, get_resolution_status
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError:
        print("[labeler] swebench not installed", flush=True)
        return None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(eval_log)
        log_path = f.name

    try:
        test_spec = make_test_spec(instance)
        eval_sm, found = get_logs_eval(test_spec, log_path)
        if not found:
            return None

        passed, failed, error = [], [], []
        for test_name, status in eval_sm.items():
            status_str = status.value if hasattr(status, "value") else str(status)
            if status_str == "PASSED":
                passed.append(test_name)
            elif status_str == "FAILED":
                failed.append(test_name)
            else:
                error.append(test_name)

        gold = {FAIL_TO_PASS: test_spec.FAIL_TO_PASS, PASS_TO_PASS: test_spec.PASS_TO_PASS}
        report = get_eval_tests_report(eval_sm, gold)
        resolved = get_resolution_status(report) == ResolvedStatus.FULL.value

        return {"passed": passed, "failed": failed, "error": error, "resolved": resolved}
    finally:
        os.unlink(log_path)


def _check_compiles(sandbox: Any, diff: str) -> bool:
    """Return *True* iff every ``.py`` file touched by *diff* compiles cleanly.

    If *diff* is empty or touches no ``.py`` files the result is *True*.
    """
    py_files = _changed_python_files(diff)
    if not py_files:
        return True

    for fname in py_files:
        process = sandbox.exec(
            "bash", "-lc", f"cd /testbed && python -m py_compile {fname}", timeout=30
        )
        stderr = process.stderr.read() if getattr(process, "stderr", None) is not None else ""
        if hasattr(process, "wait"):
            process.wait()
        returncode = getattr(process, "returncode", 0)
        if returncode != 0 or "Error" in stderr or "SyntaxError" in stderr:
            return False
    return True


def _create_sandbox(instance: Dict, app_name: str, timeout: int) -> Any:
    """Create a Modal sandbox pre-loaded with the instance's Docker image."""
    import modal
    from src.agents.swe_bench_environment import _docker_image_url

    app = modal.App.lookup(app_name, create_if_missing=True)
    image_url = _docker_image_url(instance["instance_id"])
    image = modal.Image.from_registry(image_url)
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


def _apply_patch_in_sandbox(sandbox: Any, patch: str) -> bool:
    """Upload *patch* to the sandbox and ``git apply`` it.  Returns *True* on success."""
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
        r = _exec_in_sandbox(sandbox, "cd /testbed && git apply /tmp/edit.patch", timeout=30)
        return r["returncode"] == 0
    finally:
        os.unlink(patch_local)


def _run_eval_in_sandbox(sandbox: Any, eval_script: str, timeout: int = 600) -> str:
    """Upload and run the SWE-bench eval script in *sandbox*, return stdout+stderr."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(eval_script)
        script_local = f.name

    try:
        with open(script_local) as sf:
            content = sf.read()
        upload = f"cat > /tmp/swe_eval.sh << 'SWEEVALEOF'\n{content}\nSWEEVALEOF"
        _exec_in_sandbox(sandbox, upload, timeout=30)
        r = _exec_in_sandbox(sandbox, "bash /tmp/swe_eval.sh 2>&1", timeout=timeout)
        return r["output"]
    finally:
        os.unlink(script_local)


def label_trajectory(
    trajectory_path: Path,
    *,
    instance: Dict,
    eval_script: str,
    output_path: Path,
    modal_app_name: str = "program-probes-labeler",
    sandbox_timeout: int = 3600,
    eval_timeout: int = 600,
) -> Dict:
    """Label a single trajectory by applying cumulative diffs in a Modal sandbox.

    1. Load the trajectory JSON and identify edit commands (unique cumulative diffs).
    2. Create a Modal sandbox from the instance's Docker image.
    3. For each edit, reset to clean HEAD, apply the cumulative diff, then:
       - run ``py_compile`` on changed ``.py`` files → ``compiles``
       - run the SWE-bench eval script → ``test_results``
    4. Write ``_labels.json`` alongside the trajectory.
    """
    with open(trajectory_path) as f:
        traj = json.load(f)

    command_history = traj.get("command_history", [])
    if not command_history:
        print(f"[labeler] {trajectory_path.name}: no command history", flush=True)
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
        print(f"[labeler] {trajectory_path.name}: no edits", flush=True)
        result = {
            "instance_id": instance["instance_id"],
            "trajectory_path": str(trajectory_path),
            "edits": [],
        }
        _write_labels(result, output_path)
        return result

    print(
        f"[labeler] {trajectory_path.name}: {len(edit_indices)} edits at cmds {edit_indices}",
        flush=True,
    )

    sandbox = _create_sandbox(instance, modal_app_name, sandbox_timeout)

    edits: List[Dict] = []
    try:
        # Evaluate the clean checkout first to establish a baseline for
        # currently_reduces_failing and currently_has_regressions probes.
        print(f"[labeler]   clean checkout: running baseline eval ...", flush=True)
        baseline_output = _run_eval_in_sandbox(sandbox, eval_script, timeout=eval_timeout)
        baseline_results = _parse_test_results(baseline_output, instance)
        n_passed = len(baseline_results.get("passed", [])) if baseline_results else "?"
        n_failed = len(baseline_results.get("failed", [])) if baseline_results else "?"
        n_errors = len(baseline_results.get("error", [])) if baseline_results else "?"
        resolved = baseline_results.get("resolved") if baseline_results else None
        print(
            f"[labeler]   clean checkout: compiles=True, resolved={resolved}, "
            f"passed={n_passed}, failed={n_failed}, errors={n_errors}",
            flush=True,
        )
        edits.append({
            "cmd_idx": -1,
            "compiles": True,
            "test_results": baseline_results,
        })

        for cmd_idx in edit_indices:
            cumulative_diff = command_history[cmd_idx].get("diff", "")

            # Reset to clean HEAD, then apply the cumulative diff
            _exec_in_sandbox(sandbox, "cd /testbed && git checkout HEAD -- .", timeout=30)

            if cumulative_diff.strip():
                if not _apply_patch_in_sandbox(sandbox, cumulative_diff):
                    edits.append({
                        "cmd_idx": cmd_idx,
                        "compiles": None,
                        "test_results": None,
                        "apply_error": True,
                    })
                    print(f"[labeler]   cmd[{cmd_idx}]: git apply failed", flush=True)
                    continue

            compiles = _check_compiles(sandbox, cumulative_diff)
            eval_output = _run_eval_in_sandbox(sandbox, eval_script, timeout=eval_timeout)
            test_results = _parse_test_results(eval_output, instance)

            edits.append({
                "cmd_idx": cmd_idx,
                "compiles": compiles,
                "test_results": test_results,
            })

            n_passed = len(test_results.get("passed", [])) if test_results else "?"
            n_failed = len(test_results.get("failed", [])) if test_results else "?"
            n_errors = len(test_results.get("error", [])) if test_results else "?"
            resolved = test_results.get("resolved") if test_results else None
            print(
                f"[labeler]   cmd[{cmd_idx}]: compiles={compiles}, "
                f"resolved={resolved}, passed={n_passed}, failed={n_failed}, errors={n_errors}",
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
    print(f"[labeler] wrote {output_path}", flush=True)
    return result


def _write_labels(result: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def load_instance(instance_id: str) -> Dict:
    """Load a single SWE-bench Verified instance from HuggingFace."""
    from datasets import load_dataset
    from src.tasks.swe_bench_verified import _make_eval_script

    data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    matches = [inst for inst in data if inst["instance_id"] == instance_id]
    if not matches:
        raise ValueError(f"Instance {instance_id!r} not found in SWE-bench Verified")
    instance = dict(matches[0])
    instance["eval_script"] = _make_eval_script(instance)
    return instance


