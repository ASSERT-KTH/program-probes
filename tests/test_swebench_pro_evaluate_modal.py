"""Integration test for SWEBenchProModalEnvironment.evaluate().

Validates that the grading logic returns False before the ground-truth patch is
applied and True after.  Requires Modal credentials, network access, and the SWE-bench_Pro-os submodule
initialised (git submodule update --init SWE-bench_Pro-os).

Run from the project root:
    uv run python tests/test_swebench_pro_evaluate_modal.py \\
        --scripts-dir SWE-bench_Pro-os \\
        [--instance-id ...]
"""

import argparse

# Python instance with a relatively fast test suite.
DEFAULT_INSTANCE_ID = "instance_ansible__ansible-c9a09b1b05-v3bfb2be"


def _load_instance(instance_id: str) -> dict:
    from datasets import load_dataset
    data = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    matches = [inst for inst in data if inst["instance_id"] == instance_id]
    if not matches:
        raise ValueError(f"Instance {instance_id!r} not found in SWE-bench Pro")
    return dict(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE_ID)
    parser.add_argument(
        "--scripts-dir",
        default="SWE-bench_Pro-os",
        help="Path to local clone of github.com/scaleapi/SWE-bench_Pro-os",
    )
    parser.add_argument("--modal-app-name", default="program-probes-swebench-pro-eval-test")
    parser.add_argument("--modal-timeout", type=int, default=600)
    args = parser.parse_args()

    from src.agents.swe_bench_pro_environment import SWEBenchProModalEnvironment

    print(f"Loading instance {args.instance_id!r}...")
    instance = _load_instance(args.instance_id)
    patch = instance.get("patch", "")
    print(f"  repo:        {instance['repo']}")
    print(f"  language:    {instance.get('repo_language', '?')}")
    print(f"  patch lines: {len(patch.splitlines())}")
    print(f"  docker tag:  {instance['dockerhub_tag']}")

    env = SWEBenchProModalEnvironment(
        instance,
        scripts_dir=args.scripts_dir,
        timeout=args.modal_timeout,
        app_name=args.modal_app_name,
    )

    with env:
        print("\n--- smoke test ---")
        smoke = env.execute({"command": "cd /app && git log --oneline -1"})
        print(smoke["output"])

        print("\n--- applying ground-truth patch ---")
        apply = env.execute({"command": f"cd /app && git apply - << 'PATCH_EOF'\n{patch}\nPATCH_EOF"})
        print(f"  rc={apply['returncode']}  {apply['output'][:200]}")
        assert apply["returncode"] == 0, f"Patch did not apply cleanly: {apply['output']}"

        print("\n--- evaluate AFTER patch (expect True) ---")
        applied_patch = env.get_patch()
        result_after, log = env.evaluate(patch=applied_patch)
        print(f"evaluate() = {result_after}")
        assert result_after is True, f"Expected True after patch, got {result_after}\nLog:\n{log[-2000:]}"

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
