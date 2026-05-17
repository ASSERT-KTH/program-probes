"""Integration test for SWEBenchModalEnvironment.evaluate().

Validates that the grading logic correctly returns False before the ground-truth
patch is applied and True after.  Requires Modal credentials and network access.

Run from the project root:
    uv run python tests/test_swebench_evaluate_modal.py [--instance-id ...]
"""

import argparse

# Small instance with a fast test suite (~0.4s), confirmed to fail without patch.
DEFAULT_INSTANCE_ID = "astropy__astropy-13033"


def _load_instance(instance_id: str) -> dict:
    from datasets import load_dataset
    from swebench.harness.test_spec.test_spec import make_test_spec

    data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    matches = [inst for inst in data if inst["instance_id"] == instance_id]
    if not matches:
        raise ValueError(f"Instance {instance_id!r} not found in SWE-bench Verified")
    instance = dict(matches[0])
    instance["eval_script"] = make_test_spec(instance).eval_script
    return instance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE_ID)
    parser.add_argument("--modal-app-name", default="program-probes-swebench-eval-test")
    parser.add_argument("--modal-timeout", type=int, default=300)
    args = parser.parse_args()

    from src.agents.swe_bench_environment import SWEBenchModalEnvironment

    print(f"Loading instance {args.instance_id!r}...")
    instance = _load_instance(args.instance_id)
    patch = instance.get("patch", "")
    print(f"  repo:        {instance['repo']}")
    print(f"  patch lines: {len(patch.splitlines())}")

    env = SWEBenchModalEnvironment(
        instance,
        timeout=args.modal_timeout,
        app_name=args.modal_app_name,
    )

    with env:
        print("\n--- smoke test ---")
        smoke = env.execute({"command": "cd /testbed && git log --oneline -1"})
        print(smoke["output"])

        print("\n--- evaluate BEFORE patch (expect False) ---")
        result_before = env.evaluate(instance["eval_script"])
        print(f"evaluate() = {result_before}")
        assert result_before is False, f"Expected False before patch, got {result_before}"

        print("\n--- applying ground-truth patch ---")
        apply = env.execute({"command": f"cd /testbed && git apply - << 'PATCH_EOF'\n{patch}\nPATCH_EOF"})
        print(f"  rc={apply['returncode']}  {apply['output'][:200]}")
        assert apply["returncode"] == 0, f"Patch did not apply cleanly: {apply['output']}"

        print("\n--- evaluate AFTER patch (expect True) ---")
        result_after = env.evaluate(instance["eval_script"])
        print(f"evaluate() = {result_after}")
        assert result_after is True, f"Expected True after patch, got {result_after}"

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
