"""Run mini-SWE-agent on a single SWE-bench Verified instance in Modal.

Run inside a Berzelius GPU allocation after repository setup and Modal auth:

    export MODAL_TOKEN_ID="..."
    export MODAL_TOKEN_SECRET="..."

    uv run python tests/run_swebench_single_instance.py \
      --model-config configs/models/qwen3_8b.yaml \
      --generation-config configs/generation.yaml \
      --vllm-config configs/agents/vllm_launch.yaml \
      --agent-config configs/agents/mini_swe_swebench.yaml

The agent runs against a single instance (default: astropy__astropy-12907, a small
realistic bug).  Pass --instance-id to target a different one.
"""

import argparse
import json

from src.agents.mini_swe import MiniSweAgentAdapter
from src.agents.swe_bench_environment import SWEBenchModalEnvironment
from src.agents.vllm_server import VllmServer
from src.configs import (
    GenerationConfig,
    MiniSweAgentConfig,
    ModelConfig,
    VllmServerConfig,
    load_config,
)

DEFAULT_INSTANCE_ID = "astropy__astropy-12907"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/models/qwen3_8b.yaml")
    parser.add_argument("--generation-config", default="configs/generation.yaml")
    parser.add_argument("--vllm-config", default="configs/agents/vllm_launch.yaml")
    parser.add_argument("--agent-config", default="configs/agents/mini_swe_swebench.yaml")
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE_ID)
    parser.add_argument("--modal-app-name", default="program-probes-swebench")
    parser.add_argument("--modal-timeout", type=int, default=180)
    parser.add_argument("--trajectory-output", default="outputs/agent_trajectories/swebench_single.json")
    args = parser.parse_args()

    model_cfg = load_config(args.model_config, ModelConfig)
    gen_cfg = load_config(args.generation_config, GenerationConfig)
    vllm_cfg = load_config(args.vllm_config, VllmServerConfig)
    agent_cfg = load_config(args.agent_config, MiniSweAgentConfig)

    print(f"Loading instance {args.instance_id!r}...")
    instance = _load_instance(args.instance_id)
    print(f"  repo: {instance['repo']}")
    print(f"  problem_statement: {instance['problem_statement'][:200]}...")

    env = SWEBenchModalEnvironment(
        instance,
        timeout=args.modal_timeout,
        app_name=args.modal_app_name,
    )

    print("Starting vLLM server...")
    with VllmServer(model_cfg, vllm_cfg) as server, env:
        print(f"vLLM ready at {server.base_url} with served model {server.model_name!r}")

        print("Smoke-testing sandbox...")
        smoke = env.execute({"command": "cd /testbed && git log --oneline -3 && echo sandbox_ok"})
        print(smoke["output"])

        agent = MiniSweAgentAdapter(
            model_name=server.model_name,
            base_url=server.base_url,
            api_key=vllm_cfg.api_key,
            agent_config=agent_cfg,
            generation_config=gen_cfg,
            environment=env,
        )

        task = (
            f"Repository: {instance['repo']}\n\n"
            f"Issue:\n{instance['problem_statement']}\n\n"
            "The repository is checked out at /testbed. Fix the issue."
        )

        print("Running mini-SWE-agent...")
        agent.run_task(task)

        print("\nCommand history (with diffs):")
        for idx, item in enumerate(agent.command_history, start=1):
            diff_summary = f" [diff: {len(item.get('diff', ''))} chars]" if item.get("diff") else ""
            print(f"  {idx}. rc={item['returncode']}{diff_summary} $ {item['command']!r:.120}")

        patch = env.get_patch()
        print(f"\nFinal patch ({len(patch)} chars):")
        print(patch[:2000] or "(no changes)")

        print("\nRunning SWE-bench evaluation...")
        outcome, eval_log = env.evaluate(instance["eval_script"])
        print(f"Outcome: {'PASS' if outcome else 'FAIL'}")

    agent.save_trajectory(
        args.trajectory_output,
        tokenizer_name=model_cfg.model_id,
        metadata={
            "instance_id": instance["instance_id"],
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "outcome": outcome,
            "patch": patch,
            "agent_config": args.agent_config,
        },
    )
    print(f"\nSaved trajectory to {args.trajectory_output}")


if __name__ == "__main__":
    main()
