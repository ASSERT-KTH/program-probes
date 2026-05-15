"""Run mini-SWE-agent on SWE-bench Verified instances with Modal sandboxes.

Each instance gets a fresh Modal sandbox pre-loaded with its Docker image (the repo
is already at /testbed).  vLLM stays running for the full job.  After the agent
finishes each instance, the git diff is captured and the SWE-bench eval script is
run inside the sandbox to determine pass/fail.  Trajectories are saved as JSON.

Usage (single shard):
    export MODAL_TOKEN_ID="..."
    export MODAL_TOKEN_SECRET="..."

    uv run python run_swebench_agent.py \\
      --model-config configs/models/qwen3_8b.yaml \\
      --generation-config configs/generation.yaml \\
      --vllm-config configs/agents/vllm_launch.yaml \\
      --agent-config configs/agents/mini_swe_swebench.yaml \\
      --output-dir generations/swebench/

Usage (array job — pass SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT):
    uv run python run_swebench_agent.py ... \\
      --shard-rank 2 --num-shards 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agents.mini_swe import MiniSweAgentAdapter
from src.agents.swe_bench_environment import SWEBenchModalEnvironment
from src.agents.vllm_server import VllmServer
from src.configs import (
    GenerationConfig,
    MiniSweAgentConfig,
    ModelConfig,
    TaskConfig,
    VllmServerConfig,
    load_config,
)
from src.tasks.swe_bench_verified import SWEBenchVerifiedAdapter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/models/qwen3_8b.yaml")
    parser.add_argument("--generation-config", default="configs/generation.yaml")
    parser.add_argument("--vllm-config", default="configs/agents/vllm_launch.yaml")
    parser.add_argument("--agent-config", default="configs/agents/mini_swe_swebench.yaml")
    parser.add_argument("--task-config", default="configs/tasks/swe_bench_verified.yaml")
    parser.add_argument("--output-dir", default="generations/swebench")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--n-instances", type=int, default=-1, help="Max instances to run (-1 = all)")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip already-completed instances")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--modal-app-name", default="program-probes-swebench")
    parser.add_argument("--modal-timeout", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    model_cfg = load_config(args.model_config, ModelConfig)
    gen_cfg = load_config(args.generation_config, GenerationConfig)
    vllm_cfg = load_config(args.vllm_config, VllmServerConfig)
    agent_cfg = load_config(args.agent_config, MiniSweAgentConfig)
    task_cfg = load_config(args.task_config, TaskConfig)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.jsonl"

    task_adapter = SWEBenchVerifiedAdapter()
    all_instances = task_adapter.load_dataset(task_cfg)

    # Shard by index so each instance always belongs to the same shard
    instances = [inst for i, inst in enumerate(all_instances) if i % args.num_shards == args.shard_rank]
    print(f"[swe-bench] shard {args.shard_rank}/{args.num_shards}: {len(instances)} instances", flush=True)

    if args.resume:
        n_before = len(instances)
        instances = [inst for inst in instances if not (output_dir / f"{inst['instance_id']}.json").exists()]
        print(f"[swe-bench] resume: skipped {n_before - len(instances)}, {len(instances)} remaining", flush=True)

    if args.n_instances > 0:
        instances = instances[: args.n_instances]

    if not instances:
        print("[swe-bench] nothing to do.", flush=True)
        return

    print(f"[swe-bench] starting vLLM server...", flush=True)
    with VllmServer(model_cfg, vllm_cfg) as server:
        print(f"[swe-bench] vLLM ready at {server.base_url} ({server.model_name})", flush=True)

        for idx, instance in enumerate(instances):
            instance_id = instance["instance_id"]
            out_path = output_dir / f"{instance_id}.json"
            print(f"\n[swe-bench] [{idx+1}/{len(instances)}] {instance_id}", flush=True)

            env = SWEBenchModalEnvironment(
                instance,
                timeout=args.modal_timeout,
                app_name=args.modal_app_name,
            )
            agent = MiniSweAgentAdapter(
                model_name=server.model_name,
                base_url=server.base_url,
                api_key=vllm_cfg.api_key,
                agent_config=agent_cfg,
                generation_config=gen_cfg,
                environment=env,
            )

            try:
                prompt = task_adapter.format_prompt(instance)
                task_str = prompt.user_content
                agent.run_task(task_str)

                patch = env.get_patch()
                print(f"[swe-bench] patch length: {len(patch)} chars", flush=True)

                eval_script = instance.get("eval_script", "")
                outcome = env.evaluate(eval_script) if eval_script else None
                print(f"[swe-bench] outcome: {outcome}", flush=True)

                agent.save_trajectory(
                    out_path,
                    tokenizer_name=model_cfg.model_id,
                    metadata={
                        "instance_id": instance_id,
                        "repo": instance.get("repo", ""),
                        "base_commit": instance.get("base_commit", ""),
                        "outcome": outcome,
                        "patch": patch,
                        "model_config": args.model_config,
                        "agent_config": args.agent_config,
                    },
                )

                summary = {"instance_id": instance_id, "outcome": outcome, "patch_len": len(patch)}
                with index_path.open("a") as f:
                    f.write(json.dumps(summary) + "\n")

                print(f"[swe-bench] saved {out_path}", flush=True)

            except Exception as exc:
                print(f"[swe-bench] ERROR on {instance_id}: {exc}", flush=True)
                summary = {"instance_id": instance_id, "outcome": None, "error": str(exc)}
                with index_path.open("a") as f:
                    f.write(json.dumps(summary) + "\n")
            finally:
                env.close()

    print(f"\n[swe-bench] done. index at {index_path}", flush=True)


if __name__ == "__main__":
    main()
