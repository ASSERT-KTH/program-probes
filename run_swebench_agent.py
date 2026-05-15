"""Run mini-SWE-agent on SWE-bench Verified instances with Modal sandboxes.

Each instance gets a fresh Modal sandbox loaded with its per-instance Docker
image (the repo is already at /testbed).  A single vLLM server stays up for
the full job; all parallel agent threads share it, and vLLM batches their
requests automatically.

Usage (single shard, all defaults from run config):
    export MODAL_TOKEN_ID="..."
    export MODAL_TOKEN_SECRET="..."

    uv run python run_swebench_agent.py \\
      --run-config configs/runs/qwen3_8b_swebench_test.yaml

Usage (array job — driven by swebench_run.sh):
    uv run python run_swebench_agent.py \\
      --run-config configs/runs/qwen3_8b_swebench_test.yaml \\
      --shard-rank 2 --num-shards 8
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import wandb

from src.agents.mini_swe import MiniSweAgentAdapter
from src.agents.swe_bench_environment import SWEBenchModalEnvironment
from src.agents.vllm_server import VllmServer
from src.configs import SWEBenchRunConfig, load_config
from src.tasks.swe_bench_verified import SWEBenchVerifiedAdapter, _make_eval_script


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True, help="Path to SWEBenchRunConfig YAML")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def _out_path(output_dir: Path, instance_id: str, run_idx: int) -> Path:
    if run_idx == 0:
        return output_dir / f"{instance_id}.json"
    return output_dir / f"{instance_id}_run{run_idx:02d}.json"


def _run_one(
    instance: dict,
    run_idx: int,
    *,
    cfg: SWEBenchRunConfig,
    server: VllmServer,
    output_dir: Path,
    index_lock: threading.Lock,
    index_path: Path,
) -> dict:
    """Run one (instance, run_idx) job. Called from a worker thread."""
    instance_id = instance["instance_id"]
    label = f"{instance_id}[run{run_idx}]"
    out_path = _out_path(output_dir, instance_id, run_idx)

    env = SWEBenchModalEnvironment(
        instance,
        timeout=cfg.modal_timeout,
        app_name=cfg.modal_app_name,
    )
    agent = MiniSweAgentAdapter(
        model_name=server.model_name,
        base_url=server.base_url,
        api_key="EMPTY",
        agent_config=cfg.to_agent_config(),
        generation_config=cfg.to_generation_config(),
        environment=env,
    )

    outcome = None
    patch = ""
    error = None

    try:
        task = (
            f"Repository: {instance.get('repo', '')}\n\n"
            f"Issue:\n{instance['problem_statement']}\n\n"
            "The repository is checked out at /testbed. Fix the issue."
        )
        agent.run_task(task)
        patch = env.get_patch()
        eval_script = instance.get("eval_script", "")
        outcome = env.evaluate(eval_script) if eval_script else None
        print(f"[{label}] outcome={'PASS' if outcome else 'FAIL'} patch={len(patch)}chars", flush=True)
    except Exception as exc:
        error = str(exc)
        print(f"[{label}] ERROR: {exc}", flush=True)
    finally:
        env.close()

    if error is None:
        agent.save_trajectory(
            out_path,
            tokenizer_name=cfg.model_id,
            metadata={
                "instance_id": instance_id,
                "run_idx": run_idx,
                "repo": instance.get("repo", ""),
                "base_commit": instance.get("base_commit", ""),
                "outcome": outcome,
                "patch": patch,
                "run_config": cfg.model_dump(),
            },
        )

    summary = {
        "instance_id": instance_id,
        "run_idx": run_idx,
        "outcome": outcome,
        "patch_len": len(patch),
        **({"error": error} if error else {}),
    }
    with index_lock:
        with index_path.open("a") as f:
            f.write(json.dumps(summary) + "\n")

    return summary


def main() -> None:
    args = _parse_args()
    cfg: SWEBenchRunConfig = load_config(args.run_config, SWEBenchRunConfig)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.jsonl"
    index_lock = threading.Lock()

    # Load and shard instances
    from src.configs import TaskConfig
    task_cfg = TaskConfig(dataset="princeton-nlp/SWE-bench_Verified", adapter="swe_bench_verified")
    adapter = SWEBenchVerifiedAdapter()
    all_instances = adapter.load_dataset(task_cfg)

    instances = [inst for i, inst in enumerate(all_instances) if i % args.num_shards == args.shard_rank]
    print(f"[swe-bench] shard {args.shard_rank}/{args.num_shards}: {len(instances)} instances", flush=True)

    # Build full job list: (instance, run_idx) pairs
    jobs = [
        (inst, run_idx)
        for inst in instances
        for run_idx in range(cfg.n_runs)
    ]

    if args.resume:
        n_before = len(jobs)
        jobs = [
            (inst, run_idx) for inst, run_idx in jobs
            if not _out_path(output_dir, inst["instance_id"], run_idx).exists()
        ]
        print(f"[swe-bench] resume: skipped {n_before - len(jobs)}, {len(jobs)} remaining", flush=True)

    if cfg.n_instances > 0:
        # Limit by unique instances, not total jobs
        seen: set[str] = set()
        limited = []
        for inst, run_idx in jobs:
            seen.add(inst["instance_id"])
            if len(seen) <= cfg.n_instances:
                limited.append((inst, run_idx))
        jobs = limited

    if not jobs:
        print("[swe-bench] nothing to do.", flush=True)
        return

    print(f"[swe-bench] {len(jobs)} jobs, {cfg.n_workers} workers", flush=True)

    run_name = f"{Path(args.run_config).stem}_shard{args.shard_rank}of{args.num_shards}"
    wandb.init(
        project="program-probes",
        name=run_name,
        config={**cfg.model_dump(), "shard_rank": args.shard_rank, "num_shards": args.num_shards},
        resume="allow",
    )

    vllm_cfg = cfg.to_vllm_server_config()
    model_cfg = cfg.to_model_config()

    n_total = len(jobs)
    n_done = n_passed = n_failed = n_errored = 0

    with VllmServer(model_cfg, vllm_cfg) as server:
        print(f"[swe-bench] vLLM ready at {server.base_url} ({server.model_name})", flush=True)

        with ThreadPoolExecutor(max_workers=cfg.n_workers) as pool:
            futures = {
                pool.submit(
                    _run_one,
                    inst,
                    run_idx,
                    cfg=cfg,
                    server=server,
                    output_dir=output_dir,
                    index_lock=index_lock,
                    index_path=index_path,
                ): (inst["instance_id"], run_idx)
                for inst, run_idx in jobs
            }
            for future in as_completed(futures):
                instance_id, run_idx = futures[future]
                try:
                    summary = future.result()
                except Exception as exc:
                    print(f"[swe-bench] unhandled exception for {instance_id}[run{run_idx}]: {exc}", flush=True)
                    summary = {"instance_id": instance_id, "run_idx": run_idx, "outcome": None, "patch_len": 0, "error": str(exc)}

                n_done += 1
                errored = "error" in summary
                passed = (not errored) and bool(summary.get("outcome"))
                if errored:
                    n_errored += 1
                elif passed:
                    n_passed += 1
                else:
                    n_failed += 1

                wandb.log({
                    "instance/outcome": 1 if passed else 0,
                    "instance/patch_len": summary.get("patch_len", 0),
                    "instance/errored": int(errored),
                    "progress/completed": n_done,
                    "progress/passed": n_passed,
                    "progress/failed": n_failed,
                    "progress/errored": n_errored,
                    "progress/pass_rate": n_passed / n_done,
                    "progress/remaining": n_total - n_done,
                })

    wandb.summary.update({
        "total": n_total,
        "passed": n_passed,
        "failed": n_failed,
        "errored": n_errored,
        "pass_rate": n_passed / n_done if n_done else 0.0,
    })
    wandb.finish()

    print(f"\n[swe-bench] done. index at {index_path}", flush=True)


if __name__ == "__main__":
    main()
