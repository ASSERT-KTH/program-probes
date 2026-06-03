"""Run mini-SWE-agent on SWE-bench Pro instances with Modal sandboxes.

Each instance gets a fresh Modal sandbox loaded with its per-instance Docker
image from jefzda/sweap-images (the repo is already at /app).  A single vLLM
server stays up for the full job; all parallel agent threads share it.

Prerequisite: clone the official harness repo next to this project (or point
--scripts-dir at it):
    git clone https://github.com/scaleapi/SWE-bench_Pro-os

Usage (single shard):
    export MODAL_TOKEN_ID="..."
    export MODAL_TOKEN_SECRET="..."

    uv run python run_swebench_pro_agent.py \\
      --run-config configs/runs/qwen36_27b_swebench_pro_test.yaml

Usage (array job — driven by swebench_pro_run.sh):
    uv run python run_swebench_pro_agent.py \\
      --run-config configs/runs/qwen36_27b_swebench_pro_test.yaml \\
      --shard-rank 2 --num-shards 8
"""

from __future__ import annotations

import argparse
import re
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import wandb

from src.agents.mini_swe import MiniSweAgentAdapter
from src.agents.swe_bench_pro_environment import SWEBenchProModalEnvironment
from src.agents.vllm_server import VllmServer
from src.configs import SWEBenchRunConfig, load_config
from src.tasks.swe_bench_pro import SWEBenchProAdapter


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


def _extract_agent_metrics(agent: MiniSweAgentAdapter) -> dict:
    messages = agent.last_messages or []
    result = agent.last_result or {}

    prompt_tokens = 0
    completion_tokens = 0
    n_steps = 0
    timestamps: list[float] = []

    for msg in messages:
        extra = msg.get("extra") or {}
        if "actions" not in extra:
            continue
        n_steps += 1
        usage = (extra.get("response") or {}).get("usage") or {}
        prompt_tokens += usage.get("prompt_tokens", 0)
        completion_tokens += usage.get("completion_tokens", 0)
        ts = extra.get("timestamp")
        if ts is not None:
            timestamps.append(ts)

    duration_s = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0

    return {
        "agent/n_steps": n_steps,
        "agent/exit_status": result.get("exit_status", "unknown"),
        "agent/prompt_tokens": prompt_tokens,
        "agent/completion_tokens": completion_tokens,
        "agent/total_tokens": prompt_tokens + completion_tokens,
        "agent/tool_calls": len(agent.command_history),
        "agent/duration_s": duration_s,
    }


_VLLM_METRICS = {
    "vllm:num_requests_running": "vllm/requests_running",
    "vllm:num_requests_waiting": "vllm/requests_waiting",
    "vllm:gpu_cache_usage_perc": "vllm/gpu_cache_usage_perc",
    "vllm:prompt_tokens_total": "vllm/prompt_tokens_total",
    "vllm:generation_tokens_total": "vllm/generation_tokens_total",
}


def _scrape_vllm_metrics(base_url: str) -> dict[str, float]:
    try:
        url = f"{base_url.rstrip('/')}/metrics"
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read().decode()
    except Exception:
        return {}

    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^([\w:]+)(?:\{[^}]*\})?\s+([\d.e+\-]+)", line)
        if m and m.group(1) in _VLLM_METRICS:
            try:
                out[_VLLM_METRICS[m.group(1)]] = float(m.group(2))
            except ValueError:
                pass
    return out


def _vllm_metrics_poller(base_url: str, stop: threading.Event, interval: int = 30) -> None:
    while not stop.wait(interval):
        metrics = _scrape_vllm_metrics(base_url)
        if metrics:
            wandb.log(metrics)


def _run_one(
    instance: dict,
    run_idx: int,
    *,
    cfg: SWEBenchRunConfig,
    server: VllmServer,
    output_dir: Path,
) -> dict:
    instance_id = instance["instance_id"]
    label = f"{instance_id}[run{run_idx}]"
    out_path = _out_path(output_dir, instance_id, run_idx)

    env = SWEBenchProModalEnvironment(
        instance,
        scripts_dir=cfg.swe_bench_pro_scripts_dir,
        timeout=cfg.modal_timeout,
        app_name=cfg.modal_app_name,
    )
    agent = MiniSweAgentAdapter(
        model_name=server.model_name,
        base_url=server.base_url,
        api_key="EMPTY",
        agent_config=cfg.to_pro_agent_config(),
        generation_config=cfg.to_generation_config(),
        environment=env,
    )

    outcome = None
    patch = ""
    error = None
    eval_log = ""
    agent_metrics: dict = {}

    try:
        task = (
            f"Repository: {instance.get('repo', '')}\n\n"
            f"Issue:\n{instance['problem_statement']}"
        )
        agent.run_task(task)
        agent_metrics = _extract_agent_metrics(agent)
        patch = env.get_patch()
        outcome, eval_log = env.evaluate(patch=patch)
        print(
            f"[{label}] outcome={'PASS' if outcome else 'FAIL'} "
            f"patch={len(patch)}chars steps={agent_metrics.get('agent/n_steps', '?')} "
            f"exit={agent_metrics.get('agent/exit_status', '?')}",
            flush=True,
        )
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
                "eval_log": eval_log[-5000:] if eval_log else "",
                "patch": patch,
                "run_config": cfg.model_dump(),
            },
        )

    return {
        "instance_id": instance_id,
        "run_idx": run_idx,
        "outcome": outcome,
        "patch_len": len(patch),
        **agent_metrics,
        **({"error": error} if error else {}),
    }


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.run_config, SWEBenchRunConfig)
    assert isinstance(cfg, SWEBenchRunConfig)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.configs import TaskConfig
    task_cfg = TaskConfig(dataset="ScaleAI/SWE-bench_Pro", adapter="swe_bench_pro")
    adapter = SWEBenchProAdapter()
    all_instances = adapter.load_dataset(task_cfg)

    instances = [inst for i, inst in enumerate(all_instances) if i % args.num_shards == args.shard_rank]
    print(f"[swe-bench-pro] shard {args.shard_rank}/{args.num_shards}: {len(instances)} instances", flush=True)

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
        print(f"[swe-bench-pro] resume: skipped {n_before - len(jobs)}, {len(jobs)} remaining", flush=True)

    if cfg.n_instances > 0:
        seen: set[str] = set()
        limited = []
        for inst, run_idx in jobs:
            seen.add(inst["instance_id"])
            if len(seen) <= cfg.n_instances:
                limited.append((inst, run_idx))
        jobs = limited

    if not jobs:
        print("[swe-bench-pro] nothing to do.", flush=True)
        return

    print(f"[swe-bench-pro] {len(jobs)} jobs, {cfg.n_workers} workers", flush=True)

    run_name = f"{Path(args.run_config).stem}_shard{args.shard_rank}of{args.num_shards}"
    wandb.init(
        project="program-probes",
        name=run_name,
        config={**cfg.model_dump(), "shard_rank": args.shard_rank, "num_shards": args.num_shards},
        resume="allow",
    )

    vllm_cfg = cfg.to_vllm_server_config()
    if vllm_cfg.log_path:
        stem = vllm_cfg.log_path.replace(".log", "")
        vllm_cfg = vllm_cfg.model_copy(update={"log_path": f"{stem}_shard{args.shard_rank}.log"})
    model_cfg = cfg.to_model_config()

    n_total = len(jobs)
    n_done = n_passed = n_failed = n_errored = 0

    with VllmServer(model_cfg, vllm_cfg) as server:
        print(f"[swe-bench-pro] vLLM ready at {server.base_url} ({server.model_name})", flush=True)

        stop_poller = threading.Event()
        poller = threading.Thread(
            target=_vllm_metrics_poller,
            args=(server.base_url, stop_poller),
            daemon=True,
        )
        poller.start()

        try:
            with ThreadPoolExecutor(max_workers=cfg.n_workers) as pool:
                futures = {
                    pool.submit(
                        _run_one,
                        inst,
                        run_idx,
                        cfg=cfg,
                        server=server,
                        output_dir=output_dir,
                    ): (inst["instance_id"], run_idx)
                    for inst, run_idx in jobs
                }
                for future in as_completed(futures):
                    instance_id, run_idx = futures[future]
                    try:
                        summary = future.result()
                    except Exception as exc:
                        print(f"[swe-bench-pro] unhandled exception for {instance_id}[run{run_idx}]: {exc}", flush=True)
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
                        "instance/n_steps": summary.get("agent/n_steps", 0),
                        "instance/prompt_tokens": summary.get("agent/prompt_tokens", 0),
                        "instance/completion_tokens": summary.get("agent/completion_tokens", 0),
                        "instance/total_tokens": summary.get("agent/total_tokens", 0),
                        "instance/tool_calls": summary.get("agent/tool_calls", 0),
                        "instance/duration_s": summary.get("agent/duration_s", 0),
                        "instance/exit_status": summary.get("agent/exit_status", "unknown"),
                        "progress/completed": n_done,
                        "progress/passed": n_passed,
                        "progress/failed": n_failed,
                        "progress/errored": n_errored,
                        "progress/pass_rate": n_passed / n_done,
                        "progress/remaining": n_total - n_done,
                    })
        finally:
            stop_poller.set()
            poller.join(timeout=5)

    wandb.summary.update({
        "total": n_total,
        "passed": n_passed,
        "failed": n_failed,
        "errored": n_errored,
        "pass_rate": n_passed / n_done if n_done else 0.0,
    })
    wandb.finish()

    print(f"\n[swe-bench-pro] done.", flush=True)


if __name__ == "__main__":
    main()
