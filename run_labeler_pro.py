"""Run the SWE-bench Pro labeler on generated agent trajectories.

Replays each edit step of agent trajectories in Modal sandboxes, computing
probe labels (compiles, test_results) at every intermediate code state using
the Pro eval harness (entryscript.sh → output.json).

Usage:
    uv run python run_labeler_pro.py --config configs/labeling/laguna_xs2_swebench_pro_test_labeler.yaml

The config YAML fields are documented in SwebenchProLabelerConfig (src/configs.py).
"""

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.configs import SwebenchProLabelerConfig, load_config
from src.labeling.swebench_pro_labeler import label_pro_trajectory


def _process(
    tf: Path,
    instance: dict,
    cfg: SwebenchProLabelerConfig,
    out_dir: Path,
    scripts_dir: Path,
    jitter: float = 0.0,
) -> str:
    label_path = out_dir / f"{tf.stem}_labels.json"
    if cfg.resume and label_path.exists():
        return f"{tf.name}: skipped (labels exist)"
    if jitter > 0:
        time.sleep(random.uniform(0, jitter))
    label_pro_trajectory(
        tf,
        instance=instance,
        scripts_dir=scripts_dir,
        output_path=label_path,
        modal_app_name=cfg.modal_app_name,
        sandbox_timeout=cfg.sandbox_timeout,
        eval_timeout=cfg.eval_timeout,
    )
    return f"{tf.name}: done"


def main() -> None:
    parser = argparse.ArgumentParser(description="Label SWE-bench Pro agent trajectories")
    parser.add_argument(
        "--config", required=True,
        help="Path to SwebenchProLabelerConfig YAML (see configs/labeling/)",
    )
    args = parser.parse_args()

    cfg: SwebenchProLabelerConfig = load_config(args.config, SwebenchProLabelerConfig)

    traj_dir = Path(cfg.trajectory_dir)
    out_dir = Path(cfg.output_dir)
    scripts_dir = Path(cfg.scripts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_files = sorted(
        f for f in traj_dir.glob("*.json")
        if not f.name.startswith("index") and "_labels" not in f.name
    )
    if cfg.single:
        traj_files = [f for f in traj_files if f.name == cfg.single]

    # Extract instance IDs: strip _run<N> suffix if present
    def _iid(tf: Path) -> str:
        stem = tf.stem
        # e.g. django__django-12345_run01 → django__django-12345
        if "_run" in stem:
            stem = stem[: stem.rfind("_run")]
        return stem

    if cfg.instances:
        traj_files = [f for f in traj_files if _iid(f) in cfg.instances]

    print(f"Labeling {len(traj_files)} trajectories from {traj_dir} → {out_dir}", flush=True)

    # Pre-load dataset once to avoid N concurrent load_dataset() calls in threads
    print("Loading SWE-bench Pro dataset ...", flush=True)
    from datasets import load_dataset as _hf_load
    _data = _hf_load("ScaleAI/SWE-bench_Pro", split="test")
    instances: dict[str, dict] = {inst["instance_id"]: dict(inst) for inst in _data}
    print(f"Loaded {len(instances)} instances.", flush=True)

    def _get_instance(iid: str) -> dict:
        if iid not in instances:
            raise ValueError(f"Instance {iid!r} not found in SWE-bench Pro")
        return instances[iid]

    jitter = 2.0 if cfg.n_workers > 1 else 0.0

    if cfg.n_workers == 1:
        for tf in traj_files:
            result = _process(tf, _get_instance(_iid(tf)), cfg, out_dir, scripts_dir, jitter=0.0)
            print(result, flush=True)
    else:
        with ThreadPoolExecutor(max_workers=cfg.n_workers) as pool:
            futures = {
                pool.submit(_process, tf, _get_instance(_iid(tf)), cfg, out_dir, scripts_dir, jitter): tf.name
                for tf in traj_files
            }
            for fut in as_completed(futures):
                try:
                    print(fut.result(), flush=True)
                except Exception as exc:
                    print(f"{futures[fut]}: ERROR — {exc}", flush=True)


if __name__ == "__main__":
    main()
