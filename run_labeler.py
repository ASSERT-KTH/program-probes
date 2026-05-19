"""Run the SWE-bench labeler on generated agent trajectories.

Replays each edit step of agent trajectories in Modal sandboxes, computing
probe labels (compiles, test_results) at every intermediate code state.

Usage:
    uv run python run_labeler.py --config configs/labeling/swebench_labeler.yaml

The config YAML fields are documented in SwebenchLabelerConfig (src/configs.py).
"""

import argparse
from pathlib import Path

from src.configs import SwebenchLabelerConfig, load_config
from src.labeling.swebench_labeler import label_trajectory, load_instance


def main() -> None:
    parser = argparse.ArgumentParser(description="Label SWE-bench agent trajectories")
    parser.add_argument(
        "--config", required=True,
        help="Path to SwebenchLabelerConfig YAML (see configs/labeling/)",
    )
    args = parser.parse_args()

    cfg: SwebenchLabelerConfig = load_config(args.config, SwebenchLabelerConfig)

    traj_dir = Path(cfg.trajectory_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_files = sorted(
        f for f in traj_dir.glob("*.json")
        if not f.name.startswith("index") and "_labels" not in f.name
    )
    if cfg.single:
        traj_files = [traj_dir / cfg.single]
    if cfg.instances:
        traj_files = [
            tf for tf in traj_files
            if any(iid in tf.stem for iid in cfg.instances)
        ]

    # Group by instance_id so we create one sandbox per instance
    by_instance: dict[str, list[Path]] = {}
    for tf in traj_files:
        with open(tf) as fh:
            import json
            meta = json.load(fh).get("metadata", {})
        iid = meta.get("instance_id", tf.stem.split("_run")[0])
        by_instance.setdefault(iid, []).append(tf)

    print(
        f"[labeler] {len(traj_files)} trajectories across {len(by_instance)} instances",
        flush=True,
    )

    for iid, files in by_instance.items():
        print(f"[labeler] loading instance {iid} ...", flush=True)
        instance = load_instance(iid)
        eval_script = instance.get("eval_script", "")

        for tf in files:
            label_path = out_dir / f"{tf.stem}_labels.json"
            if cfg.resume and label_path.exists():
                print(f"[labeler] {tf.name}: labels exist, skipping", flush=True)
                continue

            print(f"[labeler] processing {tf.name} ...", flush=True)
            label_trajectory(
                tf,
                instance=instance,
                eval_script=eval_script,
                output_path=label_path,
                modal_app_name=cfg.modal_app_name,
                sandbox_timeout=cfg.sandbox_timeout,
                eval_timeout=cfg.eval_timeout,
            )

    print("[labeler] done.", flush=True)


if __name__ == "__main__":
    main()
