"""Attach probe labels to activation .pt files produced by run_extract_swebench.py.

CPU-only step: reads activation .pt files and trajectory JSON files, computes
probe labels via carry-forward, and writes new .pt files with a `labels` dict
added.  Re-run this whenever label logic changes — no GPU needed.

Example usage:
    uv run python run_attach_labels_swebench.py \
        --input-dir outputs/swebench/qwen36_27b_test \
        --traj-dir generations/swebench/qwen36_27b_test \
        --label-dir generations/swebench/qwen36_27b_test/labels \
        --output-dir outputs/swebench/qwen36_27b_test_labeled \
        --probe will_resolve currently_correct \
        --generation-config configs/generation.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.configs import GenerationConfig, load_config
from src.probes.base import EditEvent, TrajectoryContext
from src.tasks.swe_bench_extract import load_trajectories
from src.tasks.swe_bench_label_map import map_edit_labels_to_positions


def _load_probe(probe_name: str):
    if probe_name == "will_resolve":
        from src.probes.will_resolve import WillResolveProbe
        return WillResolveProbe()
    if probe_name == "will_be_correct":
        from src.probes.will_be_correct import WillBeCorrectProbe
        return WillBeCorrectProbe()
    if probe_name == "currently_compiles":
        from src.probes.currently_compiles import CurrentlyCompilesProbe
        return CurrentlyCompilesProbe()
    if probe_name == "currently_correct":
        from src.probes.currently_correct import CurrentlyCorrectProbe
        return CurrentlyCorrectProbe()
    if probe_name == "currently_has_regressions":
        from src.probes.currently_has_regressions import CurrentlyHasRegressionsProbe
        return CurrentlyHasRegressionsProbe()
    if probe_name == "currently_reduces_failing":
        from src.probes.currently_reduces_failing import CurrentlyReducesFailingProbe
        return CurrentlyReducesFailingProbe()
    raise ValueError(f"Unknown probe: {probe_name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True,
                        help="Directory of activation .pt files from run_extract_swebench.py")
    parser.add_argument("--traj-dir", required=True,
                        help="Directory of trajectory JSON files")
    parser.add_argument("--label-dir", required=True,
                        help="Directory of _labels.json files from run_labeler.py")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write labeled .pt files")
    parser.add_argument("--probe", nargs="+", required=True,
                        help="Probe name(s) to attach")
    parser.add_argument("--generation-config", required=True,
                        help="Path to generation YAML (for stride)")
    parser.add_argument("--run-id", default=None,
                        help="Optional run ID (used for logging only)")
    args = parser.parse_args()

    gen_config: GenerationConfig = load_config(args.generation_config, GenerationConfig)
    stride = gen_config.stride

    probes = [_load_probe(name) for name in args.probe]

    # Index trajectories by instance_id for fast lookup
    print(f"Loading trajectories from {args.traj_dir}...")
    all_trajs = load_trajectories(args.traj_dir)
    traj_by_id = {t.instance_id: t for t in all_trajs}
    print(f"  {len(traj_by_id)} trajectories loaded")

    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(input_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} activation files in {input_dir}")

    n_ok = 0
    n_skip = 0
    for pt_path in pt_files:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        instance_id: str = data["instance_id"]

        traj = traj_by_id.get(instance_id)
        if traj is None:
            print(f"  SKIP {pt_path.name}: no matching trajectory for {instance_id!r}")
            n_skip += 1
            continue

        # Load _labels.json
        label_path = label_dir / f"{instance_id}_labels.json"
        sorted_edits: list[dict] = []
        edit_history: list[EditEvent] = []
        if label_path.exists():
            label_data = json.loads(label_path.read_text())
            sorted_edits = sorted(label_data.get("edits", []), key=lambda e: e["cmd_idx"])
            edit_history = [
                EditEvent(
                    step_idx=e["cmd_idx"],
                    code="",
                    test_results=e.get("test_results"),
                    compiles=e.get("compiles"),
                )
                for e in sorted_edits
            ]
        else:
            print(f"  WARN {pt_path.name}: no _labels.json found at {label_path}")

        n_steps: int = data.get("n_captured_steps", 0)
        ctx = TrajectoryContext(
            sample={"outcome": traj.outcome, "instance_id": instance_id},
            generated_text="",
            edit_history=edit_history,
            n_captured_steps=n_steps,
        )

        labels: dict = {}
        for probe in probes:
            try:
                raw = probe.compute_label(ctx)
            except NotImplementedError:
                raw = [None] * len(sorted_edits) if probe.is_dynamic else None

            if probe.is_dynamic and isinstance(raw, list):
                labels[probe.name] = map_edit_labels_to_positions(
                    segments=traj.segments,
                    messages=traj.messages,
                    sorted_edits=sorted_edits,
                    extraction_mask=traj.extraction_mask,
                    stride=stride,
                    edit_labels=raw,
                )
            else:
                labels[probe.name] = raw

        out = dict(data)
        out["labels"] = labels

        out_path = out_dir / pt_path.name
        torch.save(out, out_path)
        print(f"  Saved {out_path.name}  (n_steps={n_steps}, outcome={traj.outcome})")
        n_ok += 1

    print(f"\nDone: {n_ok} labeled, {n_skip} skipped")


if __name__ == "__main__":
    main()
