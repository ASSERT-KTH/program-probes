"""Extract hidden states from SWE-bench agent trajectories for probe training.

Each trajectory is a single linearized token sequence.  An extraction_mask marks
assistant-turn tokens; hidden states are extracted at masked positions strided by
gen_config.stride.  One .pt file is written per trajectory.

Example usage:
    uv run python run_extract_swebench.py \
        --model-config configs/models/qwen36_27b.yaml \
        --generation-config configs/generation.yaml \
        --traj-dir generations/swebench/qwen36_27b_test \
        --probe will_resolve \
        --output-dir outputs/swebench \
        --shard-rank 0 --num-shards 1
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.configs import GenerationConfig, ModelConfig, load_config
from src.probes.base import TrajectoryContext
from src.tasks.swe_bench_extract import load_trajectories
from src.tasks.swe_bench_label_map import build_label_sequence


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model_adapter(adapter_name: str):
    if adapter_name == "qwen":
        from src.models.qwen import QwenAdapter
        return QwenAdapter()
    if adapter_name == "qwen35":
        from src.models.qwen35 import Qwen35Adapter
        return Qwen35Adapter()
    if adapter_name == "cwm":
        from src.models.cwm import CwmAdapter
        return CwmAdapter()
    raise ValueError(f"Unknown model adapter: {adapter_name!r}")


def _load_probe(probe_name: str):
    if probe_name == "will_resolve":
        from src.probes.will_resolve import WillResolveProbe
        return WillResolveProbe()
    if probe_name == "currently_compiles_swe":
        from src.probes.currently_compiles_swe import CurrentlyCompilesSwEProbe
        return CurrentlyCompilesSwEProbe()
    if probe_name == "currently_correct_swe":
        from src.probes.currently_correct_swe import CurrentlyCorrectSweProbe
        return CurrentlyCorrectSweProbe()
    raise ValueError(f"Unknown probe: {probe_name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--generation-config", required=True)
    parser.add_argument("--traj-dir", required=True, help="Directory of trajectory JSON files")
    parser.add_argument("--probe", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/swebench")
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--extraction-batch-size", type=int, default=None,
                        help="Override gen_config.extraction_batch_size")
    parser.add_argument("--label-dir", default=None,
                        help="Directory of _labels.json files from run_labeler.py")
    args = parser.parse_args()

    model_config: ModelConfig = load_config(args.model_config, ModelConfig)
    gen_config: GenerationConfig = load_config(args.generation_config, GenerationConfig)
    _set_seeds(gen_config.seed)

    probes = [_load_probe(name) for name in args.probe]

    print(f"Loading trajectories from {args.traj_dir}...")
    all_trajs = load_trajectories(args.traj_dir)
    trajs = all_trajs[args.shard_rank::args.num_shards]
    print(f"  {len(trajs)} trajectories (shard {args.shard_rank}/{args.num_shards})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_adapter = _load_model_adapter(model_config.adapter)
    model_adapter.load_for_extraction(model_config, gen_config)

    layer_indices = model_config.probe_layers
    stride = gen_config.stride
    batch_size = args.extraction_batch_size if args.extraction_batch_size is not None else gen_config.extraction_batch_size

    for batch_start in range(0, len(trajs), batch_size):
        batch = trajs[batch_start: batch_start + batch_size]

        pending = []
        for traj in batch:
            fname = out_dir / f"{traj.instance_id.replace('/', '_')}.pt"
            if not fname.exists():
                pending.append((traj, fname))
        if not pending:
            continue

        print(f"  Extracting batch {batch_start}–{batch_start + len(pending) - 1}...")
        for traj, fname in pending:
            try:
                per_seq_hs = model_adapter.extract_hidden_states(
                    [traj.token_ids], [0], layer_indices, stride,
                    extraction_masks=[traj.extraction_mask],
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"    OOM — skipping {fname.name} (n_tokens={len(traj.token_ids)})")
                continue

            hs = per_seq_hs[0]
            n_steps = min(len(v) for v in hs.values()) if hs else 0

            # Load per-edit labels if available and build label sequences
            label_sequences: dict[str, list] = {}
            if args.label_dir is not None:
                label_path = Path(args.label_dir) / f"{traj.instance_id}_labels.json"
                if label_path.exists():
                    label_data = json.loads(label_path.read_text())
                    for probe_key in ("currently_compiles", "currently_correct"):
                        label_sequences[f"label_sequence_{probe_key}"] = build_label_sequence(
                            segments=traj.segments,
                            messages=traj.messages,
                            edits=label_data.get("edits", []),
                            extraction_mask=traj.extraction_mask,
                            stride=stride,
                            probe=probe_key,
                        )

            ctx = TrajectoryContext(
                sample={
                    "outcome": traj.outcome,
                    "instance_id": traj.instance_id,
                    **label_sequences,
                },
                generated_text="",
                n_captured_steps=n_steps,
            )

            labels = {}
            for probe in probes:
                try:
                    labels[probe.name] = probe.compute_label(ctx)
                except NotImplementedError:
                    labels[probe.name] = [None] * n_steps if probe.is_dynamic else None

            out = {
                "activations": hs,
                "labels": labels,
                "instance_id": traj.instance_id,
                "sample_id": traj.instance_id,
                "group_id": traj.instance_id,
                "outcome": traj.outcome,
                "n_captured_steps": n_steps,
                "n_tokens": len(traj.token_ids),
                "n_turns": len(traj.step_segment_indices),
            }
            torch.save(out, fname)
            print(f"    Saved {fname.name}  (n_steps={n_steps}, outcome={traj.outcome})")


if __name__ == "__main__":
    main()
