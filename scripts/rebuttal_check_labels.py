"""Validate re-attached *_labels.pt files and report real turn-length statistics.

For every activation file in --input-dir, checks that the companion
*_labels.pt has `step_idx` and `turn_labels`, and that
len(step_idx) == len(labels[probe]) == activation rows for every dynamic probe.
Turn length = number of extracted positions whose step_idx equals that turn.

Usage:
    uv run python scripts/rebuttal_check_labels.py --input-dir outputs/swebench/laguna_xs2_full \
        --probe currently_compiles currently_correct --output reports/labels_laguna.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--probe", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    pt_files = sorted(f for f in in_dir.glob("*.pt") if not f.stem.endswith("_labels"))
    errors: list[str] = []
    turn_lengths: list[int] = []
    n_turns_per_traj: list[int] = []
    n_empty_per_traj: list[int] = []
    n_files = n_empty_acts = 0

    for fi, f in enumerate(pt_files):
        if fi % 500 == 0:
            print(f"  {fi}/{len(pt_files)} {f.name}", flush=True)
        label_path = f.parent / f"{f.stem}_labels.pt"
        if not label_path.exists():
            errors.append(f"{f.name}: missing labels file")
            continue
        lab = torch.load(label_path, weights_only=False)
        data = torch.load(f, weights_only=False, mmap=True)
        T = next(iter(data["activations"].values())).shape[0]
        del data
        n_files += 1
        if T == 0:
            n_empty_acts += 1
        if "step_idx" not in lab or "turn_labels" not in lab:
            errors.append(f"{label_path.name}: missing step_idx/turn_labels")
            continue
        step_idx = list(lab["step_idx"])
        if len(step_idx) != T:
            errors.append(f"{f.name}: {T} rows but {len(step_idx)} step_idx")
        K = None
        for probe in args.probe:
            y = lab["labels"].get(probe)
            if not isinstance(y, list):
                continue
            if len(y) != T:
                errors.append(f"{f.name}: {T} rows but {len(y)} labels for {probe}")
            tl = lab["turn_labels"].get(probe)
            if tl is None:
                errors.append(f"{label_path.name}: no turn_labels for {probe}")
                continue
            if K is None:
                K = len(tl)
            elif K != len(tl):
                errors.append(f"{label_path.name}: turn_labels length differs across probes")
        if K is None or K == 0:
            continue
        if step_idx and max(step_idx) >= K:
            errors.append(f"{f.name}: step_idx max {max(step_idx)} >= n_turns {K}")
            continue
        counts = np.bincount(np.asarray(step_idx, dtype=np.int64), minlength=K)
        turn_lengths.extend(counts.tolist())
        n_turns_per_traj.append(K)
        n_empty_per_traj.append(int((counts == 0).sum()))

    tl = np.asarray(turn_lengths)
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else None  # noqa: E731
    summary = {
        "input_dir": str(in_dir),
        "n_files": n_files,
        "n_files_zero_rows": n_empty_acts,
        "n_errors": len(errors),
        "errors_head": errors[:50],
        "n_turns_total": int(len(tl)),
        "n_turns_empty": int((tl == 0).sum()),
        "frac_turns_empty": float((tl == 0).mean()) if len(tl) else None,
        "turn_len_mean": float(tl.mean()) if len(tl) else None,
        "turn_len_nonempty_mean": float(tl[tl > 0].mean()) if (tl > 0).any() else None,
        "turn_len_quantiles": {p: q(tl, p) for p in (0, 10, 25, 50, 75, 90, 99, 100)},
        "turns_per_traj_quantiles": {p: q(np.asarray(n_turns_per_traj), p) for p in (0, 25, 50, 75, 100)},
        "trajs_with_any_empty_turn": int(sum(1 for n in n_empty_per_traj if n > 0)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
