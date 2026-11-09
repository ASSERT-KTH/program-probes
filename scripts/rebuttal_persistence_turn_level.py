"""Turn-level view of the persistence baseline at k=1 (test split).

Token-level persistence AUC weights each turn by its number of extracted tokens.
This script collapses the k=1 shift cache to one row per (trajectory, source turn)
and reports, for each probe and window:
  - number of test turns, positive turns at t and at t+1, flipped turns
  - mean tokens per turn: all turns, flipped turns, positive (y_t = 1) turns
  - turn-level persistence TPR, TNR and balanced accuracy

Usage:
    uv run python scripts/rebuttal_persistence_turn_level.py --bench swebench_pro \
        --prefix laguna_xs2_pro_full --output reports/rerun/persistence_turn_level_laguna_xs2_pro_full.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.probe import _split_groups

PROBES = ["currently_compiles", "currently_correct", "currently_has_regressions", "currently_reduces_failing"]


def turn_level(path: Path) -> dict:
    d = torch.load(path, weights_only=False, mmap=True)
    y, yo, st = d["y"].numpy(), d["y_original"].numpy(), d["step_idx"].numpy()
    sids, gids = d["sample_id"], d["group_id"]
    del d
    _, _, test = _split_groups(gids, 42)
    turns: dict = defaultdict(lambda: [0, 0, 0])  # (sid, step) -> [n_tokens, y_t, y_t+1]
    for s, g, a, vo, v in zip(sids, gids, st, yo, y):
        if g in test and v >= 0 and vo >= 0:
            t = turns[(s, int(a))]
            t[0] += 1
            t[1], t[2] = int(vo), int(v)
    arr = np.array(list(turns.values()))
    n, o, v = arr[:, 0], arr[:, 1], arr[:, 2]
    flip = o != v
    tpr = float(((o == 1) & (v == 1)).sum() / (v == 1).sum()) if (v == 1).any() else None
    tnr = float(((o == 0) & (v == 0)).sum() / (v == 0).sum()) if (v == 0).any() else None
    return {
        "n_turns": int(len(arr)), "n_pos_turns_t": int((o == 1).sum()), "n_pos_turns_t1": int((v == 1).sum()),
        "n_flip_turns": int(flip.sum()), "n_flip_turns_0to1": int(((o == 0) & (v == 1)).sum()),
        "n_flip_turns_1to0": int(((o == 1) & (v == 0)).sum()),
        "tokens_per_turn": float(n.mean()),
        "tokens_per_flipped_turn": float(n[flip].mean()) if flip.any() else None,
        "tokens_per_pos_turn": float(n[o == 1].mean()) if (o == 1).any() else None,
        "tpr": tpr, "tnr": tnr,
        "balanced_acc": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--prefix", required=True, help="Shift-cache prefix, e.g. laguna_xs2_pro_full")
    ap.add_argument("--max-k", type=int, nargs="+", default=[50, 15])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out: dict = {"bench": args.bench, "prefix": args.prefix, "k": 1, "split": "test", "rows": {}}
    for probe in PROBES:
        for K in args.max_k:
            r = turn_level(Path("cache", args.bench, f"{args.prefix}_shift1_max{K}", probe, "layer_0.pt"))
            out["rows"][f"{probe}_max{K}"] = r
            print(probe, K, json.dumps(r), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
