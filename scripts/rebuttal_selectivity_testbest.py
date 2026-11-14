"""Request SEL-CI: k=0 selectivity at the best TEST layers (the convention of Table 5 / Figure 3).

Eval-only. For one (run, probe) on the pooled k=0 cache (task split, test tokens with y >= 0):
  real   <pooled> at its best layer on test
  traj   <pooled>_shuffled_traj (trajectory shuffle, aligned by relative position) at ITS best test layer
  token  <pooled>_shuffled (token shuffle) at ITS best test layer
Best layers come from the stored results.pt test AUCs (one pooled bin per layer). The recomputed AUCs
must reproduce the stored ones, i.e. paper_rebuttal/figures/auc_table.tex (asserted, 1e-3).
CIs: 1000 task-level bootstrap resamples (multinomial task weights), paired on the same test tokens,
for real - traj and real - token (scripts/rebuttal_phase_confound.py::bootstrap).

Usage:
    uv run python scripts/rebuttal_selectivity_testbest.py --bench swebench --pooled laguna_xs2_full_pooled \
        --cache-prefix laguna_xs2_full --probe currently_correct --output paper_rebuttal_candidate/selectivity/...json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuttal_persistence_analysis import Metric  # noqa: E402
from rebuttal_phase_confound import bootstrap, load_probe, log, results, score_layer, split_masks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--pooled", required=True)
    ap.add_argument("--cache-prefix", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rroot = Path("results", args.bench)
    cdir = Path("cache", args.bench, args.cache_prefix, args.probe)
    meta = torch.load(cdir / "layer_0.pt", weights_only=False, mmap=True)
    sids, gids, y_all = meta["sample_id"], meta["group_id"], meta["y"].numpy().copy()
    del meta
    _, _, in_te = split_masks(gids, args.seed)
    test_idx = np.flatnonzero(in_te & (y_all >= 0))
    y = y_all[test_idx]

    runs = {"real": args.pooled, "traj": f"{args.pooled}_shuffled_traj", "token": f"{args.pooled}_shuffled"}
    best, stored = {}, {}
    for name, rid in runs.items():
        res = results(rroot / rid / args.probe / "results.pt")
        if any(len(v) != 1 for v in res.values() if v):
            raise ValueError(f"{rid}: expected one pooled bin per layer")
        stored[name] = {int(L): float(v[0].test_auc) for L, v in res.items() if v}
        best[name] = max(stored[name], key=stored[name].get)
        if res[best[name]][0].n_test != len(y):
            raise ValueError(f"{rid}: stored n_test {res[best[name]][0].n_test} != {len(y)} test tokens")
    log("tokens", len(y), "best test layers", best)

    sc: dict[str, np.ndarray] = {}
    by_layer: dict[int, dict] = {}
    for name, L in best.items():
        by_layer.setdefault(L, {})[name] = load_probe(rroot / runs[name] / args.probe / "weights.pt", L)
    for L, probes in sorted(by_layer.items()):
        H = torch.load(cdir / f"layer_{L}.pt", weights_only=False, mmap=True)["H"]
        sc.update(score_layer(H, test_idx, probes))
        del H
    full = np.ones(len(y), dtype=bool)
    metrics = {n: Metric(v, y, full) for n, v in sc.items()}
    point, cis = bootstrap(metrics, {"real-traj": ("real", "traj"), "real-token": ("real", "token")},
                           [sids[i] for i in test_idx], args.n_boot, args.seed)
    for n in runs:
        if abs(point[n] - stored[n][best[n]]) > 1e-3:
            raise ValueError(f"{n}: recomputed {point[n]:.4f} != stored {stored[n][best[n]]:.4f}")
    out = {"bench": args.bench, "pooled": args.pooled, "probe": args.probe, "runs": runs,
           "best_test_layer": best, "best_test_layer_1indexed": {n: L + 1 for n, L in best.items()},
           "stored_test_auc_by_layer": {n: {str(L): v for L, v in d.items()} for n, d in stored.items()},
           "n_test": int(len(y)), "n_pos_test": int((y == 1).sum()), "n_boot": args.n_boot,
           "auc": point, "ci95": cis}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    log(json.dumps({k: round(v, 4) for k, v in point.items()}), cis["real-traj"], cis["real-token"])


if __name__ == "__main__":
    main()
