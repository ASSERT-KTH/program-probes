"""Reference curves for the paper's lookahead figures (paper_rebuttal/, brief P-FIG A.6).

For one (run, probe, window K) and every k = 0..K, on the window's test tokens (task split):
  persistence_pooled   oracle persistence: the TRUE current label y_t as the score for y_{t+k}
                       (hard 0/1 score, so the AUC is the balanced accuracy); tokens with
                       y_{t+k} >= 0 and y_t >= 0; one pooled AUC over all tokens
  k0_pooled            E2: the k=0 probe of the window (<pooled>_shift0_max{K}) at its
                       validation-best layer, scored against y_{t+k}; tokens with y_{t+k} >= 0
                       (the k-trained probe's test tokens); one pooled AUC
  probe_pooled         the k-trained probe's stored results.pt test AUC per layer (single bin,
                       i.e. already pooled), copied for reference
  persistence_binavg_stored / _drop_single
                       the stored persistence_baseline.pt summary as the submitted figures drew
                       it (n_test-weighted mean over 10 relative-position bins, single-class
                       bins kept at their 0.5 placeholder), and the same with those bins dropped
No bootstrap (reference curves only). The shift caches of a window share their token set
(checked here, as in rebuttal_phase_confound.py), so the k=0 probe is scored once.

Usage:
    uv run python scripts/rebuttal_paper_lookahead_refs.py --bench swebench --pooled laguna_xs2_full_pooled \
        --cache-prefix laguna_xs2_full --probe currently_correct --max-k 15 --output reports/rerun/lookahead_refs/...json
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
from rebuttal_phase_confound import best_layer, load_probe, log, results, score_layer, split_masks  # noqa: E402


def auc(score: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
        return float("nan")
    return float(Metric(score, y, mask)(np.ones(len(y))))


def _get(r, f):
    return r[f] if isinstance(r, dict) else getattr(r, f)


def binavg(rows, drop_single: bool) -> float:
    num = den = 0.0
    for r in rows:
        if drop_single and _get(r, "n_pos_test") in (0, _get(r, "n_test")):
            continue
        num += _get(r, "test_auc") * _get(r, "n_test")
        den += _get(r, "n_test")
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--pooled", required=True)
    ap.add_argument("--cache-prefix", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--max-k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    K, probe = args.max_k, args.probe
    croot, rroot = Path("cache", args.bench), Path("results", args.bench)
    c0 = croot / f"{args.cache_prefix}_shift0_max{K}" / probe
    meta = torch.load(c0 / "layer_0.pt", weights_only=False, mmap=True)
    sids, gids, step = meta["sample_id"], meta["group_id"], meta["step_idx"].numpy().copy()
    del meta
    _, _, in_te = split_masks(gids, args.seed)
    test_idx = np.flatnonzero(in_te)
    log("tokens", len(sids), "test", len(test_idx))

    r0 = results(rroot / f"{args.pooled}_shift0_max{K}" / probe / "results.pt")
    l0 = best_layer(r0, "val_auc")
    H = torch.load(c0 / f"layer_{l0}.pt", weights_only=False, mmap=True)["H"]
    k0 = score_layer(H, test_idx, {"k0": load_probe(rroot / f"{args.pooled}_shift0_max{K}" / probe / "weights.pt", l0)})["k0"]
    del H

    rows = []
    for k in range(K + 1):
        d = torch.load(croot / f"{args.cache_prefix}_shift{k}_max{K}" / probe / "layer_0.pt", weights_only=False, mmap=True)
        if len(d["y"]) != len(sids) or not np.array_equal(d["step_idx"].numpy(), step):
            raise ValueError(f"shift{k} cache does not share the shift0 token set")
        y = d["y"].numpy()[test_idx].copy()
        yo = (d["y_original"] if "y_original" in d else d["y"]).numpy()[test_idx].copy()
        del d
        tok = y >= 0
        row = {"k": k, "n_test": int(tok.sum()), "n_pos_test": int((y[tok] == 1).sum()),
               "k0_layer": int(l0), "k0_pooled": auc(k0, y, tok),
               "persistence_pooled": auc(yo.astype(np.float64), y, tok & (yo >= 0)),
               "n_persistence": int((tok & (yo >= 0)).sum())}
        rid = f"{args.pooled}_shift{k}_max{K}"
        rk = results(rroot / rid / probe / "results.pt")
        if any(len(v) != 1 for v in rk.values() if v):
            raise ValueError(f"{rid}: expected one (pooled) result per layer")
        row["probe_pooled"] = {str(L): float(v[0].test_auc) for L, v in rk.items() if v}
        if k == 0:  # recomputed k0 AUC must match the stored one
            if abs(row["k0_pooled"] - rk[l0][0].test_auc) > 2e-3:
                raise ValueError(f"k0 recomputed {row['k0_pooled']:.4f} != stored {rk[l0][0].test_auc:.4f}")
        pb = rroot / rid / probe / "persistence_baseline.pt"
        if pb.exists():
            br = torch.load(pb, weights_only=False)
            row["persistence_binavg_stored"] = binavg(br, False)
            row["persistence_binavg_drop_single"] = binavg(br, True)
            row["persistence_n_bins"] = len(br)
            row["persistence_n_single_class_bins"] = sum(1 for r in br if _get(r, "n_pos_test") in (0, _get(r, "n_test")))
        log(f"k={k}", json.dumps({n: (round(v, 3) if isinstance(v, float) else v) for n, v in row.items()
                                  if n != "probe_pooled"}))
        rows.append(row)
    out = {"bench": args.bench, "pooled": args.pooled, "probe": probe, "max_k": K, "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    log("wrote", args.output)


if __name__ == "__main__":
    main()
