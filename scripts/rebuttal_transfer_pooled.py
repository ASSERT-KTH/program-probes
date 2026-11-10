"""Brief T-POOL: cross-benchmark transfer scored as one pooled AUC over the target's test tokens.

Eval-only. One job per (model, direction, probe). For every layer L in 0, 10, 20, 30, 39:
  transfer  the SOURCE benchmark's pooled k=0 probe at layer L (results/<src>/<src_pooled>/<probe>/weights.pt)
  indist    the TARGET benchmark's pooled k=0 probe at layer L (results/<tgt>/<tgt_pooled>/<probe>/weights.pt)
both applied to the TARGET cache's layer-L hidden states, exactly as src/probe.py::run_eval does:
the hidden state is centred on the probe's own training mean (`mean` stored with its weights: the
SOURCE training mean for the transfer probe; no std scaling), then the linear probe's softmax.
Test tokens: the target cache's task-split test set (_split_groups(group_ids, 42)) with y >= 0.

Per layer:
  pooled AUC of both probes; the in-dist one must equal the stored <tgt_pooled> results.pt test_auc
  and its n_test the number of test tokens (token-set check);
  binned transfer AUC recomputed as run_eval does (10 relative-position bins, a bin is skipped if it
  has no val or no test tokens), summarised as the n_test-weighted mean over bins with both classes
  in test (paper_rebuttal's _weighted_mean); it must equal the stored transfer results.pt summary,
  and the stored bins must cover the same test tokens;
  delta = transfer - indist (pooled), 95% CI from 1000 task-level bootstrap resamples (paired).
Also the delta at each probe's best layer on test (the paper's convention), with its paired CI.

Usage:
    uv run python scripts/rebuttal_transfer_pooled.py --model laguna --direction v2p --probe currently_correct \
        --output paper_rebuttal_candidate/transfer/json/transfer_laguna_v2p_currently_correct.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuttal_persistence_analysis import Metric  # noqa: E402
from rebuttal_phase_confound import bootstrap, load_probe, log, score_layer, split_masks  # noqa: E402

LAYERS = [0, 10, 20, 30, 39]
N_BINS = 10
MODELS = {"laguna": {"swebench": ("laguna_xs2_full", "laguna_xs2_full_pooled", "laguna_xs2_full_pro_transfer"),
                     "swebench_pro": ("laguna_xs2_full", "laguna_xs2_pro_full_pooled", "laguna_xs2_full_verified_transfer")},
          "qwen": {"swebench": ("qwen36_35b_a3b_full", "qwen36_35b_a3b_full_pooled", "qwen36_35b_a3b_full_pro_transfer"),
                   "swebench_pro": ("qwen36_35b_a3b_full", "qwen36_35b_a3b_pro_full_pooled", "qwen36_35b_a3b_full_verified_transfer")}}
DIRECTIONS = {"v2p": ("swebench", "swebench_pro"), "p2v": ("swebench_pro", "swebench")}


def binavg(rows, drop_single: bool = True) -> tuple[float, int]:
    num = den = n_all = 0
    for auc, n, npos in rows:
        n_all += n
        if drop_single and npos in (0, n):
            continue
        num += auc * n
        den += n
    return (num / den if den else float("nan")), n_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--direction", choices=list(DIRECTIONS), required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    src, tgt = DIRECTIONS[args.direction]
    src_pooled = MODELS[args.model][src][1]
    tgt_cache, tgt_pooled, transfer_run = MODELS[args.model][tgt]
    rr = Path("results")
    cdir = Path("cache", tgt, tgt_cache, args.probe)

    meta = torch.load(cdir / "layer_0.pt", weights_only=False, mmap=True)
    sids, gids = meta["sample_id"], meta["group_id"]
    y_all, rel = meta["y"].numpy().copy(), meta["rel_pos"].numpy().copy()
    del meta
    _, in_va, in_te = split_masks(gids, args.seed)
    valid = y_all >= 0
    test_idx = np.flatnonzero(in_te & valid)
    y = y_all[test_idx]
    tsids = [sids[i] for i in test_idx]
    tasks_split = {re.sub(r"_run\d+$", "", s) for s, t in zip(sids, in_te) if t}
    tasks_valid = {re.sub(r"_run\d+$", "", s) for s in tsids}
    bins_all = np.minimum(np.floor(rel * N_BINS).astype(int), N_BINS - 1)
    bins_te = bins_all[test_idx]
    val_bins = set(np.unique(bins_all[in_va & valid]).tolist())
    log("target", tgt, tgt_cache, "test tokens", len(test_idx), "test tasks", len(tasks_valid))

    stored_in = torch.load(rr / tgt / tgt_pooled / args.probe / "results.pt", weights_only=False)
    stored_tr = torch.load(rr / tgt / transfer_run / args.probe / "results.pt", weights_only=False)

    scores: dict[str, np.ndarray] = {}
    per_layer = {}
    for L in LAYERS:
        H = torch.load(cdir / f"layer_{L}.pt", weights_only=False, mmap=True)["H"]
        sc = score_layer(H, test_idx, {
            "transfer": load_probe(rr / src / src_pooled / args.probe / "weights.pt", L),
            "indist": load_probe(rr / tgt / tgt_pooled / args.probe / "weights.pt", L)})
        del H
        scores[f"transfer|{L}"], scores[f"indist|{L}"] = sc["transfer"], sc["indist"]
        full = np.ones(len(y), dtype=bool)
        ones = np.ones(len(y))
        pooled_in = Metric(sc["indist"], y, full)(ones)
        pooled_tr = Metric(sc["transfer"], y, full)(ones)
        # Token-set and stored-value checks.
        s_in = stored_in[L][0]
        if s_in.n_test != len(y) or abs(s_in.test_auc - pooled_in) > 2e-3:
            raise ValueError(f"L{L}: in-dist n/AUC {len(y)}/{pooled_in:.4f} != stored {s_in.n_test}/{s_in.test_auc:.4f}")
        rows = []
        for b in range(N_BINS):
            m = bins_te == b
            if not m.any() or b not in val_bins:
                continue
            ab = Metric(sc["transfer"], y, m)(ones) if len(np.unique(y[m])) == 2 else 0.5
            rows.append((ab, int(m.sum()), int((y[m] == 1).sum())))
        binned_tr, n_binned = binavg(rows)
        st_rows = [(r.test_auc, r.n_test, r.n_pos_test) for r in stored_tr[L]]
        stored_binned, n_stored = binavg(st_rows)
        if n_binned != len(y) or n_stored != len(y):
            raise ValueError(f"L{L}: binned token count {n_binned} / stored {n_stored} != pooled {len(y)}")
        if abs(binned_tr - stored_binned) > 2e-3:
            raise ValueError(f"L{L}: recomputed binned {binned_tr:.4f} != stored {stored_binned:.4f}")
        per_layer[L] = {"indist_pooled": float(pooled_in), "transfer_pooled": float(pooled_tr),
                        "transfer_binned": float(stored_binned),
                        "transfer_binned_submitted_mean": binavg(st_rows, drop_single=False)[0],
                        "n_bins": len(st_rows), "n_single_class_bins": sum(1 for a, n, p in st_rows if p in (0, n)),
                        "n_test": int(len(y)), "n_pos_test": int((y == 1).sum())}
        log(f"L{L}", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in per_layer[L].items()}))

    best_in = max(LAYERS, key=lambda L: per_layer[L]["indist_pooled"])
    best_tr = max(LAYERS, key=lambda L: per_layer[L]["transfer_pooled"])
    best_tr_binned = max(LAYERS, key=lambda L: per_layer[L]["transfer_binned"])
    full = np.ones(len(y), dtype=bool)
    metrics = {k: Metric(v, y, full) for k, v in scores.items()}
    diffs = {f"delta|{L}": (f"transfer|{L}", f"indist|{L}") for L in LAYERS}
    diffs["delta|best"] = (f"transfer|{best_tr}", f"indist|{best_in}")
    point, cis = bootstrap(metrics, diffs, tsids, args.n_boot, args.seed)
    for L in LAYERS:
        per_layer[L].update({"delta_pooled": point[f"delta|{L}"], "delta_pooled_ci95": cis[f"delta|{L}"],
                             "transfer_pooled_ci95": cis[f"transfer|{L}"], "indist_pooled_ci95": cis[f"indist|{L}"]})
    out = {"model": args.model, "direction": args.direction, "probe": args.probe, "source": src, "target": tgt,
           "source_weights": f"results/{src}/{src_pooled}/{args.probe}/weights.pt",
           "indist_weights": f"results/{tgt}/{tgt_pooled}/{args.probe}/weights.pt",
           "target_cache": str(cdir), "stored_transfer_run": f"results/{tgt}/{transfer_run}",
           "normalization": "h - mean, mean = the probe's own training mean stored with its weights "
                            "(source training mean for the transfer probe); no std scaling; as src/probe.py::run_eval",
           "layer_pairing": "source layer L -> target layer L (same index)",
           "n_test": int(len(y)), "n_pos_test": int((y == 1).sum()),
           "n_test_tasks_in_split": len(tasks_split), "n_test_tasks_with_valid_tokens": len(tasks_valid),
           "test_tasks_without_valid_tokens": sorted(tasks_split - tasks_valid),
           "n_boot": args.n_boot,
           "layers": {str(L): v for L, v in per_layer.items()},
           "best_layer_test": {"indist_pooled": best_in, "transfer_pooled": best_tr, "transfer_binned": best_tr_binned},
           "delta_best": point["delta|best"], "delta_best_ci95": cis["delta|best"]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    log("delta_best", round(point["delta|best"], 4), cis["delta|best"], "wrote", args.output)


if __name__ == "__main__":
    main()
