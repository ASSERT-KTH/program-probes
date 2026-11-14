"""Brief E12 (1nZa-W4): per-trajectory and per-version AUC of the k=0 probes. Eval only, no training.

Cells: the 16 k=0 cells of the AUC table (Table 6 here, Table 5 in the paper): 4 runs x 4 properties, the rerun's
k=0 pooled probe (results/<bench>/<pooled>/<probe>/weights.pt) at its best TEST layer (E17 real_test_best_layer),
on that table's test tokens: pooled k=0 cache (cache/<bench>/<run>/<probe>/), task-level test split (seed 42),
y >= 0. Probe scores = P(y=1) from the stored weights (per-token scores were not stored; recomputed exactly as in
E17 / PO k0 mode).
Metrics:
  pooled      AUC over test tokens; must equal the stored test AUC (asserted, |diff| <= 2e-4)
  per_traj    weighted AUC, token weight 1 / (number of test tokens in its trajectory), so every trajectory has
              the same total weight
  per_version one point per program version: a version is the run of turns between two edits
              (edit_step_index.pt: sid -> [N_e + 1], the first turn whose label reflects edit e; turn t belongs
              to version #{starts <= t}; the original repository before the first edit is version 0).
              Score = mean probe score P(y=1) over the version's test tokens; label = the version's k=0 label.
              Versions whose test-token labels are not constant are reported and excluded.
CIs: 95% task-level bootstrap (1000 resamples, multinomial task weights, seed 42; rebuttal_phase_confound.bootstrap),
for each metric and paired for (metric - pooled). A version's bootstrap weight is its task's weight.

Usage: uv run python scripts/rebuttal_per_unit_auc.py --run laguna_verified --probe currently_correct \
           --output paper_rebuttal_candidate/per_unit_auc/json/pu_laguna_verified_currently_correct.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuttal_persistence_analysis import sort_blocks, weighted_auc  # noqa: E402
from rebuttal_phase_confound import bootstrap, load_probe, score_layer, split_masks  # noqa: E402
from rebuttal_prompt_only import RUNS, Trajs, log, task  # noqa: E402


class WMetric:
    """Weighted AUC over n points; point i gets bootstrap weight token_w[rep[i]] * base[i]."""

    def __init__(self, scores: np.ndarray, y: np.ndarray, rep: np.ndarray, base: np.ndarray):
        self.y, self.rep, self.base = y, rep, base
        self.order, self.starts = sort_blocks(scores)

    def __call__(self, token_w: np.ndarray) -> float:
        return weighted_auc(self.order, self.starts, self.y, token_w[self.rep] * self.base)


def run(args) -> dict:
    bench, pooled, _, k0dir = RUNS[args.run]
    probe = args.probe
    cdir = Path("cache", bench, k0dir, probe)
    meta = torch.load(cdir / "layer_0.pt", weights_only=False, mmap=True)
    sids, gids = meta["sample_id"], meta["group_id"]
    y = meta["y"].numpy().copy()
    step = meta["step_idx"].numpy().astype(np.int64)
    del meta
    trajs = Trajs(sids, step)
    _, _, in_te = split_masks(gids, args.seed)
    test_idx = np.flatnonzero(in_te & (y >= 0))
    e17 = json.loads(Path("reports/rerun/phase", f"k0_{bench}_{pooled}_{probe}.json").read_text())
    L = int(e17["real_test_best_layer"])
    stored = e17["stored_test_auc"]["real"][str(L)]
    s = score_layer(torch.load(cdir / f"layer_{L}.pt", weights_only=False, mmap=True)["H"], test_idx,
                    {"real": load_probe(Path("results", bench, pooled, probe, "weights.pt"), L)})["real"]
    yt = y[test_idx].astype(np.int64)
    tsids = [sids[i] for i in test_idx]
    traj = trajs.of_row[test_idx]
    n = len(test_idx)
    log(f"{args.run} {probe} L{L}: {n} test tokens, {len(set(traj))} trajectories")

    # Per-trajectory weights.
    _, tinv, tcnt = np.unique(traj, return_inverse=True, return_counts=True)
    w_traj = 1.0 / tcnt[tinv]

    # Versions.
    index = torch.load(Path("cache", bench, k0dir, "edit_step_index.pt"), weights_only=False)
    vid = np.empty(n, np.int64)
    if np.any(np.diff(traj) < 0):
        raise ValueError("test tokens are not grouped by trajectory")
    uj, jfirst = np.unique(traj, return_index=True)
    for j, a, b in zip(uj, jfirst, np.r_[jfirst[1:], n]):
        starts = np.array(sorted(index.get(trajs.sids[j], [])), dtype=np.int64)
        vid[a:b] = np.searchsorted(starts, step[test_idx[a:b]], side="right")
    key = traj * 1_000_000 + vid
    ukey, vinv, vcnt = np.unique(key, return_inverse=True, return_counts=True)
    nv = len(ukey)
    vsum = np.bincount(vinv, weights=s, minlength=nv)
    vpos = np.bincount(vinv, weights=yt, minlength=nv)
    v_score = vsum / vcnt
    const = (vpos == 0) | (vpos == vcnt)
    v_label = (vpos == vcnt).astype(np.int64)
    rep = np.empty(nv, np.int64)
    rep[vinv] = np.arange(n)  # any token of the version (same task)
    violations = [{"sample_id": trajs.sids[int(k // 1_000_000)], "version": int(k % 1_000_000), "n_tokens": int(c),
                   "n_pos": int(p)} for k, c, p, ok in zip(ukey, vcnt, vpos, const) if not ok]
    keep = np.flatnonzero(const)
    log(f"versions {nv}, non-constant label {len(violations)}")

    ones_n = np.ones(n)
    metrics = {"pooled": WMetric(s, yt, np.arange(n), ones_n),
               "per_traj": WMetric(s, yt, np.arange(n), w_traj),
               "per_version": WMetric(v_score[keep], v_label[keep], rep[keep], np.ones(len(keep)))}
    diffs = {"per_traj-pooled": ("per_traj", "pooled"), "per_version-pooled": ("per_version", "pooled")}
    point, cis = bootstrap(metrics, diffs, tsids, args.n_boot, args.seed)
    if abs(point["pooled"] - stored) > 2e-4:
        raise ValueError(f"pooled AUC {point['pooled']:.5f} != stored test AUC {stored:.5f}")
    # Cross-check the weighted AUCs with sklearn.
    from sklearn.metrics import roc_auc_score
    sk = {"pooled": roc_auc_score(yt, s), "per_traj": roc_auc_score(yt, s, sample_weight=w_traj),
          "per_version": roc_auc_score(v_label[keep], v_score[keep])}
    for m, v in sk.items():
        if abs(v - point[m]) > 1e-9:
            raise ValueError(f"{m}: {point[m]} != sklearn {v}")

    nvk = len(keep)
    vpt = np.unique(ukey // 1_000_000, return_counts=True)[1]
    tasks = {task(x) for x in tsids}
    counts = {"n_test_tokens": int(n), "n_pos_tokens": int(yt.sum()), "n_test_tasks": len(tasks),
              "n_test_trajectories": int(len(tcnt)), "n_test_versions": int(nv),
              "n_versions_label_not_constant": len(violations), "n_test_versions_used": int(nvk),
              "share_versions_label1": float(v_label[keep].mean()) if nvk else None,
              "share_versions_label0": float(1 - v_label[keep].mean()) if nvk else None,
              "n_versions_label1": int(v_label[keep].sum()), "n_versions_label0": int(nvk - v_label[keep].sum()),
              "tokens_per_trajectory": {"median": float(np.median(tcnt)), "max": int(tcnt.max())},
              "tokens_per_version": {"median": float(np.median(vcnt)), "max": int(vcnt.max())},
              "versions_per_trajectory": {"median": float(np.median(vpt)), "max": int(vpt.max())}}
    log(json.dumps({m: round(v, 4) for m, v in point.items()}), json.dumps(counts))
    return {"mode": "per_unit_auc", "run": args.run, "bench": bench, "pooled": pooled, "probe": probe, "layer": L,
            "stored_test_auc": stored, "n_boot": args.n_boot, "score": "P(y=1)", "counts": counts,
            "auc": point, "ci95": cis, "label_violations": violations[:50]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=sorted(RUNS))
    ap.add_argument("--probe", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = run(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    log("wrote", args.output)
    os._exit(0)


if __name__ == "__main__":
    main()
