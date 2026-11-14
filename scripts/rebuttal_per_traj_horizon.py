"""Brief E12H (1nZa-W4): per-trajectory-weighted version of the Section 4.2 horizon numbers. Eval only.

One job per (run, probe), window 15, k in {1, 2, 3, 5, 10, 15}: exactly the E16 / E18 / PO / T1K1 setup
(scripts/rebuttal_prompt_only.py horizon mode, scripts/rebuttal_table1_k1.py): same test tokens (test split seed 42,
y_{t+k} >= 0), the k-step probe at its validation-best layer (real_val), k0, traj, pos_causal, pos_rel, prompt_hs
(PO's fit_prompt refit on y_{t+k} at real_val's layer) and oracle_yT (defined where y_T >= 0). prompt_tfidf is not
computed. Per-state scores were not stored and are recomputed from the stored weights / refits.
Subsets: S0 = y_t = 0; S2 = y_t = 0 and y_T = 1.
Weighting: per metric subset (the stratum; for oracle_yT the stratum and y_T >= 0), each test state gets weight
1 / (number of that subset's test states in its trajectory), so every trajectory with >= 1 such state has total
weight 1. Keys "w:<scorer>|<stratum>" (weighted) next to the unweighted "<scorer>|<stratum>".
Bootstrap: task-level, 1000 resamples, seed 42 (rebuttal_phase_confound.bootstrap; a state's resample weight is its
task count times its trajectory weight). Paired CIs of real_val - {k0, traj, pos_causal, pos_rel, prompt_hs}, both
weightings. The resamples are the same draws as in PO / E18 / T1K1.
Checks (asserted): unweighted AUCs, CIs and counts reproduce PO (S0 all k; S2 k >= 2), E18 (k >= 2, incl. oracle_yT) and T1K1 (k = 1, incl. oracle_yT),
tolerance 1e-3 for the refit position baselines, else 1e-6; the weighted point AUCs match sklearn roc_auc_score with
sample_weight.

Usage: uv run python scripts/rebuttal_per_traj_horizon.py --run laguna_verified --probe currently_correct \
           --output paper_rebuttal_candidate/per_traj_horizon/json/pth_laguna_verified_currently_correct.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuttal_edit_outcome_strata import MIN_TASKS, last_turn_label  # noqa: E402
from rebuttal_persistence_analysis import Metric, sort_blocks, weighted_auc  # noqa: E402
from rebuttal_phase_confound import bootstrap, lookahead_cell, lookahead_context, split_masks  # noqa: E402
from rebuttal_prompt_only import KS, RUNS, Trajs, first_token_states, fit_prompt, log, task  # noqa: E402

K = 15
CONTROLS = ["k0", "traj", "pos_causal", "pos_rel", "prompt_hs"]


class WMetric:
    """Weighted AUC over a subset; state i gets bootstrap weight token_w[i] * base[i]."""

    def __init__(self, scores: np.ndarray, y: np.ndarray, subset: np.ndarray, base: np.ndarray):
        self.idx = np.flatnonzero(subset)
        self.y, self.base = y[self.idx], base[self.idx]
        self.order, self.starts = sort_blocks(scores[self.idx])

    def __call__(self, token_w: np.ndarray) -> float:
        if len(self.idx) == 0:
            return float("nan")
        return weighted_auc(self.order, self.starts, self.y, token_w[self.idx] * self.base)


def traj_weights(traj: np.ndarray, subset: np.ndarray) -> np.ndarray:
    cnt = np.bincount(traj[subset], minlength=traj.max() + 1)
    w = np.zeros(len(traj))
    w[subset] = 1.0 / cnt[traj[subset]]
    return w


def tol(key: str) -> float:
    return 1e-3 if "pos_" in key.split("|")[0] else 1e-6


def check(point: dict, cis: dict, counts: dict, ref: dict, strata: list[str], what: str) -> float:
    worst = 0.0
    for st in strata:
        if ref["counts"][st] != counts[st]:
            raise ValueError(f"{what} {st}: counts {counts[st]} != {ref['counts'][st]}")
        for key, rv in ref["auc"].items():
            if not key.endswith(f"|{st}") or key not in point or rv != rv:
                continue
            d = abs(point[key] - rv)
            if ref["ci95"][key][0] is not None:
                d = max(d, *(abs(a - b) for a, b in zip(cis[key], ref["ci95"][key])))
            if d > tol(key):
                raise ValueError(f"{what} {key}: {point[key]:.6f} CI {cis[key]} != {rv:.6f} CI {ref['ci95'][key]}")
            worst = max(worst, d)
    return worst


def run(args) -> dict:
    bench, pooled, prefix, k0dir = RUNS[args.run]
    probe = args.probe
    ns = argparse.Namespace(bench=bench, pooled=pooled, cache_prefix=prefix, probe=probe, max_k=K, seed=args.seed)
    ctx = lookahead_context(ns, KS, extra_label_ks=(1,))
    c0 = Path("cache", bench, f"{prefix}_shift0_max{K}", probe)
    meta = torch.load(c0 / "layer_0.pt", weights_only=False, mmap=True)
    sids, gids, step = meta["sample_id"], meta["group_id"], meta["step_idx"].numpy().copy()
    del meta
    if list(sids) != list(ctx["sids"]):
        raise ValueError("sample ids differ from lookahead_context")
    trajs = Trajs(sids, step)
    in_tr, in_va, _ = split_masks(gids, args.seed)
    if not np.array_equal(in_tr, ctx["in_tr"]):
        raise ValueError("train split differs")
    test_sids = sorted({sids[i] for i in ctx["test_idx"]})
    yT_of = {s: last_turn_label(Path("outputs", bench, k0dir), s, probe) for s in test_sids}
    po = {r["k"]: r for r in json.loads(Path("paper_rebuttal_candidate/prompt_only/json",
                                             f"horizon_{args.run}_{probe}.json").read_text())["rows"]}
    e18 = {r["k"]: r for r in json.loads(Path("reports/rerun/phase",
                                              f"e18_{bench}_{pooled}_{probe}_max{K}.json").read_text())["rows"]}
    t1k1 = json.loads(Path("paper_rebuttal_candidate/table1_k1/json", f"k1_{args.run}_{probe}.json").read_text())["rows"][0]

    X_cache: dict[int, np.ndarray] = {}
    rows = []
    for k in KS:
        L = ctx["layers"][k]["real_val"]
        if L not in X_cache:
            X_cache[L] = first_token_states(c0, L, trajs)
        yk = ctx["ylab"][k]
        ok = yk >= 0
        p_hs, fit_hs = fit_prompt(X_cache[L], trajs.of_row, yk, in_tr & ok, in_va & ok, dense=True)
        cell = lookahead_cell(ctx, k, args.seed)
        y, yo, tsids, sc, tok = cell["y"], cell["yo"], cell["tsids"], dict(cell["sc"]), cell["tok"]
        sc.pop("real_test")
        sc["prompt_hs"] = p_hs[trajs.of_row[tok]]
        yT = np.array([yT_of[s] for s in tsids])
        sc["oracle_yT"] = yT.astype(np.float64)
        traj = trajs.of_row[tok]
        t0 = yo == 0
        strata = {"S0": t0, "S2": t0 & (yT == 1)}
        tasks = np.array([task(s) for s in tsids])
        counts = {}
        for st, m in strata.items():
            pos = m & (y == 1)
            counts[st] = {"n": int(m.sum()), "n_pos": int(pos.sum()), "n_tasks": int(len(set(tasks[m]))),
                          "n_tasks_pos": int(len(set(tasks[pos]))), "underpowered": bool(len(set(tasks[pos])) < MIN_TASKS)}
        wcounts = {}
        metrics, sk = {}, {}
        for st, m in strata.items():
            for s, v in sc.items():
                mm = m & (yT >= 0) if s == "oracle_yT" else m
                base = traj_weights(traj, mm)
                metrics[f"{s}|{st}"] = Metric(v, y, mm)
                metrics[f"w:{s}|{st}"] = WMetric(v, y, mm, base)
                if len(np.unique(y[mm])) == 2:
                    sk[f"w:{s}|{st}"] = roc_auc_score(y[mm], v[mm], sample_weight=base[mm])
            wcounts[st] = {"n_trajectories": int(len(np.unique(traj[m]))),
                           "n_trajectories_with_pos": int(len(np.unique(traj[m & (y == 1)])))}
        diffs = {}
        for st in strata:
            for o in CONTROLS:
                if o in sc:
                    diffs[f"real_val-{o}|{st}"] = (f"real_val|{st}", f"{o}|{st}")
                    diffs[f"w:real_val-{o}|{st}"] = (f"w:real_val|{st}", f"w:{o}|{st}")
        point, cis = bootstrap(metrics, diffs, tsids, args.n_boot, args.seed)
        for key, v in sk.items():
            if abs(v - point[key]) > 1e-9:
                raise ValueError(f"k={k} {key}: {point[key]} != sklearn {v}")
        worst = check(point, cis, counts, po[k], ["S0"] if k == 1 else ["S0", "S2"], f"k={k} vs PO")
        if k == 1:
            worst = max(worst, check(point, cis, counts, t1k1, ["S0", "S2"], "k=1 vs T1K1"))
        else:
            worst = max(worst, check(point, cis, counts, e18[k], ["S0", "S2"], f"k={k} vs E18"))
        log(f"k={k} L{L} reproduces PO/T1K1 (max abs diff {worst:.2g})",
            json.dumps({n: round(v, 3) for n, v in point.items() if v == v and n.split("|")[0] in ("real_val", "w:real_val")}))
        rows.append({"k": k, "layer": L, "layers": ctx["layers"][k], "counts": counts, "traj_counts": wcounts,
                     "fit": {"prompt_hs": fit_hs}, "max_abs_diff_unweighted_vs_po_t1k1": worst, "auc": point, "ci95": cis})
    return {"mode": "per_traj_horizon", "run": args.run, "bench": bench, "pooled": pooled, "probe": probe, "max_k": K,
            "n_boot": args.n_boot, "min_tasks": MIN_TASKS, "rows": rows}


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
