"""Brief T1K1: Table 1 at k=1 for strata S0 (y_t=0) and S2 (y_t=0 and y_T=1), window 15.

Exactly the E18 / PO horizon setup (scripts/rebuttal_edit_outcome_strata.py, scripts/rebuttal_prompt_only.py), at
k=1 only (E18 and PO computed S1-S4 only for k >= 2, because S1 needs y_{t+1} = 0, which is empty at k=1):
  tokens     test split, y_{t+1} >= 0, window 15 shift caches (rebuttal_phase_confound.lookahead_cell)
  scorers    real_val (k=1 probe at its validation-best layer), k0, traj, pos_causal, pos_rel,
             prompt_hs / prompt_tfidf (PO's fit_prompt, refit on y_{t+1} at real_val's layer), and
             oracle_yT (the score y_T, defined where y_T >= 0; constant inside S2), as in E18
  strata     S0 = y_t = 0; S2 = y_t = 0 and y_T = 1 (y_T = turn_labels[probe][-1])
  bootstrap  paired task-level, 1000 resamples, seed 42: CIs of every metric and of real_val - every control
             (k0, traj, pos_causal, pos_rel, prompt_hs, prompt_tfidf) and of prompt_hs - prompt_tfidf, as in PO;
             additionally real_val - oracle_yT (an extra key; E18 reported oracle_yT without a paired difference)
  counts     n, n_pos, n_tasks, n_tasks_pos, underpowered (< 20 test tasks with a positive token)
Check (asserted): every S0 AUC, CI and count reproduces the PO k=1 row
(paper_rebuttal_candidate/prompt_only/json/horizon_<run>_<probe>.json; position baselines are refit, tolerance
1e-3, else 1e-6), and the pre-existing scorers reproduce E16 (reports/rerun/phase/lookahead_*_max15.json, yt0).

Usage: uv run python scripts/rebuttal_table1_k1.py --run laguna_verified --probe currently_correct \
           --output paper_rebuttal_candidate/table1_k1/json/k1_laguna_verified_currently_correct.json
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
from rebuttal_edit_outcome_strata import MIN_TASKS, last_turn_label  # noqa: E402
from rebuttal_persistence_analysis import Metric  # noqa: E402
from rebuttal_phase_confound import bootstrap, lookahead_cell, lookahead_context, split_masks  # noqa: E402
from rebuttal_prompt_only import (C_GRID, OLD_CONTROLS, PROMPT, RUNS, Trajs, first_token_states, fit_prompt,  # noqa: E402
                                  log, task, tfidf_matrix)

K, k = 15, 1


def tol(name: str) -> float:
    return 1e-3 if name.split("|")[0].split("-")[-1].startswith("pos_") else 1e-6


def run(args) -> dict:
    bench, pooled, prefix, k0dir = RUNS[args.run]
    probe = args.probe
    ns = argparse.Namespace(bench=bench, pooled=pooled, cache_prefix=prefix, probe=probe, max_k=K, seed=args.seed)
    ctx = lookahead_context(ns, [k], extra_label_ks=(1,))
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
    texts = json.loads(Path("tmp", f"po_texts_{bench}_{k0dir}.json").read_text())
    Xt = tfidf_matrix(trajs, texts, in_tr[trajs.first])
    log("trajectories", trajs.n, trajs.check(), "tfidf", Xt.shape)

    test_sids = sorted({sids[i] for i in ctx["test_idx"]})
    yT_of = {s: last_turn_label(Path("outputs", bench, k0dir), s, probe) for s in test_sids}
    log("y_T loaded for", len(yT_of), "test trajectories;", sum(v == 1 for v in yT_of.values()), "with y_T = 1")

    L = ctx["layers"][k]["real_val"]
    yk = ctx["ylab"][k]
    ok = yk >= 0
    p_hs, fit_hs = fit_prompt(first_token_states(c0, L, trajs), trajs.of_row, yk, in_tr & ok, in_va & ok, dense=True)
    p_tf, fit_tf = fit_prompt(Xt, trajs.of_row, yk, in_tr & ok, in_va & ok, dense=False)
    log(f"k={k} L{L} prompt_hs C={fit_hs.get('C')} val={fit_hs.get('best_val_auc')}; "
        f"tfidf C={fit_tf.get('C')} val={fit_tf.get('best_val_auc')}")

    cell = lookahead_cell(ctx, k, args.seed)
    y, yo, tsids, sc, tok = cell["y"], cell["yo"], cell["tsids"], dict(cell["sc"]), cell["tok"]
    sc.pop("real_test")
    sc["prompt_hs"], sc["prompt_tfidf"] = p_hs[trajs.of_row[tok]], p_tf[trajs.of_row[tok]]
    yT = np.array([yT_of[s] for s in tsids])
    sc["oracle_yT"] = yT.astype(np.float64)
    t0 = yo == 0
    strata = {"S0": t0, "S2": t0 & (yT == 1)}
    tasks = np.array([task(s) for s in tsids])
    counts = {}
    for st, m in strata.items():
        pos = m & (y == 1)
        counts[st] = {"n": int(m.sum()), "n_pos": int(pos.sum()), "n_tasks": int(len(set(tasks[m]))),
                      "n_tasks_pos": int(len(set(tasks[pos]))), "underpowered": bool(len(set(tasks[pos])) < MIN_TASKS)}
    metrics = {}
    for st, m in strata.items():
        for s, v in sc.items():
            metrics[f"{s}|{st}"] = Metric(v, y, m & (yT >= 0) if s == "oracle_yT" else m)
    diffs = {f"real_val-{o}|{st}": (f"real_val|{st}", f"{o}|{st}")
             for o in OLD_CONTROLS + PROMPT + ["oracle_yT"] if o in sc for st in strata}
    diffs.update({f"prompt_hs-prompt_tfidf|{st}": (f"prompt_hs|{st}", f"prompt_tfidf|{st}") for st in strata})
    point, cis = bootstrap(metrics, diffs, tsids, args.n_boot, args.seed)

    # Check S0 against PO (k=1 row: every AUC, CI and count) and E16.
    po = {r["k"]: r for r in json.loads(Path("paper_rebuttal_candidate/prompt_only/json",
                                             f"horizon_{args.run}_{probe}.json").read_text())["rows"]}[k]
    e16 = {r["k"]: r for r in json.loads(Path("reports/rerun/phase",
                                              f"lookahead_{bench}_{pooled}_{probe}_max{K}.json").read_text())["rows"]}[k]
    if po["layer"] != L or po["counts"]["S0"] != counts["S0"]:
        raise ValueError(f"layer / S0 counts differ from PO: L{L} vs L{po['layer']}, {counts['S0']} vs {po['counts']['S0']}")
    checks = {}
    for key, ref in po["auc"].items():
        if not key.endswith("|S0"):
            continue
        d = abs(point[key] - ref) if ref == ref else (0.0 if point[key] != point[key] else np.inf)
        dc = max(abs(a - b) for a, b in zip(cis[key], po["ci95"][key])) if ref == ref else 0.0
        checks[key] = max(d, dc)
        if checks[key] > tol(key):
            raise ValueError(f"{key}: AUC {point[key]:.6f} CI {cis[key]} != PO {ref:.6f} CI {po['ci95'][key]}")
    if len(checks) != sum(kk.endswith("|S0") for kk in point) - 2:  # oracle_yT|S0 and real_val-oracle_yT|S0 are new
        raise ValueError(f"S0 keys differ from PO: {sorted(set(kk for kk in point if kk.endswith('|S0')) - set(checks))}")
    for s in ("real_val",) + tuple(OLD_CONTROLS):
        ref = e16["auc"].get(f"{s}|yt0")
        if ref is not None and abs(point[f"{s}|S0"] - ref) > tol(s):
            raise ValueError(f"{s}|S0: {point[f'{s}|S0']:.6f} != E16 {ref:.6f}")
    log("S0 reproduces PO k=1 (max abs diff", max(checks.values()), ") and E16")
    log(json.dumps(counts), json.dumps({n: round(v, 3) for n, v in point.items() if v == v and "-" not in n}))
    row = {"k": k, "layer": L, "layers": ctx["layers"][k], "counts": counts, "fit": {"prompt_hs": fit_hs, "prompt_tfidf": fit_tf},
           "n_yT_missing_in_S0": int((strata["S0"] & (yT < 0)).sum()),
           "max_abs_diff_S0_vs_po": max(checks.values()), "auc": point, "ci95": cis}
    return {"mode": "table1_k1", "run": args.run, "bench": bench, "pooled": pooled, "probe": probe, "max_k": K,
            "n_boot": args.n_boot, "min_tasks": MIN_TASKS, "C_grid": C_GRID.tolist(), "cache_check": trajs.check(),
            "rows": [row]}


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
