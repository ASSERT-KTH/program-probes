"""Brief PO: prompt-only baseline for the horizon (E16/E18 setup) and for k=0 (Table 6 setup).

Prompt-only predictors (one score per trajectory, given to every state of the trajectory):
  prompt_hs     logistic regression on the hidden state of the trajectory's FIRST CACHED TOKEN, i.e. the
                first extracted position of the first assistant turn (its first `content` token; stride 5
                starts at the first assistant token). Its context is the system prompt + task statement +
                the assistant header, and for Laguna also the turn's own <think> block (Qwen is encoded as
                plain text without reasoning). Same layer as the probe it is compared with.
  prompt_tfidf  logistic regression on TF-IDF of the task prompt text (first user message;
                tmp/po_texts_<bench>_<run>.json from scripts/rebuttal_prompt_texts.py).
Training: the probe's population and target (train split, y >= 0 of that cache), every token weighted by
one (aggregated exactly as per-trajectory counts of positive / negative tokens). The inverse
regularisation C is chosen on validation AUC (pooled over validation tokens, as for the probes) on
a log grid. Features of prompt_hs are standardised on the training trajectories.

horizon mode, one job per (run, probe), window 15, k in {1,2,3,5,10,15}: reuses
scripts/rebuttal_phase_confound.py (same tokens, probes, layers, position baselines, bootstrap) and the
E18 strata of scripts/rebuttal_edit_outcome_strata.py (S0 = all y_t=0; S1-S4 for k >= 2). Paired
task-bootstrap CIs (1000) of real - every control, now including prompt_hs and prompt_tfidf. Checks
S0 against E16 (k=1) / E18 (k >= 2) for every pre-existing scorer.

k0 mode, one job per (run, probe), pooled k=0 cache, all test tokens with y >= 0: the k=0 probe at its
test-best layer (Table 6) vs the prompt-only predictors at that layer; checks the probe's AUC against
results.pt.

Usage:
    uv run python scripts/rebuttal_prompt_only.py horizon --run laguna_verified --probe currently_correct --output ...
    uv run python scripts/rebuttal_prompt_only.py k0 --run qwen_pro --probe currently_compiles --output ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuttal_edit_outcome_strata import MIN_TASKS, last_turn_label  # noqa: E402
from rebuttal_persistence_analysis import Metric  # noqa: E402
from rebuttal_phase_confound import (bootstrap, load_probe, lookahead_cell, lookahead_context,  # noqa: E402
                                     score_layer, split_masks)

RUNS = {  # bench, pooled run id, shift-cache prefix, k=0 cache dir (= outputs/ and generations/ run name)
    "laguna_verified": ("swebench", "laguna_xs2_full_pooled", "laguna_xs2_full", "laguna_xs2_full"),
    "qwen_verified": ("swebench", "qwen36_35b_a3b_full_pooled", "qwen36_35b_a3b_full", "qwen36_35b_a3b_full"),
    "laguna_pro": ("swebench_pro", "laguna_xs2_pro_full_pooled", "laguna_xs2_pro_full", "laguna_xs2_full"),
    "qwen_pro": ("swebench_pro", "qwen36_35b_a3b_pro_full_pooled", "qwen36_35b_a3b_pro_full", "qwen36_35b_a3b_full"),
}
KS = [1, 2, 3, 5, 10, 15]
OLD_CONTROLS = ["k0", "traj", "pos_causal", "pos_rel"]
CAUSAL_OLD = ["k0", "traj", "pos_causal"]
PROMPT = ["prompt_hs", "prompt_tfidf"]
C_GRID = np.logspace(-7, 2, 19)


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def task(sid: str) -> str:
    return re.sub(r"_run\d+$", "", sid)


class Trajs:
    """Trajectory index of every cache row and the row of each trajectory's first cached token."""

    def __init__(self, sids: list[str], step: np.ndarray):
        change = np.fromiter((i == 0 or sids[i] != sids[i - 1] for i in range(len(sids))), dtype=bool, count=len(sids))
        self.first = np.flatnonzero(change)
        self.of_row = np.cumsum(change) - 1
        self.sids = [sids[i] for i in self.first]
        if len(set(self.sids)) != len(self.sids):
            raise ValueError("trajectory rows are not contiguous in the cache")
        self.first_step = step[self.first]
        self.n = len(self.first)

    def check(self) -> dict:
        return {"n_trajectories": int(self.n), "n_first_row_not_turn0": int((self.first_step != 0).sum())}


def fit_prompt(X, traj: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, dense: bool) -> tuple[np.ndarray, dict]:
    """Per-trajectory logistic regression with token-count weights; C chosen on pooled validation AUC.

    traj, y: trajectory index and label of every cache row; tr / va: row masks (already restricted to y >= 0).
    Returns P(y=1) per trajectory and a fit summary.
    """
    n = X.shape[0]
    pos_tr = np.bincount(traj[tr & (y == 1)], minlength=n).astype(float)
    neg_tr = np.bincount(traj[tr & (y == 0)], minlength=n).astype(float)
    pos_va = np.bincount(traj[va & (y == 1)], minlength=n).astype(float)
    neg_va = np.bincount(traj[va & (y == 0)], minlength=n).astype(float)
    rp, rn = np.flatnonzero(pos_tr > 0), np.flatnonzero(neg_tr > 0)
    rows = np.concatenate([rp, rn])
    yy = np.r_[np.ones(len(rp)), np.zeros(len(rn))]
    w = np.r_[pos_tr[rp], neg_tr[rn]]
    n_tr_traj = len(np.unique(rows))
    w = w * n_tr_traj / w.sum()  # total weight = number of training trajectories, so C is on a per-trajectory scale
    if dense:
        mu, sd = X[np.unique(rows)].mean(0), X[np.unique(rows)].std(0) + 1e-6
        Z = (X - mu) / sd
    else:
        Z = X
    vp, vn = np.flatnonzero(pos_va > 0), np.flatnonzero(neg_va > 0)
    vrows, vy, vw = np.r_[vp, vn], np.r_[np.ones(len(vp)), np.zeros(len(vn))], np.r_[pos_va[vp], neg_va[vn]]
    summary = {"n_train_trajectories": int(n_tr_traj), "n_train_tokens": int(pos_tr.sum() + neg_tr.sum()),
               "train_pos_rate": float(pos_tr.sum() / max(pos_tr.sum() + neg_tr.sum(), 1)), "val_auc": {}}
    if len(rp) == 0 or len(rn) == 0 or len(vp) == 0 or len(vn) == 0:
        summary["degenerate"] = True
        return np.full(n, pos_tr.sum() / max(pos_tr.sum() + neg_tr.sum(), 1)), summary
    best = (-1.0, None, None)
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=5000)
        m.fit(Z[rows], yy, sample_weight=w)
        p = m.predict_proba(Z)[:, 1]
        a = roc_auc_score(vy, p[vrows], sample_weight=vw)
        summary["val_auc"][f"{C:.0e}"] = float(a)
        if a > best[0]:
            best = (a, C, p)
    summary.update({"C": float(best[1]), "best_val_auc": float(best[0]),
                    "C_at_grid_edge": bool(best[1] in (C_GRID[0], C_GRID[-1]))})
    return best[2], summary


def tfidf_matrix(trajs: Trajs, texts: dict, train_traj: np.ndarray):
    docs = [texts[task(s)] for s in trajs.sids]
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.9, ngram_range=(1, 2), max_features=50_000)
    vec.fit([docs[i] for i in np.flatnonzero(train_traj)])
    return vec.transform(docs)


def first_token_states(cache_dir: Path, layer: int, trajs: Trajs) -> np.ndarray:
    H = torch.load(cache_dir / f"layer_{layer}.pt", weights_only=False, mmap=True)["H"]
    X = H[torch.from_numpy(trajs.first)].float().numpy().astype(np.float64)
    del H
    return X


def run_horizon(args) -> dict:
    bench, pooled, prefix, k0dir = RUNS[args.run]
    K, probe = 15, args.probe
    ns = argparse.Namespace(bench=bench, pooled=pooled, cache_prefix=prefix, probe=probe, max_k=K, seed=args.seed)
    ctx = lookahead_context(ns, KS, extra_label_ks=(1,))
    c0 = Path("cache", bench, f"{prefix}_shift0_max{K}", probe)
    meta = torch.load(c0 / "layer_0.pt", weights_only=False, mmap=True)
    sids, gids, step = meta["sample_id"], meta["group_id"], meta["step_idx"].numpy().copy()
    del meta
    if list(sids) != list(ctx["sids"]):
        raise ValueError("sample ids differ from lookahead_context")
    trajs = Trajs(sids, step)
    in_tr, in_va, in_te = split_masks(gids, args.seed)
    if not np.array_equal(in_tr, ctx["in_tr"]):
        raise ValueError("train split differs")
    train_traj = in_tr[trajs.first]
    texts = json.loads(Path("tmp", f"po_texts_{bench}_{k0dir}.json").read_text())
    Xt = tfidf_matrix(trajs, texts, train_traj)
    log("trajectories", trajs.n, trajs.check(), "tfidf", Xt.shape)

    test_sids = sorted({sids[i] for i in ctx["test_idx"]})
    yT_of = {s: last_turn_label(Path("outputs", bench, k0dir), s, probe) for s in test_sids}
    e16 = {r["k"]: r for r in json.loads(Path("reports/rerun/phase", f"lookahead_{bench}_{pooled}_{probe}_max{K}.json").read_text())["rows"]}
    e18 = {r["k"]: r for r in json.loads(Path("reports/rerun/phase", f"e18_{bench}_{pooled}_{probe}_max{K}.json").read_text())["rows"]}

    X_cache: dict[int, np.ndarray] = {}
    y1_all = ctx["ylab"][1]
    rows = []
    for k in KS:
        L = ctx["layers"][k]["real_val"]
        if L not in X_cache:
            X_cache[L] = first_token_states(c0, L, trajs)
        yk = ctx["ylab"][k]
        ok = yk >= 0
        p_hs, fit_hs = fit_prompt(X_cache[L], trajs.of_row, yk, in_tr & ok, in_va & ok, dense=True)
        p_tf, fit_tf = fit_prompt(Xt, trajs.of_row, yk, in_tr & ok, in_va & ok, dense=False)
        log(f"k={k} L{L} prompt_hs C={fit_hs.get('C')} val={fit_hs.get('best_val_auc')}; "
            f"tfidf C={fit_tf.get('C')} val={fit_tf.get('best_val_auc')}")

        cell = lookahead_cell(ctx, k, args.seed)
        y, yo, tsids, sc, tok = cell["y"], cell["yo"], cell["tsids"], dict(cell["sc"]), cell["tok"]
        sc.pop("real_test")
        sc["prompt_hs"], sc["prompt_tfidf"] = p_hs[trajs.of_row[tok]], p_tf[trajs.of_row[tok]]
        t0 = yo == 0
        strata = {"S0": t0}
        if k >= 2:
            y1 = y1_all[tok]
            yT = np.array([yT_of[s] for s in tsids])
            strata.update({"S1": t0 & (y1 == 0), "S2": t0 & (yT == 1), "S3": t0 & (y1 == 0) & (yT == 1),
                           "S4": t0 & (yT == 0)})
        tasks = np.array([task(s) for s in tsids])
        counts = {}
        for st, m in strata.items():
            pos = m & (y == 1)
            counts[st] = {"n": int(m.sum()), "n_pos": int(pos.sum()), "n_tasks": int(len(set(tasks[m]))),
                          "n_tasks_pos": int(len(set(tasks[pos]))), "underpowered": bool(len(set(tasks[pos])) < MIN_TASKS)}
        metrics = {f"{s}|{st}": Metric(v, y, m) for s, v in sc.items() for st, m in strata.items()}
        diffs = {f"real_val-{o}|{st}": (f"real_val|{st}", f"{o}|{st}")
                 for o in OLD_CONTROLS + PROMPT if o in sc for st in strata}
        diffs.update({f"prompt_hs-prompt_tfidf|{st}": (f"prompt_hs|{st}", f"prompt_tfidf|{st}") for st in strata})
        point, cis = bootstrap(metrics, diffs, tsids, args.n_boot, args.seed)

        # Reproduce E16 (k=1) / E18 (k >= 2) for every pre-existing scorer.
        checks = {}
        if k in e18:
            for st in strata:
                for s in ("real_val",) + tuple(OLD_CONTROLS):
                    ref = e18[k]["auc"].get(f"{s}|{st}")
                    if ref is None or ref != ref:
                        continue
                    tol = 1e-3 if s.startswith("pos_") else 1e-6
                    d = abs(point[f"{s}|{st}"] - ref)
                    checks[f"{s}|{st}"] = d
                    if d > tol:
                        raise ValueError(f"k={k} {s}|{st}: {point[f'{s}|{st}']:.6f} != E18 {ref:.6f}")
                if e18[k]["counts"][st] != counts[st]:
                    raise ValueError(f"k={k} {st}: counts differ from E18")
        else:
            for s in ("real_val",) + tuple(OLD_CONTROLS):
                ref = e16[k]["auc"].get(f"{s}|yt0")
                if ref is None:
                    continue
                tol = 1e-3 if s.startswith("pos_") else 1e-6
                d = abs(point[f"{s}|S0"] - ref)
                checks[f"{s}|S0"] = d
                if d > tol:
                    raise ValueError(f"k={k} {s}|S0: {point[f'{s}|S0']:.6f} != E16 {ref:.6f}")
        rows.append({"k": k, "layer": L, "layers": ctx["layers"][k], "counts": counts,
                     "fit": {"prompt_hs": fit_hs, "prompt_tfidf": fit_tf},
                     "max_abs_diff_vs_e16_e18": max(checks.values()) if checks else None,
                     "auc": point, "ci95": cis})
        log(f"k={k}", json.dumps({n: round(v, 3) for n, v in point.items()
                                  if v == v and n.split("|")[0] in ("real_val", "prompt_hs", "prompt_tfidf")}))
    return {"mode": "horizon", "run": args.run, "bench": bench, "pooled": pooled, "probe": probe, "max_k": K,
            "n_boot": args.n_boot, "min_tasks": MIN_TASKS, "C_grid": C_GRID.tolist(), "cache_check": trajs.check(),
            "rows": rows}


def run_k0(args) -> dict:
    bench, pooled, _, k0dir = RUNS[args.run]
    probe = args.probe
    cdir = Path("cache", bench, k0dir, probe)
    meta = torch.load(cdir / "layer_0.pt", weights_only=False, mmap=True)
    sids, gids = meta["sample_id"], meta["group_id"]
    y = meta["y"].numpy().copy()
    step = meta["step_idx"].numpy().copy()
    del meta
    trajs = Trajs(sids, step)
    in_tr, in_va, in_te = split_masks(gids, args.seed)
    ok = y >= 0
    test_idx = np.flatnonzero(in_te & ok)
    e17 = json.loads(Path("reports/rerun/phase", f"k0_{bench}_{pooled}_{probe}.json").read_text())
    L = int(e17["real_test_best_layer"])
    stored = e17["stored_test_auc"]["real"][str(L)]
    real = score_layer(torch.load(cdir / f"layer_{L}.pt", weights_only=False, mmap=True)["H"], test_idx,
                       {"real": load_probe(Path("results", bench, pooled, probe, "weights.pt"), L)})["real"]
    X = first_token_states(cdir, L, trajs)
    texts = json.loads(Path("tmp", f"po_texts_{bench}_{k0dir}.json").read_text())
    Xt = tfidf_matrix(trajs, texts, in_tr[trajs.first])
    log("trajectories", trajs.n, trajs.check(), "test tokens", len(test_idx), "layer", L)
    p_hs, fit_hs = fit_prompt(X, trajs.of_row, y, in_tr & ok, in_va & ok, dense=True)
    p_tf, fit_tf = fit_prompt(Xt, trajs.of_row, y, in_tr & ok, in_va & ok, dense=False)
    yt = y[test_idx]
    sc = {"real_testbest": real, "prompt_hs": p_hs[trajs.of_row[test_idx]], "prompt_tfidf": p_tf[trajs.of_row[test_idx]]}
    full = np.ones(len(yt), dtype=bool)
    metrics = {n: Metric(v, yt, full) for n, v in sc.items()}
    diffs = {"real-prompt_hs": ("real_testbest", "prompt_hs"), "real-prompt_tfidf": ("real_testbest", "prompt_tfidf"),
             "prompt_hs-prompt_tfidf": ("prompt_hs", "prompt_tfidf")}
    point, cis = bootstrap(metrics, diffs, [sids[i] for i in test_idx], args.n_boot, args.seed)
    if abs(point["real_testbest"] - stored) > 2e-4:
        raise ValueError(f"recomputed AUC {point['real_testbest']:.5f} != results.pt {stored:.5f}")
    tasks = {task(sids[i]) for i in test_idx}
    log(json.dumps({n: round(v, 3) for n, v in point.items()}))
    return {"mode": "k0", "run": args.run, "bench": bench, "pooled": pooled, "probe": probe, "layer": L,
            "stored_test_auc": stored, "n_boot": args.n_boot, "C_grid": C_GRID.tolist(),
            "cache_check": trajs.check(), "n_test": int(len(yt)), "n_pos_test": int((yt == 1).sum()),
            "n_test_tasks": len(tasks), "n_test_trajectories": int(len({sids[i] for i in test_idx})),
            "fit": {"prompt_hs": fit_hs, "prompt_tfidf": fit_tf}, "auc": point, "ci95": cis}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["horizon", "k0"])
    ap.add_argument("--run", required=True, choices=sorted(RUNS))
    ap.add_argument("--probe", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = run_horizon(args) if args.mode == "horizon" else run_k0(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    log("wrote", args.output)
    os._exit(0)


if __name__ == "__main__":
    main()
