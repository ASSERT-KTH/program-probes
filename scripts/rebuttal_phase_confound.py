"""Rebuttal E2 / E16 / E17: is the lookahead signal trajectory phase / position?

Two modes. All AUCs are on test tokens; CIs are 95% percentile intervals from a
task-level bootstrap (1000 resamples, multinomial task weights), exactly as in
scripts/rebuttal_persistence_analysis.py, whose Metric/weighted AUC is reused.
Differences use the same resample for both terms (paired).

lookahead (E2, E16), one job per (run, probe, window K), k in 1..K:
  Test tokens per k: test split, y_{t+k} >= 0, i.e. the tokens of
  rebuttal_persistence_analysis.py; strata y_t=0 / y_t=1 need y_t >= 0 as there.
  Scorers (all AUCs against y_{t+k}):
    real_val   k-trained probe at its validation-best layer (primary)
    real_test  k-trained probe at its test-best layer (lines up with the existing tables)
    k0         E2: the k=0 probe of the same window (<pooled>_shift0_max{K}) at its own
               validation-best layer, i.e. the current-state probe used as a forecast
    traj       E16(ii): trajectory-shuffle probe trained on y_{t+k} with
               --shuffle-unit trajectory, at real_val's layer
               (<pooled>_shift{k}_max{K}_shuffled_traj)
    pos_causal E16(iii)(a): logistic regression on cubic-spline features of
               log1p(step_idx) and log1p(absolute position), fit on train tokens with
               target y_{t+k}; uses only information available at time t
    pos_rel    E16(iii)(b): same on rel_pos = position / trajectory length. NON-CAUSAL
               (uses the total length): a deliberately strong upper bound
  The shift caches of one window share their tokens and hidden states (checked by
  rebuttal_check_persistence_labels.py), so every probe is scored on the shift0 cache's H.

k0 (E17), one job per (run, probe), on the pooled k=0 cache (all tokens):
  real, token-shuffle (<pooled>_shuffled) and trajectory-shuffle (<pooled>_shuffled_traj)
  probes at every layer. Each run's layer is chosen on validation AUC against the REAL
  validation labels (the shuffled runs' stored val_auc is against shuffled labels), so
  each control gets its best layer. Reports each at its own val-best layer, the gaps,
  the gaps at the real probe's val-best layer, the real probe at its test-best layer,
  and both position baselines.

Usage:
    uv run python scripts/rebuttal_phase_confound.py lookahead --bench swebench_pro \
        --pooled qwen36_35b_a3b_pro_full_pooled --cache-prefix qwen36_35b_a3b_pro_full \
        --probe currently_correct --max-k 15 --output reports/rerun/phase/...json
    uv run python scripts/rebuttal_phase_confound.py k0 --bench swebench_pro \
        --pooled qwen36_35b_a3b_pro_full_pooled --cache-prefix qwen36_35b_a3b_full \
        --probe currently_correct --output reports/rerun/phase/...json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuttal_persistence_analysis import Metric, ci  # noqa: E402

from src.probe import _build_probe, _split_groups  # noqa: E402

KS = {15: [1, 2, 3, 5, 10, 15], 50: [1, 2, 3, 5, 10, 15, 25, 50]}
CHUNK = 100_000


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def abs_positions(sids: list[str]) -> np.ndarray:
    """Index of each token within its trajectory (cache rows are contiguous and ordered per trajectory)."""
    change = np.fromiter((i == 0 or sids[i] != sids[i - 1] for i in range(len(sids))), dtype=bool, count=len(sids))
    starts = np.flatnonzero(change)
    if len(starts) != len(set(sids)):
        raise ValueError("trajectory rows are not contiguous in the cache")
    return np.arange(len(sids)) - starts[np.cumsum(change) - 1]


def split_masks(group_ids: list[str], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr, va, te = _split_groups(group_ids, seed)
    n = len(group_ids)
    return (np.fromiter((g in tr for g in group_ids), dtype=bool, count=n),
            np.fromiter((g in va for g in group_ids), dtype=bool, count=n),
            np.fromiter((g in te for g in group_ids), dtype=bool, count=n))


def results(path: Path) -> dict:
    return torch.load(path, weights_only=False)


def best_layer(res: dict, field: str) -> int:
    return max((L for L in res if res[L]), key=lambda L: getattr(res[L][0], field))


def load_probe(weights_path: Path, layer: int) -> tuple[torch.nn.Module, torch.Tensor]:
    w = torch.load(weights_path, weights_only=False)[layer][0]
    m = _build_probe("linear", w["mean"].shape[0])
    m.load_state_dict(w["state_dict"])
    m.eval()
    return m, w["mean"]


def score_layer(H: torch.Tensor, idx: np.ndarray, probes: dict[str, tuple]) -> dict[str, np.ndarray]:
    """P(y=1) of each probe on rows idx of H (one pass over the rows)."""
    out = {name: np.empty(len(idx)) for name in probes}
    with torch.no_grad():
        for s in range(0, len(idx), CHUNK):
            Hc = H[torch.from_numpy(idx[s:s + CHUNK])].float()
            for name, (m, mean) in probes.items():
                out[name][s:s + len(Hc)] = torch.softmax(m(Hc - mean), dim=1)[:, 1].double().numpy()
    return out


def fit_position(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, seed: int, max_n: int = 2_000_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(ytr) > max_n:
        sel = rng.choice(len(ytr), max_n, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(ytr.mean()) if len(ytr) else 0.5)
    model = make_pipeline(SplineTransformer(n_knots=8, degree=3, knots="quantile", extrapolation="constant"),
                          LogisticRegression(max_iter=2000))
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def pos_features(step: np.ndarray, apos: np.ndarray, rel: np.ndarray) -> dict[str, np.ndarray]:
    return {"pos_causal": np.column_stack([np.log1p(step), np.log1p(apos)]), "pos_rel": rel[:, None].astype(np.float64)}


def bootstrap(metrics: dict[str, Metric], diffs: dict[str, tuple[str, str]], sids: list[str],
              n_boot: int, seed: int) -> tuple[dict, dict]:
    """Point estimates and CIs for metrics and paired differences a - b."""
    inst = np.array([re.sub(r"_run\d+$", "", s) for s in sids])
    uniq, inst_id = np.unique(inst, return_inverse=True)
    ones = np.ones(len(sids))
    point = {n: m(ones) for n, m in metrics.items()}
    for n, (a, b) in diffs.items():
        point[n] = point[a] - point[b]
    boots: dict[str, list] = {n: [] for n in point}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        tw = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))[inst_id].astype(np.float64)
        v = {n: m(tw) for n, m in metrics.items()}
        for n, (a, b) in diffs.items():
            v[n] = v[a] - v[b]
        for n, x in v.items():
            boots[n].append(x)
    return point, {n: ci(b) for n, b in boots.items()}


def run_lookahead(args) -> dict:
    croot, rroot = Path("cache", args.bench), Path("results", args.bench)
    K, probe = args.max_k, args.probe
    ks = [k for k in (args.ks or KS[K]) if k <= K]
    c0 = croot / f"{args.cache_prefix}_shift0_max{K}" / probe
    meta = torch.load(c0 / "layer_0.pt", weights_only=False, mmap=True)
    sids, gids = meta["sample_id"], meta["group_id"]
    step, rel = meta["step_idx"].numpy().copy(), meta["rel_pos"].numpy().copy()
    del meta
    apos = abs_positions(sids)
    in_tr, _, in_te = split_masks(gids, args.seed)
    test_idx = np.flatnonzero(in_te)
    feats = pos_features(step, apos, rel)
    log("tokens", len(sids), "test", len(test_idx))

    ylab, yorig = {}, {}
    for k in ks:
        d = torch.load(croot / f"{args.cache_prefix}_shift{k}_max{K}" / probe / "layer_0.pt", weights_only=False, mmap=True)
        if len(d["y"]) != len(sids) or not np.array_equal(d["step_idx"].numpy(), step):
            raise ValueError(f"shift{k} cache does not share the shift0 token set")
        ylab[k], yorig[k] = d["y"].numpy().copy(), d["y_original"].numpy().copy()
        del d

    # Which probe at which layer.
    need: dict[int, dict[str, tuple]] = {}
    layers: dict[str, dict] = {}
    r0 = results(rroot / f"{args.pooled}_shift0_max{K}" / probe / "results.pt")
    l0 = best_layer(r0, "val_auc")
    need.setdefault(l0, {})["k0"] = load_probe(rroot / f"{args.pooled}_shift0_max{K}" / probe / "weights.pt", l0)
    for k in ks:
        rid = f"{args.pooled}_shift{k}_max{K}"
        rk = results(rroot / rid / probe / "results.pt")
        lv, lt = best_layer(rk, "val_auc"), best_layer(rk, "test_auc")
        layers[k] = {"real_val": lv, "real_test": lt, "k0": l0, "traj": lv,
                     "results_test_auc_real_test": float(rk[lt][0].test_auc),
                     "results_test_auc_real_val": float(rk[lv][0].test_auc)}
        need.setdefault(lv, {})[f"real_val_{k}"] = load_probe(rroot / rid / probe / "weights.pt", lv)
        need.setdefault(lt, {})[f"real_test_{k}"] = load_probe(rroot / rid / probe / "weights.pt", lt)
        tpath = rroot / f"{rid}_shuffled_traj" / probe
        tres = results(tpath / "results.pt") if (tpath / "results.pt").exists() else {}
        if lv in tres:
            need[lv][f"traj_{k}"] = load_probe(tpath / "weights.pt", lv)
            layers[k]["results_test_auc_traj"] = float(tres[lv][0].test_auc)
        else:
            layers[k]["traj"] = None
            log(f"WARNING: no trajectory-shuffle probe for k={k} at layer {lv}")

    scores: dict[str, np.ndarray] = {}
    for L, probes in sorted(need.items()):
        log(f"scoring layer {L}: {sorted(probes)}")
        H = torch.load(c0 / f"layer_{L}.pt", weights_only=False, mmap=True)["H"]
        scores.update(score_layer(H, test_idx, probes))
        del H

    rows = []
    for k in ks:
        sub = ylab[k][test_idx] >= 0
        tok = test_idx[sub]
        y, yo = ylab[k][tok], yorig[k][tok]
        tsids = [sids[i] for i in tok]
        sc = {"real_val": scores[f"real_val_{k}"][sub], "real_test": scores[f"real_test_{k}"][sub], "k0": scores["k0"][sub]}
        if f"traj_{k}" in scores:
            sc["traj"] = scores[f"traj_{k}"][sub]
        trn = in_tr & (ylab[k] >= 0)
        for name, X in feats.items():
            sc[name] = fit_position(X[trn], ylab[k][trn], X[tok], args.seed)
        ok = yo >= 0
        strata = {"all": np.ones(len(y), dtype=bool), "yt0": ok & (yo == 0), "yt1": ok & (yo == 1)}
        metrics = {f"{s}|{st}": Metric(v, y, m) for s, v in sc.items() for st, m in strata.items()}
        diffs = {f"real_val-{o}|{st}": (f"real_val|{st}", f"{o}|{st}")
                 for o in ("k0", "traj", "pos_causal", "pos_rel") if o in sc for st in strata}
        point, cis = bootstrap(metrics, diffs, tsids, args.n_boot, args.seed)
        row = {"k": k, "layers": layers[k], "n_test": int(len(y)),
               "n_yt0": int(strata["yt0"].sum()), "n_yt1": int(strata["yt1"].sum()),
               "n_pos_yt0": int((y[strata["yt0"]] == 1).sum()), "n_pos_yt1": int((y[strata["yt1"]] == 1).sum()),
               "auc": point, "ci95": cis}
        for name in ("real_test", "real_val"):
            if abs(point[f"{name}|all"] - layers[k][f"results_test_auc_{name}"]) > 2e-3:
                raise ValueError(f"k={k} {name}: recomputed AUC {point[f'{name}|all']:.4f} != results.pt "
                                 f"{layers[k][f'results_test_auc_{name}']:.4f}")
        log(f"k={k}", json.dumps({n: round(v, 3) for n, v in point.items() if not np.isnan(v)}))
        rows.append(row)
    return {"mode": "lookahead", "bench": args.bench, "pooled": args.pooled, "probe": probe, "max_k": K,
            "n_boot": args.n_boot, "rows": rows}


def run_k0(args) -> dict:
    croot, rroot = Path("cache", args.bench), Path("results", args.bench)
    probe = args.probe
    cdir = croot / args.cache_prefix / probe
    meta = torch.load(cdir / "layer_0.pt", weights_only=False, mmap=True)
    sids, gids = meta["sample_id"], meta["group_id"]
    y_all = meta["y"].numpy().copy()
    step, rel = meta["step_idx"].numpy().copy(), meta["rel_pos"].numpy().copy()
    del meta
    apos = abs_positions(sids)
    in_tr, in_va, in_te = split_masks(gids, args.seed)
    valid = y_all >= 0
    val_idx, test_idx = np.flatnonzero(in_va & valid), np.flatnonzero(in_te & valid)
    log("tokens", len(sids), "val", len(val_idx), "test", len(test_idx))

    runs = {"real": args.pooled, "token": f"{args.pooled}_shuffled", "traj": f"{args.pooled}_shuffled_traj"}
    res = {n: results(rroot / rid / probe / "results.pt") for n, rid in runs.items()}
    all_layers = sorted(set.intersection(*(set(r) for r in res.values())))
    both = np.concatenate([val_idx, test_idx])
    nv = len(val_idx)
    yv, yt = y_all[val_idx], y_all[test_idx]
    val_auc: dict[str, dict[int, float]] = {n: {} for n in runs}
    test_scores: dict[tuple, np.ndarray] = {}
    ones_v = np.ones(nv)
    for L in all_layers:
        log(f"scoring layer {L}")
        H = torch.load(cdir / f"layer_{L}.pt", weights_only=False, mmap=True)["H"]
        sc = score_layer(H, both, {n: load_probe(rroot / rid / probe / "weights.pt", L) for n, rid in runs.items()})
        del H
        for n, s in sc.items():
            val_auc[n][L] = Metric(s[:nv], yv, np.ones(nv, dtype=bool))(ones_v)
            test_scores[(n, L)] = s[nv:]
    lv = {n: max(all_layers, key=lambda L: val_auc[n][L]) for n in runs}
    lt_real = max(all_layers, key=lambda L: res["real"][L][0].test_auc)

    sc = {f"{n}_valbest": test_scores[(n, lv[n])] for n in runs}
    sc["real_testbest"] = test_scores[("real", lt_real)]
    for n in ("token", "traj"):
        sc[f"{n}_at_real_layer"] = test_scores[(n, lv["real"])]
    trn = in_tr & valid
    for name, X in pos_features(step, apos, rel).items():
        sc[name] = fit_position(X[trn], y_all[trn], X[test_idx], args.seed)
    full = np.ones(len(yt), dtype=bool)
    metrics = {n: Metric(v, yt, full) for n, v in sc.items()}
    diffs = {"real-token": ("real_valbest", "token_valbest"), "real-traj": ("real_valbest", "traj_valbest"),
             "real-token_same_layer": ("real_valbest", "token_at_real_layer"),
             "real-traj_same_layer": ("real_valbest", "traj_at_real_layer"),
             "real-pos_causal": ("real_valbest", "pos_causal"), "real-pos_rel": ("real_valbest", "pos_rel")}
    point, cis = bootstrap(metrics, diffs, [sids[i] for i in test_idx], args.n_boot, args.seed)
    stored = {n: {L: float(res[n][L][0].test_auc) for L in all_layers} for n in runs}
    for n in runs:  # recomputed test AUC must match results.pt
        if abs(Metric(test_scores[(n, lv[n])], yt, full)(np.ones(len(yt))) - stored[n][lv[n]]) > 2e-3:
            raise ValueError(f"{n}: recomputed test AUC does not match results.pt")
    log(json.dumps({n: round(v, 3) for n, v in point.items()}))
    return {"mode": "k0", "bench": args.bench, "pooled": args.pooled, "probe": probe, "n_boot": args.n_boot,
            "n_test": int(len(yt)), "n_pos_test": int((yt == 1).sum()),
            "val_best_layer": lv, "real_test_best_layer": lt_real,
            "val_auc_real_labels": {n: {str(L): v for L, v in d.items()} for n, d in val_auc.items()},
            "stored_test_auc": {n: {str(L): v for L, v in d.items()} for n, d in stored.items()},
            "auc": point, "ci95": cis}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["lookahead", "k0"])
    ap.add_argument("--bench", required=True)
    ap.add_argument("--pooled", required=True)
    ap.add_argument("--cache-prefix", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--max-k", type=int, default=15)
    ap.add_argument("--ks", type=int, nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = run_lookahead(args) if args.mode == "lookahead" else run_k0(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    log("wrote", args.output)


if __name__ == "__main__":
    main()
