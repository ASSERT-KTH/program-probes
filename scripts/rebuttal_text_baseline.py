"""Brief E9: text-only baselines, and k=0 AUC split by whether the current program version has run yet.

Text baselines (one score per assistant turn, given to every cached token of that turn):
  text_a1  TF-IDF of the most recent tool output in the turn's context: the observation message right before
           assistant turn t (the output of command t-1); empty for turn 0 (no tool output yet).
  text_a2  (--a2) TF-IDF of all text before turn t (system prompt, task, earlier assistant messages and
           observations, joined with role tags), truncated to the LAST A2_CHARS characters.
TF-IDF: word 1-2-grams, tokens of >= 1 word character (so "1 failed" keeps the "1"), sublinear tf,
min_df 3, at most 200k features (A2: 100k), fit once per run (features mode) on the turns of training tasks.
Logistic regression, every token
weighted by one (aggregated as per-turn token counts; total weight normalised to the number of training
turns); C chosen on pooled validation AUC on a log grid (warm-started path). Same split, population and
target as the probe it is compared with (train split, y >= 0 of that cache).
Texts and command classes: tmp/e9_text_<bench>_<run>.jsonl.gz (scripts/rebuttal_text_extract.py).

horizon mode (one job per run x probe): as scripts/rebuttal_prompt_only.py (window 15, k in {1,2,3,5,10,15},
E18 strata, paired task bootstrap, E16/E18 reproduction checks), scorers text_a1 (+ text_a2).
k0 mode (one job per run x probe): pooled k=0 cache, all test tokens, k=0 probe at its test-best layer
(Table 6) vs the text baselines; and part B: test tokens split by whether the current program version
(the one created by the last edit at or before the turn, or the original repository before any edit) has
already been run, i.e. any command from the edit command itself through command t-1 ran tests, scripts,
builds or interpreters (AE0 runs_project rule; "strict" variant: a Python editor script also counts).
Edits are the label-changing versions of edit_step_index.pt (turn N+1 for an edit at command N).
The *_post_edit subsets keep only turns whose current version comes from a real edit (drops the original
repository phase before the first edit, which is also the start of the trajectory); pre_first_edit is the rest.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuttal_edit_outcome_strata import MIN_TASKS, last_turn_label  # noqa: E402
from rebuttal_persistence_analysis import Metric  # noqa: E402
from rebuttal_phase_confound import (bootstrap, load_probe, lookahead_cell, lookahead_context,  # noqa: E402
                                     score_layer, split_masks)
from src.probe import _split_groups  # noqa: E402
from rebuttal_prompt_only import KS, OLD_CONTROLS, RUNS, task  # noqa: E402

C_GRID = np.logspace(-3, 2, 11)
A2_CHARS = 32_000


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_texts(bench: str, k0dir: str, sids: set[str]) -> dict:
    out = {}
    with gzip.open(Path("tmp", f"e9_text_{bench}_{k0dir}.jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["sid"] in sids:
                out[r["sid"]] = r
    missing = sids - set(out)
    if missing:
        raise ValueError(f"{len(missing)} cached trajectories have no text record, e.g. {sorted(missing)[:3]}")
    return out


class Units:
    """(trajectory, turn) units of the cache rows."""

    def __init__(self, sids: list[str], step: np.ndarray):
        uniq_s, sidx = np.unique(np.asarray(sids, dtype=object), return_inverse=True)
        key = sidx.astype(np.int64) * 100_000 + step
        ukey, self.of_row = np.unique(key, return_inverse=True)
        self.sid = [uniq_s[i] for i in ukey // 100_000]
        self.turn = (ukey % 100_000).astype(int)
        self.n = len(ukey)


def doc_a1(r: dict, t: int) -> str:
    if t == 0 or t >= r["n_asst"]:
        return ""
    prev = r["msgs"][r["asst_at"][t] - 1]
    return prev[1] if prev[0] == "user" else ""


def doc_a2(r: dict, t: int) -> str:
    end = r["asst_at"][min(t, r["n_asst"] - 1)]
    s = "\n".join(f"<{role}>\n{c}" for role, c in r["msgs"][:end])
    return s[-A2_CHARS:]


def fit_text(X, unit: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, dict]:
    """Logistic regression on per-unit features with token-count weights; C chosen on pooled validation AUC."""
    n = X.shape[0]
    cnt = {(s, c): np.bincount(unit[m & (y == c)], minlength=n).astype(float)
           for s, m in (("tr", tr), ("va", va)) for c in (0, 1)}
    rp, rn = np.flatnonzero(cnt["tr", 1] > 0), np.flatnonzero(cnt["tr", 0] > 0)
    rows, yy = np.r_[rp, rn], np.r_[np.ones(len(rp)), np.zeros(len(rn))]
    w = np.r_[cnt["tr", 1][rp], cnt["tr", 0][rn]]
    n_units = len(np.unique(rows))
    w = w * n_units / w.sum()
    vp, vn = np.flatnonzero(cnt["va", 1] > 0), np.flatnonzero(cnt["va", 0] > 0)
    vrows, vy, vw = np.r_[vp, vn], np.r_[np.ones(len(vp)), np.zeros(len(vn))], np.r_[cnt["va", 1][vp], cnt["va", 0][vn]]
    tot = cnt["tr", 1].sum() + cnt["tr", 0].sum()
    summary = {"n_train_units": int(n_units), "n_train_tokens": int(tot),
               "train_pos_rate": float(cnt["tr", 1].sum() / max(tot, 1)), "val_auc": {}}
    if len(rp) == 0 or len(rn) == 0 or len(vp) == 0 or len(vn) == 0:
        summary["degenerate"] = True
        return np.full(n, cnt["tr", 1].sum() / max(tot, 1)), summary
    Xr = X[rows]
    m = LogisticRegression(C=C_GRID[0], max_iter=1000, warm_start=True)
    best = (-1.0, None, None)
    for C in C_GRID:
        m.set_params(C=C)
        m.fit(Xr, yy, sample_weight=w)
        p = m.predict_proba(X)[:, 1]
        a = roc_auc_score(vy, p[vrows], sample_weight=vw)
        summary["val_auc"][f"{C:.0e}"] = float(a)
        if a > best[0]:
            best = (a, C, p)
    summary.update({"C": float(best[1]), "best_val_auc": float(best[0]),
                    "C_at_grid_edge": bool(best[1] in (C_GRID[0], C_GRID[-1]))})
    return best[2], summary


def build_features(args) -> None:
    """features mode: A1 (and A2) TF-IDF matrices for every turn of every cached trajectory of one run.

    The vectorizer is fit on the turns of tasks that are in the training split of BOTH the k=0 cache and the
    window-15 cache (each cache has its own task split, as the probes were trained), so no validation or test
    text of either split enters the vocabulary or idf.
    """
    bench, _, prefix, k0dir = RUNS[args.run]
    tr_sids, all_sids = None, set()
    for c in (Path("cache", bench, k0dir, "currently_compiles"), Path("cache", bench, f"{prefix}_shift0_max15", "currently_correct")):
        meta = torch.load(c / "layer_0.pt", weights_only=False, mmap=True)
        sids, gids = list(meta["sample_id"]), list(meta["group_id"])
        del meta
        tr_groups = _split_groups(gids, args.seed)[0]
        tr = {s for s, g in zip(sids, gids) if g in tr_groups}
        tr_sids = tr if tr_sids is None else tr_sids & tr
        all_sids |= set(sids)
    texts = load_texts(bench, k0dir, all_sids)
    usid, uturn = [], []
    for s in sorted(all_sids):
        for t in range(texts[s]["n_asst"]):
            usid.append(s)
            uturn.append(t)
    uturn = np.array(uturn)
    train = np.array([s in tr_sids for s in usid])
    info = {"n_units": len(usid), "n_trajectories": len(all_sids), "n_train_trajectories_for_vocab": len(tr_sids),
            "n_train_units_for_vocab": int(train.sum())}
    out = Path("tmp", f"e9_feat_{bench}_{k0dir}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "units.json").write_text(json.dumps({"sid": usid, "turn": uturn.tolist()}))
    kinds = [k for k in [("text_a1", doc_a1, 200_000), ("text_a2", doc_a2, 100_000)] if k[0] in args.kinds]
    for name, fn, mf in kinds:
        vec = TfidfVectorizer(sublinear_tf=True, min_df=3, max_df=0.95, ngram_range=(1, 2),
                              token_pattern=r"(?u)\b\w+\b", max_features=mf, dtype=np.float32)
        vec.fit(fn(texts[usid[i]], uturn[i]) for i in np.flatnonzero(train))
        X = vec.transform(fn(texts[usid[i]], uturn[i]) for i in range(len(usid))).tocsr()
        sp.save_npz(out / f"{name}.npz", X)
        info[name] = {"shape": list(X.shape), "nnz": int(X.nnz), "vocab": len(vec.vocabulary_)}
        if name == "text_a1":
            info[name]["empty_docs"] = int(sum(fn(texts[usid[i]], uturn[i]) == "" for i in range(len(usid))))
        else:
            info[name]["truncated_docs"] = int(sum(len(fn(texts[usid[i]], uturn[i])) >= A2_CHARS for i in range(len(usid))))
        log(name, info[name])
    ipath = out / "info.json"
    old = json.loads(ipath.read_text()) if ipath.exists() else {}
    ipath.write_text(json.dumps({**old, **info}, indent=1))


def load_text_features(bench: str, k0dir: str, sids: list[str], step: np.ndarray, a2: bool):
    """Units of the cache rows and the matching rows of the precomputed feature matrices."""
    units = Units(sids, step)
    d = Path("tmp", f"e9_feat_{bench}_{k0dir}")
    u = json.loads((d / "units.json").read_text())
    pos = {(s, t): i for i, (s, t) in enumerate(zip(u["sid"], u["turn"]))}
    rows = np.array([pos.get((units.sid[i], int(units.turn[i])), -1) for i in range(units.n)])
    if (rows < 0).any():
        raise ValueError(f"{int((rows < 0).sum())} cache units have no text features")
    feats = {"text_a1": sp.load_npz(d / "text_a1.npz")[rows]}
    if a2:
        feats["text_a2"] = sp.load_npz(d / "text_a2.npz")[rows]
    info = json.loads((d / "info.json").read_text())
    info["n_units_this_cache"] = units.n
    log("features", {n: X.shape for n, X in feats.items()})
    return units, feats, info


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
    in_tr, in_va, _ = split_masks(gids, args.seed)
    units, feats, info = load_text_features(bench, k0dir, sids, step, args.a2)
    test_sids = sorted({sids[i] for i in ctx["test_idx"]})
    yT_of = {s: last_turn_label(Path("outputs", bench, k0dir), s, probe) for s in test_sids}
    e16 = {r["k"]: r for r in json.loads(Path("reports/rerun/phase", f"lookahead_{bench}_{pooled}_{probe}_max{K}.json").read_text())["rows"]}
    e18 = {r["k"]: r for r in json.loads(Path("reports/rerun/phase", f"e18_{bench}_{pooled}_{probe}_max{K}.json").read_text())["rows"]}
    y1_all = ctx["ylab"][1]
    rows = []
    for k in KS:
        yk = ctx["ylab"][k]
        ok = yk >= 0
        fits, pu = {}, {}
        for name, X in feats.items():
            pu[name], fits[name] = fit_text(X, units.of_row, yk, in_tr & ok, in_va & ok)
            log(f"k={k} {name} C={fits[name].get('C')} val={fits[name].get('best_val_auc')}")
        cell = lookahead_cell(ctx, k, args.seed)
        y, yo, tsids, sc, tok = cell["y"], cell["yo"], cell["tsids"], dict(cell["sc"]), cell["tok"]
        sc.pop("real_test")
        for name in feats:
            sc[name] = pu[name][units.of_row[tok]]
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
                 for o in OLD_CONTROLS + list(feats) if o in sc for st in strata}
        point, cis = bootstrap(metrics, diffs, tsids, args.n_boot, args.seed)
        checks = {}
        for st in strata:
            for s in ("real_val",) + tuple(OLD_CONTROLS):
                ref = (e18[k]["auc"].get(f"{s}|{st}") if k in e18 else
                       (e16[k]["auc"].get(f"{s}|yt0") if st == "S0" else None))
                if ref is None or ref != ref:
                    continue
                d = abs(point[f"{s}|{st}"] - ref)
                checks[f"{s}|{st}"] = d
                if d > (1e-3 if s.startswith("pos_") else 1e-6):
                    raise ValueError(f"k={k} {s}|{st}: {point[f'{s}|{st}']:.6f} != E16/E18 {ref:.6f}")
            if k in e18 and e18[k]["counts"][st] != counts[st]:
                raise ValueError(f"k={k} {st}: counts differ from E18")
        rows.append({"k": k, "layer": ctx["layers"][k]["real_val"], "layers": ctx["layers"][k], "counts": counts,
                     "fit": fits, "max_abs_diff_vs_e16_e18": max(checks.values()) if checks else None,
                     "auc": point, "ci95": cis})
        log(f"k={k}", json.dumps({n: round(v, 3) for n, v in point.items()
                                  if v == v and n.split("|")[0] in ("real_val",) + tuple(feats)}))
    return {"mode": "horizon", "run": args.run, "bench": bench, "pooled": pooled, "probe": probe, "max_k": K,
            "n_boot": args.n_boot, "min_tasks": MIN_TASKS, "C_grid": C_GRID.tolist(), "a2_chars": A2_CHARS,
            "text_info": info, "rows": rows}


def already_run(units: Units, texts: dict, index: dict, key: str) -> np.ndarray:
    """Per unit: has the program version current at turn t been run by command e-1 .. t-1?"""
    out = np.zeros(units.n, dtype=bool)
    for i in range(units.n):
        r, t = texts[units.sid[i]], units.turn[i]
        edits = [e for e in index.get(units.sid[i], []) if e <= t]
        start = edits[-1] - 1 if edits else 0
        runs = r[key]
        out[i] = any(runs[j] for j in range(start, min(t, len(runs))))
    return out


def post_edit(units: Units, index: dict) -> np.ndarray:
    """Per unit: is the current version one created by a real edit (not the original repository)?"""
    return np.array([any(e <= units.turn[i] for e in index.get(units.sid[i], [])) for i in range(units.n)])


def run_k0(args) -> dict:
    bench, pooled, _, k0dir = RUNS[args.run]
    probe = args.probe
    cdir = Path("cache", bench, k0dir, probe)
    meta = torch.load(cdir / "layer_0.pt", weights_only=False, mmap=True)
    sids, gids = meta["sample_id"], meta["group_id"]
    y = meta["y"].numpy().copy()
    step = meta["step_idx"].numpy().copy()
    del meta
    in_tr, in_va, in_te = split_masks(gids, args.seed)
    ok = y >= 0
    test_idx = np.flatnonzero(in_te & ok)
    e17 = json.loads(Path("reports/rerun/phase", f"k0_{bench}_{pooled}_{probe}.json").read_text())
    L = int(e17["real_test_best_layer"])
    stored = e17["stored_test_auc"]["real"][str(L)]
    real = score_layer(torch.load(cdir / f"layer_{L}.pt", weights_only=False, mmap=True)["H"], test_idx,
                       {"real": load_probe(Path("results", bench, pooled, probe, "weights.pt"), L)})["real"]
    units, feats, info = load_text_features(bench, k0dir, sids, step, args.a2)
    texts = load_texts(bench, k0dir, set(units.sid))
    index = torch.load(Path("cache", bench, k0dir, "edit_step_index.pt"), weights_only=False)
    ran = {v: already_run(units, texts, index, key)[units.of_row[test_idx]]
           for v, key in (("primary", "runs"), ("strict", "strict"))}
    edited = post_edit(units, index)[units.of_row[test_idx]]
    fits, sc = {}, {"real_testbest": real}
    for name, X in feats.items():
        p, fits[name] = fit_text(X, units.of_row, y, in_tr & ok, in_va & ok)
        sc[name] = p[units.of_row[test_idx]]
        log(f"{name} C={fits[name].get('C')} val={fits[name].get('best_val_auc')}")
    yt = y[test_idx]
    tsids = [sids[i] for i in test_idx]
    tasks = np.array([task(s) for s in tsids])
    subsets = {"all": np.ones(len(yt), dtype=bool)}
    for v, r in ran.items():
        sfx = "" if v == "primary" else "_strict"
        subsets[f"not_yet_run{sfx}"], subsets[f"already_run{sfx}"] = ~r, r
        subsets[f"not_yet_run{sfx}_post_edit"], subsets[f"already_run{sfx}_post_edit"] = ~r & edited, r & edited
    subsets["pre_first_edit"] = ~edited
    metrics = {f"{n}|{s}": Metric(v, yt, m) for n, v in sc.items() for s, m in subsets.items()}
    diffs = {f"real-{n}|{s}": (f"real_testbest|{s}", f"{n}|{s}") for n in feats for s in subsets}
    diffs.update({f"real|already-not_yet{sfx}": (f"real_testbest|already_run{sfx}", f"real_testbest|not_yet_run{sfx}")
                  for sfx in ("", "_strict", "_post_edit", "_strict_post_edit")})
    point, cis = bootstrap(metrics, diffs, tsids, args.n_boot, args.seed)
    if abs(point["real_testbest|all"] - stored) > 2e-4:
        raise ValueError(f"recomputed AUC {point['real_testbest|all']:.5f} != results.pt {stored:.5f}")
    counts = {s: {"n": int(m.sum()), "share": float(m.mean()), "n_pos": int((yt[m] == 1).sum()),
                  "pos_rate": float((yt[m] == 1).mean()) if m.any() else None, "n_tasks": int(len(set(tasks[m])))}
              for s, m in subsets.items()}
    log(json.dumps({n: round(v, 3) for n, v in point.items() if v == v}))
    return {"mode": "k0", "run": args.run, "bench": bench, "pooled": pooled, "probe": probe, "layer": L,
            "stored_test_auc": stored, "n_boot": args.n_boot, "C_grid": C_GRID.tolist(), "a2_chars": A2_CHARS,
            "text_info": info, "counts": counts, "fit": fits, "auc": point, "ci95": cis}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["features", "horizon", "k0"])
    ap.add_argument("--run", required=True, choices=sorted(RUNS))
    ap.add_argument("--probe")
    ap.add_argument("--a2", action="store_true", help="horizon/k0: also score text_a2")
    ap.add_argument("--kinds", nargs="+", default=["text_a1"], help="features: which matrices to build")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output")
    args = ap.parse_args()
    if args.mode == "features":
        build_features(args)
        os._exit(0)
    out = run_horizon(args) if args.mode == "horizon" else run_k0(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    log("wrote", args.output)
    os._exit(0)


if __name__ == "__main__":
    main()
