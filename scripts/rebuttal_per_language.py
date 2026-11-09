"""Rebuttal E11: per-language and per-repo AUC of the k=0 probes on SWE-bench Pro.

Eval-only. For one (Pro run, probe):
  - probe: the rerun's pooled k=0 probe (<pooled>, task split) at its validation-best layer,
    taken from the E17 output (reports/rerun/phase/k0_<bench>_<pooled>_<probe>.json, layer chosen
    on validation AUC against real labels). Test tokens and AUC definition are E17's: every test
    token with y >= 0, one pooled AUC (no position bins). The pooled AUC is checked against E17.
  - language = `repo_language` and repo = `repo` of ScaleAI/SWE-bench_Pro (HF cache), joined
    on instance_id = sample_id without the _runNN suffix.
  - per stratum (language, repo): AUC, tokens, positives, positive rate, test tasks, test tasks
    with both classes (underpowered if < 20).
  - stratified AUC: counts only positive/negative pairs within the same stratum
    (sum_s P_s N_s AUC_s / sum_s P_s N_s, strata with one class contribute nothing).
  - stratum-only baselines: score = the training positive rate of the token's language
    (resp. repo; a repo absent from train falls back to its language rate, counted).
  - with --transfer-pooled: the Verified-trained k=0 probe (rerun, task split) at its Verified
    validation-best layer (from E17's Verified output), with its Verified training mean, applied
    to the same Pro test tokens: pooled, per-language and language-stratified AUC.
CIs: 1000 task-level bootstrap resamples (multinomial task weights), differences paired.

Usage:
    uv run python scripts/rebuttal_per_language.py --pooled laguna_xs2_pro_full_pooled \
        --cache-prefix laguna_xs2_full --probe currently_compiles \
        --transfer-pooled laguna_xs2_full_pooled --output reports/rerun/lang/...json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuttal_persistence_analysis import Metric, weighted_auc  # noqa: E402,F401
from rebuttal_phase_confound import bootstrap, load_probe, log, score_layer, split_masks  # noqa: E402

MIN_TASKS = 20


def task(sid: str) -> str:
    return re.sub(r"_run\d+$", "", sid)


class Stratified:
    """Weighted AUC over same-stratum positive/negative pairs only."""

    def __init__(self, scores: np.ndarray, y: np.ndarray, strata: np.ndarray):
        self.parts = []
        for s in np.unique(strata):
            m = strata == s
            if (y[m] == 1).any() and (y[m] == 0).any():
                self.parts.append((Metric(scores, y, m), m & (y == 1), m & (y == 0)))

    def __call__(self, w: np.ndarray) -> float:
        num = den = 0.0
        for metric, pos, neg in self.parts:
            pn = w[pos].sum() * w[neg].sum()
            if pn > 0:
                num += pn * metric(w)
                den += pn
        return num / den if den > 0 else float("nan")


def pro_metadata() -> dict[str, tuple[str, str]]:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset
    d = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    return {r["instance_id"]: (r["repo_language"], r["repo"]) for r in d}


def stratum_counts(name: str, values: np.ndarray, y: np.ndarray, tasks: np.ndarray) -> dict:
    out = {}
    for v in np.unique(values):
        m = values == v
        t = tasks[m]
        both = sum(1 for tt in np.unique(t) if len(np.unique(y[m][t == tt])) == 2)
        out[str(v)] = {"n": int(m.sum()), "n_pos": int((y[m] == 1).sum()), "pos_rate": float((y[m] == 1).mean()),
                       "n_tasks": int(len(np.unique(t))), "n_tasks_both_classes": int(both),
                       "underpowered": bool(both < MIN_TASKS)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="swebench_pro")
    ap.add_argument("--pooled", required=True)
    ap.add_argument("--cache-prefix", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--transfer-pooled", default=None, help="Verified pooled run id (results/swebench/<id>)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    phase = Path("reports/rerun/phase")
    e17 = json.loads((phase / f"k0_{args.bench}_{args.pooled}_{args.probe}.json").read_text())
    layer = int(e17["val_best_layer"]["real"])
    cdir = Path("cache", args.bench, args.cache_prefix, args.probe)
    meta = torch.load(cdir / "layer_0.pt", weights_only=False, mmap=True)
    sids, gids, y_all = meta["sample_id"], meta["group_id"], meta["y"].numpy().copy()
    del meta
    in_tr, _, in_te = split_masks(gids, args.seed)
    valid = y_all >= 0
    test_idx, train_idx = np.flatnonzero(in_te & valid), np.flatnonzero(in_tr & valid)
    md = pro_metadata()
    tasks_all = np.array([task(s) for s in sids])
    missing = sorted(set(tasks_all) - set(md))
    if missing:
        raise ValueError(f"{len(missing)} tasks missing from the Pro metadata, e.g. {missing[:3]}")
    lang_all = np.array([md[t][0] for t in tasks_all])
    repo_all = np.array([md[t][1] for t in tasks_all])
    log("tokens", len(sids), "test", len(test_idx), "layer", layer)

    # Scores.
    H = torch.load(cdir / f"layer_{layer}.pt", weights_only=False, mmap=True)["H"]
    sc = {"real": score_layer(H, test_idx, {"real": load_probe(Path("results", args.bench, args.pooled, args.probe, "weights.pt"), layer)})["real"]}
    del H
    transfer_layer = None
    if args.transfer_pooled:
        e17v = json.loads((phase / f"k0_swebench_{args.transfer_pooled}_{args.probe}.json").read_text())
        transfer_layer = int(e17v["val_best_layer"]["real"])
        H = torch.load(cdir / f"layer_{transfer_layer}.pt", weights_only=False, mmap=True)["H"]
        sc["transfer"] = score_layer(H, test_idx, {"t": load_probe(
            Path("results", "swebench", args.transfer_pooled, args.probe, "weights.pt"), transfer_layer)})["t"]
        del H

    # Stratum-only baselines from training positive rates.
    ytr = y_all[train_idx]
    lang_rate = {l: float((ytr[lang_all[train_idx] == l] == 1).mean()) for l in np.unique(lang_all[train_idx])}
    repo_rate = {r: float((ytr[repo_all[train_idx] == r] == 1).mean()) for r in np.unique(repo_all[train_idx])}
    y = y_all[test_idx]
    lang, repo, tasks = lang_all[test_idx], repo_all[test_idx], tasks_all[test_idx]
    test_repos = sorted(set(repo))
    repo_fallback = [r for r in test_repos if r not in repo_rate]
    sc["lang_only"] = np.array([lang_rate[l] for l in lang])
    sc["repo_only"] = np.array([repo_rate.get(r, lang_rate[l]) for r, l in zip(repo, lang)])

    full = np.ones(len(y), dtype=bool)
    metrics = {f"{s}|pooled": Metric(v, y, full) for s, v in sc.items()}
    metrics["real|strat_lang"] = Stratified(sc["real"], y, lang)
    metrics["real|strat_repo"] = Stratified(sc["real"], y, repo)
    for l in np.unique(lang):
        metrics[f"real|lang={l}"] = Metric(sc["real"], y, lang == l)
        if "transfer" in sc:
            metrics[f"transfer|lang={l}"] = Metric(sc["transfer"], y, lang == l)
    for r in test_repos:
        metrics[f"real|repo={r}"] = Metric(sc["real"], y, repo == r)
    if "transfer" in sc:
        metrics["transfer|strat_lang"] = Stratified(sc["transfer"], y, lang)
        metrics["transfer|strat_repo"] = Stratified(sc["transfer"], y, repo)
    diffs = {"real-lang_only": ("real|pooled", "lang_only|pooled"),
             "real-repo_only": ("real|pooled", "repo_only|pooled"),
             "real_pooled-real_strat_lang": ("real|pooled", "real|strat_lang"),
             "real_pooled-real_strat_repo": ("real|pooled", "real|strat_repo")}
    if "transfer" in sc:
        diffs["real-transfer"] = ("real|pooled", "transfer|pooled")
        for l in np.unique(lang):
            diffs[f"real-transfer|lang={l}"] = (f"real|lang={l}", f"transfer|lang={l}")
    point, cis = bootstrap(metrics, diffs, [sids[i] for i in test_idx], args.n_boot, args.seed)
    if abs(point["real|pooled"] - e17["auc"]["real_valbest"]) > 1e-6:
        raise ValueError(f"pooled AUC {point['real|pooled']:.6f} != E17 {e17['auc']['real_valbest']:.6f}")
    out = {"bench": args.bench, "pooled": args.pooled, "probe": args.probe, "layer": layer,
           "transfer_pooled": args.transfer_pooled, "transfer_layer": transfer_layer,
           "auc_definition": "pooled AUC over all test tokens with y >= 0 (E17); no position bins",
           "language_field": "repo_language (ScaleAI/SWE-bench_Pro)", "repo_field": "repo (ScaleAI/SWE-bench_Pro)",
           "n_boot": args.n_boot, "min_tasks_both_classes": MIN_TASKS,
           "n_test": int(len(y)), "n_pos_test": int((y == 1).sum()), "n_test_tasks": int(len(set(tasks))),
           "train_pos_rate_by_language": lang_rate, "train_pos_rate_by_repo": repo_rate,
           "test_repos_without_train_rate": repo_fallback,
           "n_test_tokens_repo_fallback": int(sum(1 for r in repo if r not in repo_rate)),
           "counts_by_language": stratum_counts("lang", lang, y, tasks),
           "counts_by_repo": stratum_counts("repo", repo, y, tasks),
           "auc": point, "ci95": cis}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    log(json.dumps({k: round(v, 3) for k, v in point.items() if v == v}))
    log("wrote", args.output)


if __name__ == "__main__":
    main()
