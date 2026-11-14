"""Persistence analysis for the lookahead probes (rebuttal, reviewer R1).

For one (benchmark, run, probe, max_k) and each k, on the test split used in
training (src.probe._split_groups(group_ids, seed=42)):

  1. flip rate: fraction of test tokens with y (label at target turn) != y_original
     (label at the source turn), and the fraction of source/target pairs with at
     least one edit turn e in (step_idx, target_step_idx] (edit turns from
     edit_step_index.pt, i.e. the first turn after each edit);
  2. persistence AUC: AUC of the hard prediction y_original for y (equals balanced
     accuracy);
  3. probe AUC overall (should match results.pt), within y_original == 0, within
     y_original == 1, and on pairs with an edit in between.

Uncertainty: bootstrap over tasks (sample_id without the _runNN suffix), the
same unit the train/val/test split uses.
Resampling tasks (SWE-bench instances) with replacement is implemented as
multinomial task weights, with a weighted AUC computed from one sort of the scores.

The layer is the one with the best test AUC for that k, the rule used by
run_paper_figures.py, so the numbers line up with the figures.

Usage:
    uv run python scripts/rebuttal_persistence_analysis.py --bench swebench \
        --pooled laguna_xs2_full_pooled --cache-prefix laguna_xs2_full \
        --edit-index-run-id laguna_xs2_full --probe currently_correct \
        --model-config configs/models/laguna_xs2.yaml --max-k 50 \
        --output reports/rerun/persistence/swebench_laguna_xs2_full_pooled_currently_correct_max50.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from src.probe import _build_probe, _split_groups

DEFAULT_KS = [0, 1, 2, 3, 5, 10, 15, 25, 50]


def weighted_auc(order: np.ndarray, uniq_start: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """ROC AUC with per-sample weights; ties count 1/2.

    `order` sorts the scores ascending and `uniq_start` marks where each block
    of tied scores begins in that order (both computed once per score vector).
    """
    ys, ws = y[order], w[order]
    wp = np.add.reduceat(ws * (ys == 1), uniq_start)
    wn = np.add.reduceat(ws * (ys == 0), uniq_start)
    tot_p, tot_n = wp.sum(), wn.sum()
    if tot_p <= 0 or tot_n <= 0:
        return float("nan")
    neg_below = np.cumsum(wn) - wn
    return float((wp * (neg_below + 0.5 * wn)).sum() / (tot_p * tot_n))


def sort_blocks(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]]) if len(s) else np.array([], dtype=int)
    return order, starts


class Metric:
    """A weighted AUC over a fixed subset of test tokens."""

    def __init__(self, scores: np.ndarray, y: np.ndarray, subset: np.ndarray):
        self.idx = np.flatnonzero(subset)
        self.y = y[self.idx]
        self.order, self.starts = sort_blocks(scores[self.idx])

    def __call__(self, token_w: np.ndarray) -> float:
        if len(self.idx) == 0:
            return float("nan")
        return weighted_auc(self.order, self.starts, self.y, token_w[self.idx])


def ci(vals: list[float]) -> list[float | None]:
    a = np.asarray([v for v in vals if not np.isnan(v)])
    if len(a) == 0:
        return [None, None]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def best_layer(results_path: Path) -> tuple[int, float]:
    res = torch.load(results_path, weights_only=False)
    best = max(res, key=lambda L: res[L][0].test_auc if res[L] else -1)
    return best, float(res[best][0].test_auc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--pooled", required=True, help="Results run-id prefix, e.g. laguna_xs2_full_pooled")
    parser.add_argument("--cache-prefix", required=True, help="Shift-cache prefix, e.g. laguna_xs2_full")
    parser.add_argument("--edit-index-run-id", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--max-k", type=int, default=50)
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-arch", default="linear")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache_root = Path("cache") / args.bench
    results_root = Path("results") / args.bench
    edit_index = torch.load(cache_root / args.edit_index_run_id / "edit_step_index.pt", weights_only=False)
    rows = []

    for k in [k for k in args.ks if k <= args.max_k]:
        run_id = f"{args.pooled}_shift{k}_max{args.max_k}"
        cache_run = f"{args.cache_prefix}_shift{k}_max{args.max_k}"
        layer, res_auc = best_layer(results_root / run_id / args.probe / "results.pt")
        data = torch.load(cache_root / cache_run / args.probe / f"layer_{layer}.pt", weights_only=False, mmap=True)
        y_all = data["y"]
        sample_ids, group_ids = data["sample_id"], data["group_id"]
        _, _, test_groups = _split_groups(group_ids, args.seed)
        in_test = np.fromiter((g in test_groups for g in group_ids), dtype=bool, count=len(group_ids))
        has_orig = "y_original" in data
        valid = in_test & (y_all.numpy() >= 0)
        probe_idx = np.flatnonzero(valid)  # tokens scored by the probe run (y >= 0)

        # Probe probabilities on the probe's test tokens.
        w = torch.load(results_root / run_id / args.probe / "weights.pt", weights_only=False)[layer][0]
        H = data["H"]
        model = _build_probe(args.probe_arch, H.shape[1])
        model.load_state_dict(w["state_dict"])
        model.eval()
        probs = np.empty(len(probe_idx), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, len(probe_idx), 200_000):
                sl = torch.from_numpy(probe_idx[s:s + 200_000])
                probs[s:s + len(sl)] = torch.softmax(model(H[sl].float() - w["mean"]), dim=1)[:, 1].double().numpy()
        del H

        y = y_all.numpy()[probe_idx]
        sids = [sample_ids[i] for i in probe_idx]
        inst = np.array([re.sub(r"_run\d+$", "", s) for s in sids])
        uniq_inst, inst_id = np.unique(inst, return_inverse=True)
        ones = np.ones(len(y))
        overall = Metric(probs, y, np.ones(len(y), dtype=bool))
        row: dict = {
            "k": k, "layer": int(layer), "results_test_auc": res_auc,
            "n_test": int(len(y)), "n_instances": int(len(uniq_inst)),
            "probe_auc": overall(ones),
        }
        metrics = {"probe_auc": overall}

        if has_orig and k > 0:
            y_orig = data["y_original"].numpy()[probe_idx]
            step = data["step_idx"].numpy()[probe_idx]
            target = data["target_step_idx"].numpy()[probe_idx]
            edit_between = np.array([
                any(s < e <= t for e in edit_index.get(sid, ()))
                for sid, s, t in zip(sids, step, target)
            ])
            ok = y_orig >= 0  # persistence is undefined where the source label is missing
            row.update({
                "n_orig_missing": int((~ok).sum()),
                "flip_rate": float((y[ok] != y_orig[ok]).mean()),
                "frac_edit_between": float(edit_between[ok].mean()),
                "flip_rate_edit_between": float((y[ok & edit_between] != y_orig[ok & edit_between]).mean()) if (ok & edit_between).any() else None,
                "flip_rate_no_edit_between": float((y[ok & ~edit_between] != y_orig[ok & ~edit_between]).mean()) if (ok & ~edit_between).any() else None,
                "n_orig0": int((ok & (y_orig == 0)).sum()), "n_orig1": int((ok & (y_orig == 1)).sum()),
                "n_edit_between": int((ok & edit_between).sum()),
            })
            metrics.update({
                "persistence_auc": Metric(np.where(ok, y_orig, 0).astype(np.float64), y, ok),
                "probe_auc_persist_subset": Metric(probs, y, ok),
                "probe_auc_orig0": Metric(probs, y, ok & (y_orig == 0)),
                "probe_auc_orig1": Metric(probs, y, ok & (y_orig == 1)),
                "probe_auc_edit_between": Metric(probs, y, ok & edit_between),
                "persistence_auc_edit_between": Metric(np.where(ok, y_orig, 0).astype(np.float64), y, ok & edit_between),
            })
            for name, m in metrics.items():
                if name != "probe_auc":
                    row[name] = m(ones)

        rng = np.random.default_rng(args.seed)
        boots: dict[str, list[float]] = {name: [] for name in metrics}
        diff_boot: list[float] = []
        for _ in range(args.n_boot):
            counts = rng.multinomial(len(uniq_inst), np.full(len(uniq_inst), 1 / len(uniq_inst)))
            tw = counts[inst_id].astype(np.float64)
            vals = {name: m(tw) for name, m in metrics.items()}
            for name, v in vals.items():
                boots[name].append(v)
            if "persistence_auc" in vals:
                diff_boot.append(vals["probe_auc_persist_subset"] - vals["persistence_auc"])
        row["ci95"] = {name: ci(v) for name, v in boots.items()}
        if diff_boot:
            row["probe_minus_persistence"] = row["probe_auc_persist_subset"] - row["persistence_auc"]
            row["ci95"]["probe_minus_persistence"] = ci(diff_boot)
        print(json.dumps({kk: vv for kk, vv in row.items() if kk != "ci95"}), flush=True)
        rows.append(row)
        del data

    out = {"bench": args.bench, "pooled": args.pooled, "probe": args.probe, "max_k": args.max_k,
           "n_boot": args.n_boot, "layer_rule": "best test AUC per k", "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
