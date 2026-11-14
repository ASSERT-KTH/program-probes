"""Sanity checks for the persistence baseline (reviewer comment on Pro persistence AUC).

Part A, label files (outputs/<bench>/<run>/*_labels.pt), every dynamic probe:
  - len(labels[probe]) == len(step_idx)
  - labels[probe][t] == turn_labels[probe][step_idx[t]] for every extracted position t
    (y_original in the shift caches is labels[t]; the target y is turn_labels[step + k])
  - turn-level label dynamics per trajectory: number of label changes, reverts
    (a change that undoes the previous change, e.g. 0->1->0), and for each k the
    fraction of turn pairs (s, s+k), both labelled, with turn_label[s+k] != turn_label[s].

Part B, shift caches (cache/<bench>/<prefix>_shift{k}_max{K}/<probe>/layer_0.pt):
  - y_original and y agree with the k=0 cache of the same window, looked up by
    (sample_id, turn): y_original == y0[(sid, step_idx)], y == y0[(sid, target_step_idx)]
    wherever that turn has extracted tokens
  - group_id is the task id
  - on all tokens and on the test split: flip rate (tokens with y != y_original),
    fraction of trajectories with at least one flipped token, confusion matrix of
    persistence (predict y_original), TPR, TNR, balanced accuracy (= persistence AUC).

Usage (SLURM; see slurm/check_persistence_labels.sh):
    uv run python scripts/rebuttal_check_persistence_labels.py --output reports/rerun/persistence_label_check.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.probe import _split_groups

PROBES = ["currently_compiles", "currently_correct", "currently_has_regressions", "currently_reduces_failing"]
KS = [1, 2, 3, 5, 10, 15, 25, 50]
# (bench, outputs run dir, cache prefix, display name)
COMBOS = [
    ("swebench", "laguna_xs2_full", "laguna_xs2_full", "Laguna XS.2 / Verified"),
    ("swebench", "qwen36_35b_a3b_full", "qwen36_35b_a3b_full", "Qwen3.6-35B-A3B / Verified"),
    ("swebench_pro", "laguna_xs2_full", "laguna_xs2_pro_full", "Laguna XS.2 / Pro"),
    ("swebench_pro", "qwen36_35b_a3b_full", "qwen36_35b_a3b_pro_full", "Qwen3.6-35B-A3B / Pro"),
]


def _b(v) -> int:
    return -1 if v is None else int(bool(v))


def check_label_files(bench: str, run: str) -> dict:
    files = sorted(Path("outputs", bench, run).glob("*_labels.pt"))
    out: dict = {"n_files": len(files), "probes": {}}
    acc = {p: {"n_pos": 0, "n_mismatch": 0, "n_len_mismatch": 0, "mismatch_examples": [],
               "n_traj": 0, "n_traj_any_change": 0, "changes": Counter(), "reverts": 0, "n_changes": 0,
               "pairs": {k: [0, 0] for k in KS}, "n_turns_labelled": 0, "n_turns": 0}
           for p in PROBES}
    for f in files:
        d = torch.load(f, weights_only=False)
        pos_turns = d["step_idx"]
        for p in PROBES:
            lab, tl = d["labels"].get(p), d.get("turn_labels", {}).get(p)
            if not isinstance(lab, list) or tl is None:
                continue
            a = acc[p]
            if len(lab) != len(pos_turns):
                a["n_len_mismatch"] += 1
                continue
            for t, (l, s) in enumerate(zip(lab, pos_turns)):
                a["n_pos"] += 1
                if s >= len(tl) or _b(l) != _b(tl[s]):
                    a["n_mismatch"] += 1
                    if len(a["mismatch_examples"]) < 5:
                        a["mismatch_examples"].append([f.name, t, s, len(tl)])
            seq = [_b(v) for v in tl]
            a["n_traj"] += 1
            a["n_turns"] += len(seq)
            defined = [v for v in seq if v >= 0]
            a["n_turns_labelled"] += len(defined)
            ch = [i for i in range(1, len(defined)) if defined[i] != defined[i - 1]]
            a["changes"][min(len(ch), 10)] += 1
            a["n_changes"] += len(ch)
            a["n_traj_any_change"] += bool(ch)
            # a revert: two consecutive changes (so the label returns to its earlier value)
            a["reverts"] += max(len(ch) - 1, 0)
            for k in KS:
                for s in range(len(seq) - k):
                    if seq[s] >= 0 and seq[s + k] >= 0:
                        a["pairs"][k][0] += 1
                        a["pairs"][k][1] += seq[s] != seq[s + k]
    for p, a in acc.items():
        out["probes"][p] = {
            "n_positions": a["n_pos"], "n_label_vs_turn_label_mismatch": a["n_mismatch"],
            "n_files_len_mismatch": a["n_len_mismatch"], "mismatch_examples": a["mismatch_examples"],
            "n_traj": a["n_traj"], "frac_traj_any_label_change": a["n_traj_any_change"] / max(a["n_traj"], 1),
            "mean_label_changes_per_traj": a["n_changes"] / max(a["n_traj"], 1),
            "frac_changes_that_revert": a["reverts"] / max(a["n_changes"], 1),
            "hist_label_changes_per_traj": {str(k): v for k, v in sorted(a["changes"].items())},
            "frac_turns_labelled": a["n_turns_labelled"] / max(a["n_turns"], 1),
            "turn_pair_flip_rate": {str(k): (v[1] / v[0] if v[0] else None) for k, v in a["pairs"].items()},
        }
    return out


def confusion(yo: np.ndarray, y: np.ndarray) -> dict:
    ok = (yo >= 0) & (y >= 0)
    yo, y = yo[ok], y[ok]
    tp = int(((yo == 1) & (y == 1)).sum()); fn = int(((yo == 0) & (y == 1)).sum())
    fp = int(((yo == 1) & (y == 0)).sum()); tn = int(((yo == 0) & (y == 0)).sum())
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    return {"n": int(ok.sum()), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "flip_rate": (fn + fp) / max(int(ok.sum()), 1), "tpr": tpr, "tnr": tnr,
            "balanced_acc": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None}


def check_caches(bench: str, prefix: str, probe: str, max_k: int) -> dict:
    root = Path("cache", bench)
    d0 = torch.load(root / f"{prefix}_shift0_max{max_k}" / probe / "layer_0.pt", weights_only=False, mmap=True)
    sid0, y0, st0 = d0["sample_id"], d0["y"].numpy(), d0["step_idx"].numpy()
    turn_y: dict = {}
    n_inconsistent_within_turn = 0
    for s, t, v in zip(sid0, st0, y0):
        key = (s, int(t))
        if turn_y.setdefault(key, int(v)) != int(v):
            n_inconsistent_within_turn += 1
    out: dict = {"n_tokens_k0": len(y0), "n_inconsistent_within_turn_k0": n_inconsistent_within_turn, "ks": {}}
    del d0
    for k in [k for k in KS if k <= max_k]:
        d = torch.load(root / f"{prefix}_shift{k}_max{max_k}" / probe / "layer_0.pt", weights_only=False, mmap=True)
        y, yo = d["y"].numpy(), d["y_original"].numpy()
        st, tg = d["step_idx"].numpy(), d["target_step_idx"].numpy()
        sids, gids = d["sample_id"], d["group_id"]
        del d
        r: dict = {"n_tokens": len(y), "same_tokens_as_k0": bool(len(y) == len(y0) and list(sids) == list(sid0) and np.array_equal(st, st0)),
                   "target_minus_step_eq_k": bool(np.all(tg - st == k)),
                   "group_id_is_task": all(g == re.sub(r"_run\d+$", "", s) for s, g in zip(sids, gids))}
        n_orig_chk = n_orig_bad = n_tgt_chk = n_tgt_bad = 0
        for s, a, b, vo, v in zip(sids, st, tg, yo, y):
            ref = turn_y.get((s, int(a)))
            if ref is not None:
                n_orig_chk += 1; n_orig_bad += ref != int(vo)
            ref = turn_y.get((s, int(b)))
            if ref is not None:
                n_tgt_chk += 1; n_tgt_bad += ref != int(v)
        r.update({"y_original_checked": n_orig_chk, "y_original_mismatch": n_orig_bad,
                  "y_target_checked": n_tgt_chk, "y_target_mismatch": n_tgt_bad})
        _, _, test_groups = _split_groups(gids, 42)
        in_test = np.fromiter((g in test_groups for g in gids), dtype=bool, count=len(gids))
        valid = y >= 0
        sid_arr = np.asarray(sids)
        for name, m in (("all", valid), ("test", valid & in_test)):
            c = confusion(yo[m], y[m])
            flipped = m & (yo >= 0) & (y != yo)
            trajs = set(sid_arr[m & (yo >= 0)])
            c["n_traj"] = len(trajs)
            c["frac_traj_with_flip"] = len(set(sid_arr[flipped])) / max(len(trajs), 1)
            r[name] = c
        out["ks"][str(k)] = r
        print(bench, prefix, probe, max_k, k, json.dumps({kk: vv for kk, vv in r.items() if kk not in ("all",)}), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-k", type=int, nargs="+", default=[50])
    ap.add_argument("--only", nargs="*", default=None, help="Cache prefixes to check, e.g. laguna_xs2_pro_full")
    args = ap.parse_args()
    res: dict = {}
    for bench, run, prefix, name in COMBOS:
        if args.only and prefix not in args.only:
            continue
        key = f"{bench}/{prefix}"
        res[key] = {"name": name, "label_files": check_label_files(bench, run), "caches": {}}
        print(key, json.dumps(res[key]["label_files"]["probes"], default=str)[:2000], flush=True)
        for probe in PROBES:
            for K in args.max_k:
                res[key]["caches"][f"{probe}_max{K}"] = check_caches(bench, prefix, probe, K)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
