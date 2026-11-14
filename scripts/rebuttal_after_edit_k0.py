"""Brief AE0 (HaQJ-Q3, wokJ-1b, 1nZa-W2): k=0 probe AUC on the states right after an edit,
before anything has run on the new program version. Eval only, no training.

States: for every real edit (cmd_idx N >= 0 in the labeler's edits, i.e. a command that changed the
cumulative diff and applied), the tokens of assistant turn N+1, the first turn whose label reflects the
new version (edit_step_index.pt, as in the E5 *_after_edit runs). These tokens' context ends with the
edit command and its own output. Extraction covers assistant tokens only, so turn N+1's context holds no
execution of the new version unless command N itself ran something.
Filter (scripts/after_edit_cmd_filter.py) on command N's string:
  primary  drop if command N runs anything beyond the edit (interpreter, test runner, build / lint /
           type-check, package manager, script), except a Python editor script (stdlib file I/O only);
  strict   drop if command N invokes anything (Python editor scripts dropped too);
  none     E5's selection (no command filter).
Probe: the rerun's k=0 pooled weights (results/<bench>/<pooled>/<probe>/weights.pt), on the k=0 pooled
cache's task-level test split (seed 42), at the layer with the best all-state test AUC (tab:auc_roc).
Also every probed layer (best after-edit layer as a side note).
AUC pooled over the selected test tokens; 95% task-level bootstrap CI (1000 resamples of test tasks).
Checks: the all-state test AUC recomputed at the best layer equals the stored test_auc; the unfiltered
selection's test tokens equal the stored *_after_edit eval's n_test summed over bins.

Usage: uv run python scripts/rebuttal_after_edit_k0.py --run laguna_verified \
           --output paper_rebuttal_candidate/after_edit_k0/json/ae0_laguna_verified.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.probe import _build_probe, _split_groups  # noqa: E402
from after_edit_cmd_filter import classify  # noqa: E402

PROBES = ["currently_compiles", "currently_correct", "currently_reduces_failing", "currently_has_regressions"]
LAYERS = [0, 10, 20, 30, 39]
RUNS = {  # bench, cache / generations dir, pooled run id
    "laguna_verified": ("swebench", "laguna_xs2_full", "laguna_xs2_full_pooled"),
    "laguna_pro": ("swebench_pro", "laguna_xs2_full", "laguna_xs2_pro_full_pooled"),
    "qwen_verified": ("swebench", "qwen36_35b_a3b_full", "qwen36_35b_a3b_full_pooled"),
    "qwen_pro": ("swebench_pro", "qwen36_35b_a3b_full", "qwen36_35b_a3b_pro_full_pooled"),
}
VARIANTS = ["primary", "strict", "none"]
N_BOOT = 1000
T0 = time.time()


def log(*a) -> None:
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def short(c: str) -> str:
    return c if len(c) <= 500 else c[:300] + " [...] " + c[-200:]


def classify_edits(bench: str, cache: str, index: dict) -> tuple[dict, dict]:
    """(sid, turn N+1) -> classification of command N; plus per-run alignment checks."""
    out, chk = {}, collections.Counter()
    for sid, turns in index.items():
        if not turns:
            continue
        g = json.loads(Path(f"generations/{bench}/{cache}/{sid}.json").read_text())
        ch = g["command_history"]
        n_asst = sum(m["role"] == "assistant" for m in g["messages"])
        chk["trajectories_with_edits"] += 1
        chk["command_history_longer_than_assistant_turns"] += len(ch) > n_asst
        for t in turns:
            r = classify(ch[t - 1]["command"])
            r["cmd"] = ch[t - 1]["command"]
            r["turn_exists"] = t < n_asst
            out[(sid, t)] = r
        del g
    return out, dict(chk)


def auc_ci(y: np.ndarray, p: np.ndarray, grp: np.ndarray, rng: np.random.Generator) -> dict:
    if len(y) == 0 or y.min() == y.max():
        return {"auc": None, "ci95": None, "n_boot_valid": 0}
    auc = float(roc_auc_score(y, p))
    groups, inv = np.unique(grp, return_inverse=True)
    vals = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))[inv]
        m = w > 0
        if y[m].min() == y[m].max():
            continue
        vals.append(roc_auc_score(y[m], p[m], sample_weight=w[m]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"auc": auc, "ci95": [float(lo), float(hi)], "n_boot_valid": len(vals)}


def stored(path: Path) -> dict:
    res = torch.load(path, weights_only=False)
    return {int(L): {"test_auc": (float(np.average([r.test_auc for r in rs], weights=[r.n_test for r in rs]))
                                  if rs else None),
                     "n_test": sum(int(r.n_test) for r in rs), "n_pos_test": sum(int(r.n_pos_test) for r in rs),
                     "n_bins": len(rs)} for L, rs in res.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(RUNS))
    ap.add_argument("--output", required=True)
    ap.add_argument("--edits-dump", help="optional JSON with every edit's classification (audit)")
    args = ap.parse_args()
    bench, cache, pooled = RUNS[args.run]
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(0)
    random.seed(0)

    index = torch.load(f"cache/{bench}/{cache}/edit_step_index.pt", weights_only=False)
    cls, align = classify_edits(bench, cache, index)
    log("classified", len(cls), "edits", align)
    keep = {"primary": {k for k, r in cls.items() if not r["runs_project"]},
            "strict": {k for k, r in cls.items() if not r["strict"]},
            "none": set(cls)}
    out: dict = {"run": args.run, "bench": bench, "cache": cache, "pooled": pooled, "alignment": align,
                 "filter": {}, "probes": {}}

    split_of_sid = None
    for probe in PROBES:
        st = stored(Path(f"results/{bench}/{pooled}/{probe}/results.pt"))
        st_ae = stored(Path(f"results/{bench}/{pooled}_after_edit/{probe}/results.pt"))
        best = max((L for L in LAYERS if st[L]["test_auc"] is not None), key=lambda L: st[L]["test_auc"])
        weights = torch.load(f"results/{bench}/{pooled}/{probe}/weights.pt", weights_only=False)
        pr: dict = {"best_layer_all_states": best + 1, "layers": {}}
        for L in LAYERS:
            data = torch.load(f"cache/{bench}/{cache}/{probe}/layer_{L}.pt", mmap=True, weights_only=False)
            y = data["y"].numpy()
            step = data.get("target_step_idx", data["step_idx"]).numpy().astype(np.int64)
            sids, gids = data["sample_id"], data["group_id"]
            _, _, test_groups = _split_groups(gids, 42)
            sid_names, sid_codes = np.unique(np.asarray(sids, dtype=object).astype(str), return_inverse=True)
            gid_names, gid_codes = np.unique(np.asarray(gids, dtype=object).astype(str), return_inverse=True)
            is_test = np.isin(gid_codes, [i for i, g in enumerate(gid_names) if g in test_groups])
            code_of = {s: i for i, s in enumerate(sid_names)}
            key = sid_codes.astype(np.int64) * 100_000 + step
            ae = {v: np.isin(key, np.fromiter((code_of[s] * 100_000 + t for s, t in keep[v] if s in code_of),
                                              dtype=np.int64)) for v in VARIANTS}
            valid = y >= 0
            base = valid & is_test & ae["none"]
            rows = np.nonzero(base)[0]
            b0 = weights[L][0]
            model = _build_probe("linear", data["H"].shape[1])
            model.load_state_dict(b0["state_dict"])
            model.eval()
            p_rows = torch.softmax(model(data["H"][torch.from_numpy(rows)].float() - b0["mean"]), dim=1)[:, 1].numpy()
            p = np.full(len(y), np.nan)
            p[rows] = p_rows
            # Label at turn N (before the edit), for the label-changed subset.
            uk, first = np.unique(key[valid], return_index=True)
            yv = y[valid]
            pos = np.searchsorted(uk, key[rows] - 1)
            hit = (pos < len(uk)) & (uk[np.minimum(pos, len(uk) - 1)] == key[rows] - 1)
            prev_y = np.where(hit, yv[first[np.minimum(pos, len(uk) - 1)]], -1)
            changed = np.zeros(len(y), bool)
            changed[rows] = (prev_y >= 0) & (prev_y != y[rows])
            lay: dict = {"all_states_stored_test_auc": st[L]["test_auc"], "all_states_stored_n_test": st[L]["n_test"],
                         "e5_after_edit_stored": st_ae.get(L)}
            for v in VARIANTS:
                m = base & ae[v]
                ed = {(sid_codes[i], step[i]) for i in np.nonzero(m)[0].tolist()}
                lay[v] = {"n_edits": len(ed), "n_states": int(m.sum()), "n_test_tasks": int(len(np.unique(gid_codes[m]))),
                          "n_trajectories": int(len(np.unique(sid_codes[m]))),
                          "pos_rate": float(y[m].mean()) if m.any() else None,
                          "auc": float(roc_auc_score(y[m], p[m])) if m.any() and y[m].min() != y[m].max() else None}
                mc = m & changed
                lay[v]["label_changed_by_edit"] = {
                    "n_states": int(mc.sum()), "n_edits": len({(sid_codes[i], step[i]) for i in np.nonzero(mc)[0].tolist()}),
                    "pos_rate": float(y[mc].mean()) if mc.any() else None,
                    "auc": float(roc_auc_score(y[mc], p[mc])) if mc.any() and y[mc].min() != y[mc].max() else None}
            if lay["none"]["n_states"] != st_ae[L]["n_test"] or int(y[base].sum()) != st_ae[L]["n_pos_test"]:
                log(f"WARNING {probe} L{L}: unfiltered n={lay['none']['n_states']} pos={int(y[base].sum())} vs stored "
                    f"after_edit n={st_ae[L]['n_test']} pos={st_ae[L]['n_pos_test']} ({st_ae[L]['n_bins']} bins)")
            lay["check_unfiltered_equals_e5_n_test"] = lay["none"]["n_states"] == st_ae[L]["n_test"]
            pr["layers"][L + 1] = lay
            if L == best:
                # Recompute the all-state test AUC (tab:auc_roc) with these weights: must equal the stored one.
                trows = np.nonzero(valid & is_test)[0]
                pa = np.concatenate([
                    torch.softmax(model(data["H"][torch.from_numpy(trows[i:i + 100_000])].float() - b0["mean"]),
                                  dim=1)[:, 1].numpy() for i in range(0, len(trows), 100_000)])
                re_auc = float(roc_auc_score(y[trows], pa))
                if abs(re_auc - st[L]["test_auc"]) > 2e-4 or len(trows) != st[L]["n_test"]:
                    raise ValueError(f"{probe} L{L}: recomputed all-state AUC {re_auc} n={len(trows)} vs stored "
                                     f"{st[L]['test_auc']} n={st[L]['n_test']}")
                pr["all_states_recomputed_auc"] = re_auc
                for v in VARIANTS:
                    m = base & ae[v]
                    pr.setdefault("best_layer_ci", {})[v] = auc_ci(y[m], p[m], gid_codes[m], rng)
                if split_of_sid is None:
                    t_sid = np.zeros(len(sid_names), bool)
                    t_sid[sid_codes[is_test]] = True
                    split_of_sid = dict(zip(sid_names.tolist(), t_sid.tolist()))
            log(probe, "L", L + 1, {v: (lay[v]["n_states"], lay[v]["auc"] and round(lay[v]["auc"], 4)) for v in VARIANTS})
            del data
        ae_layers = {L: pr["layers"][L]["primary"]["auc"] for L in pr["layers"] if pr["layers"][L]["primary"]["auc"] is not None}
        bl = max(ae_layers, key=ae_layers.get)
        pr["best_layer_after_edit_primary"] = bl
        if bl != best + 1:
            # CI at the best after-edit layer (side note): reload that layer's selection.
            data = torch.load(f"cache/{bench}/{cache}/{probe}/layer_{bl - 1}.pt", mmap=True, weights_only=False)
            y = data["y"].numpy()
            step = data.get("target_step_idx", data["step_idx"]).numpy().astype(np.int64)
            sid_names, sid_codes = np.unique(np.asarray(data["sample_id"], dtype=object).astype(str), return_inverse=True)
            gid_names, gid_codes = np.unique(np.asarray(data["group_id"], dtype=object).astype(str), return_inverse=True)
            _, _, test_groups = _split_groups(data["group_id"], 42)
            is_test = np.isin(gid_codes, [i for i, g in enumerate(gid_names) if g in test_groups])
            code_of = {s: i for i, s in enumerate(sid_names)}
            key = sid_codes.astype(np.int64) * 100_000 + step
            m = (y >= 0) & is_test & np.isin(key, np.fromiter(
                (code_of[s] * 100_000 + t for s, t in keep["primary"] if s in code_of), dtype=np.int64))
            rows = np.nonzero(m)[0]
            b0 = weights[bl - 1][0]
            model = _build_probe("linear", data["H"].shape[1])
            model.load_state_dict(b0["state_dict"])
            pm = torch.softmax(model(data["H"][torch.from_numpy(rows)].float() - b0["mean"]), dim=1)[:, 1].numpy()
            pr["best_after_edit_layer_ci_primary"] = auc_ci(y[rows], pm, gid_codes[rows], rng)
            del data
        out["probes"][probe] = pr

    # Filter counts, all splits and test split (by the k=0 cache's task split).
    def counts(keys) -> dict:
        rs = [cls[k] for k in keys]
        c = {"edits": len(rs), "edits_with_next_turn": sum(r["turn_exists"] for r in rs),
             "dropped_primary": sum(r["runs_project"] for r in rs), "dropped_strict": sum(r["strict"] for r in rs),
             "python_editor_scripts": sum(r["python_editor"] for r in rs),
             "dropped_primary_by_reason": dict(collections.Counter("+".join(r["reasons"]) for r in rs if r["runs_project"]))}
        c["dropped_primary_share"] = c["dropped_primary"] / c["edits"] if c["edits"] else None
        c["dropped_strict_share"] = c["dropped_strict"] / c["edits"] if c["edits"] else None
        return c

    out["filter"]["all_splits"] = counts(list(cls))
    out["filter"]["test_split"] = counts([k for k in cls if split_of_sid.get(k[0], False)])
    ex = collections.defaultdict(list)
    for k, r in cls.items():
        cat = "kept" if not r["strict"] else ("python_editor_kept" if r["python_editor"] else "dropped:" + "+".join(r["reasons"]))
        ex[cat].append(k)
    out["filter"]["examples"] = {cat: [{"sample_id": s, "edit_turn": t, "cmd": short(cls[(s, t)]["cmd"])}
                                       for s, t in random.sample(v, min(8, len(v)))] for cat, v in sorted(ex.items())}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1))
    if args.edits_dump:
        Path(args.edits_dump).write_text(json.dumps(
            [{"sample_id": s, "edit_turn": t, **{k: v for k, v in r.items() if k != "cmd"}, "cmd": short(r["cmd"])}
             for (s, t), r in cls.items()]))
    log("[done]")
    os._exit(0)


if __name__ == "__main__":
    main()
