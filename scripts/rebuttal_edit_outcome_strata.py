"""Rebuttal E18: edit-turn and eventual-outcome confounds of the within-y_t=0 lookahead signal.

Eval-only; reuses scripts/rebuttal_phase_confound.py (same test tokens, same probes at the
same layers, same position baselines, same task-level bootstrap with paired differences).

strata mode, one job per (run, probe, window K); k in {2,3,5,10,15} (+25, 50 for K=50).
All strata are within y_t = 0 (y_t = y_original >= 0):
  S0  all y_t = 0 tokens (reproduces the E16 numbers; checked)
  S1  pre-edit: y_{t+1} = 0, i.e. the action of turn t does not change the label
      (y_{t+1} is the shift-1 label of the same token; token sets are checked to align)
  S2  outcome: y_T = 1, y_T = label at the trajectory's last assistant turn
      (turn_labels[probe][-1] in outputs/<bench>/<run>/<sample>_labels.pt)
  S3  S1 and S2: fixed eventual outcome, label-changing edit not in the current turn
  S4  y_T = 0 (counts; AUCs are reported but flagged when underpowered)
Scorers: real (val-best layer), k0, traj (trajectory shuffle per k), pos_causal, pos_rel,
and the oracle score y_T (defined where y_T >= 0; constant inside S2-S4).
Per stratum: tokens, positives, test tasks with >= 1 positive token (underpowered if < 20).

outcome mode, one job per run: how the last-turn label y_T of currently_correct relates to
the final-patch evaluation (trajectory metadata.outcome), and to the label of the last edit
(which differs from y_T when the last edit is issued in the last turn).

Usage:
    uv run python scripts/rebuttal_edit_outcome_strata.py strata --bench swebench_pro \
        --pooled qwen36_35b_a3b_pro_full_pooled --cache-prefix qwen36_35b_a3b_pro_full \
        --outputs-dir outputs/swebench_pro/qwen36_35b_a3b_full --probe currently_correct --max-k 15 \
        --output reports/rerun/phase/e18_...json
    uv run python scripts/rebuttal_edit_outcome_strata.py outcome --bench swebench_pro \
        --outputs-dir outputs/swebench_pro/qwen36_35b_a3b_full --gen-dir generations/swebench_pro/qwen36_35b_a3b_full \
        --label-dir labels/swebench_pro/qwen36_35b_a3b_full --output reports/rerun/phase/e18_outcome_...json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuttal_phase_confound import bootstrap, log, lookahead_cell, lookahead_context  # noqa: E402
from rebuttal_persistence_analysis import Metric  # noqa: E402

KS = {15: [2, 3, 5, 10, 15], 50: [2, 3, 5, 10, 15, 25, 50]}
CONTROLS = ["k0", "traj", "pos_causal", "pos_rel"]
MIN_TASKS = 20


def task(sid: str) -> str:
    return re.sub(r"_run\d+$", "", sid)


def last_turn_label(outputs_dir: Path, sid: str, probe: str) -> int:
    tl = torch.load(outputs_dir / f"{sid}_labels.pt", weights_only=False)["turn_labels"][probe]
    v = tl[-1] if tl else None
    return -1 if v is None else int(bool(v))


def run_strata(args) -> dict:
    K = args.max_k
    ks = [k for k in (args.ks or KS[K]) if k <= K]
    ctx = lookahead_context(args, ks, extra_label_ks=(1,))
    sids, y1_all = ctx["sids"], ctx["ylab"][1]
    test_sids = sorted({sids[i] for i in ctx["test_idx"]})
    yT_of = {s: last_turn_label(Path(args.outputs_dir), s, args.probe) for s in test_sids}
    log("y_T loaded for", len(yT_of), "test trajectories;", sum(v == 1 for v in yT_of.values()), "with y_T = 1")
    e16 = None
    e16_path = Path("reports/rerun/phase") / f"lookahead_{args.bench}_{args.pooled}_{args.probe}_max{K}.json"
    if e16_path.exists():
        e16 = {r["k"]: r for r in json.loads(e16_path.read_text())["rows"]}

    rows = []
    for k in ks:
        cell = lookahead_cell(ctx, k, args.seed)
        y, yo, tsids, sc = cell["y"], cell["yo"], cell["tsids"], dict(cell["sc"])
        sc.pop("real_test")
        y1 = y1_all[cell["tok"]]
        yT = np.array([yT_of[s] for s in tsids])
        sc["oracle_yT"] = yT.astype(np.float64)
        t0 = (yo == 0)
        strata = {"S0": t0, "S1": t0 & (y1 == 0), "S2": t0 & (yT == 1), "S3": t0 & (y1 == 0) & (yT == 1), "S4": t0 & (yT == 0)}
        tasks = np.array([task(s) for s in tsids])
        counts = {}
        for st, m in strata.items():
            pos = m & (y == 1)
            counts[st] = {"n": int(m.sum()), "n_pos": int(pos.sum()), "n_tasks": int(len(set(tasks[m]))),
                          "n_tasks_pos": int(len(set(tasks[pos]))),
                          "underpowered": bool(len(set(tasks[pos])) < MIN_TASKS)}
        metrics = {}
        for st, m in strata.items():
            for s, v in sc.items():
                mm = m & (yT >= 0) if s == "oracle_yT" else m
                metrics[f"{s}|{st}"] = Metric(v, y, mm)
        diffs = {f"real_val-{o}|{st}": (f"real_val|{st}", f"{o}|{st}") for o in CONTROLS if o in sc for st in strata}
        point, cis = bootstrap(metrics, diffs, tsids, args.n_boot, args.seed)
        strongest = {}
        for st in strata:
            cand = [(point[f"{o}|{st}"], o) for o in CONTROLS if o in sc and point[f"{o}|{st}"] == point[f"{o}|{st}"]]
            if cand:
                strongest[st] = max(cand)[1]
        if e16 is not None and k in e16:
            ref = e16[k]["auc"]
            for s in ("real_val", "k0", "traj", "pos_causal", "pos_rel"):
                tol = 1e-3 if s.startswith("pos_") else 1e-6  # position baselines are refit
                if f"{s}|yt0" in ref and abs(point[f"{s}|S0"] - ref[f"{s}|yt0"]) > tol:
                    raise ValueError(f"k={k} {s}: S0 AUC {point[f'{s}|S0']:.6f} != E16 {ref[f'{s}|yt0']:.6f}")
        rows.append({"k": k, "layers": ctx["layers"][k], "counts": counts, "strongest_control": strongest,
                     "n_yT_missing_in_S0": int((strata["S0"] & (yT < 0)).sum()), "auc": point, "ci95": cis})
        log(f"k={k}", json.dumps(counts), json.dumps({n: round(v, 3) for n, v in point.items() if v == v and n.startswith(("real_val|", "oracle"))}))
    return {"mode": "strata", "bench": args.bench, "pooled": args.pooled, "probe": args.probe, "max_k": K,
            "n_boot": args.n_boot, "min_tasks": MIN_TASKS, "rows": rows}


def run_outcome(args) -> dict:
    """currently_correct: last-turn label vs final-patch evaluation vs last-edit label."""
    out_dir, gen_dir, lab_dir = Path(args.outputs_dir), Path(args.gen_dir), Path(args.label_dir)
    cm_turn: dict[str, int] = {}
    cm_edit: dict[str, int] = {}
    n_last_edit_in_last_turn = n_files = 0
    examples = []
    for f in sorted(out_dir.glob("*_labels.pt")):
        sid = f.stem.removesuffix("_labels")
        gpath = gen_dir / f"{sid}.json"
        if not gpath.exists():
            continue
        n_files += 1
        tl = torch.load(f, weights_only=False)["turn_labels"].get("currently_correct") or []
        yT = "none" if not tl or tl[-1] is None else str(int(bool(tl[-1])))
        meta = json.loads(gpath.read_text())
        outcome = str(int(bool(meta.get("metadata", {}).get("outcome", False))))
        n_turns = sum(1 for m in meta["messages"] if m["role"] == "assistant")
        del meta
        edits = []
        lpath = lab_dir / f"{sid}_labels.json"
        if lpath.exists():
            edits = sorted(json.loads(lpath.read_text()).get("edits", []), key=lambda e: e["cmd_idx"])
        last_edit = "none"
        if edits:
            tr = edits[-1].get("test_results")
            if tr is not None:
                last_edit = str(int(bool(tr["resolved"] if "resolved" in tr else len(tr.get("failed", [])) == 0)))
            n_last_edit_in_last_turn += edits[-1]["cmd_idx"] + 1 >= n_turns
        cm_turn[f"yT={yT},outcome={outcome}"] = cm_turn.get(f"yT={yT},outcome={outcome}", 0) + 1
        cm_edit[f"last_edit={last_edit},outcome={outcome}"] = cm_edit.get(f"last_edit={last_edit},outcome={outcome}", 0) + 1
        if yT in ("0", "1") and yT != outcome and len(examples) < 10:
            examples.append({"sample_id": sid, "yT": yT, "outcome": outcome, "last_edit": last_edit,
                             "n_turns": n_turns, "last_edit_cmd_idx": edits[-1]["cmd_idx"] if edits else None})

    def agree(cm: dict, a: str) -> dict:
        tot = sum(v for kk, v in cm.items() if f"{a}=none" not in kk)
        same = sum(v for kk, v in cm.items() if kk in (f"{a}=0,outcome=0", f"{a}=1,outcome=1"))
        return {"n_defined": tot, "agree": same, "agreement": same / tot if tot else None}

    return {"mode": "outcome", "bench": args.bench, "outputs_dir": str(out_dir), "n_trajectories": n_files,
            "yT_vs_outcome": cm_turn, "yT_agreement": agree(cm_turn, "yT"),
            "last_edit_vs_outcome": cm_edit, "last_edit_agreement": agree(cm_edit, "last_edit"),
            "n_last_edit_in_last_turn": n_last_edit_in_last_turn, "disagreement_examples": examples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["strata", "outcome"])
    ap.add_argument("--bench", required=True)
    ap.add_argument("--pooled")
    ap.add_argument("--cache-prefix")
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--gen-dir")
    ap.add_argument("--label-dir")
    ap.add_argument("--probe", default="currently_correct")
    ap.add_argument("--max-k", type=int, default=15)
    ap.add_argument("--ks", type=int, nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = run_strata(args) if args.mode == "strata" else run_outcome(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    log("wrote", args.output)


if __name__ == "__main__":
    main()
