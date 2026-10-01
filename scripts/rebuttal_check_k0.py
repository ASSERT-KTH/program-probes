"""Check that a rebuilt k=0 cache matches the pre-fix snapshot.

y and sample_id must be identical to snapshots/<bench>_<run>_<probe>_k0.pt;
step_idx may change and target_step_idx must be present (== step_idx at k=0).
group_id must equal the snapshot's, or with --task-groups, the task id
(sample_id without the _runNN suffix), since splits are now by task.
Also compares the rebuilt edit_step_index.pt against the saved copy.

Usage:
    uv run python scripts/rebuttal_check_k0.py --bench swebench --run laguna_xs2_full \
        --probe currently_compiles currently_correct --output reports/k0_check_swebench_laguna.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--probe", nargs="+", required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-groups", action="store_true",
                        help="Expect group_id == task id instead of the snapshot's group_id.")
    args = parser.parse_args()

    out: dict = {"bench": args.bench, "run": args.run, "probes": {}}
    ok = True
    for probe in args.probe:
        snap = torch.load(f"snapshots/{args.bench}_{args.run}_{probe}_k0.pt", weights_only=False)
        new = torch.load(f"cache/{args.bench}/{args.run}/{probe}/layer_{args.layer}.pt",
                         weights_only=False, mmap=True)
        r = {
            "n_old": len(snap["y"]),
            "n_new": len(new["y"]),
            "y_identical": bool(len(snap["y"]) == len(new["y"]) and torch.equal(snap["y"], new["y"])),
            "sample_id_identical": list(snap["sample_id"]) == list(new["sample_id"]),
            "group_id_identical": list(snap["group_id"]) == list(new["group_id"]),
            "group_id_is_task": all(g == re.sub(r"_run\d+$", "", s) for s, g in zip(new["sample_id"], new["group_id"])),
            "has_target_step_idx": "target_step_idx" in new,
            "n_groups": len(set(new["group_id"])),
            "n_samples": len(set(new["sample_id"])),
        }
        if r["has_target_step_idx"]:
            r["target_eq_step_idx"] = bool(torch.equal(new["target_step_idx"], new["step_idx"]))
        if "step_idx_legacy" in snap and len(snap["step_idx_legacy"]) == len(new["step_idx"]):
            diff = new["step_idx"] != snap["step_idx_legacy"]
            r["step_idx_frac_changed"] = float(diff.float().mean())
            r["step_idx_mean_abs_change"] = float((new["step_idx"] - snap["step_idx_legacy"]).abs().float().mean())
        groups_ok = r["group_id_is_task"] if args.task_groups else r["group_id_identical"]
        ok &= r["y_identical"] and r["sample_id_identical"] and groups_ok
        out["probes"][probe] = r
        print(probe, r, flush=True)
        del new

    old_idx = torch.load(f"snapshots/{args.bench}_{args.run}_edit_step_index.pt", weights_only=False)
    new_idx = torch.load(f"cache/{args.bench}/{args.run}/edit_step_index.pt", weights_only=False)
    out["edit_step_index_identical"] = old_idx == new_idx
    out["all_identical"] = bool(ok)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print("ALL_IDENTICAL" if ok else "MISMATCH", "edit_index_identical:", out["edit_step_index_identical"])
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
