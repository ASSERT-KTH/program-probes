"""Brief E12H: summary of scripts/rebuttal_per_traj_horizon.py (paper_rebuttal_candidate/per_traj_horizon/json/*.json).

For S0 and S2, unweighted ("u") and per-trajectory weighted ("w"), over the 72 settings (4 runs x 3 properties x
k in {1, 2, 3, 5, 10, 15}):
  beats_each          settings where the paired CI of probe - baseline is > 0, for k0, traj, pos_rel, pos_causal
                      (and prompt_hs)
  gap4 / gap5         probe - max point AUC of {k0, traj, pos_causal, pos_rel} (+ prompt_hs): median, IQR
  strict4 / strict5   settings where the CI is > 0 against every baseline of the set
  oracle_higher       settings where oracle_yT has a higher point AUC than the probe
  any_sig_better4/5   settings where any baseline of the set is significantly better (CI < 0)
Settings with an undefined AUC (single-class subset) are left out of that number; n_defined says how many remain.
Writes paper_rebuttal_candidate/per_traj_horizon/summary.json (and the per-setting cells) and prints the table.
Usage: uvx python@3.12 scripts/paper_rebuttal_candidate_per_traj_horizon.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path("paper_rebuttal_candidate/per_traj_horizon")
RUNS = ["laguna_verified", "qwen_verified", "laguna_pro", "qwen_pro"]
PROBES = ["currently_correct", "currently_reduces_failing", "currently_has_regressions"]
B4 = ["k0", "traj", "pos_rel", "pos_causal"]
B5 = B4 + ["prompt_hs"]


def sig(c) -> int:
    if c is None or c[0] is None or c[0] != c[0]:
        return 0
    return 1 if c[0] > 0 else (-1 if c[1] < 0 else 0)


def ok(v) -> bool:
    return v is not None and v == v


def quart(xs: list[float]) -> dict:
    if not xs:
        return {"median": None, "q25": None, "q75": None, "n": 0}
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return {"median": statistics.median(xs), "q25": q[0], "q75": q[2], "n": len(xs)}


def main() -> None:
    cells = []
    for run in RUNS:
        for probe in PROBES:
            d = json.loads((OUT / "json" / f"pth_{run}_{probe}.json").read_text())
            for r in d["rows"]:
                for st in ("S0", "S2"):
                    for wt, pre in (("u", ""), ("w", "w:")):
                        a, c = r["auc"], r["ci95"]
                        cell = {"run": run, "probe": probe, "k": r["k"], "stratum": st, "weighting": wt,
                                "n": r["counts"][st]["n"], "n_tasks_pos": r["counts"][st]["n_tasks_pos"],
                                "n_trajectories": r["traj_counts"][st]["n_trajectories"],
                                "auc": {s: a[f"{pre}{s}|{st}"] for s in ["real_val", "oracle_yT"] + B5},
                                "sig": {b: sig(c[f"{pre}real_val-{b}|{st}"]) for b in B5}}
                        p = cell["auc"]["real_val"]
                        for name, bs in (("4", B4), ("5", B5)):
                            vals = [cell["auc"][b] for b in bs if ok(cell["auc"][b])]
                            cell[f"gap{name}"] = p - max(vals) if ok(p) and vals else None
                            cell[f"strict{name}"] = all(cell["sig"][b] == 1 for b in bs)
                            cell[f"any_sig_better{name}"] = any(cell["sig"][b] == -1 for b in bs)
                        cell["oracle_higher"] = ok(p) and ok(cell["auc"]["oracle_yT"]) and cell["auc"]["oracle_yT"] > p
                        cells.append(cell)
    summ = {}
    for st in ("S0", "S2"):
        for wt in ("u", "w"):
            cs = [x for x in cells if x["stratum"] == st and x["weighting"] == wt]
            summ[f"{st}|{wt}"] = {
                "n_settings": len(cs), "n_defined": sum(ok(x["auc"]["real_val"]) for x in cs),
                "beats_each": {b: sum(x["sig"][b] == 1 for x in cs) for b in B5},
                "gap4": quart([x["gap4"] for x in cs if x["gap4"] is not None]),
                "strict4": sum(x["strict4"] for x in cs),
                "gap5": quart([x["gap5"] for x in cs if x["gap5"] is not None]),
                "strict5": sum(x["strict5"] for x in cs),
                "oracle_higher": sum(x["oracle_higher"] for x in cs),
                "any_sig_better4": sum(x["any_sig_better4"] for x in cs),
                "any_sig_better5": sum(x["any_sig_better5"] for x in cs),
                "probe_auc_range": [min(x["auc"]["real_val"] for x in cs if ok(x["auc"]["real_val"])),
                                    max(x["auc"]["real_val"] for x in cs if ok(x["auc"]["real_val"]))]}
    (OUT / "summary.json").write_text(json.dumps({"summary": summ, "cells": cells}, indent=1))
    f = lambda q: f"{q['median']:+.3f} ({q['q25']:+.3f}, {q['q75']:+.3f})"  # noqa: E731
    for k, e in summ.items():
        print(f"{k}: n {e['n_defined']}/{e['n_settings']} | beats k0/traj/pos_rel/pos_causal/prompt_hs "
              f"{'/'.join(str(e['beats_each'][b]) for b in B5)} | gap4 {f(e['gap4'])} strict4 {e['strict4']} | "
              f"gap5 {f(e['gap5'])} strict5 {e['strict5']} | oracle higher {e['oracle_higher']} | "
              f"any sig better 4: {e['any_sig_better4']} 5: {e['any_sig_better5']} | probe {e['probe_auc_range']}")


if __name__ == "__main__":
    main()
