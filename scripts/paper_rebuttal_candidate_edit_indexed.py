"""Brief EI: summary of scripts/rebuttal_edit_indexed.py (paper_rebuttal_candidate/edit_indexed/json/*.json).

Writes paper_rebuttal_candidate/edit_indexed/summary.json and prints, per m and population (A, B, C), over the
12 settings (4 runs x 3 properties):
  - probe AUC range and every baseline's AUC range;
  - settings where the probe beats each baseline (paired 95% task-bootstrap CI of probe - baseline > 0);
  - settings where the probe beats ALL baselines available at step t (1-6: persistence [A only], k0, pos_causal,
    pos_rel, edits_so_far, observable, traj, prompt_hs), i.e. CI > 0 against the strongest of them by point AUC;
    also the causal variant without pos_rel (which uses the trajectory length);
  - settings where any baseline (oracle y_T excluded) is significantly better (CI < 0);
  - the same counts over adequately powered settings (>= 20 test tasks with a positive token).
Usage: uvx python@3.12 scripts/paper_rebuttal_candidate_edit_indexed.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("paper_rebuttal_candidate/edit_indexed")
RUNS = ["laguna_verified", "qwen_verified", "laguna_pro", "qwen_pro"]
PROBES = ["currently_correct", "currently_reduces_failing", "currently_has_regressions"]
BASE = ["persistence", "k0", "pos_causal", "pos_rel", "edits_so_far", "observable", "traj", "prompt_hs"]
CAUSAL = [b for b in BASE if b != "pos_rel"]


def sig(ci) -> int:
    if ci is None or ci[0] is None:
        return 0
    return 1 if ci[0] > 0 else (-1 if ci[1] < 0 else 0)


def ok(v) -> bool:
    return v is not None and v == v


def main() -> None:
    cells, agg = [], {}
    for run in RUNS:
        for probe in PROBES:
            d = json.loads((OUT / "json" / f"ei_{run}_{probe}.json").read_text())
            for r in d["rows"]:
                a, c, m = r["auc"], r["ci95"], r["m"]
                for pop in "ABC":
                    n = r["counts"][pop]
                    cell = {"run": run, "probe": probe, "m": m, "pop": pop, "layer": r["layer"],
                            "weight_decay": r["fit"]["hparams_selected"][str(r["layer"])]["weight_decay"],
                            "n": n["n"], "n_tasks_pos": n["n_tasks_pos"], "pos_rate": n["pos_rate"],
                            "powered": not n["underpowered"], "share_target_edits_change_label": n["share_target_edits_change_label"],
                            "median_turns_to_e_m": n["median_turns_to_e_m"], "auc": {}, "diff": {}, "diff_ci95": {}, "sig": {}}
                    for s in ["probe", "yT"] + BASE:
                        if ok(a.get(f"{s}|{pop}")):
                            cell["auc"][s] = a[f"{s}|{pop}"]
                    for b in BASE + ["yT"]:
                        k = f"probe-{b}|{pop}"
                        if k in a and ok(a[k]):
                            cell["diff"][b], cell["diff_ci95"][b], cell["sig"][b] = a[k], c[k], sig(c[k])
                    for name, ctrl in (("all", BASE), ("causal", CAUSAL)):
                        cand = [(cell["auc"][b], b) for b in ctrl if b in cell["auc"]]
                        best = max(cand)[1] if cand else None
                        cell[f"strongest_{name}"] = best
                        cell[f"beats_{name}"] = best is not None and cell["sig"].get(best) == 1
                    cell["any_baseline_better"] = any(cell["sig"].get(b) == -1 for b in BASE)
                    cells.append(cell)
    for m in (1, 2, 3):
        for pop in "ABC":
            cs = [x for x in cells if x["m"] == m and x["pop"] == pop and "probe" in x["auc"]]
            e = {"n_settings": len(cs), "n_powered": sum(x["powered"] for x in cs)}
            e["probe_auc_range"] = [min(x["auc"]["probe"] for x in cs), max(x["auc"]["probe"] for x in cs)]
            e["baseline_auc_range"] = {b: [min(x["auc"][b] for x in cs if b in x["auc"]),
                                           max(x["auc"][b] for x in cs if b in x["auc"])]
                                       for b in BASE + ["yT"] if any(b in x["auc"] for x in cs)}
            for scope, sel in (("all", cs), ("powered", [x for x in cs if x["powered"]])):
                e[scope] = {"beats_each": {b: sum(x["sig"].get(b) == 1 for x in sel) for b in BASE},
                            "worse_than_each": {b: sum(x["sig"].get(b) == -1 for x in sel) for b in BASE},
                            "beats_all_1_6": sum(x["beats_all"] for x in sel),
                            "beats_all_causal": sum(x["beats_causal"] for x in sel),
                            "any_baseline_better": sum(x["any_baseline_better"] for x in sel), "n": len(sel)}
            e["strongest_count"] = {}
            for x in cs:
                e["strongest_count"][x["strongest_all"]] = e["strongest_count"].get(x["strongest_all"], 0) + 1
            e["median_turns_to_e_m_range"] = [min(x["median_turns_to_e_m"] for x in cs), max(x["median_turns_to_e_m"] for x in cs)]
            e["share_target_edits_change_label_range"] = [min(x["share_target_edits_change_label"] for x in cs),
                                                          max(x["share_target_edits_change_label"] for x in cs)]
            agg[f"m{m}|{pop}"] = e
    (OUT / "summary.json").write_text(json.dumps({"headline": agg, "cells": cells}, indent=1))
    for k, e in agg.items():
        a, p = e["all"], e["powered"]
        print(f"{k}: probe {e['probe_auc_range'][0]:.3f}-{e['probe_auc_range'][1]:.3f} | beats all(1-6) {a['beats_all_1_6']}/{a['n']} "
              f"[{p['beats_all_1_6']}/{p['n']}] causal {a['beats_all_causal']} [{p['beats_all_causal']}] | any baseline better "
              f"{a['any_baseline_better']} [{p['any_baseline_better']}] | each {a['beats_each']} | worse {a['worse_than_each']} | "
              f"strongest {e['strongest_count']} | gap {e['median_turns_to_e_m_range']} chg {[round(v, 2) for v in e['share_target_edits_change_label_range']]}")
        print("    ranges", {b: [round(v, 3) for v in r] for b, r in e["baseline_auc_range"].items()})


if __name__ == "__main__":
    main()
