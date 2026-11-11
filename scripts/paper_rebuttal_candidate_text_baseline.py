"""Brief E9: summary of scripts/rebuttal_text_baseline.py, combined with brief PO for the Table 1 recounts.

Reads paper_rebuttal_candidate/text_baseline/json/*.json (and prompt_only/json/horizon_*.json) and writes
paper_rebuttal_candidate/text_baseline/summary.json:
  k0          16 cells: k=0 probe (test-best layer) vs text_a1, AUCs with CIs and the CI of the difference
  part_b      16 cells: probe and text_a1 AUC [CI] on test tokens whose current program version has not yet /
              already been run (primary and strict command rule), share of test tokens in each subset
  horizon_72  within y_t = 0, window 15: text_a1 AUC and probe - text_a1 significance per setting
  table1      E18 strata, k >= 2, 60 settings: "beats all baselines" / "beats baselines available at step t"
              with the old controls, + PO (prompt_hs), + E9 (text_a1), + both. Text and prompt are available
              at step t. The paired bootstrap of PO and E9 uses the same tokens and resamples (seed 42); the
              shared old-control AUCs and CIs are checked to be identical in both files.
Usage: uvx python@3.12 scripts/paper_rebuttal_candidate_text_baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("paper_rebuttal_candidate/text_baseline")
PO = Path("paper_rebuttal_candidate/prompt_only/json")
RUNS = ["laguna_verified", "qwen_verified", "laguna_pro", "qwen_pro"]
H_PROBES = ["currently_correct", "currently_reduces_failing", "currently_has_regressions"]
K0_PROBES = ["currently_compiles"] + H_PROBES
OLD_ALL, OLD_CAUSAL = ["k0", "traj", "pos_causal", "pos_rel"], ["k0", "traj", "pos_causal"]
STRATA = ["S0", "S1", "S2", "S3", "S4"]
SETS = {"old": [], "po": ["prompt_hs"], "e9": ["text_a1"], "both": ["prompt_hs", "text_a1"]}


def sig(ci: list) -> int:
    if ci is None or ci[0] is None or ci[0] != ci[0]:
        return 0  # undefined (single-class subset)
    return 1 if ci[0] > 0 else (-1 if ci[1] < 0 else 0)


def main() -> None:
    k0, part_b = [], []
    for run in RUNS:
        for probe in K0_PROBES:
            d = json.loads((OUT / "json" / f"k0_{run}_{probe}.json").read_text())
            a, c, n = d["auc"], d["ci95"], d["counts"]
            k0.append({"run": run, "probe": probe, "layer": d["layer"], "real": a["real_testbest|all"],
                       "real_ci95": c["real_testbest|all"], "text_a1": a["text_a1|all"], "text_a1_ci95": c["text_a1|all"],
                       "real-text_a1": a["real-text_a1|all"], "real-text_a1_ci95": c["real-text_a1|all"],
                       "C": d["fit"]["text_a1"].get("C"), "C_at_grid_edge": d["fit"]["text_a1"].get("C_at_grid_edge")})
            row = {"run": run, "probe": probe}
            for s in ("not_yet_run", "already_run", "not_yet_run_strict", "already_run_strict", "not_yet_run_post_edit",
                      "already_run_post_edit", "not_yet_run_strict_post_edit", "already_run_strict_post_edit", "pre_first_edit"):
                row[s] = {"share": n[s]["share"], "n": n[s]["n"], "pos_rate": n[s]["pos_rate"], "n_tasks": n[s]["n_tasks"],
                          "real": a[f"real_testbest|{s}"], "real_ci95": c[f"real_testbest|{s}"],
                          "text_a1": a[f"text_a1|{s}"], "text_a1_ci95": c[f"text_a1|{s}"],
                          "real-text_a1": a[f"real-text_a1|{s}"], "real-text_a1_ci95": c[f"real-text_a1|{s}"]}
            for sfx in ("", "_strict", "_post_edit", "_strict_post_edit"):
                row[f"real_already-not_yet{sfx}"] = a[f"real|already-not_yet{sfx}"]
                row[f"real_already-not_yet{sfx}_ci95"] = c[f"real|already-not_yet{sfx}"]
            part_b.append(row)

    cells, t1 = [], {}
    for run in RUNS:
        for probe in H_PROBES:
            e9 = {r["k"]: r for r in json.loads((OUT / "json" / f"horizon_{run}_{probe}.json").read_text())["rows"]}
            po = {r["k"]: r for r in json.loads((PO / f"horizon_{run}_{probe}.json").read_text())["rows"]}
            for k, r in e9.items():
                p = po[k]
                for key in r["auc"]:
                    if key in p["auc"] and r["auc"][key] == r["auc"][key] and abs(r["auc"][key] - p["auc"][key]) > 1e-3:
                        raise ValueError(f"{run} {probe} k={k} {key}: E9 {r['auc'][key]} != PO {p['auc'][key]}")
                a = {**p["auc"], **r["auc"]}
                c = {**p["ci95"], **r["ci95"]}
                cells.append({"run": run, "probe": probe, "k": k, "real": a["real_val|S0"], "text_a1": a["text_a1|S0"],
                              "prompt_hs": a["prompt_hs|S0"], "real-text_a1": a["real_val-text_a1|S0"],
                              "real-text_a1_ci95": c["real_val-text_a1|S0"], "C": r["fit"]["text_a1"].get("C"),
                              "C_at_grid_edge": r["fit"]["text_a1"].get("C_at_grid_edge")})
                if k < 2:
                    continue
                for st in STRATA:
                    powered = not r["counts"][st]["underpowered"]
                    for sname, extra in SETS.items():
                        for kind, base in (("all", OLD_ALL), ("causal", OLD_CAUSAL)):
                            ctrl = base + extra
                            cand = [(a[f"{o}|{st}"], o) for o in ctrl if a.get(f"{o}|{st}") == a.get(f"{o}|{st}")]
                            s = max(cand)[1]
                            v = sig(c[f"real_val-{s}|{st}"])
                            e = t1.setdefault(st, {}).setdefault(f"{kind}_{sname}", {
                                "all": {"ci>0": 0, "ci<0": 0, "n": 0}, "powered": {"ci>0": 0, "ci<0": 0, "n": 0}})
                            for scope in (["all", "powered"] if powered else ["all"]):
                                e[scope]["n"] += 1
                                e[scope]["ci>0"] += v == 1
                                e[scope]["ci<0"] += v == -1
                    best = max((a[f"{o}|{st}"], o) for o in OLD_ALL + ["prompt_hs", "text_a1"] if a.get(f"{o}|{st}") == a.get(f"{o}|{st}"))[1]
                    t1[st].setdefault("strongest_control_count", {}).setdefault(best, 0)
                    t1[st]["strongest_control_count"][best] += 1
    s = [sig(x["real-text_a1_ci95"]) for x in cells]
    h72 = {"n": len(cells), "probe_better_ci>0": s.count(1), "text_better_ci<0": s.count(-1), "not_significant": s.count(0),
           "text_auc_range": [min(x["text_a1"] for x in cells), max(x["text_a1"] for x in cells)],
           "by_run": {run: {"ci>0": sum(sig(x["real-text_a1_ci95"]) == 1 for x in cells if x["run"] == run),
                            "ci<0": sum(sig(x["real-text_a1_ci95"]) == -1 for x in cells if x["run"] == run),
                            "text_auc_range": [min(x["text_a1"] for x in cells if x["run"] == run),
                                               max(x["text_a1"] for x in cells if x["run"] == run)]} for run in RUNS},
           "C_at_grid_edge": sum(bool(x["C_at_grid_edge"]) for x in cells)}
    summary = {"k0": k0, "part_b": part_b, "horizon_72": h72, "table1": t1, "horizon_cells": cells}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    print("HORIZON 72", json.dumps(h72))
    for st in STRATA:
        e = t1[st]
        line = " ".join(f"{kind}: " + " / ".join(f"{e[f'{kind}_{n}']['all']['ci>0']}({e[f'{kind}_{n}']['powered']['ci>0']}/{e[f'{kind}_{n}']['powered']['n']})"
                                                 for n in SETS) for kind in ("all", "causal"))
        print(st, "old/po/e9/both:", line, "| ci<0 both:", e["all_both"]["all"]["ci<0"], e["causal_both"]["all"]["ci<0"],
              "| strongest:", e["strongest_control_count"])
    for x in k0:
        print(f"K0 {x['run']:16s} {x['probe']:27s} L{x['layer']:2d} probe {x['real']:.3f} text {x['text_a1']:.3f} "
              f"diff {x['real-text_a1']:+.3f} [{x['real-text_a1_ci95'][0]:+.3f},{x['real-text_a1_ci95'][1]:+.3f}] C {x['C']}")
    for x in part_b:
        n, a = x["not_yet_run"], x["already_run"]
        print(f"B  {x['run']:16s} {x['probe']:27s} notyet {n['share']:.2f}: probe {n['real']:.3f} [{n['real_ci95'][0]:.2f},{n['real_ci95'][1]:.2f}] "
              f"text {n['text_a1']:.3f} [{n['text_a1_ci95'][0]:.2f},{n['text_a1_ci95'][1]:.2f}] | already {a['share']:.2f}: "
              f"probe {a['real']:.3f} [{a['real_ci95'][0]:.2f},{a['real_ci95'][1]:.2f}] text {a['text_a1']:.3f} "
              f"[{a['text_a1_ci95'][0]:.2f},{a['text_a1_ci95'][1]:.2f}]")




def part_b_counts() -> None:
    """Significance counts for part B (printed)."""
    s = json.loads((OUT / "summary.json").read_text())
    for sub in ("not_yet_run", "already_run", "not_yet_run_post_edit", "already_run_post_edit", "pre_first_edit",
                "not_yet_run_strict", "already_run_strict", "not_yet_run_strict_post_edit", "already_run_strict_post_edit"):
        xs = s["part_b"]
        ok = [x for x in xs if x[sub]["real"] == x[sub]["real"] and x[sub]["real"] is not None]
        print(f"{sub:30s} share {min(x[sub]['share'] for x in xs):.2f}-{max(x[sub]['share'] for x in xs):.2f} "
              f"defined in {len(ok)} cells; probe {min(x[sub]['real'] for x in ok):.3f}-{max(x[sub]['real'] for x in ok):.3f} "
              f"text {min(x[sub]['text_a1'] for x in ok):.3f}-{max(x[sub]['text_a1'] for x in ok):.3f} "
              f"probe-text CI>0 {sum(sig(x[sub]['real-text_a1_ci95']) == 1 for x in xs)} "
              f"CI<0 {sum(sig(x[sub]['real-text_a1_ci95']) == -1 for x in xs)}")
    for sfx in ("", "_strict", "_post_edit", "_strict_post_edit"):
        v = [sig(x[f"real_already-not_yet{sfx}_ci95"]) for x in s["part_b"]]
        print(f"probe already - not yet{sfx}: CI>0 {v.count(1)} CI<0 {v.count(-1)}; "
              f"range {min(x[f'real_already-not_yet{sfx}'] for x in s['part_b']):+.3f} to "
              f"{max(x[f'real_already-not_yet{sfx}'] for x in s['part_b']):+.3f}")


if __name__ == "__main__":
    main()
    part_b_counts()
