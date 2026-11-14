"""Brief PO: summary of scripts/rebuttal_prompt_only.py (paper_rebuttal_candidate/prompt_only/json/*.json).

Writes paper_rebuttal_candidate/prompt_only/summary.json and prints the headline counts:
  horizon_72   within y_t = 0 (S0), window 15, 4 runs x 3 probes x k in {1,2,3,5,10,15}: prompt-only AUC and
               whether the probe or the prompt-only predictor is significantly better (paired 95% task-bootstrap CI)
  table1       E18 strata, k >= 2 (60 settings per stratum): "beats all baselines" (real - strongest control
               by point AUC, CI > 0) and "beats baselines available at step t" (same over the causal controls:
               k=0 probe, traj-shuffle, causal position), old control set vs old + prompt-only (prompt_hs; and
               a variant with prompt_tfidf too). Prompt-only counts as available at step t. Counts over all 60
               settings and over the adequately powered ones (>= 20 test tasks with a positive token).
  k0           16 cells: k=0 probe at its test-best layer vs prompt-only, CI of the difference.
Usage: uvx python@3.12 scripts/paper_rebuttal_candidate_prompt_only.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("paper_rebuttal_candidate/prompt_only")
RUNS = ["laguna_verified", "qwen_verified", "laguna_pro", "qwen_pro"]
H_PROBES = ["currently_correct", "currently_reduces_failing", "currently_has_regressions"]
K0_PROBES = ["currently_compiles"] + H_PROBES
OLD_ALL, OLD_CAUSAL = ["k0", "traj", "pos_causal", "pos_rel"], ["k0", "traj", "pos_causal"]
STRATA = ["S0", "S1", "S2", "S3", "S4"]


def sig(ci: list) -> int:
    lo, hi = ci
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def strongest(auc: dict, st: str, controls: list[str]) -> str:
    cand = [(auc[f"{o}|{st}"], o) for o in controls if f"{o}|{st}" in auc and auc[f"{o}|{st}"] == auc[f"{o}|{st}"]]
    return max(cand)[1]


def main() -> None:
    cells, t1, k0 = [], {}, []
    for run in RUNS:
        for probe in H_PROBES:
            d = json.loads((OUT / "json" / f"horizon_{run}_{probe}.json").read_text())
            for r in d["rows"]:
                a, c, k = r["auc"], r["ci95"], r["k"]
                cells.append({"run": run, "probe": probe, "k": k, "layer": r["layer"],
                              "n_S0": r["counts"]["S0"]["n"], "n_tasks_pos_S0": r["counts"]["S0"]["n_tasks_pos"],
                              "real": a["real_val|S0"],
                              **{p: a[f"{p}|S0"] for p in ("prompt_hs", "prompt_tfidf")},
                              **{f"real-{p}": a[f"real_val-{p}|S0"] for p in ("prompt_hs", "prompt_tfidf")},
                              **{f"real-{p}_ci95": c[f"real_val-{p}|S0"] for p in ("prompt_hs", "prompt_tfidf")},
                              "C_prompt_hs": r["fit"]["prompt_hs"].get("C"),
                              "C_prompt_tfidf": r["fit"]["prompt_tfidf"].get("C")})
                if k < 2:
                    continue
                for st in STRATA:
                    powered = not r["counts"][st]["underpowered"]
                    for name, ctrl in {"all_old": OLD_ALL, "causal_old": OLD_CAUSAL,
                                       "all_new": OLD_ALL + ["prompt_hs"], "causal_new": OLD_CAUSAL + ["prompt_hs"],
                                       "all_new_tfidf": OLD_ALL + ["prompt_hs", "prompt_tfidf"],
                                       "causal_new_tfidf": OLD_CAUSAL + ["prompt_hs", "prompt_tfidf"]}.items():
                        s = strongest(a, st, ctrl)
                        v = sig(c[f"real_val-{s}|{st}"])
                        e = t1.setdefault(st, {}).setdefault(name, {"all": {"ci>0": 0, "ci<0": 0, "n": 0},
                                                                    "powered": {"ci>0": 0, "ci<0": 0, "n": 0}})
                        for scope in (["all", "powered"] if powered else ["all"]):
                            e[scope]["n"] += 1
                            e[scope]["ci>0"] += v == 1
                            e[scope]["ci<0"] += v == -1
                    pe = t1[st].setdefault("prompt_hs_is_strongest", {"all": 0, "powered": 0})
                    if strongest(a, st, OLD_ALL + ["prompt_hs"]) == "prompt_hs":
                        pe["all"] += 1
                        pe["powered"] += powered
                    pe = t1[st].setdefault("prompt_hs_is_strongest_causal", {"all": 0, "powered": 0})
                    if strongest(a, st, OLD_CAUSAL + ["prompt_hs"]) == "prompt_hs":
                        pe["all"] += 1
                        pe["powered"] += powered
    h72 = {}
    for p in ("prompt_hs", "prompt_tfidf"):
        s = [sig(x[f"real-{p}_ci95"]) for x in cells]
        h72[p] = {"n": len(cells), "probe_better_ci>0": s.count(1), "prompt_better_ci<0": s.count(-1),
                  "not_significant": s.count(0), "prompt_auc_range": [min(x[p] for x in cells), max(x[p] for x in cells)],
                  "diff_range": [min(x[f"real-{p}"] for x in cells), max(x[f"real-{p}"] for x in cells)]}
    for run in RUNS:
        for probe in K0_PROBES:
            d = json.loads((OUT / "json" / f"k0_{run}_{probe}.json").read_text())
            a, c = d["auc"], d["ci95"]
            k0.append({"run": run, "probe": probe, "layer": d["layer"], "real": a["real_testbest"],
                       "real_ci95": c["real_testbest"],
                       **{p: a[p] for p in ("prompt_hs", "prompt_tfidf")},
                       **{f"{p}_ci95": c[p] for p in ("prompt_hs", "prompt_tfidf")},
                       **{f"real-{p}": a[f"real-{p}"] for p in ("prompt_hs", "prompt_tfidf")},
                       **{f"real-{p}_ci95": c[f"real-{p}"] for p in ("prompt_hs", "prompt_tfidf")},
                       "n_test": d["n_test"], "n_test_tasks": d["n_test_tasks"], "cache_check": d["cache_check"],
                       "C_prompt_hs": d["fit"]["prompt_hs"].get("C"), "C_prompt_tfidf": d["fit"]["prompt_tfidf"].get("C")})
    summary = {"horizon_72": h72, "table1": t1, "k0": k0, "horizon_cells": cells}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(h72, indent=1))
    for st, e in t1.items():
        print(st, {n: (v["all"], v["powered"]) if isinstance(v, dict) and "all" in v and isinstance(v["all"], dict)
                   else v for n, v in e.items()})
    for x in k0:
        print(f"{x['run']:16s} {x['probe']:28s} L{x['layer']:2d} real {x['real']:.3f} hs {x['prompt_hs']:.3f} "
              f"tfidf {x['prompt_tfidf']:.3f} real-hs {x['real-prompt_hs']:+.3f} "
              f"[{x['real-prompt_hs_ci95'][0]:+.3f},{x['real-prompt_hs_ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
