"""Brief AE0: candidate appendix table from scripts/rebuttal_after_edit_k0.py.

Reads paper_rebuttal_candidate/after_edit_k0/json/ae0_<run>.json and writes, next to json/:
  after_edit_table.tex  rows = model x benchmark, columns = the 4 properties (tab:auc_roc order); cell =
                        after-edit AUC [95% task-bootstrap CI] and, in grey, the all-state AUC of tab:auc_roc
                        (same probe and layer); a counts line per row (CANDIDATE: not in paper_rebuttal/)
  summary.json          the 16 cells (primary filter, plus strict / no filter), counts, positive rates,
                        best after-edit layer side note, label-changed subset, filter counts per run
Usage: uvx python@3.12 scripts/paper_rebuttal_candidate_after_edit.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("paper_rebuttal_candidate/after_edit_k0")
RUNS = [("laguna_verified", "Laguna-XS.2", "Verified"), ("qwen_verified", "Qwen3.6-35B-A3B", "Verified"),
        ("laguna_pro", "Laguna-XS.2", "Pro"), ("qwen_pro", "Qwen3.6-35B-A3B", "Pro")]
PROBES = [("currently_compiles", r"\Syntactic{}"), ("currently_correct", r"\Semantic{}"),
          ("currently_reduces_failing", r"\RedFail{}"), ("currently_has_regressions", r"\Regressions{}")]


def pct(x: float) -> str:
    return r"$<$1\%" if x < 0.01 else f"{100 * x:.0f}\\%"


def k(n: int) -> str:
    return f"{n / 1e3:.1f}k" if n >= 1e4 else f"{n:,}"


def main() -> None:
    summary: dict = {"cells": {}, "filter": {}}
    rows = []
    prev_bench = None
    for run, model, bench in RUNS:
        d = json.loads((OUT / "json" / f"ae0_{run}.json").read_text())
        summary["filter"][run] = {s: {kk: v for kk, v in d["filter"][s].items()} for s in ("all_splits", "test_split")}
        if bench != prev_bench:
            if prev_bench is not None:
                rows.append(r"\midrule")
            rows.append(rf"\multicolumn{{5}}{{l}}{{\textsc{{SWE-bench {bench}}}}} \\")
            prev_bench = bench
        cells, cnt = [], []
        for probe, _ in PROBES:
            pr = d["probes"][probe]
            bl = pr["best_layer_all_states"]
            lay = pr["layers"][str(bl)]
            ci = pr["best_layer_ci"]
            c = {"layer": bl, "all_states_auc": lay["all_states_stored_test_auc"],
                 **{v: {**{kk: lay[v][kk] for kk in ("auc", "n_edits", "n_states", "n_test_tasks", "n_trajectories",
                                                    "pos_rate")},
                        "ci95": ci[v]["ci95"], "label_changed_by_edit": lay[v]["label_changed_by_edit"]}
                    for v in ("primary", "strict", "none")},
                 "best_after_edit_layer_primary": pr["best_layer_after_edit_primary"],
                 "auc_at_best_after_edit_layer_primary": pr["layers"][str(pr["best_layer_after_edit_primary"])]["primary"]["auc"],
                 "ci95_at_best_after_edit_layer_primary": (pr.get("best_after_edit_layer_ci_primary") or ci["primary"])["ci95"],
                 "auc_primary_by_layer": {L: v["primary"]["auc"] for L, v in pr["layers"].items()}}
            summary["cells"][f"{run}|{probe}"] = c
            p = c["primary"]
            lo, hi = p["ci95"]
            cells.append(rf"{p['auc']:.3f} {{\scriptsize[{lo:.2f}, {hi:.2f}]}} "
                         rf"{{\scriptsize\textcolor{{gray}}{{{c['all_states_auc']:.3f}}}}}")
            cnt.append(p)
        same = all((x["n_edits"], x["n_states"], x["n_test_tasks"]) == (cnt[0]["n_edits"], cnt[0]["n_states"],
                                                                      cnt[0]["n_test_tasks"]) for x in cnt)
        summary.setdefault("counts_same_for_all_properties", {})[run] = same
        rows.append(rf"\quad\textit{{{model}}} & " + " & ".join(cells) + r" \\")
        n = cnt[0]
        f = d["filter"]["test_split"]
        rows.append(rf"\quad{{\scriptsize\textcolor{{gray}}{{{n['n_edits']:,} edits, {k(n['n_states'])} $h_t$, "
                    rf"{n['n_test_tasks']} tasks; {pct(f['dropped_primary_share'])} of test edits dropped}}}}"
                    r" & \multicolumn{4}{l}{{\scriptsize\textcolor{gray}{positive rate "
                    + " / ".join(f"{100 * x['pos_rate']:.0f}\\%" for x in cnt) + r"}}} \\")
    cap = (r"AUC of the $k{=}0$ probes on the hidden states right after a code edit, before anything has run on "
           r"the new program version: the tokens of the first assistant turn whose label reflects the edit, whose "
           r"context ends with the edit command and its own output. Edits whose command also ran something "
           r"(tests, scripts, interpreters, builds, linters; Python scripts that only rewrite files are kept) are "
           r"dropped. Same probe, test split and layer as Table~\ref{tab:auc_roc} (best layer on all test "
           r"tokens); AUC pooled over the selected test tokens, with a 95\% task-level bootstrap CI (1000 "
           r"resamples). Grey: the all-token AUC of Table~\ref{tab:auc_roc}. Counts are for \Syntactic{}; "
           r"positive rates per property.")
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\caption{" + cap + "}", r"\label{tab:after_edit_k0}",
        r"\resizebox{\columnwidth}{!}{", r"\begin{tabular}{lcccc}", r"\toprule",
        " & " + " & ".join(lbl for _, lbl in PROBES) + r" \\", r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular}", "}", r"\end{table}", ""])
    if all(summary["counts_same_for_all_properties"].values()):
        tex = tex.replace(r"Counts are for \Syntactic{}; positive rates per property.",
                          r"Counts (test split) are the same for all four properties; positive rates per property.")
    (OUT / "after_edit_table.tex").write_text(tex)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    for key, c in summary["cells"].items():
        p, s = c["primary"], c["strict"]
        print(f"{key:48s} L{c['layer']:2d} AE {p['auc']:.3f} [{p['ci95'][0]:.3f},{p['ci95'][1]:.3f}] all {c['all_states_auc']:.3f} "
              f"| strict {s['auc'] if s['auc'] is None else round(s['auc'], 3)} | none {c['none']['auc']:.3f} "
              f"| n {p['n_edits']}/{p['n_states']}/{p['n_test_tasks']} pos {p['pos_rate']:.2f} "
              f"| bestAE L{c['best_after_edit_layer_primary']} {c['auc_at_best_after_edit_layer_primary']:.3f} "
              f"| changed n={p['label_changed_by_edit']['n_states']} auc={p['label_changed_by_edit']['auc']}")


if __name__ == "__main__":
    main()
