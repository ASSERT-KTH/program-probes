"""Brief E14: candidate appendix table and summary from scripts/rebuttal_population_counts.py.

Reads paper_rebuttal_candidate/counts/json/counts_<run>.json and writes, next to json/:
  counts_table.tex   one row per run x filter: test tasks / trajectories (share of the split's
                     trajectories) / hidden states, and hidden states train / val / test
                     (CANDIDATE: not in paper_rebuttal/)
  summary.json       the compact tables, the trajectory lengths and the legacy (submitted) max50
                     decomposition
Usage: uv run python scripts/paper_rebuttal_candidate_counts.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path("paper_rebuttal_candidate/counts")
RUNS = [("laguna_verified", "Laguna", "Verified"), ("laguna_pro", "Laguna", "Pro"),
        ("qwen_verified", "Qwen3.6", "Verified"), ("qwen_pro", "Qwen3.6", "Pro")]
PROBES = ["currently_compiles", "currently_correct", "currently_reduces_failing", "currently_has_regressions"]
FILTERS = [("k0", r"$k{=}0$ (no window)"), ("max15", r"$k_{\max}{=}15$"), ("max50", r"$k_{\max}{=}50$")]
SPLITS = ["train", "val", "test"]


def fmt_states(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n / 1e3:.0f}k"


def one(d: dict, filt: str) -> dict:
    """The run's counts under a filter; identical across probes for the windowed caches (checked)."""
    per = d["rerun"][filt]
    ref = per[PROBES[0]]
    same = all(all(per[p][sp][k] == ref[sp][k] for sp in SPLITS for k in ("tasks", "trajectories", "states"))
               for p in PROBES)
    r = {sp: {k: ref[sp][k] for k in ("tasks", "trajectories", "states", "trajectories_in_split",
                                       "frac_trajectories_kept")} for sp in SPLITS}
    r["all"] = ref["all"]
    r["split_groups"] = ref["split_groups"]
    r["test_tasks_shared_with_k0_test"] = ref.get("test_tasks_shared_with_k0_test")
    r["same_for_all_probes"] = same
    if not same:
        r["per_probe_states"] = {p: {sp: per[p][sp]["states"] for sp in SPLITS} for p in PROBES}
    r["test_pos_rate"] = {p: per[p]["test"]["pos_rate"] for p in PROBES}
    return r


def main() -> None:
    data = {run: json.loads((OUT / "json" / f"counts_{run}.json").read_text()) for run, _, _ in RUNS}
    summary: dict = {"rerun": {}, "length": {}, "legacy": {}}
    rows = []
    for i, (run, model, bench) in enumerate(RUNS):
        d = data[run]
        summary["rerun"][run] = {f: one(d, f) for f, _ in FILTERS}
        lt, ls = d["length"]["turns_label_files"], d["length"]["steps_generations"]
        summary["length"][run] = {"median_turns": lt["median"], "frac_turns_ge15": lt["frac_ge15"],
                                  "frac_turns_ge50": lt["frac_ge50"], "frac_turns_ge16": lt["frac_ge16"],
                                  "frac_turns_ge51": lt["frac_ge51"], "n_label_files": lt["n"],
                                  "median_steps_generations": ls["median"], "frac_steps_ge50": ls["frac_ge50"],
                                  "frac_steps_ge15": ls["frac_ge15"], "n_generations": ls["n"],
                                  "turns_equal_steps": d["length"]["turns_equal_steps"],
                                  "n_common": d["length"]["n_common"]}
        for j, (f, flabel) in enumerate(FILTERS):
            r = summary["rerun"][run][f]
            t = r["test"]
            mcell = rf"\multirow{{6}}{{*}}{{\texttt{{{model}}}}}" if (i % 2 == 0 and j == 0) else ""
            bcell = rf"\multirow{{3}}{{*}}{{{bench}}}" if j == 0 else ""
            rows.append(
                rf"{mcell} & {bcell} & {flabel} & {t['tasks']:,} & {t['trajectories']:,} "
                rf"({100 * t['frac_trajectories_kept']:.0f}\%) & {fmt_states(t['states'])} & "
                + " / ".join(fmt_states(r[sp]["states"]) for sp in SPLITS) + r" \\")
        if i == 1:
            rows.append(r"\midrule")
        elif i < len(RUNS) - 1:
            rows.append(r"\addlinespace")

        # legacy (submitted) populations
        lg = d["legacy"]
        leg = {"filters": {}}
        for f, _ in FILTERS:
            ref = lg[PROBES[0]][f]
            leg["filters"][f] = {
                "split_groups": ref["split_groups"], "test_groups": ref.get("test_groups"),
                "tasks_by_split": {sp: ref[sp]["tasks"] for sp in SPLITS},
                "test": {k: ref["test"][k] for k in ("tasks", "trajectories", "states", "trajectories_in_split",
                                                      "frac_trajectories_kept")},
                "all": ref["all"],
                "states_by_split": {sp: ref[sp]["states"] for sp in SPLITS},
                "per_probe_test_states": {p: lg[p][f]["test"]["states"] for p in PROBES},
                "matches_stored_all_probes": all(lg[p][f]["matches_stored"] for p in PROBES),
                "stored_test": {p: (lg[p][f]["stored"] or {}).get("test") for p in PROBES}}
        summary["legacy"][run] = leg

    # Decomposition of the submitted max50 test count, Verified vs Pro, per model:
    # states = (trajectories in test) x (share kept by the window) x (states per kept trajectory).
    dec = {}
    for model in ("laguna", "qwen"):
        parts = {}
        for bench in ("verified", "pro"):
            L = summary["legacy"][f"{model}_{bench}"]["filters"]["max50"]["test"]
            parts[bench] = {"test_trajectories": L["trajectories_in_split"],
                            "window_keep_share": L["frac_trajectories_kept"],
                            "states_per_kept_trajectory": L["states"] / L["trajectories"] if L["trajectories"] else math.nan,
                            "test_states": L["states"]}
        v, p = parts["verified"], parts["pro"]
        ratio = {k: v[k] / p[k] for k in ("test_trajectories", "window_keep_share", "states_per_kept_trajectory",
                                         "test_states")}
        logs = {k: math.log(ratio[k]) for k in ratio}
        dec[model] = {"verified": v, "pro": p, "ratio_verified_over_pro": ratio,
                      "share_of_log_ratio": {k: logs[k] / logs["test_states"] for k in
                                             ("test_trajectories", "window_keep_share", "states_per_kept_trajectory")}}
    summary["legacy_max50_decomposition"] = dec

    cap = (r"Population after filtering, per model, benchmark and lookahead window (task-level split, "
           r"70/15/15). A hidden state $h_t$ is kept if its label is defined; with a window $k_{\max}$ only "
           r"states whose turn has at least $k_{\max}$ later turns are kept, so trajectories with at most "
           r"$k_{\max}$ turns drop out. Tasks, trajectories and $\#h_t$ are for the test split; the percentage "
           r"is the share of the test split's trajectories that keep at least one state. The last column gives "
           r"$\#h_t$ for train / validation / test. As in training, each population is split over its own tasks, "
           r"so a window that removes every trajectory of some tasks also changes which tasks are in test. "
           r"The counts are the same for all four properties.")
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\caption{" + cap + "}", r"\label{tab:population_counts}",
        r"\resizebox{\columnwidth}{!}{", r"\begin{tabular}{lllrrrr}", r"\toprule",
        r"Model & Benchmark & Filter & Tasks & Traj. & $\#h_t$ & $\#h_t$ train / val / test \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", "}", r"\end{table}", ""])
    if not all(summary["rerun"][r][f]["same_for_all_probes"] for r, _, _ in RUNS for f, _ in FILTERS):
        tex = tex.replace(r" The counts are the same for all four properties.",
                          r" Counts are for \texttt{currently\_compiles}; see the JSON for the other properties.")
    (OUT / "counts_table.tex").write_text(tex)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
