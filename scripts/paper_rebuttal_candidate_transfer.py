"""Brief T-POOL: candidate tab:transfer and transfer barplot with every AUC pooled over test tokens.

Reads paper_rebuttal_candidate/transfer/json/transfer_<model>_<dir>_<probe>.json
(scripts/rebuttal_transfer_pooled.py) and draws the table and the barplot with the submitted code
(run_paper_figures.build_transfer_table / plot_transfer_barplot), the transfer runs' results
replaced by one pooled AUC per layer. In-dist results are the stored pooled runs, unchanged.
Writes, next to the JSON:
  transfer_table.tex, transfer_barplot.pdf  candidate artifacts (CANDIDATE: not in paper_rebuttal/)
  comparison.json                          per row: best-test-layer values, binned vs pooled, deltas
  cells.json                               per cell (row x layer): binned vs pooled transfer AUC

Usage: uv run python scripts/paper_rebuttal_candidate_transfer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_paper_figures as rpf  # noqa: E402
from paper import style as _style  # noqa: E402

OUT = Path("paper_rebuttal_candidate/transfer")
LAYERS = [0, 10, 20, 30, 39]
PROBES = ["currently_compiles", "currently_correct", "currently_reduces_failing", "currently_has_regressions"]
TRANSFER_RUN = {("laguna", "v2p"): "laguna_xs2_full_verified_transfer", ("laguna", "p2v"): "laguna_xs2_full_pro_transfer",
                ("qwen", "v2p"): "qwen36_35b_a3b_full_verified_transfer", ("qwen", "p2v"): "qwen36_35b_a3b_full_pro_transfer"}


def main() -> None:
    data = {}
    for (model, d), run in TRANSFER_RUN.items():
        for probe in PROBES:
            data[(run, probe)] = json.loads((OUT / "json" / f"transfer_{model}_{d}_{probe}.json").read_text())

    original_load = rpf._load

    def load(results_dir, run_id, probe):
        if (run_id, probe) in data:
            j = data[(run_id, probe)]
            return {int(L): [{"test_auc": v["transfer_pooled"], "n_test": v["n_test"], "n_pos_test": v["n_pos_test"]}]
                    for L, v in j["layers"].items()}
        return original_load(results_dir, run_id, probe)

    rpf._load = load
    _style.apply()
    labels = ["Laguna-XS.2", "Qwen3.6-35B-A3B"]
    kw = dict(verified_results_dir=Path("results/swebench"), pro_results_dir=Path("results/swebench_pro"),
              probes=PROBES, model_labels=labels,
              verified_pooled_run_ids=["laguna_xs2_full_pooled", "qwen36_35b_a3b_full_pooled"],
              pro_pooled_run_ids=["laguna_xs2_pro_full_pooled", "qwen36_35b_a3b_pro_full_pooled"],
              verified_to_pro_run_ids=[TRANSFER_RUN[("laguna", "v2p")], TRANSFER_RUN[("qwen", "v2p")]],
              pro_to_verified_run_ids=[TRANSFER_RUN[("laguna", "p2v")], TRANSFER_RUN[("qwen", "p2v")]])
    tex = rpf.build_transfer_table(layers=LAYERS, **kw)
    old = r"\caption{Cross-dataset transfer AUC. "
    assert tex.count(old) == 1
    tex = tex.replace(old, r"\caption{Cross-dataset transfer AUC, pooled over test tokens. ")
    (OUT / "transfer_table.tex").write_text(tex)
    rpf.plot_transfer_barplot(figures_dir=OUT, **kw)

    # Check: the binned values reproduce paper_rebuttal/figures/transfer_table.tex.
    current = Path("paper_rebuttal/figures/transfer_table.tex").read_text()
    for (run, probe), j in data.items():
        for L, v in j["layers"].items():
            if f"{v['transfer_binned']:.3f}" not in current:
                raise ValueError(f"{run}/{probe} L{L}: binned {v['transfer_binned']:.3f} not in the current table")

    comparison, cells = {}, []
    for (run, probe), j in data.items():
        lay = j["layers"]
        b = j["best_layer_test"]
        key = f"{j['model']}|{j['direction']}|{probe}"
        comparison[key] = {
            "indist_pooled_best": lay[str(b["indist_pooled"])]["indist_pooled"], "indist_best_layer": b["indist_pooled"] + 1,
            "transfer_binned_best": lay[str(b["transfer_binned"])]["transfer_binned"], "transfer_binned_best_layer": b["transfer_binned"] + 1,
            "transfer_pooled_best": lay[str(b["transfer_pooled"])]["transfer_pooled"], "transfer_pooled_best_layer": b["transfer_pooled"] + 1,
            "delta_pooled_best": j["delta_best"], "delta_pooled_best_ci95": j["delta_best_ci95"],
            "delta_binned_vs_pooled_indist_best": lay[str(b["transfer_binned"])]["transfer_binned"] - lay[str(b["indist_pooled"])]["indist_pooled"],
            "n_test_tasks_in_split": j["n_test_tasks_in_split"], "test_tasks_without_valid_tokens": j["test_tasks_without_valid_tokens"]}
        for L, v in lay.items():
            cells.append({"model": j["model"], "direction": j["direction"], "probe": probe, "layer": int(L) + 1,
                          "transfer_binned": v["transfer_binned"], "transfer_pooled": v["transfer_pooled"],
                          "pooled_minus_binned": v["transfer_pooled"] - v["transfer_binned"],
                          "pooled_lower": v["transfer_pooled"] < v["transfer_binned"],
                          "indist_pooled": v["indist_pooled"], "delta_pooled": v["delta_pooled"],
                          "delta_pooled_ci95": v["delta_pooled_ci95"]})
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2))
    (OUT / "cells.json").write_text(json.dumps(cells, indent=2))
    ch = [c["pooled_minus_binned"] for c in cells]
    print(f"pooled - binned: min {min(ch):+.3f} max {max(ch):+.3f}; pooled lower in {sum(c['pooled_lower'] for c in cells)}/{len(cells)} cells")


if __name__ == "__main__":
    main()
