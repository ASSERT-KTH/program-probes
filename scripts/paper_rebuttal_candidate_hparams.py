"""Brief HP: candidate tab:probe-hparams from the rerun's task-split sweeps (metadata only).

Reads paper_rebuttal_candidate/hparams/json/sweeps.json (scripts/rebuttal_hparam_provenance.py) and writes
next to it:
  hparam_table.tex          same format as paper/hparam_table.tex (run_paper_figures.build_hparam_table),
                            middle probed layer (index 20, shown as 21); caption updated (CANDIDATE)
  json/hparams_per_layer.json   chosen config per benchmark x model x probe x layer (+ best mean val AUC)
  json/layer_variation.json     per benchmark x model x probe: spread of each hyperparameter over the 5 layers
Usage: uv run python scripts/paper_rebuttal_candidate_hparams.py
"""
from __future__ import annotations

import json
import math

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_paper_figures as rpf  # noqa: E402

OUT = Path("paper_rebuttal_candidate/hparams")
PROBES = [("currently_compiles", "Syntactic correctness"), ("currently_correct", "Semantic correctness"),
          ("currently_has_regressions", "Has regressions"), ("currently_reduces_failing", "Reduces failures")]
RUNS = [("Verified", "Laguna-XS.2", "laguna_xs2_full_pooled"), ("Verified", "Qwen3.6-35B-A3B", "qwen36_35b_a3b_full_pooled"),
        ("Pro", "Laguna-XS.2", "laguna_xs2_pro_full_pooled"), ("Pro", "Qwen3.6-35B-A3B", "qwen36_35b_a3b_pro_full_pooled")]


def main() -> None:
    sweeps = json.loads((OUT / "json" / "sweeps.json").read_text())
    by = {(s["run"], s["probe"], s["layer"]): (sid, s) for sid, s in sweeps.items()}
    hp, per_layer, variation = {}, {}, {}
    # Flatness of the objective: best trial's mean validation AUC minus the 2nd / 5th / median trial's.
    flat = {}
    for sid, sw in sweeps.items():
        a = sorted((t["mean_val_auc"] for t in sw["trials"] if t["mean_val_auc"] is not None), reverse=True)
        flat[sid] = {"n_trials": len(a), "best": a[0], "best_minus_2nd": a[0] - a[1], "best_minus_5th": a[0] - a[4],
                     "best_minus_median": a[0] - float(np.median(a)), "best_minus_worst": a[0] - a[-1]}
    for ds, model, run in RUNS:
        for probe, label in PROBES:
            layers = {}
            for L in (0, 10, 20, 30, 39):
                sid, s = by[(run, probe, L)]
                c = s["config"]
                layers[str(L)] = {"lr": float(c["lr"]), "weight_decay": float(c["weight_decay"]),
                                  "batch_size": int(c["batch_size"]), "patience": int(c["patience"]),
                                  "sweep": sid, "best_run": s["best_run"], "best_mean_val_auc": s["best_mean_val_auc"]}
            hp.setdefault(ds, {}).setdefault(model, {})[label] = {
                L: {k: v[k] for k in ("lr", "weight_decay", "batch_size", "patience")} for L, v in layers.items()}
            key = f"{ds}|{model}|{probe}"
            per_layer[key] = {str(int(L) + 1): v for L, v in layers.items()}  # 1-indexed layer keys
            lrs = [v["lr"] for v in layers.values()]
            wds = [v["weight_decay"] for v in layers.values()]
            variation[key] = {
                "lr_min": min(lrs), "lr_max": max(lrs), "lr_decades": math.log10(max(lrs) / min(lrs)),
                "wd_min": min(wds), "wd_max": max(wds), "wd_decades": math.log10(max(wds) / min(wds)),
                "batch_sizes": sorted({v["batch_size"] for v in layers.values()}),
                "patience": sorted({v["patience"] for v in layers.values()}),
                "layer21_lr_rank_among_5": sorted(lrs).index(layers["20"]["lr"]) + 1,
                "val_auc_best_minus_5th_per_layer": {str(int(L) + 1): flat[by[(run, probe, int(L))][0]]["best_minus_5th"]
                                                     for L in layers}}
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "hparam_table.tex"
    rpf.build_hparam_table(hparams=hp, model_labels=["Laguna-XS.2", "Qwen3.6-35B-A3B"],
                           dataset_labels=["Verified", "Pro"], output_path=tmp)
    tex = tmp.read_text()
    old = (r"selected by 20-trial random search maximising mean validation AUC. "
           r"Sweeps were run independently per probe and per layer; "
           r"table shows the middle probed layer as representative.}")
    assert tex.count(old) == 1
    tex = tex.replace(old, (
        r"selected by a 20-trial random search (learning rate and weight decay log-uniform in "
        r"$[10^{-4}, 10^{-1}]$ and $[10^{-5}, 10^{-1}]$, batch size $\in \{256, 512, 1024\}$, "
        r"patience $\in \{10, 25, 50\}$) maximising validation AUC on the task-level split. "
        r"Sweeps were run independently per probe and per layer on the $k{=}0$ data and reused for every "
        r"horizon and control; the table shows the middle probed layer (21) as representative.}"))
    tmp.write_text(tex)
    (OUT / "json" / "hparams_per_layer.json").write_text(json.dumps(per_layer, indent=1))
    summ = {k: {q: float(np.quantile([f[k] for f in flat.values()], q)) for q in (0, 0.25, 0.5, 0.75, 1)}
            for k in ("best_minus_2nd", "best_minus_5th", "best_minus_median", "best_minus_worst")}
    (OUT / "json" / "layer_variation.json").write_text(json.dumps(
        {"objective_flatness_quantiles_over_80_sweeps": summ, "per_sweep_flatness": flat, "per_probe": variation}, indent=1))
    print(json.dumps(summ, indent=1))
    for k, v in variation.items():
        print(f"{k:60s} lr {v['lr_min']:.1e}-{v['lr_max']:.1e} ({v['lr_decades']:.1f} dec)  "
              f"wd {v['wd_min']:.1e}-{v['wd_max']:.1e} ({v['wd_decades']:.1f} dec)  bs {v['batch_sizes']}  pat {v['patience']}")
    print("[done]", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
