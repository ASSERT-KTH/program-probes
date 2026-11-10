"""Paper figures and tables for the revision (brief P-FIG), written to paper_rebuttal/.

Paths mirror the paper repo, with the submitted filenames (drop-in replacements):
  paper_rebuttal/figures/auc_table.tex                          tab:auc_roc (+ "Traj. shuffle" column)
  paper_rebuttal/figures/transfer_table.tex                     tab:transfer
  paper_rebuttal/figures/dataset_stats_table.tex                tab:dataset_stats (only #h_t can change)
  paper_rebuttal/figures/figures/auc_summary_barplot.pdf        + token- and trajectory-shuffle markers
  paper_rebuttal/figures/figures/combined/layer_auc_grid.pdf
  paper_rebuttal/figures/figures/transfer_barplot.pdf
  paper_rebuttal/figures/figures/<run>/currently_*_lookahead[_pro]_max{15,50}.pdf
                                                                + oracle persistence and E2 k=0 curves
  paper_rebuttal/figures/figures/horizon/horizon_<prop>_yt0.pdf (new, from the E16 output)
  paper_rebuttal/summary.json                                   _weighted_mean check, headline numbers, plotted values

Conventions (as submitted): the headline k=0 numbers are the best layer on TEST; results come
from the rerun (task split, real agent turns). The submitted run_paper_figures._weighted_mean
averages per-bin AUCs weighted by n_test and keeps single-class bins, which src/probe.py
(_clf_metrics) and src/baselines.py store as a 0.5 placeholder. Here it is replaced by a
version that drops those bins; for single-bin (pooled) runs nothing changes.

Inputs beyond results/: reports/rerun/lookahead_refs/*.json (scripts/rebuttal_paper_lookahead_refs.py)
and reports/rerun/phase/lookahead_*_max15.json (E16, scripts/rebuttal_phase_confound.py).

Usage (compute node): uv run python scripts/paper_rebuttal_figures.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_paper_figures as rpf  # noqa: E402
from paper import style as _style  # noqa: E402

OUT = Path("paper_rebuttal")
TEX = OUT / "figures"
FIGS = TEX / "figures"
RES = {"Verified": Path("results/swebench"), "Pro": Path("results/swebench_pro")}
LAYERS = [0, 10, 20, 30, 39]
PROBES = ["currently_compiles", "currently_correct", "currently_reduces_failing", "currently_has_regressions"]
POOLED = {"Verified": ["laguna_xs2_full_pooled", "qwen36_35b_a3b_full_pooled"],
          "Pro": ["laguna_xs2_pro_full_pooled", "qwen36_35b_a3b_pro_full_pooled"]}
TRANSFER = {"v2p": ["laguna_xs2_full_verified_transfer", "qwen36_35b_a3b_full_verified_transfer"],
            "p2v": ["laguna_xs2_full_pro_transfer", "qwen36_35b_a3b_full_pro_transfer"]}
BENCH = {"Verified": "swebench", "Pro": "swebench_pro"}
LEGACY = "_legacyturns"

_submitted_weighted_mean = rpf._weighted_mean


def _single(r) -> bool:
    n = r.n_test if hasattr(r, "n_test") else r["n_test"]
    p = r.n_pos_test if hasattr(r, "n_pos_test") else r["n_pos_test"]
    return p == 0 or p == n


def weighted_mean_drop_single(results_for_layer, field: str) -> float:
    """n_test-weighted mean of per-bin values over bins with both classes in test."""
    return _submitted_weighted_mean([r for r in results_for_layer if not _single(r)], field)


def best_layer(res: dict) -> tuple[int, float]:
    vals = {L: rpf._weighted_mean(v, "test_auc") for L, v in res.items() if v}
    L = max(vals, key=lambda l: (vals[l] if not math.isnan(vals[l]) else -1))
    return L, vals[L]


# ---------------------------------------------------------------------------
# _weighted_mean check
# ---------------------------------------------------------------------------

def weighted_mean_check() -> dict:
    """Old vs new summary for every multi-bin run used here (transfer; persistence is in the refs JSON)."""
    out = {}
    for ds_dir, runs in ((RES["Pro"], TRANSFER["v2p"]), (RES["Verified"], TRANSFER["p2v"])):
        for run in runs:
            for probe in PROBES:
                res = rpf._load(ds_dir, run, probe)
                if res is None:
                    continue
                cell = {}
                for L in LAYERS:
                    rows = res.get(L, [])
                    cell[str(L)] = {"n_bins": len(rows), "n_single_class_bins": sum(_single(r) for r in rows),
                                    "n_test_single_class": int(sum(r.n_test for r in rows if _single(r))),
                                    "submitted": _submitted_weighted_mean(rows, "test_auc"),
                                    "drop_single": weighted_mean_drop_single(rows, "test_auc")}
                out[f"{run}/{probe}"] = cell
    # Pooled runs: one bin per layer -> no change (asserted).
    for ds, runs in POOLED.items():
        for run in runs:
            for sfx in ("", "_shuffled", "_shuffled_traj"):
                for probe in PROBES:
                    res = rpf._load(RES[ds], run + sfx, probe)
                    if res is None:
                        continue
                    if any(len(v) != 1 for v in res.values() if v):
                        raise ValueError(f"{run}{sfx}/{probe}: expected one bin per layer")
                    if any(_single(v[0]) for v in res.values() if v):
                        raise ValueError(f"{run}{sfx}/{probe}: single-class test set")
    return out


def transfer_reference_check() -> dict:
    """The transfer runs are scored as the mean over 10 relative-position bins, the in-dist reference
    of tab:transfer as one pooled AUC. The <pooled>_bineval runs score the in-dist probe in the same
    10 bins on the same test tokens, which gives a like-for-like reference."""
    out = {}
    for ds in ("Verified", "Pro"):
        for run in POOLED[ds]:
            for probe in PROBES:
                pooled = rpf._load(RES[ds], run, probe)
                binned = rpf._load(RES[ds], f"{run}_bineval", probe)
                if binned is None:
                    continue
                out[f"{ds}|{run}|{probe}"] = {str(L): {"pooled": rpf._weighted_mean(pooled.get(L, []), "test_auc"),
                                                     "binavg": rpf._weighted_mean(binned.get(L, []), "test_auc"),
                                                     "n_bins": len(binned.get(L, []))} for L in LAYERS}
    return out


# ---------------------------------------------------------------------------
# A1: AUC table with a trajectory-shuffle column
# ---------------------------------------------------------------------------

def build_auc_table() -> tuple[str, dict]:
    n_layer = len(LAYERS)
    n_cols = 1 + n_layer + 2
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{AUC per probe, model, and benchmark across transformer layers."
        r" \textbf{Bold} marks the best layer per row."
        r" \emph{Shuffled}: probe trained on labels permuted across tokens; \emph{Traj.\ shuffle}:"
        r" probe trained with each trajectory's labels replaced by another trajectory's label sequence,"
        r" aligned by relative position. Both are evaluated on the real test labels at their best layer.}",
        r"\label{tab:auc_roc}",
        rf"\begin{{tabular}}{{l{'c' * n_layer}cc}}",
        r"\toprule",
        rf" & \multicolumn{{{n_layer}}}{{c}}{{AUC $\uparrow$}} & Shuffled & Traj.\ shuffle \\",
        rf"\cmidrule(lr){{2-{1 + n_layer}}}",
        "Probe & " + " & ".join(str(rpf._layer_label(l)) for l in LAYERS) + r" & (best layer) & (best layer) \\",
        r"\midrule",
    ]
    headline = {}
    for di, (ds, label) in enumerate((("Verified", "SWE-bench Verified"), ("Pro", "SWE-bench Pro"))):
        if di:
            lines.append(r"\midrule")
        lines += [rf"\multicolumn{{{n_cols}}}{{l}}{{\textsc{{{label}}}}} \\", r"\addlinespace[2pt]"]
        for ri, run in enumerate(POOLED[ds]):
            if ri:
                lines.append(r"\addlinespace[3pt]")
            lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\quad\textit{{{rpf.MODEL_LABELS[rpf._model_key(run)]}}}}} \\")
            for probe in PROBES:
                res = rpf._load(RES[ds], run, probe)
                aucs = rpf._layer_means(res, "test_auc", LAYERS)
                best = max(aucs.values())
                cells = [rf"\textbf{{{v:.3f}}}" if abs(v - best) < 1e-9 else f"{v:.3f}" for v in aucs.values()]
                ctrl = []
                for sfx in ("_shuffled", "_shuffled_traj"):
                    r = rpf._load(RES[ds], run + sfx, probe)
                    ctrl.append(f"{rpf._best_layer_mean(r, 'test_auc'):.3f}" if r else "—")
                lines.append(f"{rpf.PROBE_LABELS_TEX[probe]} & " + " & ".join(cells + ctrl) + r" \\")
                legacy = rpf._load(RES[ds], run + LEGACY, probe)
                lb = best_layer(legacy) if legacy else (None, float("nan"))
                bl = max(aucs, key=aucs.get)
                headline[f"{ds}|{run}|{probe}"] = {"rerun": best, "rerun_layer": rpf._layer_label(bl),
                                                   "submitted": lb[1],
                                                   "submitted_layer": rpf._layer_label(lb[0]) if lb[0] is not None else None,
                                                   "token_shuffle": float(ctrl[0]) if ctrl[0] != "—" else None,
                                                   "traj_shuffle": float(ctrl[1]) if ctrl[1] != "—" else None}
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines), headline


# ---------------------------------------------------------------------------
# A3: summary barplot with token- and trajectory-shuffle markers
# ---------------------------------------------------------------------------

def plot_summary_barplot() -> None:
    series = []
    for i, model in enumerate(("laguna", "qwen")):
        label = rpf.MODEL_LABELS[rpf._model_key(POOLED["Verified"][i])]
        series.append((f"{model}_verified", f"{label} (Verified)", "Verified", POOLED["Verified"][i]))
        series.append((f"{model}_pro", f"{label} (Pro)", "Pro", POOLED["Pro"][i]))
    n = len(series)
    bw = 0.15
    offsets = [(j - (n - 1) / 2) * bw for j in range(n)]
    x = list(range(len(PROBES)))
    fig, ax = plt.subplots(figsize=(_style.FULL_WIDTH, _style.FIG_HEIGHT_BAR))
    handles = []
    for i, (ckey, label, ds, run) in enumerate(series):
        dataset = "pro" if ds == "Pro" else "verified"
        color = _style.COLORS[ckey]
        ec = _style.COLORS[ckey.replace("_pro", "_verified")]
        for pi, probe in enumerate(PROBES):
            xc = x[pi] + offsets[i]
            v = rpf._best_layer_mean(rpf._load(RES[ds], run, probe), "test_auc")
            ax.bar(xc, v, width=bw, color=color, hatch=_style.HATCH[dataset], edgecolor=ec, linewidth=0.6, zorder=3)
            tok = rpf._best_layer_mean(rpf._load(RES[ds], run + "_shuffled", probe), "test_auc")
            trj = rpf._best_layer_mean(rpf._load(RES[ds], run + "_shuffled_traj", probe), "test_auc")
            ax.hlines(tok, xc - bw / 2, xc + bw / 2, colors="black", linewidth=1.6, zorder=4)
            ax.plot(xc, trj, marker="o", markersize=3.6, markerfacecolor="white", markeredgecolor="black",
                    markeredgewidth=0.9, linestyle="", zorder=5)
        handles.append(mpatches.Patch(facecolor=color, hatch=_style.HATCH[dataset], edgecolor=ec, linewidth=0.6,
                                      label=label))
    handles.append(Line2D([0], [0], color="black", linewidth=1.6, label="Token shuffle"))
    handles.append(Line2D([0], [0], marker="o", markersize=3.6, markerfacecolor="white", markeredgecolor="black",
                          markeredgewidth=0.9, linestyle="", label="Trajectory shuffle (aligned by relative position)"))
    ax.axhline(0.5, linestyle="--", color=_style.COLORS["baseline"], linewidth=1.0, zorder=2)
    handles.append(Line2D([0], [0], linestyle="--", color=_style.COLORS["baseline"], linewidth=1.0,
                          label="Random (0.5)"))
    ax.set_xticks(x)
    ax.set_xticklabels([rpf._prop_tick_label(p) for p in PROBES], fontsize=8.5)
    for tick, p in zip(ax.get_xticklabels(), PROBES):
        tick.set_color(_style.PROPERTY_COLORS[rpf.PROBE_TO_PROPERTY[p]])
    ax.set_ylabel("Best-layer AUC")
    ax.set_ylim(0.45, 1.0)
    ax.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.6,
              columnspacing=1.0, handlelength=1.6, frameon=False)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "auc_summary_barplot.pdf")
    plt.close(fig)
    print("  [fig]", FIGS / "auc_summary_barplot.pdf")


# ---------------------------------------------------------------------------
# A6: lookahead figures with oracle persistence and the E2 k=0 curve
# ---------------------------------------------------------------------------

def plot_lookahead(ds: str, run: str, probe: str, K: int, out_dir: Path, fname: str) -> dict | None:
    refs_path = Path("reports/rerun/lookahead_refs") / f"la_refs_{BENCH[ds]}_{run}_{probe}_max{K}.json"
    if not refs_path.exists():
        print("  [skip] missing", refs_path)
        return None
    rows = json.loads(refs_path.read_text())["rows"]
    ks = [r["k"] for r in rows]
    prop_marker = _style.PROPERTY_MARKERS[rpf.PROBE_TO_PROPERTY[probe]]
    blues = plt.get_cmap("Blues")
    colors = [blues(0.85 - 0.45 * i / (len(LAYERS) - 1)) for i in range(len(LAYERS))]
    fig_width = max(_style.FULL_WIDTH, _style.FULL_WIDTH + (len(ks) - 16) * 0.12)
    fig, ax = plt.subplots(figsize=(fig_width, _style.FIG_HEIGHT_LINE + 0.5))
    for L, color in zip(LAYERS, colors):
        ys = [r["probe_pooled"].get(str(L), float("nan")) for r in rows]
        ax.plot(ks, ys, marker=prop_marker, markersize=5 if K <= 15 else 3, linewidth=1.6, color=color,
                label=f"Layer {rpf._layer_label(L)}", zorder=3)
        ax.annotate(f"L{rpf._layer_label(L)}", xy=(ks[0], ys[0]), xytext=(-5, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8, color=color)
    ax.axhline(0.5, linestyle="--", color=_style.COLORS["baseline"], linewidth=1.0, zorder=1)
    ax.text(ks[-1], 0.5, "  random", va="top", ha="left", fontsize=7, color="#999", transform=ax.transData)
    pers = [r["persistence_pooled"] for r in rows]
    k0 = [r["k0_pooled"] for r in rows]
    l0 = rows[0]["k0_layer"]
    ax.plot(ks, pers, linestyle=":", marker="s", markersize=3.5 if K <= 15 else 2.5, color="#555555",
            linewidth=1.4, zorder=2, label=r"Persistence oracle (true $y_t$ as score)")
    ax.plot(ks, k0, linestyle="--", color="black", linewidth=1.3, zorder=2.5,
            label=rf"$k{{=}}0$ probe scored on $y_{{t+k}}$ (layer {rpf._layer_label(l0)})")
    ax.set_xlabel("Horizon k (steps)")
    ax.set_ylabel("AUC")
    lo = min(min(v for v in pers + k0 if v == v), 0.48)
    ax.set_ylim(bottom=max(lo - 0.02, 0.0))
    model_str = rpf.MODEL_LABELS[rpf._model_key(run)]
    rpf._apply_title(ax, probe, model_str, ds)
    ax.margins(x=0.06)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    s = rpf._fmt_n(rows[0]["n_test"])
    if s:
        ax.annotate(f"n = {s} $h_t$", xy=(1, 1), xycoords="axes fraction", xytext=(-4, -4),
                    textcoords="offset points", ha="right", va="top", fontsize=7, color="#999999",
                    annotation_clip=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-2:], labels[-2:], loc="lower left", fontsize=7, frameon=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / fname)
    plt.close(fig)
    print("  [fig]", out_dir / fname)
    return {"persistence_pooled": pers, "k0_pooled": k0, "ks": ks,
            "persistence_binavg_submitted": [r.get("persistence_binavg_stored") for r in rows],
            "persistence_binavg_drop_single": [r.get("persistence_binavg_drop_single") for r in rows]}


# ---------------------------------------------------------------------------
# B: horizon figure (E16, window 15, within y_t = 0)
# ---------------------------------------------------------------------------

HORIZON_PANELS = [("Verified", "laguna_xs2_full_pooled", "Laguna · Verified", "laguna"),
                  ("Verified", "qwen36_35b_a3b_full_pooled", "Qwen · Verified", "qwen"),
                  ("Pro", "laguna_xs2_pro_full_pooled", "Laguna · Pro", "laguna"),
                  ("Pro", "qwen36_35b_a3b_pro_full_pooled", "Qwen · Pro", "qwen")]
HORIZON_WIDTH = 5.5
HORIZON_HEIGHT = 2.5


def plot_horizon(probe: str) -> dict:
    fig, axes = plt.subplots(1, 4, figsize=(HORIZON_WIDTH, HORIZON_HEIGHT), sharey=True)
    summary = {}
    allv = []
    for ax, (ds, run, label, slug) in zip(axes, HORIZON_PANELS):
        d = json.loads((Path("reports/rerun/phase") / f"lookahead_{BENCH[ds]}_{run}_{probe}_max15.json").read_text())
        rows = d["rows"]
        ks = [r["k"] for r in rows]
        get = lambda key: [r["auc"][f"{key}|yt0"] for r in rows]  # noqa: E731
        real = get("real_val")
        lo = [r["ci95"]["real_val|yt0"][0] for r in rows]
        hi = [r["ci95"]["real_val|yt0"][1] for r in rows]
        color = _style.COLORS[f"{slug}_verified"]
        ax.fill_between(ks, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(ks, real, color=color, linewidth=1.5, marker="o", markersize=2.6, zorder=4)
        ax.plot(ks, get("k0"), color=color, linewidth=1.1, linestyle="--", zorder=3)
        ax.plot(ks, get("traj"), color="#444444", linewidth=1.1, linestyle=":", zorder=3)
        ax.plot(ks, get("pos_causal"), color="#444444", linewidth=1.0, linestyle="-.", zorder=3)
        ax.plot(ks, get("pos_rel"), color="#9E9E9E", linewidth=0.7, linestyle="-", zorder=2)
        ax.axhline(0.5, color="black", linewidth=0.5, zorder=1)
        ax.text(0.04, 0.97, label, transform=ax.transAxes, ha="left", va="top", fontsize=7)
        ax.set_xticks([1, 5, 10, 15])
        ax.tick_params(labelsize=7, pad=1.5)
        ax.set_xlabel("k (turns ahead)", fontsize=7.5, labelpad=1.5)
        allv += real + lo + hi + get("k0") + get("traj") + get("pos_causal") + get("pos_rel")
        summary[f"{ds}|{run}"] = {"ks": ks, "real_val": real, "ci_lo": lo, "ci_hi": hi, "k0": get("k0"),
                                  "traj": get("traj"), "pos_causal": get("pos_causal"), "pos_rel": get("pos_rel"),
                                  "real_val_layer": [r["layers"]["real_val"] for r in rows]}
    vals = [v for v in allv if v == v]
    axes[0].set_ylim(math.floor(min(vals) * 20) / 20 - 0.01, math.ceil(max(vals) * 20) / 20 + 0.03)
    axes[0].set_ylabel(r"AUC for $y_{t+k}$ ($y_t = 0$)", fontsize=7.5)
    handles = [
        (Line2D([0], [0], color="#333333", linewidth=1.5, marker="o", markersize=2.6), "$k$-step probe (95% CI)"),
        (Line2D([0], [0], color="#333333", linewidth=1.1, linestyle="--"), "$k{=}0$ probe"),
        (Line2D([0], [0], color="#444444", linewidth=1.1, linestyle=":"), "trajectory shuffle (aligned by relative position)"),
        (Line2D([0], [0], color="#444444", linewidth=1.0, linestyle="-."), "step index"),
        (Line2D([0], [0], color="#9E9E9E", linewidth=0.7), "relative position (uses final length)"),
    ]
    fig.legend([h for h, _ in handles], [l for _, l in handles], loc="lower center", ncol=3, fontsize=6.3,
               frameon=False, handlelength=1.9, columnspacing=0.7, handletextpad=0.35, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.98, bottom=0.335, wspace=0.08)
    out = FIGS / "horizon"
    out.mkdir(parents=True, exist_ok=True)
    name = f"horizon_{probe.removeprefix('currently_')}_yt0"
    with plt.rc_context({"savefig.bbox": "standard"}):  # keep the exact HORIZON_WIDTH x HORIZON_HEIGHT
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print("  [fig]", out / f"{name}.pdf")
    return summary


# ---------------------------------------------------------------------------

def main() -> None:
    _style.apply()
    TEX.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    print("[check] _weighted_mean")
    summary["weighted_mean_check"] = weighted_mean_check()
    rpf._weighted_mean = weighted_mean_drop_single  # used by _layer_means / _best_layer_mean from here on

    print("[table] auc_table")
    tex, headline = build_auc_table()
    (TEX / "auc_table.tex").write_text(tex)
    summary["headline"] = headline

    print("[table] transfer_table")
    labels = [rpf.MODEL_LABELS[rpf._model_key(r)] for r in POOLED["Verified"]]
    (TEX / "transfer_table.tex").write_text(rpf.build_transfer_table(
        verified_results_dir=RES["Verified"], pro_results_dir=RES["Pro"], probes=PROBES, layers=LAYERS,
        model_labels=labels, verified_pooled_run_ids=POOLED["Verified"], pro_pooled_run_ids=POOLED["Pro"],
        verified_to_pro_run_ids=TRANSFER["v2p"], pro_to_verified_run_ids=TRANSFER["p2v"]))

    print("[fig] auc_summary_barplot")
    plot_summary_barplot()

    print("[fig] layer_auc_grid")
    rpf.plot_layer_auc_grid(
        results_dir=RES["Verified"], run_ids=POOLED["Verified"],
        run_labels=[f"{l} (Verified)" for l in labels], probes=PROBES, layers=LAYERS, figures_dir=FIGS,
        out_run_id="combined",
        extra_series=[(RES["Pro"], r, f"{l} (Pro)") for r, l in zip(POOLED["Pro"], labels)],
        shuffled_ids={r: f"{r}_shuffled" for r in POOLED["Verified"] + POOLED["Pro"]})

    print("[fig] transfer_barplot")
    rpf.plot_transfer_barplot(
        verified_results_dir=RES["Verified"], pro_results_dir=RES["Pro"], probes=PROBES, figures_dir=FIGS,
        model_labels=labels, verified_pooled_run_ids=POOLED["Verified"], pro_pooled_run_ids=POOLED["Pro"],
        verified_to_pro_run_ids=TRANSFER["v2p"], pro_to_verified_run_ids=TRANSFER["p2v"])

    print("[fig] lookahead")
    summary["lookahead"] = {}
    for ds in ("Verified", "Pro"):
        for run in POOLED[ds]:
            out_dir = FIGS / (run if ds == "Verified" else f"{run}_pooled_pro")
            for probe in PROBES:
                for K in (15, 50):
                    fname = f"{probe}_lookahead_{'pro_' if ds == 'Pro' else ''}max{K}.pdf"
                    s = plot_lookahead(ds, run, probe, K, out_dir, fname)
                    if s:
                        summary["lookahead"][f"{ds}|{run}|{probe}|max{K}"] = s

    print("[fig] horizon")
    summary["horizon"] = {p: plot_horizon(p) for p in PROBES[1:]}

    print("[table] dataset_stats_table (only #h_t depends on the rerun)")

    class _A:  # the arguments _generate_dataset_stats reads
        generations_dir, labels_dir, dataset_stats_cache_dir = "generations", "labels", "paper/cache"
    tmp_figs = OUT / "_dataset_stats_figs_unused"  # the stats figures do not depend on the rerun; not kept
    rpf._generate_dataset_stats(_A, RES["Verified"], RES["Pro"], tmp_figs, TEX, PROBES)
    shutil.rmtree(tmp_figs)

    print("[check] transfer reference: pooled in-dist vs in-dist evaluated in 10 position bins (<pooled>_bineval)")
    summary["transfer_reference_check"] = transfer_reference_check()

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("[done]", flush=True)
    os._exit(0)  # the interpreter otherwise hangs at exit on the compute nodes


if __name__ == "__main__":
    main()
