"""Generate paper-ready tables and figures from probe results.

Outputs
-------
- paper/auc_table.tex          : LaTeX AUC-ROC table (probes × models × layers)
- paper/calibration_table.tex  : LaTeX ECE + Brier table
- paper/figures/<run_id>/<probe>_auc_heatmap.png  : AUC heatmap (position bins)
- paper/figures/<run_id>/<probe>_auc_heatmap_step.png : AUC heatmap (step bins)
- paper/figures/<run_id>/<probe>_layer_auc.png    : AUC vs layer line plot
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load(results_dir: Path, run_id: str, probe: str) -> dict | None:
    p = results_dir / run_id / probe / "results.pt"
    if not p.exists():
        return None
    return torch.load(p, weights_only=False)


def _weighted_mean(results_for_layer, field: str) -> float:
    """Weighted mean of `field` across bins, weighted by n_test."""
    total_n, total_w = 0, 0.0
    for r in results_for_layer:
        n = r.n_test if hasattr(r, "n_test") else r["n_test"]
        v = getattr(r, field) if hasattr(r, field) else r[field]
        if v is None:
            continue
        total_w += v * n
        total_n += n
    return total_w / total_n if total_n > 0 else float("nan")


def _best_layer_mean(all_results: dict, field: str) -> float:
    """Return the mean `field` for the best layer (highest mean AUC)."""
    best = float("nan")
    for layer_results in all_results.values():
        m = _weighted_mean(layer_results, field)
        if math.isnan(best) or m > best:
            best = m
    return best


def _layer_means(all_results: dict, field: str, layers: list[int]) -> dict[int, float]:
    return {l: _weighted_mean(all_results.get(l, []), field) for l in layers}


def _auc_grid(all_results: dict, layers: list[int], n_bins: int = 10) -> np.ndarray:
    grid = np.full((len(layers), n_bins), np.nan)
    for li, layer in enumerate(layers):
        for r in all_results.get(layer, []):
            b = r.bin_idx if hasattr(r, "bin_idx") else r["bin_idx"]
            v = r.test_auc if hasattr(r, "test_auc") else r["test_auc"]
            if b < n_bins:
                grid[li, b] = v
    return grid


# ---------------------------------------------------------------------------
# AUC table
# ---------------------------------------------------------------------------

PROBE_LABELS = {
    "currently_compiles":       "Syntactic correctness",
    "currently_correct":        "Functional correctness",
    "currently_reduces_failing": "Reduces failures",
    "currently_has_regressions": "Has regressions",
    "will_resolve":             "Will resolve",
}

MODEL_LABELS = {
    "laguna_xs2_full":       "Laguna-XS2",
    "qwen36_35b_a3b_full":   "Qwen3.6-35B-A3B",
}


def build_auc_table(
    results_dir: Path,
    probes: list[str],
    model_run_ids: list[str],
    shuffled_run_ids: list[str],
    layers: list[int],
    pooled_run_ids: list[str] | None = None,
) -> str:
    """Return a LaTeX tabular string for AUC-ROC.

    If pooled_run_ids is given, the per-layer AUC columns use those (one global
    bin per layer) and the Shuffled column still comes from shuffled_run_ids.
    """
    n_layer_cols = len(layers)
    col_spec = "l" + "c" * n_layer_cols + "c"
    layer_header = " & ".join(str(l) for l in layers)
    n_cols = 1 + n_layer_cols + 1  # probe + layers + shuffled

    lines = []
    lines += [
        r"\begin{table*}[h]",
        r"\centering",
        r"\caption{Test AUC-ROC for each probe and model across transformer layers."
        r" The \emph{Shuffled} column uses label-shuffled data as a sanity baseline.}",
        r"\label{tab:auc_roc}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf" & \multicolumn{{{n_layer_cols}}}{{c}}{{AUC-ROC $\uparrow$}} & Shuffled \\",
        rf"\cmidrule(lr){{2-{1 + n_layer_cols}}}",
        "Probe & " + layer_header + r" & (best layer) \\",
        r"\midrule",
    ]

    probe_run_ids = pooled_run_ids if pooled_run_ids else model_run_ids

    # Pre-compute global best (layer, model) per probe across all models
    global_best: dict[str, float] = {}  # probe -> best AUC value across all models & layers
    for probe_run_id in probe_run_ids:
        for probe in probes:
            all_res = _load(results_dir, probe_run_id, probe)
            if all_res is None:
                continue
            for l in layers:
                v = _weighted_mean(all_res.get(l, []), "test_auc")
                if not math.isnan(v) and v > global_best.get(probe, float("-inf")):
                    global_best[probe] = v

    for probe_run_id, run_id, shuf_id in zip(probe_run_ids, model_run_ids, shuffled_run_ids):
        model_label = MODEL_LABELS.get(run_id, run_id)
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{model_label}}}}} \\")

        for probe in probes:
            all_res = _load(results_dir, probe_run_id, probe)
            shuf_res = _load(results_dir, shuf_id, probe)
            if all_res is None:
                print(f"  [missing] {probe_run_id}/{probe}")
                continue

            layer_aucs = _layer_means(all_res, "test_auc", layers)
            shuf_auc = _best_layer_mean(shuf_res, "test_auc") if shuf_res else float("nan")
            best_global = global_best.get(probe, float("nan"))

            cells = []
            for l in layers:
                v = layer_aucs.get(l, float("nan"))
                if math.isnan(v):
                    cells.append("—")
                elif not math.isnan(best_global) and abs(v - best_global) < 1e-9:
                    cells.append(rf"\textbf{{{v:.3f}}}")
                else:
                    cells.append(f"{v:.3f}")

            shuf_str = f"{shuf_auc:.3f}" if not math.isnan(shuf_auc) else "—"
            probe_label = PROBE_LABELS.get(probe, probe)
            lines.append(f"{probe_label} & " + " & ".join(cells) + f" & {shuf_str} \\\\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Calibration table
# ---------------------------------------------------------------------------

def build_calibration_table(
    results_dir: Path,
    probes: list[str],
    model_run_ids: list[str],
    layers: list[int],
    pooled_run_ids: list[str] | None = None,
) -> str:
    col_spec = "l" + "cc" * len(layers)
    layer_header = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{Layer {l}}}" for l in layers
    )
    sub_header = " & ".join([r"ECE $\downarrow$ & Brier $\downarrow$"] * len(layers))
    n_metric_cols = len(layers) * 2

    lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\caption{Calibration metrics (ECE and Brier score) for each probe across layers."
        r" Lower is better for both metrics.}",
        r"\label{tab:calibration}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Probe & {layer_header} \\",
        rf"\cmidrule(lr){{2-{1 + n_metric_cols}}}",
        rf"& {sub_header} \\",
        r"\midrule",
    ]

    probe_run_ids = pooled_run_ids if pooled_run_ids else model_run_ids

    for probe_run_id, run_id in zip(probe_run_ids, model_run_ids):
        model_label = MODEL_LABELS.get(run_id, run_id)
        n_cols = 1 + n_metric_cols
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{model_label}}}}} \\")

        for probe in probes:
            all_res = _load(results_dir, probe_run_id, probe)
            if all_res is None:
                continue
            cells = []
            for l in layers:
                ece = _weighted_mean(all_res.get(l, []), "test_ece")
                brier = _weighted_mean(all_res.get(l, []), "test_brier")
                ece_str = f"{ece:.3f}" if not math.isnan(ece) else "—"
                brier_str = f"{brier:.3f}" if not math.isnan(brier) else "—"
                cells.append(f"{ece_str} & {brier_str}")
            probe_label = PROBE_LABELS.get(probe, probe)
            lines.append(f"{probe_label} & " + " & ".join(cells) + r" \\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AUC heatmap
# ---------------------------------------------------------------------------

def plot_auc_heatmap(
    results_dir: Path,
    run_id: str,
    probe: str,
    layers: list[int],
    figures_dir: Path,
    n_bins: int = 10,
    x_label: str = "Relative position bin",
    suffix: str = "",
) -> None:
    all_res = _load(results_dir, run_id, probe)
    if all_res is None:
        print(f"  [skip] {run_id}/{probe} — no results.pt")
        return

    grid = _auc_grid(all_res, layers, n_bins)

    out_dir = figures_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
    im = ax.imshow(grid, aspect="auto", cmap="Blues", norm=norm, origin="lower")
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(
        [f"{i/n_bins:.1f}–{(i+1)/n_bins:.1f}" for i in range(n_bins)],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {l}" for l in layers])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Layer")
    probe_label = PROBE_LABELS.get(probe, probe)
    ax.set_title(f"{probe_label} — AUC-ROC ({MODEL_LABELS.get(run_id.replace('_shuffled','').replace('_step_rel','').replace('_pooled_bineval',''), run_id)})")
    plt.colorbar(im, ax=ax, label="AUC-ROC")

    # Annotate cells
    for li in range(len(layers)):
        for bi in range(n_bins):
            v = grid[li, bi]
            if not math.isnan(v):
                ax.text(bi, li, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if v > 0.8 else "black")

    plt.tight_layout()
    fname = f"{probe}_auc_heatmap{suffix}.png"
    plt.savefig(out_dir / fname, dpi=150)
    plt.close()
    print(f"  [fig] {out_dir / fname}")


# ---------------------------------------------------------------------------
# AUC vs layer line plot
# ---------------------------------------------------------------------------

def plot_layer_auc(
    results_dir: Path,
    run_ids: list[str],
    run_labels: list[str],
    probe: str,
    layers: list[int],
    figures_dir: Path,
    out_run_id: str,
) -> None:
    out_dir = figures_dir / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (run_id, label) in enumerate(zip(run_ids, run_labels)):
        all_res = _load(results_dir, run_id, probe)
        if all_res is None:
            continue
        layer_aucs = _layer_means(all_res, "test_auc", layers)
        xs = [l for l in layers if not math.isnan(layer_aucs.get(l, float("nan")))]
        ys = [layer_aucs[l] for l in xs]
        ax.plot(xs, ys, marker="o", color=colours[i % len(colours)], label=label)

    ax.axhline(0.5, linestyle="--", color="#aaa", linewidth=1, label="Random baseline")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test AUC-ROC")
    probe_label = PROBE_LABELS.get(probe, probe)
    ax.set_title(f"{probe_label} — AUC-ROC vs layer")
    ax.set_xticks(layers)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0.45, top=1.0)
    plt.tight_layout()
    fname = f"{probe}_layer_auc.png"
    plt.savefig(out_dir / fname, dpi=150)
    plt.close()
    print(f"  [fig] {out_dir / fname}")


# ---------------------------------------------------------------------------
# Transfer table (cross-dataset generalisation)
# ---------------------------------------------------------------------------

TRANSFER_ROW_LABELS = {
    "in_dist_verified": "In-distribution (Verified)",
    "in_dist_pro":      "In-distribution (Pro)",
    "verified_to_pro":  "Verified $\\rightarrow$ Pro",
    "pro_to_verified":  "Pro $\\rightarrow$ Verified",
}


def build_transfer_table(
    verified_results_dir: Path,
    pro_results_dir: Path,
    probes: list[str],
    layers: list[int],
    verified_pooled_run_id: str = "laguna_xs2_full_pooled",
    pro_pooled_run_id: str = "laguna_xs2_full",
    verified_to_pro_run_id: str = "laguna_xs2_full_verified_transfer",
    pro_to_verified_run_id: str = "laguna_xs2_full_pro_transfer",
) -> str:
    """Return a LaTeX table comparing in-distribution vs cross-dataset transfer AUC.

    Rows: in-dist Verified, in-dist Pro, Verified→Pro, Pro→Verified.
    Columns: probe × layer.
    """
    n_layer_cols = len(layers)
    col_spec = "l" + "c" * n_layer_cols
    layer_header = " & ".join(f"Layer {l}" for l in layers)

    rows = [
        ("in_dist_verified", verified_results_dir, verified_pooled_run_id),
        ("in_dist_pro",      pro_results_dir,      pro_pooled_run_id),
        ("verified_to_pro",  pro_results_dir,      verified_to_pro_run_id),
        ("pro_to_verified",  verified_results_dir, pro_to_verified_run_id),
    ]

    lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\caption{Cross-dataset transfer AUC-ROC for Laguna-XS2. "
        r"\emph{In-distribution} probes are trained and evaluated on the same dataset. "
        r"\emph{Transfer} probes use weights trained on one dataset and evaluated on the other.}",
        r"\label{tab:transfer}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf" & {layer_header} \\",
        r"\midrule",
    ]

    for probe in probes:
        probe_label = PROBE_LABELS.get(probe, probe)
        lines.append(rf"\multicolumn{{{1 + n_layer_cols}}}{{l}}{{\textit{{{probe_label}}}}} \\")

        # Find global best AUC across all rows for bolding
        best: dict[int, float] = {}
        for _, res_dir, run_id in rows:
            res = _load(res_dir, run_id, probe)
            if res is None:
                continue
            for l in layers:
                v = _weighted_mean(res.get(l, []), "test_auc")
                if not math.isnan(v) and v > best.get(l, float("-inf")):
                    best[l] = v

        for row_key, res_dir, run_id in rows:
            res = _load(res_dir, run_id, probe)
            row_label = TRANSFER_ROW_LABELS[row_key]
            cells = []
            for l in layers:
                if res is None:
                    cells.append("—")
                    continue
                v = _weighted_mean(res.get(l, []), "test_auc")
                if math.isnan(v):
                    cells.append("—")
                elif not math.isnan(best.get(l, float("nan"))) and abs(v - best[l]) < 1e-9:
                    cells.append(rf"\textbf{{{v:.3f}}}")
                else:
                    cells.append(f"{v:.3f}")
            lines.append(f"\\quad {row_label} & " + " & ".join(cells) + r" \\")

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate paper tables and figures.")
    parser.add_argument("--results-dir", default="results/swebench")
    parser.add_argument("--figures-dir", default="paper/figures")
    parser.add_argument("--output-dir", default="paper")
    parser.add_argument("--layers", nargs="+", type=int, default=[10, 20, 30, 39])
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--probes", nargs="+",
        default=["currently_compiles", "currently_correct",
                 "currently_reduces_failing", "currently_has_regressions"],
    )
    parser.add_argument(
        "--model-run-ids", nargs="+",
        default=["laguna_xs2_full", "qwen36_35b_a3b_full"],
    )
    parser.add_argument(
        "--shuffled-run-ids", nargs="+",
        default=["laguna_xs2_full_shuffled", "qwen36_35b_a3b_full_shuffled"],
    )
    parser.add_argument(
        "--step-rel-run-ids", nargs="+",
        default=["laguna_xs2_full_pooled_step_rel", "qwen36_35b_a3b_full_pooled_step_rel"],
    )
    parser.add_argument(
        "--bineval-run-ids", nargs="+",
        default=["laguna_xs2_full_pooled_bineval", "qwen36_35b_a3b_full_pooled_bineval"],
        help="Run IDs for position-bin heatmaps (pooled probe evaluated per position bin).",
    )
    parser.add_argument(
        "--pooled-run-ids", nargs="+", default=None,
        help="Run IDs with n_bins=1 pooled probes for table AUC/calibration numbers. "
             "Must match --model-run-ids in order. Falls back to --model-run-ids if omitted.",
    )
    # Transfer table args
    parser.add_argument("--pro-results-dir", default="results/swebench_pro",
                        help="Results directory for SWE-bench Pro runs.")
    parser.add_argument("--verified-pooled-run-id", default="laguna_xs2_full_pooled")
    parser.add_argument("--pro-pooled-run-id", default="laguna_xs2_full")
    parser.add_argument("--verified-to-pro-run-id", default="laguna_xs2_full_verified_transfer")
    parser.add_argument("--pro-to-verified-run-id", default="laguna_xs2_full_pro_transfer")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    layers = args.layers
    probes = args.probes

    # --- AUC table ---
    print("[table] building AUC table...")
    auc_tex = build_auc_table(
        results_dir=results_dir,
        probes=probes,
        model_run_ids=args.model_run_ids,
        shuffled_run_ids=args.shuffled_run_ids,
        layers=layers,
        pooled_run_ids=args.pooled_run_ids,
    )
    (out_dir / "auc_table.tex").write_text(auc_tex)
    print(f"  [tex] {out_dir / 'auc_table.tex'}")

    # --- Calibration table ---
    print("[table] building calibration table...")
    cal_tex = build_calibration_table(
        results_dir=results_dir,
        probes=probes,
        model_run_ids=args.model_run_ids,
        layers=layers,
        pooled_run_ids=args.pooled_run_ids,
    )
    (out_dir / "calibration_table.tex").write_text(cal_tex)
    print(f"  [tex] {out_dir / 'calibration_table.tex'}")

    # --- AUC heatmaps (position bins) ---
    print("[fig] AUC position-bin heatmaps...")
    for run_id in args.bineval_run_ids:
        for probe in probes:
            plot_auc_heatmap(
                results_dir=results_dir,
                run_id=run_id,
                probe=probe,
                layers=layers,
                figures_dir=figures_dir,
                n_bins=args.n_bins,
                x_label="Relative position bin",
            )

    # --- AUC heatmaps (step_relative bins) ---
    print("[fig] AUC step-relative heatmaps...")
    for run_id in args.step_rel_run_ids:
        for probe in probes:
            plot_auc_heatmap(
                results_dir=results_dir,
                run_id=run_id,
                probe=probe,
                layers=layers,
                figures_dir=figures_dir,
                n_bins=args.n_bins,
                x_label="Relative step bin",
                suffix="_step",
            )

    # --- AUC vs layer ---
    print("[fig] AUC vs layer plots...")
    all_run_ids = args.model_run_ids + args.shuffled_run_ids
    all_labels = (
        [MODEL_LABELS.get(r, r) for r in args.model_run_ids]
        + [MODEL_LABELS.get(r.replace("_shuffled", ""), r) + " (shuffled)" for r in args.shuffled_run_ids]
    )
    for probe in probes:
        plot_layer_auc(
            results_dir=results_dir,
            run_ids=all_run_ids,
            run_labels=all_labels,
            probe=probe,
            layers=layers,
            figures_dir=figures_dir,
            out_run_id="combined",
        )

    # --- Transfer table ---
    print("[table] building transfer table...")
    transfer_tex = build_transfer_table(
        verified_results_dir=results_dir,
        pro_results_dir=Path(args.pro_results_dir),
        probes=probes,
        layers=layers,
        verified_pooled_run_id=args.verified_pooled_run_id,
        pro_pooled_run_id=args.pro_pooled_run_id,
        verified_to_pro_run_id=args.verified_to_pro_run_id,
        pro_to_verified_run_id=args.pro_to_verified_run_id,
    )
    (out_dir / "transfer_table.tex").write_text(transfer_tex)
    print(f"  [tex] {out_dir / 'transfer_table.tex'}")

    print("[done]")


if __name__ == "__main__":
    main()
