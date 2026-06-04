import math
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path


def plot_probe_heatmap(
    run_id: str,
    probe_name: str,
    probe_layers: list[int],
    results_dir: str = "results",
    figures_dir: str = "figures",
    n_bins: int = 10,
) -> None:
    results_path = Path(results_dir) / run_id / probe_name / "results.pt"
    all_results = torch.load(results_path, weights_only=False)

    n_layers = len(probe_layers)
    acc_grid = np.full((n_layers, n_bins), np.nan)
    # Per-bin majority baseline: max(pos, neg) / n per bin, averaged across layers
    per_bin_majority = np.full(n_bins, np.nan)

    for li, layer_idx in enumerate(probe_layers):
        results = all_results.get(layer_idx, [])
        for r in results:
            bin_idx = r.bin_idx if hasattr(r, "bin_idx") else r["bin_idx"]
            test_acc = r.test_acc if hasattr(r, "test_acc") else r["test_acc"]
            acc_grid[li, bin_idx] = test_acc
            if np.isnan(per_bin_majority[bin_idx]):
                n = r.n_test if hasattr(r, "n_test") else r["n_test"]
                n_pos = r.n_pos_test if hasattr(r, "n_pos_test") else n // 2
                per_bin_majority[bin_idx] = max(n_pos, n - n_pos) / n if n > 0 else 0.5

    out_dir = Path(figures_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_heat, ax_line) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 2]})

    avg_majority = float(np.nanmean(per_bin_majority))
    norm = mcolors.Normalize(vmin=avg_majority, vmax=1.0)
    im = ax_heat.imshow(acc_grid, aspect="auto", cmap="Blues", norm=norm, origin="lower")
    ax_heat.set_xticks(range(n_bins))
    ax_heat.set_xticklabels([f"{i/n_bins:.1f}–{(i+1)/n_bins:.1f}" for i in range(n_bins)], rotation=45, ha="right")
    ax_heat.set_yticks(range(n_layers))
    ax_heat.set_yticklabels([f"Layer {li}" for li in probe_layers])
    ax_heat.set_xlabel("Relative position bin")
    ax_heat.set_ylabel("Layer")
    ax_heat.set_title(f"{probe_name} — test accuracy heatmap ({run_id})")
    plt.colorbar(im, ax=ax_heat, label="Test accuracy")

    bin_centers = [(i + 0.5) / n_bins for i in range(n_bins)]
    for li, layer_idx in enumerate(probe_layers):
        accs = acc_grid[li]
        valid = ~np.isnan(accs)
        if valid.any():
            ax_line.plot(np.array(bin_centers)[valid], accs[valid], marker="o", label=f"Layer {layer_idx}")
    # Per-bin majority baseline as step function
    for b in range(n_bins):
        if not np.isnan(per_bin_majority[b]):
            ax_line.hlines(per_bin_majority[b], b / n_bins, (b + 1) / n_bins,
                           colors="gray", linestyles="--", linewidth=1,
                           label="Per-bin majority" if b == 0 else None)
    ax_line.set_xlim(0, 1)
    ax_line.set_ylim(0, 1)
    ax_line.set_xlabel("Relative position")
    ax_line.set_ylabel("Test accuracy")
    ax_line.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / f"{probe_name}_heatmap.png", dpi=150)
    plt.close()


_PROBE_DISPLAY_NAME = {
    "currently_compiles": "Syntactic Correctness",
    "currently_correct": "Semantic Correctness",
    "currently_reduces_failing": "Reduced Failing Tests",
    "currently_has_regressions": "Introduced Regressions",
    "will_resolve": "Will Resolve",
}


def plot_lookahead_horizon(
    base_run_id: str,
    shift_run_ids: list[str],
    k_values: list[int],
    probe_name: str,
    probe_layers: list[int],
    results_dir: str = "results",
    figures_dir: str = "figures",
    filename_suffix: str = "",
) -> None:
    """Plot AUC vs lookahead horizon k (in assistant turns), k=0 on left."""
    all_k = list(k_values)
    all_run_ids = list(shift_run_ids)

    # layer → list of (k, lift, auc, n_test) across k values
    data: dict[int, list[tuple]] = {li: [] for li in probe_layers}

    for k, run_id in zip(all_k, all_run_ids):
        results_path = Path(results_dir) / run_id / probe_name / "results.pt"
        if not results_path.exists():
            print(f"  [lookahead] missing {results_path}, skipping k={k}")
            continue
        all_results = torch.load(results_path, weights_only=False)

        for layer_idx in probe_layers:
            results = all_results.get(layer_idx, [])
            if not results:
                continue
            total_n, total_correct, total_majority_w, total_auc_w = 0, 0, 0, 0.0
            for r in results:
                n = r.n_test if hasattr(r, "n_test") else r["n_test"]
                acc = r.test_acc if hasattr(r, "test_acc") else r["test_acc"]
                auc = r.test_auc if hasattr(r, "test_auc") else r["test_auc"]
                n_pos = r.n_pos_test if hasattr(r, "n_pos_test") else (r["n_pos_test"] if isinstance(r, dict) and "n_pos_test" in r else n // 2)
                total_correct += acc * n
                total_majority_w += max(n_pos, n - n_pos)
                total_auc_w += auc * n
                total_n += n
            if total_n == 0:
                continue
            agg_acc = total_correct / total_n
            agg_auc = total_auc_w / total_n
            majority = total_majority_w / total_n
            lift = agg_acc - majority
            data[layer_idx].append((k, lift, agg_auc, total_n))

    out_dir = Path(figures_dir) / base_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    k_to_n: dict[int, int] = {}
    for layer_idx in probe_layers:
        for k, _lift, _auc, n in data[layer_idx]:
            if k not in k_to_n:
                k_to_n[k] = n

    ks_present = sorted(k_to_n.keys())
    n_k = len(ks_present)
    fig_width = max(9, 9 + (n_k - 16) * 0.12)  # wider for max50

    _MARKERS = ["o", "s", "^", "D"]
    # Sequential blues: sample 4 points from dark→light so layer ordering is visible
    _cmap = plt.get_cmap("Blues")
    _seq_colours = [_cmap(v) for v in [0.85, 0.65, 0.45, 0.30]]

    with plt.style.context("seaborn-v0_8-white"):
        fig, ax_auc = plt.subplots(1, 1, figsize=(fig_width, 4.5))

        for li, layer_idx in enumerate(probe_layers):
            pts = sorted(data[layer_idx], key=lambda x: x[0])
            if not pts:
                continue
            ks = [p[0] for p in pts]
            aucs = [p[2] for p in pts]
            col = _seq_colours[li % len(_seq_colours)]
            marker = _MARKERS[li % len(_MARKERS)]
            ax_auc.plot(ks, aucs, marker=marker, markersize=5, linewidth=2,
                        color=col, label=f"Layer {layer_idx}", zorder=3)
            ax_auc.annotate(
                f"L{layer_idx}",
                xy=(ks[-1], aucs[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left", va="center", fontsize=8, color=col,
            )

        # Baseline with direct label
        ax_auc.axhline(0.5, linestyle="--", color="#bbb", linewidth=1, zorder=1)
        ax_auc.text(
            ks_present[-1], 0.5, " random",
            va="top", ha="left", fontsize=7, color="#999",
            transform=ax_auc.transData,
        )

        ax_auc.set_ylabel("AUC-ROC", fontsize=11)
        ax_auc.set_ylim(bottom=0.48)

        n_vals = [k_to_n[k] for k in ks_present if k in k_to_n]
        n_str = f"  (n = {n_vals[0]:,})" if n_vals else ""
        ax_auc.set_xlabel(f"Horizon k (turns){n_str}", fontsize=10)

        title = _PROBE_DISPLAY_NAME.get(probe_name, probe_name)
        ax_auc.set_title(title, fontsize=13, fontweight="bold")

        # Auto-select ~10 clean tick positions
        from matplotlib.ticker import MaxNLocator
        ax_auc.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax_auc.tick_params(axis="both", labelsize=9)
        ax_auc.margins(x=0.06)

        # Despine
        ax_auc.spines["top"].set_visible(False)
        ax_auc.spines["right"].set_visible(False)

        plt.tight_layout()
        suffix = f"_{filename_suffix}" if filename_suffix else ""
        plt.savefig(out_dir / f"{probe_name}_lookahead{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()
