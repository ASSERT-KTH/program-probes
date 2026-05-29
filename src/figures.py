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


def plot_lookahead_horizon(
    base_run_id: str,
    shift_run_ids: list[str],
    k_values: list[int],
    probe_name: str,
    probe_layers: list[int],
    results_dir: str = "results",
    figures_dir: str = "figures",
) -> None:
    """Plot probe lift and AUC vs lookahead horizon k (in assistant turns).

    X-axis is inverted: k=0 (at the label flip) is on the right; larger k
    (earlier prediction) is on the left — matching trajectory time direction.
    """
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Collect per-k, per-layer metrics
    # all_run_ids[0] is k=0 (base run), rest are shift runs
    all_k = [0] + list(k_values)
    all_run_ids = [base_run_id] + list(shift_run_ids)

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
            # Aggregate across bins weighted by n_test
            # majority baseline = weighted average of per-bin majority baselines
            total_n, total_correct, total_majority_w, total_auc_w = 0, 0, 0, 0.0
            for r in results:
                n = r.n_test if hasattr(r, "n_test") else r["n_test"]
                acc = r.test_acc if hasattr(r, "test_acc") else r["test_acc"]
                auc = r.test_auc if hasattr(r, "test_auc") else r["test_auc"]
                n_pos = r.n_pos_test if hasattr(r, "n_pos_test") else (r["n_pos_test"] if isinstance(r, dict) and "n_pos_test" in r else n // 2)
                total_correct += acc * n
                total_majority_w += max(n_pos, n - n_pos)  # per-bin majority
                total_auc_w += auc * n
                total_n += n
            if total_n == 0:
                continue
            agg_acc = total_correct / total_n
            agg_auc = total_auc_w / total_n
            majority = total_majority_w / total_n  # weighted avg per-bin majority baseline
            lift = agg_acc - majority
            data[layer_idx].append((k, lift, agg_auc, total_n))

    out_dir = Path(figures_dir) / base_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-k n_test (same across layers; use first available layer)
    k_to_n: dict[int, int] = {}
    for layer_idx in probe_layers:
        for k, _lift, _auc, n in data[layer_idx]:
            if k not in k_to_n:
                k_to_n[k] = n

    fig, (ax_lift, ax_auc) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for li, layer_idx in enumerate(probe_layers):
        pts = sorted(data[layer_idx], key=lambda x: x[0])
        if not pts:
            continue
        ks = [p[0] for p in pts]
        lifts = [p[1] for p in pts]
        aucs = [p[2] for p in pts]
        col = colours[li % len(colours)]
        label = f"Layer {layer_idx}"

        ax_lift.plot(ks, lifts, marker="o", color=col, label=label)
        ax_auc.plot(ks, aucs, marker="o", color=col, label=label)

    ax_lift.axhline(0.0, linestyle="--", color="#aaa", linewidth=1)
    ax_lift.set_ylabel("Accuracy − per-bin majority baseline")
    ax_lift.set_title(f"{probe_name} — lookahead horizon ({base_run_id})")
    ax_lift.legend(fontsize=8, loc="upper left")

    ax_auc.axhline(0.5, linestyle="--", color="#aaa", linewidth=1)
    ax_auc.set_ylabel("AUC  (random = 0.5)")
    ax_auc.set_ylim(bottom=0.48)  # anchor near random so drop-off is visible
    ax_auc.set_xlabel("Turns ahead (k)  ←earlier prediction    at flip→")

    # Invert x-axis: k=0 (at flip) on the right, larger k (earlier) on the left
    ax_lift.invert_xaxis()
    ks_present = sorted(k_to_n.keys())
    ax_auc.set_xticks(ks_present)
    ax_auc.set_xticklabels([])  # replaced by staggered annotations below

    # Staggered tick labels: alternate between two vertical offsets to avoid overlap
    for i, k in enumerate(ks_present):
        pad = 18 if i % 2 == 0 else 34
        ax_auc.annotate(
            f"{k}\n(n={k_to_n[k]:,})",
            xy=(k, ax_auc.get_ylim()[0]),
            xytext=(0, -pad),
            textcoords="offset points",
            ha="center", va="top", fontsize=7,
            annotation_clip=False,
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(out_dir / f"{probe_name}_lookahead.png", dpi=150)
    plt.close()
