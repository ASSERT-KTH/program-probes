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
    majority_baseline = 0.5

    for li, layer_idx in enumerate(probe_layers):
        results = all_results.get(layer_idx, [])
        for r in results:
            bin_idx = r.bin_idx if hasattr(r, "bin_idx") else r["bin_idx"]
            test_acc = r.test_acc if hasattr(r, "test_acc") else r["test_acc"]
            acc_grid[li, bin_idx] = test_acc

    out_dir = Path(figures_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_heat, ax_line) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 2]})

    norm = mcolors.Normalize(vmin=majority_baseline, vmax=1.0)
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
    ax_line.axhline(majority_baseline, linestyle="--", color="gray", label="Majority baseline")
    ax_line.set_xlim(0, 1)
    ax_line.set_ylim(0, 1)
    ax_line.set_xlabel("Relative position")
    ax_line.set_ylabel("Test accuracy")
    ax_line.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / f"{probe_name}_heatmap.png", dpi=150)
    plt.close()
