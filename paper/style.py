"""Visual style constants for paper figures.

Adjust FULL_WIDTH for the target venue:
  NeurIPS : 5.50 in
  ICML    : 6.75 in
  ICLR    : 6.00 in (approx)
"""

import matplotlib.pyplot as plt

# --- Figure dimensions ---
FULL_WIDTH      = 6.75   # inches; change per venue
HALF_WIDTH      = 3.3
FIG_HEIGHT_BAR  = 3.0
FIG_HEIGHT_LINE = 2.6
FIG_HEIGHT_HEAT = 2.8

# --- Color palette ---
# Model identity = color family; dataset identity = shade + hatch
# solid fill = Verified,  hatched (///) = Pro
COLORS = {
    "laguna_verified": "#1565C0",   # dark blue
    "laguna_pro":      "#64B5F6",   # light blue
    "qwen_verified":   "#BF360C",   # dark coral
    "qwen_pro":        "#FF8A65",   # light coral
    "shuffled":        "#9E9E9E",   # neutral gray (both models)
    "baseline":        "#AAAAAA",   # random-baseline dashed line
}

HATCH = {
    "verified": "",     # solid fill
    "pro":      "///",  # diagonal hatch
}

# --- Heatmap ---
HEATMAP_CMAP = "Blues"
HEATMAP_VMIN = 0.55
HEATMAP_VMAX = 0.85


def apply() -> None:
    """Apply paper-wide rcParams. Call once at the top of main()."""
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":          9,
        "axes.titlesize":     11,
        "axes.titleweight":   "normal",
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    8,
        "legend.framealpha":  0.9,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.color":         "#E0E0E0",
        "grid.linewidth":     0.6,
        "axes.axisbelow":     True,
    })
