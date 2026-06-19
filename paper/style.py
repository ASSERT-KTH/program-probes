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


# --- Program-property identity (color + marker) ---
# Mirrors the LaTeX macros in main.tex (\Syntactic, \Semantic, \RedFail,
# \Regressions). Color encodes the *property* and is reserved for property
# labels (subplot titles / axis tick labels), NOT data series: data series
# still use the model colors above. Markers make the encoding survive
# grayscale and colorblindness. Keep these hex values in sync with main.tex.
PROPERTY_COLORS = {
    "syntactic":   "#E69F00",   # orange
    "semantic":    "#009E73",   # green
    "reduced":     "#CC79A7",   # purple  (reduced failing tests)
    "regressions": "#8C6D31",   # brown
}
PROPERTY_MARKERS = {
    "syntactic":   "^",   # triangle
    "semantic":    "o",   # circle
    "reduced":     "s",   # square
    "regressions": "D",   # diamond
}
PROPERTY_LABELS = {
    "syntactic":   "Syntactic Correctness",
    "semantic":    "Semantic Correctness",
    "reduced":     "Reduced Failing Tests",
    "regressions": "Introduced Regressions",
}

# --- Heatmap ---
HEATMAP_CMAP = "Blues"
HEATMAP_VMIN = 0.5
HEATMAP_VMAX = 1.0


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
