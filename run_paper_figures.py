"""Generate all paper-ready tables and figures from probe results.

Outputs
-------
- paper/figures/auc_summary_barplot.pdf         : Main paper figure — best-layer AUC per model & dataset
- paper/figures/<run_id>/<probe>_auc_heatmap.pdf
- paper/figures/<run_id>/<probe>_auc_heatmap_step.pdf
- paper/figures/<run_id>/<probe>_layer_auc.pdf
- paper/figures/<run_id>/<probe>_lookahead[_suffix].pdf
- paper/figures/<run_id>/<probe>_tool_nll_layer<N>[_suffix].pdf
- paper/auc_table.tex                           : AUC table (appendix)
- paper/calibration_table.tex                  : ECE + Brier table (appendix)
- paper/figures/transfer_barplot.pdf            : Cross-dataset transfer figure
- paper/transfer_table.tex                     : Cross-dataset transfer table (appendix)
- paper/figures/manifest.json                  : Figure index for the dashboard
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).parent))
from paper import style as _style


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(results_dir: Path, run_id: str, probe: str) -> dict | None:
    p = results_dir / run_id / probe / "results.pt"
    if not p.exists():
        return None
    return torch.load(p, weights_only=False)


def _weighted_mean(results_for_layer, field: str) -> float:
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
    best = float("nan")
    for layer_results in all_results.values():
        m = _weighted_mean(layer_results, field)
        if math.isnan(best) or m > best:
            best = m
    return best


def _layer_means(all_results: dict, field: str, layers: list[int]) -> dict[int, float]:
    return {l: _weighted_mean(all_results.get(l, []), field) for l in layers}


def _auc_grid(all_results: dict, layers: list[int], n_bins: int = 10) -> tuple[np.ndarray, int]:
    all_bins = [r.bin_idx if hasattr(r, "bin_idx") else r["bin_idx"]
                for lr in all_results.values() for r in lr]
    min_b    = min(all_bins) if all_bins else 0
    eff_bins = max(max(all_bins) - min_b + 1, n_bins) if all_bins and min_b == 0 else (max(all_bins) - min_b + 1 if all_bins else n_bins)
    grid = np.full((len(layers), eff_bins), np.nan)
    for li, layer in enumerate(layers):
        for r in all_results.get(layer, []):
            b = r.bin_idx if hasattr(r, "bin_idx") else r["bin_idx"]
            v = r.test_auc if hasattr(r, "test_auc") else r["test_auc"]
            idx = b - min_b
            if 0 <= idx < eff_bins:
                grid[li, idx] = v
    return grid, eff_bins


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

PROBE_LABELS = {
    "currently_compiles":        "Syntactic correctness",
    "currently_correct":         "Semantic correctness",
    "currently_reduces_failing": "Reduces failures",
    "currently_has_regressions": "Has regressions",
    "will_resolve":              "Will resolve",
}

MODEL_LABELS = {
    "laguna_xs2_full":         "Laguna-XS.2",
    "laguna_xs2_pro_full":     "Laguna-XS.2",
    "qwen36_35b_a3b_full":     "Qwen3.6-35B-A3B",
    "qwen36_35b_a3b_pro_full": "Qwen3.6-35B-A3B",
}

_MODEL_KEYS = list(MODEL_LABELS.keys())


def _model_key(run_id: str) -> str:
    """Strip run-suffix to get the canonical MODEL_LABELS key."""
    for key in _MODEL_KEYS:
        if run_id == key or run_id.startswith(key + "_"):
            return key
    return run_id


def _layer_label(layer: int) -> int:
    """Convert 0-indexed transformer layer to 1-indexed display label."""
    return layer + 1


# ---------------------------------------------------------------------------
# AUC summary barplot  (main paper figure)
# ---------------------------------------------------------------------------

def plot_generalization_barplot(
    verified_results_dir: Path,
    pro_results_dir: Path,
    verified_run_ids: list[str],
    pro_run_ids: list[str],
    model_labels: list[str],
    probes: list[str],
    layers: list[int],
    figures_dir: Path,
    verified_pooled_run_ids: list[str] | None = None,
    pro_pooled_run_ids: list[str] | None = None,
) -> None:
    """Grouped barplot: best-layer AUC per probe × model × dataset."""
    # Build series list: (color_key, label, results_dir, effective_run_id)
    series = []
    for i, (model_id, model_label) in enumerate(zip(verified_run_ids, model_labels)):
        model_key = "laguna" if "laguna" in model_id else "qwen"
        v_run = (verified_pooled_run_ids or verified_run_ids)[i]
        _pro_list = pro_pooled_run_ids or pro_run_ids
        p_run = _pro_list[i] if pro_run_ids and i < len(_pro_list) else None
        series.append((f"{model_key}_verified", f"{model_label} (Verified)",
                        verified_results_dir, v_run))
        series.append((f"{model_key}_pro",      f"{model_label} (Pro)",
                        pro_results_dir,      p_run))

    n_series  = len(series)
    bar_width = 0.15
    offsets   = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * bar_width
    x         = np.arange(len(probes))

    fig, ax = plt.subplots(figsize=(_style.FULL_WIDTH, _style.FIG_HEIGHT_BAR))

    legend_handles = []
    for i, (color_key, label, res_dir, run_id) in enumerate(series):
        dataset   = "pro" if color_key.endswith("_pro") else "verified"
        color     = _style.COLORS[color_key]
        hatch     = _style.HATCH[dataset]
        model_key = color_key.replace("_pro", "").replace("_verified", "")
        ec        = _style.COLORS[f"{model_key}_verified"]   # dark shade for hatch visibility

        rendered = False
        for pi, probe in enumerate(probes):
            all_res = _load(res_dir, run_id, probe) if run_id else None
            if all_res is None:
                continue
            v = _best_layer_mean(all_res, "test_auc")
            if math.isnan(v):
                continue
            ax.bar(x[pi] + offsets[i], v, width=bar_width,
                   color=color, hatch=hatch, edgecolor=ec, linewidth=0.6, zorder=3)
            rendered = True

        if rendered:
            legend_handles.append(mpatches.Patch(
                facecolor=color, hatch=hatch, edgecolor=ec, linewidth=0.6, label=label,
            ))

    ax.axhline(0.5, linestyle="--", color=_style.COLORS["baseline"],
               linewidth=1.0, zorder=2)
    legend_handles.append(plt.Line2D(
        [0], [0], linestyle="--", color=_style.COLORS["baseline"],
        linewidth=1.0, label="Random (0.5)",
    ))

    ax.set_xticks(x)
    ax.set_xticklabels(
        [PROBE_LABELS.get(p, p).replace(" ", "\n") for p in probes],
        fontsize=8.5,
    )
    ax.set_ylabel("Best-layer AUC")
    ax.set_ylim(0.45, 1.0)
    ax.legend(handles=legend_handles, ncol=2, loc="upper right")
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "auc_summary_barplot.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [fig] {out}")


# ---------------------------------------------------------------------------
# AUC table
# ---------------------------------------------------------------------------

def build_auc_table(
    results_dir: Path,
    probes: list[str],
    model_run_ids: list[str],
    shuffled_run_ids: list[str],
    layers: list[int],
    pooled_run_ids: list[str] | None = None,
    pro_results_dir: Path | None = None,
    pro_run_ids: list[str] | None = None,
    pro_shuffled_run_ids: list[str] | None = None,
    pro_pooled_run_ids: list[str] | None = None,
) -> str:
    n_layer_cols = len(layers)
    col_spec     = "l" + "c" * n_layer_cols + "c"
    layer_header = " & ".join(str(_layer_label(l)) for l in layers)
    n_cols       = 1 + n_layer_cols + 1

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{AUC per probe, model, and benchmark across transformer layers."
        r" \textbf{Bold} marks the best layer per row."
        r" \emph{Shuffled} uses label-permuted data as a sanity baseline.}",
        r"\label{tab:auc_roc}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf" & \multicolumn{{{n_layer_cols}}}{{c}}{{AUC $\uparrow$}} & Shuffled \\",
        rf"\cmidrule(lr){{2-{1 + n_layer_cols}}}",
        "Probe & " + layer_header + r" & (best layer) \\",
        r"\midrule",
    ]

    def _probe_rows(res_dir: Path, probe_run_id: str, shuf_id: str | None) -> list[str]:
        rows = []
        for probe in probes:
            all_res  = _load(res_dir, probe_run_id, probe)
            shuf_res = _load(res_dir, shuf_id, probe) if shuf_id else None
            if all_res is None:
                rows.append(
                    f"{PROBE_LABELS.get(probe, probe)} & "
                    + " & ".join(["—"] * n_layer_cols) + " & — \\\\"
                )
                continue
            layer_aucs  = _layer_means(all_res, "test_auc", layers)
            shuf_auc    = _best_layer_mean(shuf_res, "test_auc") if shuf_res else float("nan")
            best_in_row = max(
                (v for v in layer_aucs.values() if not math.isnan(v)),
                default=float("nan"),
            )
            cells = []
            for l in layers:
                v = layer_aucs.get(l, float("nan"))
                if math.isnan(v):
                    cells.append("—")
                elif not math.isnan(best_in_row) and abs(v - best_in_row) < 1e-9:
                    cells.append(rf"\textbf{{{v:.3f}}}")
                else:
                    cells.append(f"{v:.3f}")
            shuf_str = f"{shuf_auc:.3f}" if not math.isnan(shuf_auc) else "—"
            rows.append(
                f"{PROBE_LABELS.get(probe, probe)} & "
                + " & ".join(cells) + f" & {shuf_str} \\\\"
            )
        return rows

    def _dataset_block(
        dataset_label: str,
        res_dir: Path,
        run_ids: list[str],
        shuf_ids: list[str] | None,
        probe_run_ids: list[str],
    ) -> list[str]:
        if shuf_ids is None:
            shuf_ids = [None] * len(run_ids)
        block = [
            rf"\multicolumn{{{n_cols}}}{{l}}{{\textsc{{{dataset_label}}}}} \\",
            r"\addlinespace[2pt]",
        ]
        for run_id, shuf_id, probe_run_id in zip(run_ids, shuf_ids, probe_run_ids):
            model_label = MODEL_LABELS.get(_model_key(run_id), run_id)
            block.append(
                rf"\multicolumn{{{n_cols}}}{{l}}{{\quad\textit{{{model_label}}}}} \\"
            )
            block += _probe_rows(res_dir, probe_run_id, shuf_id)
            block.append(r"\addlinespace[3pt]")
        return block

    lines += _dataset_block(
        "SWE-bench Verified",
        results_dir,
        model_run_ids,
        shuffled_run_ids,
        pooled_run_ids or model_run_ids,
    )

    if pro_results_dir and pro_run_ids:
        while lines and lines[-1] == r"\addlinespace[3pt]":
            lines.pop()
        lines.append(r"\midrule")
        lines += _dataset_block(
            "SWE-bench Pro",
            pro_results_dir,
            pro_run_ids,
            pro_shuffled_run_ids,
            pro_pooled_run_ids or pro_run_ids,
        )

    while lines and lines[-1] == r"\addlinespace[3pt]":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
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
    pro_results_dir: Path | None = None,
    pro_run_ids: list[str] | None = None,
    pro_pooled_run_ids: list[str] | None = None,
) -> str:
    col_spec      = "l" + "cc" * len(layers)
    layer_header  = " & ".join(rf"\multicolumn{{2}}{{c}}{{Layer {_layer_label(l)}}}" for l in layers)
    sub_header    = " & ".join([r"ECE $\downarrow$ & Brier $\downarrow$"] * len(layers))
    n_metric_cols = len(layers) * 2
    n_cols        = 1 + n_metric_cols

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Calibration metrics (ECE and Brier score) per probe across layers."
        r" Lower is better for both metrics.}",
        r"\label{tab:calibration}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Probe & {layer_header} \\",
        rf"\cmidrule(lr){{2-{1 + n_metric_cols}}}",
        rf"& {sub_header} \\",
        r"\midrule",
    ]

    def _probe_rows(res_dir: Path, probe_run_id: str) -> list[str]:
        rows = []
        for probe in probes:
            all_res = _load(res_dir, probe_run_id, probe)
            if all_res is None:
                rows.append(
                    f"{PROBE_LABELS.get(probe, probe)} & "
                    + " & ".join(["— & —"] * len(layers)) + r" \\"
                )
                continue
            cells = []
            for l in layers:
                ece   = _weighted_mean(all_res.get(l, []), "test_ece")
                brier = _weighted_mean(all_res.get(l, []), "test_brier")
                cells.append(
                    f"{'—' if math.isnan(ece) else f'{ece:.3f}'}"
                    f" & {'—' if math.isnan(brier) else f'{brier:.3f}'}"
                )
            rows.append(
                f"{PROBE_LABELS.get(probe, probe)} & " + " & ".join(cells) + r" \\"
            )
        return rows

    def _dataset_block(
        dataset_label: str,
        res_dir: Path,
        run_ids: list[str],
        probe_run_ids: list[str],
    ) -> list[str]:
        block = [
            rf"\multicolumn{{{n_cols}}}{{l}}{{\textsc{{{dataset_label}}}}} \\",
            r"\addlinespace[2pt]",
        ]
        for run_id, probe_run_id in zip(run_ids, probe_run_ids):
            model_label = MODEL_LABELS.get(_model_key(run_id), run_id)
            block.append(
                rf"\multicolumn{{{n_cols}}}{{l}}{{\quad\textit{{{model_label}}}}} \\"
            )
            block += _probe_rows(res_dir, probe_run_id)
            block.append(r"\addlinespace[3pt]")
        return block

    lines += _dataset_block(
        "SWE-bench Verified",
        results_dir,
        model_run_ids,
        pooled_run_ids or model_run_ids,
    )

    if pro_results_dir and pro_run_ids:
        while lines and lines[-1] == r"\addlinespace[3pt]":
            lines.pop()
        lines.append(r"\midrule")
        lines += _dataset_block(
            "SWE-bench Pro",
            pro_results_dir,
            pro_run_ids,
            pro_pooled_run_ids or pro_run_ids,
        )

    while lines and lines[-1] == r"\addlinespace[3pt]":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hyperparameter table
# ---------------------------------------------------------------------------

def _sci_notation(v: float) -> str:
    """Format a float as LaTeX scientific notation: $X.XX \\times 10^{N}$."""
    exp = math.floor(math.log10(abs(v)))
    mantissa = v / 10 ** exp
    return rf"${mantissa:.2f} \times 10^{{{exp}}}$"


def build_hparam_table(
    hparams: dict,
    model_labels: list[str],
    dataset_labels: list[str],
    output_path: Path,
) -> None:
    """Write a LaTeX table of chosen probe hyperparameters to output_path.

    hparams format:
      {dataset_label: {model_label: hp_entry | None}}
    where hp_entry is either:
      {lr, weight_decay, batch_size, patience}                            — one set shared across probes
      {probe_label: {lr, weight_decay, batch_size, patience}, ...}        — per-probe
      {probe_label: {layer_str: {lr, weight_decay, batch_size, patience}, ...}, ...}  — per-probe per-layer

    Per-probe per-layer entries are collapsed to per-probe for display using the middle layer.
    """
    def _hp_cells(hp: dict) -> str:
        return (
            f"{_sci_notation(hp['lr'])} & {_sci_notation(hp['weight_decay'])} "
            f"& {hp['batch_size']} & {hp['patience']}"
        )

    def _is_per_layer(entry: dict) -> bool:
        """True when entry values are dicts of layer→hp (per-probe per-layer)."""
        first = next(iter(entry.values()), None)
        if not isinstance(first, dict):
            return False
        first_inner = next(iter(first.values()), None)
        return isinstance(first_inner, dict)

    def _collapse_layers(entry: dict) -> dict:
        """Collapse per-probe per-layer dict to per-probe by picking the middle layer."""
        result = {}
        for probe_label, layer_hps in entry.items():
            layers_sorted = sorted(layer_hps.keys(), key=lambda x: int(x))
            mid = layers_sorted[len(layers_sorted) // 2]
            result[probe_label] = layer_hps[mid]
        return result

    def _is_per_probe(entry) -> bool:
        if entry is None:
            return False
        if _is_per_layer(entry):
            return True
        return isinstance(next(iter(entry.values())), dict)

    def _resolve_hp(entry) -> dict:
        """Return a per-probe dict, collapsing layers if needed."""
        if _is_per_layer(entry):
            return _collapse_layers(entry)
        return entry

    # Count total rows to size multirow for the dataset column
    def _n_rows_for_dataset(dataset_label: str) -> int:
        dataset_hparams = hparams.get(dataset_label, {})
        total = 0
        for model_label in model_labels:
            hp = dataset_hparams.get(model_label)
            if _is_per_probe(hp):
                total += len(_resolve_hp(hp))
            else:
                total += 1
        return total

    # Determine if we need a Probe column (any dataset has per-probe entries)
    need_probe_col = any(
        _is_per_probe(hparams.get(ds, {}).get(m))
        for ds in dataset_labels
        for m in model_labels
    )

    col_spec   = "lllrrrr" if need_probe_col else "llrrrr"
    header_row = (
        r"Dataset & Model & Probe & Learning rate & Weight decay & Batch size & Patience \\"
        if need_probe_col else
        r"Dataset & Model & Learning rate & Weight decay & Batch size & Patience \\"
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{Chosen hyperparameters for each model and benchmark, "
            r"selected by 20-trial random search maximising mean validation AUC. "
            r"Sweeps were run independently per probe and per layer; "
            r"table shows the middle probed layer as representative.}"
        ),
        r"\label{tab:probe-hparams}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_row,
        r"\midrule",
    ]

    for di, dataset_label in enumerate(dataset_labels):
        dataset_hparams = hparams.get(dataset_label, {})
        n_ds_rows = _n_rows_for_dataset(dataset_label)
        ds_multirow = rf"\multirow{{{n_ds_rows}}}{{*}}{{\textsc{{{dataset_label}}}}}"
        ds_printed = False

        for model_label in model_labels:
            hp = dataset_hparams.get(model_label)
            ds_cell = ds_multirow if not ds_printed else ""

            if _is_per_probe(hp):
                resolved = _resolve_hp(hp)
                probe_labels = list(resolved.keys())
                n_probe_rows = len(probe_labels)
                model_multirow = rf"\multirow{{{n_probe_rows}}}{{*}}{{{model_label}}}"
                for pi, probe_label in enumerate(probe_labels):
                    probe_hp = resolved[probe_label]
                    model_cell = model_multirow if pi == 0 else ""
                    d_cell = ds_cell if pi == 0 else ""
                    if need_probe_col:
                        lines.append(
                            f"{d_cell} & {model_cell} & {probe_label} & {_hp_cells(probe_hp)} \\\\"
                        )
                    ds_printed = True
            else:
                cells = _hp_cells(hp) if hp is not None else "— & — & — & —"
                if need_probe_col:
                    lines.append(f"{ds_cell} & {model_label} & & {cells} \\\\")
                else:
                    lines.append(f"{ds_cell} & {model_label} & {cells} \\\\")
                ds_printed = True

        if di < len(dataset_labels) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.write_text("\n".join(lines) + "\n")
    print(f"  [tex] {output_path}")


# ---------------------------------------------------------------------------
# Transfer barplot
# ---------------------------------------------------------------------------

def plot_transfer_barplot(
    verified_results_dir: Path,
    pro_results_dir: Path,
    probes: list[str],
    figures_dir: Path,
    verified_pooled_run_id: str = "laguna_xs2_full_pooled",
    pro_pooled_run_id: str = "laguna_xs2_full",
    verified_to_pro_run_id: str = "laguna_xs2_full_verified_transfer",
    pro_to_verified_run_id: str = "laguna_xs2_full_pro_transfer",
) -> None:
    """Grouped barplot: in-dist vs transfer AUC (best layer) per probe."""
    def _best(res_dir: Path, run_id: str, probe: str) -> float:
        res = _load(res_dir, run_id, probe)
        return _best_layer_mean(res, "test_auc") if res is not None else float("nan")

    color_v  = _style.COLORS["laguna_verified"]
    color_p  = "#5C6BC0"   # indigo — distinguishes Pro in-dist from Verified
    gray     = "#BDBDBD"
    gray_p   = "#9E9E9E"

    bar_width = 0.14
    pair_gap  = 0.02
    group_gap = 0.08
    # Two pairs per probe: [in-dist V | V→P] and [in-dist P | P→V]
    pair_w = bar_width * 2 + pair_gap
    total  = 2 * pair_w + group_gap
    left   = -(total / 2)
    offsets = {
        "indist_v":  left,
        "vp":        left + bar_width + pair_gap,
        "indist_p":  left + pair_w + group_gap,
        "pv":        left + pair_w + group_gap + bar_width + pair_gap,
    }

    x   = np.arange(len(probes))
    fig, ax = plt.subplots(figsize=(_style.FULL_WIDTH, _style.FIG_HEIGHT_BAR))

    for pi, probe in enumerate(probes):
        v_auc  = _best(verified_results_dir, verified_pooled_run_id,  probe)
        p_auc  = _best(pro_results_dir,      pro_pooled_run_id,       probe)
        vp_auc = _best(pro_results_dir,      verified_to_pro_run_id,  probe)
        pv_auc = _best(verified_results_dir, pro_to_verified_run_id,  probe)

        for val, offset, color, ec in [
            (v_auc,  offsets["indist_v"], gray,    "#9E9E9E"),
            (vp_auc, offsets["vp"],       color_v, color_v),
            (p_auc,  offsets["indist_p"], gray_p,  "#757575"),
            (pv_auc, offsets["pv"],       color_p, color_p),
        ]:
            if not math.isnan(val):
                ax.bar(x[pi] + offset, val, width=bar_width,
                       color=color, edgecolor=ec, linewidth=0.6, zorder=3)

    ax.axhline(0.5, linestyle="--", color=_style.COLORS["baseline"],
               linewidth=1.0, zorder=2)

    legend_handles = [
        mpatches.Patch(facecolor=gray,    edgecolor="#9E9E9E", linewidth=0.6, label="In-dist (Verified)"),
        mpatches.Patch(facecolor=color_v, edgecolor=color_v,  linewidth=0.6, label="Verified → Pro"),
        mpatches.Patch(facecolor=gray_p,  edgecolor="#757575", linewidth=0.6, label="In-dist (Pro)"),
        mpatches.Patch(facecolor=color_p, edgecolor=color_p,  linewidth=0.6, label="Pro → Verified"),
        plt.Line2D([0], [0], linestyle="--", color=_style.COLORS["baseline"],
                   linewidth=1.0, label="Random (0.5)"),
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(
        [PROBE_LABELS.get(p, p).replace(" ", "\n") for p in probes],
        fontsize=8.5,
    )
    ax.set_ylabel("Best-layer AUC")
    ax.set_ylim(0.45, 1.0)
    ax.legend(handles=legend_handles, ncol=3, loc="upper right")

    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "transfer_barplot.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [fig] {out}")


# ---------------------------------------------------------------------------
# Transfer table
# ---------------------------------------------------------------------------

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
    n_layer_cols = len(layers)
    col_spec     = "l" + "c" * n_layer_cols
    layer_header = " & ".join(f"Layer {_layer_label(l)}" for l in layers)

    def _fmt_delta(delta: float) -> str:
        sign  = "+" if delta >= 0 else "-"
        color = r"green!50!black" if delta >= 0 else r"red!70!black"
        return rf"{{\scriptsize \textcolor{{{color}}}{{${sign}{abs(delta):.3f}$}}}}"

    def _get_aucs(res_dir: Path, run_id: str, probe: str) -> dict[int, float]:
        res = _load(res_dir, run_id, probe)
        if res is None:
            return {l: float("nan") for l in layers}
        return {l: _weighted_mean(res.get(l, []), "test_auc") for l in layers}

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-dataset transfer AUC for Laguna-XS2. "
        r"Gray rows show in-distribution reference performance. "
        r"Transfer rows show AUC when probe weights trained on one dataset are "
        r"evaluated on the other; the subscript shows the delta relative to the "
        r"in-distribution baseline on the \emph{same} evaluation set.}",
        r"\label{tab:transfer}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf" & {layer_header} \\",
        r"\midrule",
    ]

    for probe in probes:
        probe_label = PROBE_LABELS.get(probe, probe)
        lines.append(
            rf"\multicolumn{{{1 + n_layer_cols}}}{{l}}{{\textit{{{probe_label}}}}} \\"
        )

        v_aucs  = _get_aucs(verified_results_dir, verified_pooled_run_id, probe)
        p_aucs  = _get_aucs(pro_results_dir,      pro_pooled_run_id,      probe)
        vp_aucs = _get_aucs(pro_results_dir,      verified_to_pro_run_id, probe)
        pv_aucs = _get_aucs(verified_results_dir, pro_to_verified_run_id, probe)

        def _ref_cell(v: float) -> str:
            return rf"\textcolor{{gray}}{{\small {v:.3f}}}" if not math.isnan(v) else "—"

        def _transfer_cell(transfer_v: float, ref_v: float) -> str:
            if math.isnan(transfer_v):
                return "—"
            delta     = transfer_v - ref_v if not math.isnan(ref_v) else float("nan")
            delta_str = _fmt_delta(delta) if not math.isnan(delta) else ""
            return rf"{transfer_v:.3f}\,{delta_str}"

        lines.append(
            r"\quad\textcolor{gray}{\small In-dist (Verified)} & "
            + " & ".join(_ref_cell(v_aucs[l]) for l in layers) + r" \\"
        )
        lines.append(
            r"\quad\textcolor{gray}{\small In-dist (Pro)} & "
            + " & ".join(_ref_cell(p_aucs[l]) for l in layers) + r" \\"
        )
        lines.append(
            r"\quad Verified $\rightarrow$ Pro & "
            + " & ".join(_transfer_cell(vp_aucs[l], p_aucs[l]) for l in layers) + r"  \\"
        )
        lines.append(
            r"\quad Pro $\rightarrow$ Verified & "
            + " & ".join(_transfer_cell(pv_aucs[l], v_aucs[l]) for l in layers) + r"  \\"
        )
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
    dataset_label: str = "",
) -> None:
    all_res = _load(results_dir, run_id, probe)
    if all_res is None:
        print(f"  [skip] {run_id}/{probe} — no results.pt")
        return

    grid, eff_bins = _auc_grid(all_res, layers, n_bins)
    out_dir = figures_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(_style.FULL_WIDTH, _style.FIG_HEIGHT_HEAT))
    ax.grid(False)  # override global rcParam — grid lines bleed over heatmap cells

    norm = mcolors.Normalize(vmin=_style.HEATMAP_VMIN, vmax=_style.HEATMAP_VMAX)
    # pcolormesh gives crisp cell edges with no interpolation artefacts
    im = ax.pcolormesh(grid, cmap=_style.HEATMAP_CMAP, norm=norm,
                       linewidth=0.5, edgecolors="white")

    ax.set_xticks([i + 0.5 for i in range(eff_bins)])
    ax.set_xticklabels(
        [f"{i/eff_bins:.1f}–{(i+1)/eff_bins:.1f}" for i in range(eff_bins)],
        rotation=40, ha="right",
    )
    ax.set_yticks([i + 0.5 for i in range(len(layers))])
    ax.set_yticklabels([f"Layer {_layer_label(l)}" for l in layers])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Layer")
    ax.tick_params(length=0)  # hide tick marks — cells are already delimited

    model_key   = _model_key(run_id)
    probe_label = PROBE_LABELS.get(probe, probe)
    model_str   = MODEL_LABELS.get(model_key, model_key)
    title       = f"{probe_label} — {model_str}"
    if dataset_label:
        title  += f" ({dataset_label})"
    ax.set_title(title)

    cb = plt.colorbar(im, ax=ax, label="AUC")
    cb.set_ticks([0.5, 0.625, 0.75, 0.875, 1.0])
    cb.ax.tick_params(labelsize=8)

    mid = (_style.HEATMAP_VMIN + _style.HEATMAP_VMAX) / 2
    for li in range(len(layers)):
        for bi in range(eff_bins):
            v = grid[li, bi]
            if not math.isnan(v):
                ax.text(bi + 0.5, li + 0.5, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if v > mid else "black")

    fname = f"{probe}_auc_heatmap{suffix}.pdf"
    fig.savefig(out_dir / fname)
    plt.close(fig)
    print(f"  [fig] {out_dir / fname}")


# ---------------------------------------------------------------------------
# Layer AUC line plot
# ---------------------------------------------------------------------------

def plot_layer_auc(
    results_dir: Path,
    run_ids: list[str],
    run_labels: list[str],
    probe: str,
    layers: list[int],
    figures_dir: Path,
    out_run_id: str,
    extra_series: list[tuple[Path, str, str]] | None = None,
) -> None:
    out_dir = figures_dir / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(_style.FULL_WIDTH, _style.FIG_HEIGHT_LINE))

    for run_id, label in zip(run_ids, run_labels):
        all_res = _load(results_dir, run_id, probe)
        if all_res is None:
            continue
        layer_aucs = _layer_means(all_res, "test_auc", layers)
        xs = [l for l in layers if not math.isnan(layer_aucs.get(l, float("nan")))]
        ys = [layer_aucs[l] for l in xs]

        is_shuffled = "shuffled" in run_id
        is_qwen     = "qwen"     in run_id
        if is_shuffled:
            color = _style.COLORS["shuffled"]
            ls, marker, ms, alpha = "--", None, 0, 0.7
        elif is_qwen:
            color = _style.COLORS["qwen_verified"]
            ls, marker, ms, alpha = "-", "o", 5, 1.0
        else:
            color = _style.COLORS["laguna_verified"]
            ls, marker, ms, alpha = "-", "o", 5, 1.0

        ax.plot(xs, ys, linestyle=ls, marker=marker, markersize=ms,
                color=color, alpha=alpha, label=label, linewidth=1.4)

    for extra_dir, extra_run_id, extra_label in (extra_series or []):
        all_res = _load(extra_dir, extra_run_id, probe)
        if all_res is None:
            continue
        layer_aucs = _layer_means(all_res, "test_auc", layers)
        xs = [l for l in layers if not math.isnan(layer_aucs.get(l, float("nan")))]
        ys = [layer_aucs[l] for l in xs]
        color = _style.COLORS["qwen_pro" if "qwen" in extra_run_id else "laguna_pro"]
        ax.plot(xs, ys, linestyle="--", marker="s", markersize=5,
                color=color, alpha=0.85, label=extra_label, linewidth=1.4)

    ax.axhline(0.5, linestyle=":", color=_style.COLORS["baseline"],
               linewidth=1.0, label="Random")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC")
    ax.set_title(PROBE_LABELS.get(probe, probe))
    ax.set_xticks(layers)
    ax.set_xticklabels([str(_layer_label(l)) for l in layers])
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0.45, top=1.0)

    fname = f"{probe}_layer_auc.pdf"
    fig.savefig(out_dir / fname)
    plt.close(fig)
    print(f"  [fig] {out_dir / fname}")


# ---------------------------------------------------------------------------
# Lookahead horizon plot
# ---------------------------------------------------------------------------

def plot_lookahead_horizon(
    results_dir: Path,
    run_ids: list[str],
    k_values: list[int],
    probe: str,
    layers: list[int],
    figures_dir: Path,
    out_run_id: str,
    filename_suffix: str = "",
    dataset_label: str = "",
) -> None:
    """AUC vs lookahead horizon k (assistant turns) across layers."""
    data: dict[int, list[tuple]] = {l: [] for l in layers}
    k_to_n: dict[int, int] = {}

    for k, run_id in zip(k_values, run_ids):
        all_res = _load(results_dir, run_id, probe)
        if all_res is None:
            print(f"  [lookahead] missing {run_id}/{probe}, skipping k={k}")
            continue
        for layer in layers:
            layer_results = all_res.get(layer, [])
            if not layer_results:
                continue
            total_n, total_auc_w = 0, 0.0
            for r in layer_results:
                n   = r.n_test   if hasattr(r, "n_test")   else r["n_test"]
                auc = r.test_auc if hasattr(r, "test_auc") else r["test_auc"]
                total_auc_w += auc * n
                total_n     += n
            if total_n == 0:
                continue
            data[layer].append((k, total_auc_w / total_n, total_n))
            k_to_n.setdefault(k, total_n)

    ks_present = sorted(k_to_n.keys())
    if not ks_present:
        print(f"  [skip] {probe} lookahead — no data")
        return

    n_layers     = len(layers)
    blues        = plt.get_cmap("Blues")
    layer_colors = [blues(0.85 - 0.45 * i / max(n_layers - 1, 1)) for i in range(n_layers)]
    markers      = ["o", "s", "^", "D"]
    fig_width    = max(_style.FULL_WIDTH, _style.FULL_WIDTH + (len(ks_present) - 16) * 0.12)

    fig, ax = plt.subplots(figsize=(fig_width, _style.FIG_HEIGHT_LINE + 0.5))

    for li, (layer, color) in enumerate(zip(layers, layer_colors)):
        pts = sorted(data[layer], key=lambda x: x[0])
        if not pts:
            continue
        ks   = [p[0] for p in pts]
        aucs = [p[1] for p in pts]
        ax.plot(ks, aucs, marker=markers[li % len(markers)], markersize=5,
                linewidth=1.6, color=color, label=f"Layer {_layer_label(layer)}", zorder=3)
        ax.annotate(f"L{layer}", xy=(ks[-1], aucs[-1]), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8, color=color)

    ax.axhline(0.5, linestyle="--", color=_style.COLORS["baseline"], linewidth=1.0, zorder=1)
    ax.text(ks_present[-1], 0.5, "  random", va="top", ha="left",
            fontsize=7, color="#999", transform=ax.transData)

    ax.set_xlabel("Horizon k (turns)")
    ax.set_ylabel("AUC")
    ax.set_ylim(bottom=0.48)
    model_key  = _model_key(run_ids[0]) if run_ids else ""
    model_str  = MODEL_LABELS.get(model_key, model_key)
    title      = PROBE_LABELS.get(probe, probe)
    if model_str or dataset_label:
        parts  = [s for s in [model_str, dataset_label] if s]
        title += f" — {', '.join(parts)}"
    ax.set_title(title)
    ax.margins(x=0.06)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))

    out_dir = figures_dir / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    fname  = f"{probe}_lookahead{suffix}.pdf"
    fig.savefig(out_dir / fname)
    plt.close(fig)
    print(f"  [fig] {out_dir / fname}")


# ---------------------------------------------------------------------------
# Tool NLL correlation plot
# ---------------------------------------------------------------------------

def plot_tool_nll_correlation(
    results_dir: Path,
    run_id: str,
    probe: str,
    layers: list[int],
    figures_dir: Path,
    filename_suffix: str = "",
) -> None:
    """Scatter of tool-output NLL vs per-step Brier score with LOWESS trend."""
    nll_path = results_dir / run_id / probe / "nll_corr.pt"
    if not nll_path.exists():
        print(f"  [skip] {run_id}/{probe} nll_corr.pt not found")
        return
    nll_corr = torch.load(nll_path, weights_only=False)

    out_dir = figures_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_label = PROBE_LABELS.get(probe, probe)
    suffix      = f"_{filename_suffix}" if filename_suffix else ""

    for layer in layers:
        if layer not in nll_corr:
            continue
        d         = nll_corr[layer]
        nll_vals  = d["tool_nll"]
        brier_vals = d["brier"]
        rho       = d["spearman_nll_brier"]
        n_with    = d["n_with_nll"]

        valid = ~np.isnan(nll_vals)
        if valid.sum() < 2:
            continue

        x, y = nll_vals[valid], brier_vals[valid]
        rng  = np.random.default_rng(42)
        idx  = rng.choice(len(x), min(5000, len(x)), replace=False)
        xs, ys = x[idx], y[idx]

        fig, ax = plt.subplots(figsize=(_style.HALF_WIDTH * 1.5, _style.FIG_HEIGHT_LINE + 0.5))
        ax.scatter(xs, ys, alpha=0.15, s=8,
                   color=_style.COLORS["laguna_verified"], linewidths=0)

        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            order    = np.argsort(x)
            smoothed = lowess(y[order], x[order], frac=0.3, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color=_style.COLORS["qwen_verified"],
                    lw=2, label="LOWESS")
        except ImportError:
            edges   = np.percentile(x, np.linspace(0, 100, 21))
            bin_x, bin_y = [], []
            for i in range(20):
                m = (x >= edges[i]) & (x < edges[i + 1])
                if m.sum() > 0:
                    bin_x.append(x[m].mean())
                    bin_y.append(y[m].mean())
            ax.plot(bin_x, bin_y, color=_style.COLORS["qwen_verified"],
                    lw=2, marker="o", ms=4)

        rho_str = f"ρ = {rho:.3f}" if not np.isnan(rho) else "ρ = n/a"
        ax.set_xlabel("Tool output NLL (model surprise)")
        ax.set_ylabel("Brier score (per step)")
        ax.set_title(f"{probe_label} — layer {layer}\n{rho_str}  (n = {n_with:,})")

        fname = f"{probe}_tool_nll_layer{layer}{suffix}.pdf"
        fig.savefig(out_dir / fname)
        plt.close(fig)
        print(f"  [fig] {out_dir / fname}")


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def _load_generation_stats(gen_dir: Path, cache_path: Path | None = None) -> list[dict]:
    """Return one dict per trajectory JSON: {turns, n_tokens, instance_id}.

    Results are cached to cache_path (JSON) if provided, to avoid re-reading
    thousands of files on NFS on every run.
    """
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text())
    records = []
    for f in gen_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        turns    = len(d.get("command_history", []))
        n_tokens = len((d.get("tokenization") or {}).get("token_ids", []))
        records.append({
            "file":        f.name,
            "instance_id": (d.get("metadata") or {}).get("instance_id", f.stem),
            "turns":       turns,
            "n_tokens":    n_tokens,
        })
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(records))
    return records


def _compute_probe_prevalences(label_dir: Path, cache_path: Path | None = None) -> dict[str, float]:
    """Return P(y=1) per probe from label files in label_dir.

    Skips None labels and the baseline edit (cmd_idx=-1) for probes that
    define it as always False by construction (reduces_failing, has_regressions).
    Results are cached to cache_path (JSON) if provided.
    """
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text())
    counts: dict[str, list[int]] = {
        "currently_compiles":        [],
        "currently_correct":         [],
        "currently_reduces_failing": [],
        "currently_has_regressions": [],
    }
    for f in label_dir.glob("*_labels.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        edits = d.get("edits", [])
        # Identify baseline edit (cmd_idx == -1)
        baseline = next((e for e in edits if e.get("cmd_idx") == -1), None)
        baseline_failed  = set((baseline.get("test_results") or {}).get("failed", [])) if baseline else set()
        baseline_passed  = set((baseline.get("test_results") or {}).get("passed", [])) if baseline else set()
        baseline_n_fail  = len(baseline_failed)

        for edit in edits:
            cidx = edit.get("cmd_idx")

            # currently_compiles
            c = edit.get("compiles")
            if c is not None:
                counts["currently_compiles"].append(int(bool(c)))

            # currently_correct
            tr = edit.get("test_results") or {}
            resolved = tr.get("resolved")
            if resolved is not None:
                counts["currently_correct"].append(int(bool(resolved)))

            # reduces_failing: skip baseline (False by construction)
            if cidx != -1:
                if edit.get("test_results") is not None:
                    curr_fail = len(tr.get("failed", []))
                    counts["currently_reduces_failing"].append(int(curr_fail < baseline_n_fail))

            # has_regressions: skip baseline (False by construction)
            if cidx != -1:
                if edit.get("test_results") is not None:
                    curr_failed_set = set(tr.get("failed", []))
                    counts["currently_has_regressions"].append(
                        int(not curr_failed_set.isdisjoint(baseline_passed))
                    )

    result = {
        probe: (sum(v) / len(v)) if v else float("nan")
        for probe, v in counts.items()
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result))
    return result


def plot_dataset_turns(
    configs: list[dict],
    figures_dir: Path,
) -> None:
    """Panel A: violin/box plots of trajectory turn counts per (model, dataset).

    Each config dict: {gen_dir, label, color}.
    """
    all_turns  = [c["turns"] for c in configs]
    all_labels = [c["label"] for c in configs]
    all_colors = [c["color"] for c in configs]

    fig, ax = plt.subplots(figsize=(max(3.5, 1.1 * len(configs)), 3.5))

    for i, (turns, color) in enumerate(zip(all_turns, all_colors), start=1):
        arr = np.array(turns)
        if not len(arr):
            continue
        if len(arr) >= 50:
            vp = ax.violinplot([arr], positions=[i], widths=0.6, showmedians=False,
                               showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.5)
        else:
            bp = ax.boxplot([arr], positions=[i], widths=0.5, patch_artist=True,
                            medianprops=dict(visible=False), whiskerprops=dict(linewidth=0.8),
                            capprops=dict(linewidth=0.8), flierprops=dict(markersize=2))
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
        # Median line
        med = np.median(arr)
        ax.hlines(med, i - 0.3, i + 0.3, colors=color, linewidth=1.8, zorder=5)

    ax.axhline(15,  color="#555555", linestyle="--", linewidth=0.9, label="15 turns")
    ax.axhline(50, color="#222222", linestyle=":",  linewidth=0.9, label="50 turns")
    ax.set_xticks(range(1, len(all_labels) + 1))
    ax.set_xticklabels(all_labels, fontsize=8)
    ax.set_ylabel("Turns")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=7, frameon=False)

    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "dataset_turns_barplot.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


def plot_label_prevalence_heatmap(
    configs: list[dict],
    probes: list[str],
    figures_dir: Path,
) -> None:
    """Panel B: heatmap of P(y=1) per probe × (model, dataset).

    Each config dict: {label_dir, label} (or label_dir=None for placeholder).
    Rows = probes, columns = model-dataset combos.
    """
    col_labels = [c["label"] for c in configs]
    row_labels  = [PROBE_LABELS.get(p, p) for p in probes]
    n_rows, n_cols = len(probes), len(configs)
    grid = np.full((n_rows, n_cols), np.nan)

    for j, cfg in enumerate(configs):
        if cfg.get("label_dir") is None:
            continue
        prevs = _compute_probe_prevalences(cfg["label_dir"], cache_path=cfg.get("cache_path"))
        for i, probe in enumerate(probes):
            grid[i, j] = prevs.get(probe, np.nan)

    fig, ax = plt.subplots(figsize=(max(4.0, 1.3 * n_cols), max(2.5, 0.8 * n_rows)))
    cmap = plt.cm.RdYlGn
    im   = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="P(y = 1)")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8, ha="right", rotation=30)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)

    for i in range(n_rows):
        for j in range(n_cols):
            v = grid[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.2 < v < 0.8 else "white")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#888888")

    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "label_prevalence_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


def build_dataset_stats_table(
    configs: list[dict],
    probes: list[str],
    out_path: Path,
) -> None:
    """LaTeX table: rows = (model, dataset), columns = #trajectories / #tokens /
    #train states / #val states / #test states / ≥15 turns / ≥50 turns.

    States are hidden-state captures at assistant-generated tokens with stride=5
    for the currently_compiles probe, split into train/val/test.
    Each config dict: {model_label, dataset_label, gen_stats, results_pt_path, placeholder}.
    """
    caption = (
        r"Dataset statistics per (model, dataset) combination. "
        r"\#Traj.\ counts all agent runs including those that hit the step limit. "
        r"\#Tokens is the total token count across all runs. "
        r"$\geq$15 and $\geq$50 count trajectories reaching those turn thresholds, "
        r"the length filters used in the lookahead experiments. "
        r"The $k_{\max}=15$ filter is nearly lossless (90--99\,\% of runs); "
        r"the $k_{\max}=50$ filter retains 40--60\,\%."
    )
    header = (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\small" "\n"
        r"\caption{" + caption + r"}" "\n"
        r"\label{tab:dataset_stats}" "\n"
        r"\begin{tabular}{llrrrr}" "\n"
        r"\toprule" "\n"
        r"Model & Dataset & \#Traj. & \#Tokens"
        r" & \#Traj.\ $\geq$15 turns & \#Traj.\ $\geq$50 turns \\" "\n"
        r"\midrule" "\n"
    )
    rows = []
    for cfg in configs:
        if cfg.get("placeholder"):
            rows.append(
                f"{cfg['model_label']} & {cfg['dataset_label']} & "
                r"\multicolumn{4}{c}{---} \\"
            )
            continue
        stats  = cfg["gen_stats"]
        n_traj = len(stats)
        n_tok  = sum(s.get("n_tokens", 0) for s in stats)
        gt15   = sum(1 for s in stats if s["turns"] >= 15)
        gt50   = sum(1 for s in stats if s["turns"] >= 50)

        rows.append(
            f"{cfg['model_label']} & {cfg['dataset_label']} & "
            f"{n_traj:,} & {n_tok:,} & {gt15:,} & {gt50:,} \\\\"
        )

    body   = "\n".join(rows)
    footer = (
        "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )
    tex = header + body + footer
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex)
    print(f"  [tex] {out_path}")


# ---------------------------------------------------------------------------
# Dashboard manifest
# ---------------------------------------------------------------------------

def write_figures_manifest(figures_dir: Path) -> None:
    """Scan figures_dir for PDFs and write manifest.json for the dashboard."""
    entries = []
    for png in sorted(figures_dir.rglob("*.pdf")):
        rel     = png.relative_to(figures_dir)
        parts   = rel.parts
        run_id  = parts[0] if len(parts) > 1 else ""
        title   = rel.stem.replace("_", " ")
        if run_id:
            title = f"{run_id} / {title}"
        entries.append({"run_id": run_id, "path": str(rel), "title": title})
    manifest = figures_dir / "manifest.json"
    manifest.write_text(json.dumps(entries, indent=2))
    print(f"  [manifest] {manifest} ({len(entries)} figures)")


# ---------------------------------------------------------------------------
# Dataset statistics orchestration
# ---------------------------------------------------------------------------

def _generate_dataset_stats(
    args,
    results_dir: Path,
    pro_results_dir: Path,
    figures_dir: Path,
    out_dir: Path,
    probes: list[str],
) -> None:
    print("[fig/table] dataset statistics...")
    gen_root    = Path(args.generations_dir)
    label_root  = Path(args.labels_dir)
    cache_dir   = Path(args.dataset_stats_cache_dir)

    _verified_color = _style.COLORS.get("laguna_verified", "#1565C0")
    _pro_color      = _style.COLORS.get("laguna_pro",      "#64B5F6")
    _qwen_color     = _style.COLORS.get("qwen_verified",   "#BF360C")
    _qwen_pro_color = _style.COLORS.get("qwen_pro",        "#FF8A65")

    _ds_entries = [
        # (model_run_id, dataset_slug, model_label, dataset_label, color, placeholder)
        ("laguna_xs2_full",     "swebench",     "Laguna-XS.2",     "Verified", _verified_color, False),
        ("qwen36_35b_a3b_full", "swebench",     "Qwen3.6-35B-A3B", "Verified", _qwen_color,     False),
        ("laguna_xs2_full",     "swebench_pro", "Laguna-XS.2",     "Pro",      _pro_color,      False),
        ("qwen36_35b_a3b_full", "swebench_pro", "Qwen3.6-35B-A3B", "Pro",      _qwen_pro_color, True),
    ]

    ds_configs_turns: list[dict] = []
    ds_configs_prev:  list[dict] = []
    ds_configs_stats: list[dict] = []

    for model_run_id, dataset_slug, model_lbl, ds_lbl, color, placeholder in _ds_entries:
        slug       = f"{dataset_slug}__{model_run_id}"
        gen_dir    = gen_root   / dataset_slug / model_run_id
        label_dir  = label_root / dataset_slug / model_run_id
        pooled_id  = (
            f"{model_run_id.replace('_full', '_pro_full')}_pooled"
            if dataset_slug == "swebench_pro"
            else f"{model_run_id}_pooled"
        )
        rpt = (results_dir     / pooled_id / "currently_compiles" / "results.pt"
               if dataset_slug == "swebench"
               else pro_results_dir / pooled_id / "currently_compiles" / "results.pt")

        label_str = f"{model_lbl}\n({ds_lbl})"
        if not placeholder and gen_dir.exists():
            gen_stats = _load_generation_stats(
                gen_dir,
                cache_path=cache_dir / f"gen_stats__{slug}.json",
            )
        else:
            gen_stats = []
        turns = [s["turns"] for s in gen_stats]

        ld = label_dir if (not placeholder and label_dir.exists()) else None
        ds_configs_turns.append({"label": label_str, "turns": turns, "color": color})
        ds_configs_prev.append({
            "label":     label_str,
            "label_dir": ld,
            "cache_path": cache_dir / f"prevalences__{slug}.json" if ld else None,
        })
        ds_configs_stats.append({
            "model_label":    model_lbl,
            "dataset_label":  ds_lbl,
            "placeholder":    placeholder,
            "gen_stats":      gen_stats,
            "results_pt_path": str(rpt),
        })

    if any(c["turns"] for c in ds_configs_turns):
        plot_dataset_turns(ds_configs_turns, figures_dir)
    plot_label_prevalence_heatmap(ds_configs_prev, probes, figures_dir)
    build_dataset_stats_table(
        ds_configs_stats, probes,
        out_path=out_dir / "dataset_stats_table.tex",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate paper tables and figures.")
    parser.add_argument("--results-dir",  default="results/swebench")
    parser.add_argument("--figures-dir",  default="paper/figures")
    parser.add_argument("--output-dir",   default="paper")
    parser.add_argument("--layers",       nargs="+", type=int, default=[0, 10, 20, 30, 39])
    parser.add_argument("--n-bins",       type=int,  default=10)
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
        default=["laguna_xs2_full_pooled_shuffled", "qwen36_35b_a3b_full_pooled_shuffled"],
    )
    parser.add_argument(
        "--step-rel-run-ids", nargs="+",
        default=["laguna_xs2_full_pooled_step_rel", "qwen36_35b_a3b_full_pooled_step_rel"],
    )
    parser.add_argument(
        "--bineval-run-ids", nargs="+",
        default=["laguna_xs2_full_pooled_bineval", "qwen36_35b_a3b_full_pooled_bineval"],
    )
    parser.add_argument(
        "--after-edit-run-ids", nargs="+",
        default=["laguna_xs2_full_pooled_after_edit", "qwen36_35b_a3b_full_pooled_after_edit"],
    )
    parser.add_argument(
        "--pro-bineval-run-ids", nargs="+",
        default=["laguna_xs2_pro_full_pooled_bineval"],
    )
    parser.add_argument(
        "--pro-step-rel-run-ids", nargs="+",
        default=["laguna_xs2_pro_full_pooled_step_rel"],
    )
    parser.add_argument(
        "--pooled-run-ids", nargs="+",
        default=["laguna_xs2_full_pooled", "qwen36_35b_a3b_full_pooled"],
    )
    # Pro
    parser.add_argument("--pro-results-dir", default="results/swebench_pro")
    parser.add_argument(
        "--pro-model-run-ids", nargs="+",
        default=["laguna_xs2_pro_full_pooled"],
    )
    parser.add_argument("--pro-shuffled-run-ids", nargs="+",
                        default=["laguna_xs2_pro_full_pooled_shuffled"])
    parser.add_argument("--pro-pooled-run-ids",   nargs="+",
                        default=["laguna_xs2_pro_full_pooled"])
    # Transfer
    parser.add_argument("--verified-pooled-run-id",  default="laguna_xs2_full_pooled")
    parser.add_argument("--pro-pooled-run-id",        default="laguna_xs2_pro_full_pooled")
    parser.add_argument("--verified-to-pro-run-id",   default="laguna_xs2_full_verified_transfer")
    parser.add_argument("--pro-to-verified-run-id",   default="laguna_xs2_full_pro_transfer")
    # Lookahead — each variant is "max_k" or "max_k:run_suffix" where run_suffix is appended
    # after _max{k} in the run ID (e.g. "15:after_edit" → _shift{k}_max15_after_edit).
    parser.add_argument("--lookahead-variants", nargs="+",
                        default=["50", "50:after_edit", "15", "15:after_edit"],
                        help="Lookahead variants to plot, each as max_k or max_k:run_suffix.")
    parser.add_argument("--lookahead-run-ids", nargs="+", default=None,
                        help="Override auto-generated run IDs for a single model/dataset.")
    parser.add_argument("--lookahead-k-values", nargs="+", type=int, default=None,
                        help="Override k values matching --lookahead-run-ids.")
    parser.add_argument("--lookahead-probes", nargs="+", default=None,
                        help="Subset of probes for lookahead figures (default: all probes).")
    parser.add_argument("--no-lookahead", action="store_true",
                        help="Skip all lookahead figures.")
    # Hyperparameter table
    parser.add_argument("--hparams-file", default="paper/hparams.json",
                        help="JSON file with chosen HP values per model × dataset.")
    # Dataset statistics
    parser.add_argument("--generations-dir", default="generations",
                        help="Root directory for trajectory JSONs.")
    parser.add_argument("--labels-dir", default="labels",
                        help="Root directory for label JSON files.")
    parser.add_argument("--dataset-stats-cache-dir", default="paper/cache",
                        help="Directory for caching generation/label stats (avoids re-reading NFS files).")
    parser.add_argument("--no-dataset-stats", action="store_true",
                        help="Skip dataset statistics figures and table.")
    # Tool NLL
    parser.add_argument("--tool-nll-run-id", default=None,
                        help="Run ID whose nll_corr.pt to use for tool-NLL correlation figures.")
    parser.add_argument("--tool-nll-filename-suffix", default="")
    args = parser.parse_args()

    _style.apply()

    results_dir     = Path(args.results_dir)
    pro_results_dir = Path(args.pro_results_dir)
    figures_dir     = Path(args.figures_dir)
    out_dir         = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    layers       = args.layers
    probes       = args.probes
    model_labels = [MODEL_LABELS.get(_model_key(r), r) for r in args.model_run_ids]

    # --- AUC summary barplot ---
    print("[fig] AUC summary barplot...")
    plot_generalization_barplot(
        verified_results_dir  = results_dir,
        pro_results_dir       = pro_results_dir,
        verified_run_ids      = args.model_run_ids,
        pro_run_ids           = args.pro_model_run_ids,
        model_labels          = model_labels,
        probes                = probes,
        layers                = layers,
        figures_dir           = figures_dir,
        verified_pooled_run_ids = args.pooled_run_ids,
        pro_pooled_run_ids      = args.pro_pooled_run_ids,
    )

    # --- Tables ---
    print("[table] building AUC table...")
    auc_tex = build_auc_table(
        results_dir       = results_dir,
        probes            = probes,
        model_run_ids     = args.model_run_ids,
        shuffled_run_ids  = args.shuffled_run_ids,
        layers            = layers,
        pooled_run_ids    = args.pooled_run_ids,
        pro_results_dir   = pro_results_dir,
        pro_run_ids       = args.pro_model_run_ids,
        pro_shuffled_run_ids = args.pro_shuffled_run_ids,
        pro_pooled_run_ids   = args.pro_pooled_run_ids,
    )
    p = out_dir / "auc_table.tex"
    p.write_text(auc_tex)
    print(f"  [tex] {p}")

    # --- Calibration table ---
    print("[table] building calibration table...")
    cal_tex = build_calibration_table(
        results_dir     = results_dir,
        probes          = probes,
        model_run_ids   = args.model_run_ids,
        layers          = layers,
        pooled_run_ids  = args.pooled_run_ids,
        pro_results_dir = pro_results_dir,
        pro_run_ids     = args.pro_model_run_ids,
        pro_pooled_run_ids = args.pro_pooled_run_ids,
    )
    p = out_dir / "calibration_table.tex"
    p.write_text(cal_tex)
    print(f"  [tex] {p}")

    # --- Hyperparameter table ---
    hparams_file = Path(args.hparams_file)
    if hparams_file.exists():
        print("[table] building hyperparameter table...")
        hparams = json.loads(hparams_file.read_text())
        build_hparam_table(
            hparams        = hparams,
            model_labels   = model_labels,
            dataset_labels = ["Verified", "Pro"],
            output_path    = out_dir / "hparam_table.tex",
        )
    else:
        print(f"[table] skipping hparam table (no {hparams_file})")

    # --- AUC heatmaps (position bins) ---
    print("[fig] AUC position-bin heatmaps...")
    for run_id in args.bineval_run_ids:
        for probe in probes:
            plot_auc_heatmap(
                results_dir = results_dir, run_id = run_id, probe = probe,
                layers = layers, figures_dir = figures_dir,
                n_bins = args.n_bins, x_label = "Relative position bin",
                dataset_label = "Verified",
            )

    # --- AUC heatmaps (step-relative bins) ---
    print("[fig] AUC step-relative heatmaps...")
    for run_id in args.step_rel_run_ids:
        for probe in probes:
            plot_auc_heatmap(
                results_dir = results_dir, run_id = run_id, probe = probe,
                layers = layers, figures_dir = figures_dir,
                n_bins = args.n_bins, x_label = "Relative step bin", suffix = "_step",
                dataset_label = "Verified",
            )

    # --- AUC heatmaps (after-edit, Verified) ---
    print("[fig] AUC after-edit heatmaps...")
    for run_id in (args.after_edit_run_ids or []):
        for probe in probes:
            plot_auc_heatmap(
                results_dir = results_dir, run_id = run_id, probe = probe,
                layers = layers, figures_dir = figures_dir,
                n_bins = args.n_bins, x_label = "Relative position bin (after edit)",
                suffix = "_after_edit", dataset_label = "Verified",
            )

    # --- AUC heatmaps (Pro position bins) ---
    print("[fig] AUC Pro position-bin heatmaps...")
    for run_id in (args.pro_bineval_run_ids or []):
        for probe in probes:
            plot_auc_heatmap(
                results_dir = pro_results_dir, run_id = run_id, probe = probe,
                layers = layers, figures_dir = figures_dir,
                n_bins = args.n_bins, x_label = "Relative position bin",
                dataset_label = "Pro",
            )

    # --- AUC heatmaps (Pro step-relative bins) ---
    print("[fig] AUC Pro step-relative heatmaps...")
    for run_id in (args.pro_step_rel_run_ids or []):
        for probe in probes:
            plot_auc_heatmap(
                results_dir = pro_results_dir, run_id = run_id, probe = probe,
                layers = layers, figures_dir = figures_dir,
                n_bins = args.n_bins, x_label = "Relative step bin",
                suffix = "_step", dataset_label = "Pro",
            )

    # --- Layer AUC line plots (pooled runs only, Random baseline from axhline) ---
    print("[fig] AUC vs layer plots...")
    pooled_ids  = args.pooled_run_ids
    pool_labels = [MODEL_LABELS.get(_model_key(r), r) + " (Verified)" for r in pooled_ids]
    pro_extra   = [
        (pro_results_dir, r, MODEL_LABELS.get(_model_key(r), r) + " (Pro)")
        for r in (args.pro_pooled_run_ids or args.pro_model_run_ids or [])
    ] if pro_results_dir.exists() else []
    for probe in probes:
        plot_layer_auc(
            results_dir  = results_dir,
            run_ids      = pooled_ids,
            run_labels   = pool_labels,
            probe        = probe,
            layers       = layers,
            figures_dir  = figures_dir,
            out_run_id   = "combined",
            extra_series = pro_extra or None,
        )

    # --- Transfer barplot + table ---
    print("[fig] transfer barplot...")
    plot_transfer_barplot(
        verified_results_dir   = results_dir,
        pro_results_dir        = pro_results_dir,
        probes                 = probes,
        figures_dir            = figures_dir,
        verified_pooled_run_id = args.verified_pooled_run_id,
        pro_pooled_run_id      = args.pro_pooled_run_id,
        verified_to_pro_run_id = args.verified_to_pro_run_id,
        pro_to_verified_run_id = args.pro_to_verified_run_id,
    )
    print("[table] building transfer table...")
    transfer_tex = build_transfer_table(
        verified_results_dir  = results_dir,
        pro_results_dir       = pro_results_dir,
        probes                = probes,
        layers                = layers,
        verified_pooled_run_id  = args.verified_pooled_run_id,
        pro_pooled_run_id       = args.pro_pooled_run_id,
        verified_to_pro_run_id  = args.verified_to_pro_run_id,
        pro_to_verified_run_id  = args.pro_to_verified_run_id,
    )
    p = out_dir / "transfer_table.tex"
    p.write_text(transfer_tex)
    print(f"  [tex] {p}")

    # --- Lookahead horizon plots ---
    if not args.no_lookahead:
        lookahead_probes = args.lookahead_probes or probes

        def _parse_variant(spec: str) -> tuple[int, str]:
            """Parse 'max_k' or 'max_k:run_suffix' into (max_k, run_suffix)."""
            if ":" in spec:
                k_str, run_sfx = spec.split(":", 1)
                return int(k_str), run_sfx
            return int(spec), ""

        def _lookahead_run_ids(base: str, max_k: int, run_sfx: str) -> list[str]:
            sfx = f"_max{max_k}_{run_sfx}" if run_sfx else f"_max{max_k}"
            return [f"{base}_shift{k}{sfx}" for k in range(max_k + 1)]

        for variant_spec in args.lookahead_variants:
            max_k, run_sfx = _parse_variant(variant_spec)
            k_range = list(range(max_k + 1))
            file_suffix = f"max{max_k}_{run_sfx}" if run_sfx else f"max{max_k}"

            print(f"[fig] lookahead horizon plots (Verified, {file_suffix})...")
            for model_run_id in args.model_run_ids:
                run_ids  = args.lookahead_run_ids  or _lookahead_run_ids(model_run_id, max_k, run_sfx)
                k_values = args.lookahead_k_values or k_range
                for probe in lookahead_probes:
                    plot_lookahead_horizon(
                        results_dir     = results_dir,
                        run_ids         = run_ids,
                        k_values        = k_values,
                        probe           = probe,
                        layers          = layers,
                        figures_dir     = figures_dir,
                        out_run_id      = f"{model_run_id}_pooled",
                        filename_suffix = file_suffix,
                        dataset_label   = "Verified",
                    )

            print(f"[fig] lookahead horizon plots (Pro, {file_suffix})...")
            for model_run_id in args.pro_model_run_ids:
                run_ids = _lookahead_run_ids(model_run_id, max_k, run_sfx)
                for probe in lookahead_probes:
                    plot_lookahead_horizon(
                        results_dir     = pro_results_dir,
                        run_ids         = run_ids,
                        k_values        = k_range,
                        probe           = probe,
                        layers          = layers,
                        figures_dir     = figures_dir,
                        out_run_id      = f"{model_run_id}_pooled_pro",
                        filename_suffix = f"pro_{file_suffix}",
                        dataset_label   = "Pro",
                    )

    # --- Tool NLL correlation plots ---
    if args.tool_nll_run_id:
        print("[fig] tool NLL correlation plots...")
        for probe in probes:
            plot_tool_nll_correlation(
                results_dir     = results_dir,
                run_id          = args.tool_nll_run_id,
                probe           = probe,
                layers          = layers,
                figures_dir     = figures_dir,
                filename_suffix = args.tool_nll_filename_suffix,
            )

    # --- Dataset statistics ---
    if args.no_dataset_stats:
        print("[skip] dataset statistics (--no-dataset-stats)")
    else:
        _generate_dataset_stats(args, results_dir, pro_results_dir, figures_dir, out_dir, probes)

    # --- Dashboard manifest ---
    write_figures_manifest(figures_dir)

    print("[done]")


if __name__ == "__main__":
    main()
