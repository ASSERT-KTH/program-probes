"""Generate all paper-ready tables and figures from probe results.

Outputs
-------
- paper/figures/generalization_barplot.pdf      : Main paper figure (2×2 AUC)
- paper/figures/<run_id>/<probe>_auc_heatmap.pdf
- paper/figures/<run_id>/<probe>_auc_heatmap_step.pdf
- paper/figures/<run_id>/<probe>_layer_auc.pdf
- paper/figures/<run_id>/<probe>_lookahead[_suffix].pdf
- paper/figures/<run_id>/<probe>_tool_nll_layer<N>[_suffix].pdf
- paper/auc_table.tex                           : AUC-ROC table (appendix)
- paper/calibration_table.tex                  : ECE + Brier table (appendix)
- paper/transfer_table.tex                     : Cross-dataset transfer table
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
    "laguna_xs2_full":     "Laguna-XS.2",
    "qwen36_35b_a3b_full": "Qwen3.6-35B-A3B",
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
# Generalization barplot  (main paper figure)
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
        p_run = (pro_pooled_run_ids or pro_run_ids)[i] if pro_run_ids else None
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
    ax.set_ylabel("Best-layer AUC-ROC")
    ax.set_ylim(0.45, 1.0)
    ax.legend(handles=legend_handles, ncol=2, loc="upper right")
    ax.set_title("Probe generalisation across models and datasets")

    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "generalization_barplot.pdf"
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
        r"\begin{table*}[h]",
        r"\centering",
        r"\caption{Test AUC-ROC per probe, model, and benchmark across transformer layers."
        r" \textbf{Bold} marks the best layer per row."
        r" \emph{Shuffled} uses label-permuted data as a sanity baseline.}",
        r"\label{tab:auc_roc}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf" & \multicolumn{{{n_layer_cols}}}{{c}}{{AUC-ROC $\uparrow$}} & Shuffled \\",
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
        r"\begin{table*}[h]",
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
        r"\begin{table}[h]",
        r"\centering",
        (
            r"\caption{Chosen hyperparameters for each model and benchmark, "
            r"selected by 20-trial random search maximising mean validation AUC. "
            r"Verified sweeps were run independently per probe and per layer; "
            r"table shows the middle probed layer as representative. "
            r"Pro sweeps were run independently per probe.}"
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
        r"\begin{table*}[h]",
        r"\centering",
        r"\caption{Cross-dataset transfer AUC-ROC for Laguna-XS2. "
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
) -> None:
    all_res = _load(results_dir, run_id, probe)
    if all_res is None:
        print(f"  [skip] {run_id}/{probe} — no results.pt")
        return

    grid    = _auc_grid(all_res, layers, n_bins)
    out_dir = figures_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(_style.FULL_WIDTH, _style.FIG_HEIGHT_HEAT))
    ax.grid(False)  # override global rcParam — grid lines bleed over heatmap cells

    norm = mcolors.Normalize(vmin=_style.HEATMAP_VMIN, vmax=_style.HEATMAP_VMAX)
    # pcolormesh gives crisp cell edges with no interpolation artefacts
    im = ax.pcolormesh(grid, cmap=_style.HEATMAP_CMAP, norm=norm,
                       linewidth=0.5, edgecolors="white")

    ax.set_xticks([i + 0.5 for i in range(n_bins)])
    ax.set_xticklabels(
        [f"{i/n_bins:.1f}–{(i+1)/n_bins:.1f}" for i in range(n_bins)],
        rotation=40, ha="right",
    )
    ax.set_yticks([i + 0.5 for i in range(len(layers))])
    ax.set_yticklabels([f"Layer {_layer_label(l)}" for l in layers])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Layer")
    ax.tick_params(length=0)  # hide tick marks — cells are already delimited

    model_key   = _model_key(run_id)
    probe_label = PROBE_LABELS.get(probe, probe)
    ax.set_title(f"{probe_label} — {MODEL_LABELS.get(model_key, model_key)}")

    cb = plt.colorbar(im, ax=ax, label="AUC-ROC")
    cb.set_ticks([0.5, 0.625, 0.75, 0.875, 1.0])
    cb.ax.tick_params(labelsize=8)

    mid = (_style.HEATMAP_VMIN + _style.HEATMAP_VMAX) / 2
    for li in range(len(layers)):
        for bi in range(n_bins):
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

    ax.axhline(0.5, linestyle=":", color=_style.COLORS["baseline"],
               linewidth=1.0, label="Random")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test AUC-ROC")
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
) -> None:
    """AUC-ROC vs lookahead horizon k (assistant turns) across layers."""
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

    n_str = f"  (n≈{k_to_n[ks_present[0]]:,})" if ks_present else ""
    ax.set_xlabel(f"Horizon k (turns){n_str}")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(bottom=0.48)
    ax.set_title(PROBE_LABELS.get(probe, probe))
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
        default=["laguna_xs2_full_shuffled", "qwen36_35b_a3b_full_shuffled"],
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
        "--pooled-run-ids", nargs="+",
        default=["laguna_xs2_full_pooled", "qwen36_35b_a3b_full_pooled"],
    )
    # Pro
    parser.add_argument("--pro-results-dir", default="results/swebench_pro")
    parser.add_argument(
        "--pro-model-run-ids", nargs="+",
        default=["laguna_xs2_full", "qwen36_35b_a3b_full"],
    )
    parser.add_argument("--pro-shuffled-run-ids", nargs="+", default=None)
    parser.add_argument("--pro-pooled-run-ids",   nargs="+", default=None)
    # Transfer
    parser.add_argument("--verified-pooled-run-id",  default="laguna_xs2_full_pooled")
    parser.add_argument("--pro-pooled-run-id",        default="laguna_xs2_full")
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

    # --- Generalization barplot ---
    print("[fig] generalization barplot...")
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
            )

    # --- AUC heatmaps (step-relative bins) ---
    print("[fig] AUC step-relative heatmaps...")
    for run_id in args.step_rel_run_ids:
        for probe in probes:
            plot_auc_heatmap(
                results_dir = results_dir, run_id = run_id, probe = probe,
                layers = layers, figures_dir = figures_dir,
                n_bins = args.n_bins, x_label = "Relative step bin", suffix = "_step",
            )

    # --- Layer AUC line plots (pooled runs only, Random baseline from axhline) ---
    print("[fig] AUC vs layer plots...")
    pooled_ids  = args.pooled_run_ids
    pool_labels = [MODEL_LABELS.get(_model_key(r), r) for r in pooled_ids]
    for probe in probes:
        plot_layer_auc(
            results_dir = results_dir,
            run_ids     = pooled_ids,
            run_labels  = pool_labels,
            probe       = probe,
            layers      = layers,
            figures_dir = figures_dir,
            out_run_id  = "combined",
        )

    # --- Transfer table ---
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

    # --- Dashboard manifest ---
    print("[manifest] writing figures manifest...")
    write_figures_manifest(figures_dir)

    print("[done]")


if __name__ == "__main__":
    main()
