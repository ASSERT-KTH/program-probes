"""Figures for rebuttal E11 (per-language / per-repo well-formedness on SWE-bench Pro).

Reads reports/rerun/lang/lang_swebench_pro_<pooled>_currently_compiles.json and writes PDF + PNG
to reports/rerun/figures/, using the paper's style (paper/style.py, as run_paper_figures.py):
  e11_compiles_by_language    fig A: per-language AUC (dot + 95% CI) per model, with the pooled AUC
                              and the language-only baseline as reference lines; underpowered
                              cells (< 20 test tasks with both classes) in gray; n tasks per label
  e11_compiles_by_repo        fig B: the same per repo (horizontal), sorted by n test tasks
  e11_compiles_transfer_by_language  fig C: Verified-trained probe on Pro next to the in-domain probe

Usage: uv run python scripts/rebuttal_per_language_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper import style as _style  # noqa: E402

ROOT = Path("reports/rerun/lang")
OUT = Path("reports/rerun/figures")
MODELS = [("laguna_xs2_pro_full_pooled", "Laguna XS.2", "laguna", "o"),
          ("qwen36_35b_a3b_pro_full_pooled", "Qwen3.6-35B-A3B", "qwen", "s")]
LANG_ORDER = ["python", "go", "js", "ts"]
GRAY = "#BDBDBD"


def load(pooled: str) -> dict:
    return json.loads((ROOT / f"lang_swebench_pro_{pooled}_currently_compiles.json").read_text())


def point_ci(r: dict, key: str) -> tuple[float, float, float]:
    v = r["auc"].get(key, float("nan"))
    c = r["ci95"].get(key) or [None, None]
    lo = v - c[0] if c[0] is not None else 0.0
    hi = c[1] - v if c[1] is not None else 0.0
    return v, lo, hi


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def reference_handles(strat_label: str) -> list:
    return [Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.0, label="pooled AUC"),
            Line2D([0], [0], color="#555555", linestyle="-.", linewidth=1.0, label=strat_label),
            Line2D([0], [0], color="#555555", linestyle=":", linewidth=1.2, label="language-only baseline"),
            Line2D([0], [0], color="#555555", marker="o", mfc="white", linestyle="", label="< 20 tasks with both classes")]


def dot(ax, x, y, err, marker, color, under, horizontal=False, size=5):
    kw = dict(fmt=marker, color=color, mfc="white" if under else color, mec=color, markersize=size,
              capsize=2, linewidth=1.0, zorder=3, ecolor=color, elinewidth=1.0, alpha=0.9)
    if horizontal:
        ax.errorbar(x, y, xerr=err, **kw)
    else:
        ax.errorbar(x, y, yerr=err, **kw)


def fig_language(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(_style.FULL_WIDTH, _style.FIG_HEIGHT_LINE))
    langs = [l for l in LANG_ORDER if any(l in d["counts_by_language"] for d in data.values())]
    width = 0.18
    handles = []
    for i, (pooled, label, slug, marker) in enumerate(MODELS):
        r = data[pooled]
        color = _style.COLORS[f"{slug}_verified"]
        for j, l in enumerate(langs):
            if l not in r["counts_by_language"]:
                continue
            v, lo, hi = point_ci(r, f"real|lang={l}")
            under = r["counts_by_language"][l]["underpowered"]
            dot(ax, j + (i - 0.5) * 2 * width, v, [[lo], [hi]], marker, color, under)
        ax.axhline(r["auc"]["real|pooled"], color=color, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axhline(r["auc"]["real|strat_lang"], color=color, linestyle="-.", linewidth=1.0, alpha=0.8)
        ax.axhline(r["auc"]["lang_only|pooled"], color=color, linestyle=":", linewidth=1.2, alpha=0.8)
        handles.append(Line2D([0], [0], color=color, marker=marker, linestyle="", label=label))
    ax.axhline(0.5, color=_style.COLORS["baseline"], linestyle="-", linewidth=0.6, zorder=0)
    ticks = []
    for l in langs:
        n = "/".join(str(data[p]["counts_by_language"].get(l, {}).get("n_tasks", 0)) for p, *_ in MODELS)
        ticks.append(f"{l}\n(n={n})")
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels(ticks)
    ax.set_ylabel("Test AUC")
    ax.set_title("Well-formedness on SWE-bench Pro, by language")
    ax.legend(handles=handles + reference_handles("same-language pairs only"), loc="upper center",
              bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=7, frameon=False)
    ax.set_xlabel("Language (n = test tasks, Laguna/Qwen)", fontsize=8)
    save(fig, "e11_compiles_by_language")


def fig_repo(data: dict) -> None:
    repos = sorted({r for d in data.values() for r in d["counts_by_repo"]},
                   key=lambda r: -max(d["counts_by_repo"].get(r, {}).get("n_tasks", 0) for d in data.values()))
    fig, ax = plt.subplots(figsize=(_style.FULL_WIDTH, 0.32 * len(repos) + 1.0))
    off = 0.18
    handles = []
    for i, (pooled, label, slug, marker) in enumerate(MODELS):
        r = data[pooled]
        color = _style.COLORS[f"{slug}_verified"]
        for j, rp in enumerate(repos):
            if rp not in r["counts_by_repo"]:
                continue
            v, lo, hi = point_ci(r, f"real|repo={rp}")
            if v != v:
                continue
            under = r["counts_by_repo"][rp]["underpowered"]
            dot(ax, v, j + (i - 0.5) * 2 * off, [[lo], [hi]], marker, color, under, horizontal=True, size=4.5)
        ax.axvline(r["auc"]["real|pooled"], color=color, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axvline(r["auc"]["real|strat_repo"], color=color, linestyle="-.", linewidth=1.0, alpha=0.8)
        ax.axvline(r["auc"]["lang_only|pooled"], color=color, linestyle=":", linewidth=1.2, alpha=0.8)
        handles.append(Line2D([0], [0], color=color, marker=marker, linestyle="", label=label))
    ax.axvline(0.5, color=_style.COLORS["baseline"], linewidth=0.6, zorder=0)
    labels = []
    for rp in repos:
        n = "/".join(str(data[p]["counts_by_repo"].get(rp, {}).get("n_tasks", 0)) for p, *_ in MODELS)
        labels.append(f"{rp} (n={n})")
    ax.set_yticks(range(len(repos)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Test AUC")
    ax.set_ylabel("Repository (n = test tasks, Laguna/Qwen)", fontsize=8)
    ax.set_title("Well-formedness on SWE-bench Pro, by repository")
    ax.legend(handles=handles + reference_handles("same-repo pairs only"), loc="upper center",
              bbox_to_anchor=(0.4, -0.1), ncol=3, fontsize=7, frameon=False)
    save(fig, "e11_compiles_by_repo")


def fig_transfer(data: dict) -> None:
    if not all("transfer|pooled" in d["auc"] for d in data.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(_style.FULL_WIDTH, _style.FIG_HEIGHT_LINE), sharey=True)
    langs = [l for l in LANG_ORDER if any(l in d["counts_by_language"] for d in data.values())]
    for ax, (pooled, label, slug, marker) in zip(axes, MODELS):
        r = data[pooled]
        for key, color, dx, name in ((f"real", _style.COLORS[f"{slug}_verified"], -0.12, "trained on Pro"),
                                     (f"transfer", _style.COLORS[f"{slug}_pro"], 0.12, "trained on Verified")):
            for j, l in enumerate(langs):
                if l not in r["counts_by_language"]:
                    continue
                v, lo, hi = point_ci(r, f"{key}|lang={l}")
                under = r["counts_by_language"][l]["underpowered"]
                dot(ax, j + dx, v, [[lo], [hi]], marker if key == "real" else "D", color, under)
            ax.axhline(r["auc"][f"{key}|pooled"], color=color, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axhline(0.5, color=_style.COLORS["baseline"], linewidth=0.6, zorder=0)
        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels([f"{l}\n(n={r['counts_by_language'].get(l, {}).get('n_tasks', 0)})" for l in langs])
        ax.set_title(label)
        ax.legend(handles=[Line2D([0], [0], color=_style.COLORS[f"{slug}_verified"], marker=marker, linestyle="", label="trained on Pro"),
                           Line2D([0], [0], color=_style.COLORS[f"{slug}_pro"], marker="D", linestyle="", label="trained on Verified"),
                           Line2D([0], [0], color="#555555", linestyle="--", label="pooled AUC"),
                           Line2D([0], [0], color="#555555", marker="o", mfc="white", linestyle="", label="< 20 tasks w/ both classes")],
                  loc="lower left", fontsize=6.5)
    axes[0].set_ylabel("Well-formedness AUC on Pro")
    save(fig, "e11_compiles_transfer_by_language")


def main() -> None:
    _style.apply()
    data = {p: load(p) for p, *_ in MODELS}
    fig_language(data)
    fig_repo(data)
    fig_transfer(data)


if __name__ == "__main__":
    main()
