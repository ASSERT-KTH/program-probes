"""Markdown tables for reports/rebuttal_rerun.md.

Best layer = the layer with the highest test AUC, as in run_paper_figures.py.
Legacy numbers come from the *_legacyturns result directories.

Usage:
    uv run python scripts/rebuttal_report_tables.py > reports/rerun/tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

PROBES = ["currently_compiles", "currently_correct", "currently_has_regressions", "currently_reduces_failing"]
KS = [0, 1, 2, 3, 5, 10, 15, 25, 50]
COMBOS = [
    ("swebench", "laguna_xs2_full_pooled", "Laguna XS.2 / Verified"),
    ("swebench", "qwen36_35b_a3b_full_pooled", "Qwen3.6-35B-A3B / Verified"),
    ("swebench_pro", "laguna_xs2_pro_full_pooled", "Laguna XS.2 / Pro"),
    ("swebench_pro", "qwen36_35b_a3b_pro_full_pooled", "Qwen3.6-35B-A3B / Pro"),
]


MIN_POS = 50  # cells with fewer test positives are flagged with †


def best(path: Path) -> tuple[float, int, int] | str | None:
    """(AUC, layer, n_pos_test) at the best layer; 'undef' if every bin has a single class.

    A layer's AUC is the n_test-weighted mean over its evaluation bins (as in
    run_paper_figures._weighted_mean), skipping bins whose test labels are one class.
    Pooled runs have a single bin, so this is just their test AUC.
    """
    if not path.exists():
        return None
    res = torch.load(path, weights_only=False)
    cands = []
    for L, rs in res.items():
        ok = [r for r in rs if r.n_pos_test not in (0, r.n_test)]
        if ok:
            n = sum(r.n_test for r in ok)
            cands.append((sum(r.test_auc * r.n_test for r in ok) / n, L, sum(r.n_pos_test for r in ok)))
    if not cands:
        return "undef" if res else None
    return max(cands)


def fmt(x, layer: bool = False) -> str:
    if x is None:
        return "–"
    if x == "undef":
        return "n/a‡"
    flag = "†" if x[2] < MIN_POS else ""
    return f"{x[0]:.3f}{flag}" + (f" (L{x[1]})" if layer else "")


def fmt_ci(v: float | None, c: list | None) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    if not c or c[0] is None:
        return f"{v:.3f}"
    return f"{v:.3f} [{c[0]:.3f}, {c[1]:.3f}]"


def lookahead_tables() -> None:
    for max_k, ks in ((50, KS), (15, [k for k in KS if k <= 15])):
        for sfx, title in (("", "all target tokens"), ("_after_edit", "after-edit targets only")):
            print(f"\n#### Lookahead AUC, window max{max_k}, {title} — new / legacy (best layer by test AUC)\n")
            print("| Model / benchmark | Probe | " + " | ".join(f"k={k}" for k in ks) + " |")
            print("|---|---|" + "---|" * len(ks))
            for bench, pooled, name in COMBOS:
                for probe in PROBES:
                    cells = []
                    for k in ks:
                        rid = f"{pooled}_shift{k}_max{max_k}{sfx}"
                        new = best(Path("results") / bench / rid / probe / "results.pt")
                        old = best(Path("results") / bench / f"{rid}_legacyturns" / probe / "results.pt")
                        cells.append(f"{fmt(new)} / {fmt(old)}")
                    print(f"| {name} | {probe.removeprefix('currently_')} | " + " | ".join(cells) + " |")


def main_table() -> None:
    print("\n#### Current-state probes (k=0), best layer by test AUC — task split vs paper (repository split on Verified)\n")
    print("| Model / benchmark | Probe | task split (new) | paper | transfer to other benchmark (new) | transfer (paper) |")
    print("|---|---|---|---|---|---|")
    transfer = {"laguna_xs2_full_pooled": ("swebench_pro", "laguna_xs2_full_verified_transfer"),
                "qwen36_35b_a3b_full_pooled": ("swebench_pro", "qwen36_35b_a3b_full_verified_transfer"),
                "laguna_xs2_pro_full_pooled": ("swebench", "laguna_xs2_full_pro_transfer"),
                "qwen36_35b_a3b_pro_full_pooled": ("swebench", "qwen36_35b_a3b_full_pro_transfer")}
    for bench, pooled, name in COMBOS:
        tb, trid = transfer[pooled]
        for probe in PROBES:
            r = [best(Path("results") / bench / f"{pooled}{s}" / probe / "results.pt") for s in ("", "_legacyturns")]
            r += [best(Path("results") / tb / f"{trid}{s}" / probe / "results.pt") for s in ("", "_legacyturns")]
            print(f"| {name} | {probe.removeprefix('currently_')} | " + " | ".join(fmt(x, layer=True) for x in r) + " |")


def control_table() -> None:
    print("\n#### Shuffled-label controls (best layer by test AUC)\n")
    print("| Model / benchmark | Probe | real labels | token shuffle (fixed) | trajectory shuffle (fixed) | token shuffle, legacy (test labels shuffled too) |")
    print("|---|---|---|---|---|---|")
    for bench, pooled, name in COMBOS:
        for probe in PROBES:
            r = [best(Path("results") / bench / f"{pooled}{s}" / probe / "results.pt") for s in ("", "_shuffled", "_shuffled_traj", "_shuffled_legacyturns")]
            print(f"| {name} | {probe.removeprefix('currently_')} | " + " | ".join(fmt(x, layer=True) for x in r) + " |")


def persistence_tables() -> None:
    root = Path("reports/rerun/persistence")
    for max_k in (50, 15):
        print(f"\n#### Persistence analysis, window max{max_k} (test split; 95% CI from 1000 instance-level bootstrap resamples)\n")
        print("| Model / benchmark | Probe | k | layer | n_test | flip rate | pairs w/ edit between | persistence AUC | probe AUC (same tokens) | probe − persistence | probe AUC, y_t=0 | probe AUC, y_t=1 | probe AUC, edit between | persistence AUC, edit between |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for bench, pooled, name in COMBOS:
            for probe in PROBES:
                f = root / f"{bench}_{pooled}_{probe}_max{max_k}.json"
                if not f.exists():
                    print(f"| {name} | {probe.removeprefix('currently_')} | – | missing | | | | | | | | | | |")
                    continue
                for r in json.loads(f.read_text())["rows"]:
                    c = r.get("ci95", {})
                    if r["k"] == 0:
                        print(f"| {name} | {probe.removeprefix('currently_')} | 0 | {r['layer']} | {r['n_test']} | – | – | – | {fmt_ci(r['probe_auc'], c.get('probe_auc'))} | – | – | – | – | – |")
                        continue
                    print(
                        f"| {name} | {probe.removeprefix('currently_')} | {r['k']} | {r['layer']} | {r['n_test']} | "
                        f"{r['flip_rate']:.3f} | {r['frac_edit_between']:.3f} | "
                        f"{fmt_ci(r['persistence_auc'], c.get('persistence_auc'))} | "
                        f"{fmt_ci(r['probe_auc_persist_subset'], c.get('probe_auc_persist_subset'))} | "
                        f"{fmt_ci(r.get('probe_minus_persistence'), c.get('probe_minus_persistence'))} | "
                        f"{fmt_ci(r['probe_auc_orig0'], c.get('probe_auc_orig0'))} | "
                        f"{fmt_ci(r['probe_auc_orig1'], c.get('probe_auc_orig1'))} | "
                        f"{fmt_ci(r['probe_auc_edit_between'], c.get('probe_auc_edit_between'))} | "
                        f"{fmt_ci(r['persistence_auc_edit_between'], c.get('persistence_auc_edit_between'))} |"
                    )


if __name__ == "__main__":
    print(f"† fewer than {MIN_POS} positive test tokens; ‡ test split has a single class, so AUC is undefined.")
    main_table()
    lookahead_tables()
    control_table()
    persistence_tables()
