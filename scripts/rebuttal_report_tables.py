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
        print(f"\n#### Persistence analysis, window max{max_k} (test split; 95% CI from 1000 task-level bootstrap resamples)\n")
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


CHECK_PREFIX = {"laguna_xs2_full_pooled": "laguna_xs2_full", "qwen36_35b_a3b_full_pooled": "qwen36_35b_a3b_full",
                "laguna_xs2_pro_full_pooled": "laguna_xs2_pro_full", "qwen36_35b_a3b_pro_full_pooled": "qwen36_35b_a3b_pro_full"}
CHECK_KS = ["1", "2", "3", "5", "10", "15", "25", "50"]


def _f(x, d: int = 3) -> str:
    return "–" if x is None else f"{x:.{d}f}"


def persistence_check_tables() -> None:
    """Tables from scripts/rebuttal_check_persistence_labels.py and rebuttal_persistence_turn_level.py."""
    root = Path("reports/rerun")
    checks, turns = {}, {}
    for _, pooled, name in COMBOS:
        p = CHECK_PREFIX[pooled]
        f = root / f"persistence_label_check_{p}.json"
        if f.exists():
            checks[name] = next(iter(json.loads(f.read_text()).values()))
        f = root / f"persistence_turn_level_{p}.json"
        if f.exists():
            turns[name] = json.loads(f.read_text())["rows"]

    print("\n#### Label consistency (all label files and all window-50 shift caches)\n")
    print("| Model / benchmark | Probe | positions checked | `label[t] != turn_label[step_idx[t]]` | cache `y_original` / `y` mismatches vs k=0 turn labels (k=1…50) | trajectories with any label change | label changes / trajectory | changes that revert |")
    print("|---|---|---|---|---|---|---|---|")
    for name, c in checks.items():
        for probe in PROBES:
            a = c["label_files"]["probes"][probe]
            ks = c["caches"][f"{probe}_max50"]["ks"].values()
            bad = sum(r["y_original_mismatch"] + r["y_target_mismatch"] for r in ks)
            chk = sum(r["y_original_checked"] + r["y_target_checked"] for r in ks)
            print(f"| {name} | {probe.removeprefix('currently_')} | {a['n_positions']:,} | {a['n_label_vs_turn_label_mismatch'] + a['n_files_len_mismatch']} | "
                  f"{bad} of {chk:,} | {a['frac_traj_any_label_change']:.3f} | {a['mean_label_changes_per_traj']:.2f} | {a['frac_changes_that_revert']:.3f} |")

    print("\n#### Fraction of test tokens with y_{t+k} ≠ y_t, window max50 (fraction of test trajectories with at least one such token in parentheses)\n")
    print("| Model / benchmark | Probe | " + " | ".join(f"k={k}" for k in CHECK_KS) + " |")
    print("|---|---|" + "---|" * len(CHECK_KS))
    for name, c in checks.items():
        for probe in PROBES:
            ks = c["caches"][f"{probe}_max50"]["ks"]
            cells = [f"{ks[k]['test']['flip_rate']:.3f} ({ks[k]['test']['frac_traj_with_flip']:.2f})" for k in CHECK_KS]
            print(f"| {name} | {probe.removeprefix('currently_')} | " + " | ".join(cells) + " |")

    print("\n#### Persistence at k=1, window max50, test split: tokens vs turns\n")
    print("| Model / benchmark | Probe | tokens y_t=1 | tokens y_{t+1}=1 | tokens 0→1 | tokens 1→0 | token TPR | token TNR | token persistence AUC | turns | turns y_t=1 | flipped turns | tokens / turn (all, flipped) | turn persistence AUC | turn persistence AUC, max15 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, c in checks.items():
        for probe in PROBES:
            t = c["caches"][f"{probe}_max50"]["ks"]["1"]["test"]
            tl = turns.get(name, {}).get(f"{probe}_max50")
            tl15 = turns.get(name, {}).get(f"{probe}_max15")
            row = (f"| {name} | {probe.removeprefix('currently_')} | {t['tp'] + t['fp']:,} | {t['tp'] + t['fn']:,} | {t['fn']:,} | {t['fp']:,} | "
                   f"{_f(t['tpr'])} | {_f(t['tnr'])} | {_f(t['balanced_acc'])} | ")
            if tl:
                row += (f"{tl['n_turns']:,} | {tl['n_pos_turns_t']:,} | {tl['n_flip_turns']} | {tl['tokens_per_turn']:.1f}, {_f(tl['tokens_per_flipped_turn'], 1)} | "
                        f"{_f(tl['balanced_acc'])} | {_f(tl15['balanced_acc']) if tl15 else '–'} |")
            else:
                row += "– | – | – | – | – | – |"
            print(row)


PHASE_PROBES = ["currently_correct", "currently_reduces_failing", "currently_has_regressions"]
STRATUM_TITLE = {"yt0": "within y_t = 0", "all": "all test tokens", "yt1": "within y_t = 1"}


def _a(d: dict, key: str) -> str:
    v = d["auc"].get(key)
    return "–" if v is None or v != v else f"{v:.3f}"


def _d(d: dict, key: str) -> str:
    v, c = d["auc"].get(key), d["ci95"].get(key)
    if v is None or v != v:
        return "–"
    s = f"{v:+.3f}"
    if c and c[0] is not None:
        s += f" [{c[0]:+.2f}, {c[1]:+.2f}]"
        if c[0] > 0 or c[1] < 0:
            s = f"**{s}**"
    return s


def phase_tables(root: Path = Path("reports/rerun/phase")) -> None:
    """Tables from scripts/rebuttal_phase_confound.py (E2, E16, E17)."""
    for max_k, strata in ((15, ("yt0", "all", "yt1")), (50, ("yt0", "all"))):
        for st in strata:
            print(f"\n#### Phase confound, window max{max_k}, {STRATUM_TITLE[st]}: AUC for y_(t+k) (test split; "
                  "differences with 95% task-bootstrap CI, bold if the CI excludes 0)\n")
            print("| Model / benchmark | Probe | k | n (pos) | real, val-best layer | real, test-best layer | k=0 probe (E2) | "
                  "traj-shuffle | position, causal | position, rel_pos (non-causal) | real − k=0 probe | real − traj-shuffle | "
                  "real − position causal | real − position rel_pos |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for bench, pooled, name in COMBOS:
                for probe in PHASE_PROBES:
                    f = root / f"lookahead_{bench}_{pooled}_{probe}_max{max_k}.json"
                    if not f.exists():
                        print(f"| {name} | {probe.removeprefix('currently_')} | – | missing |" + " |" * 10)
                        continue
                    for r in json.loads(f.read_text())["rows"]:
                        L = r["layers"]
                        n = r["n_test"] if st == "all" else r[f"n_{st}"]
                        npos = "" if st == "all" else f" ({r[f'n_pos_{st}']:,})"
                        print(f"| {name} | {probe.removeprefix('currently_')} | {r['k']} | {n:,}{npos} | "
                              f"{_a(r, f'real_val|{st}')} (L{L['real_val']}) | {_a(r, f'real_test|{st}')} (L{L['real_test']}) | "
                              f"{_a(r, f'k0|{st}')} (L{L['k0']}) | {_a(r, f'traj|{st}')} | {_a(r, f'pos_causal|{st}')} | "
                              f"{_a(r, f'pos_rel|{st}')} | {_d(r, f'real_val-k0|{st}')} | {_d(r, f'real_val-traj|{st}')} | "
                              f"{_d(r, f'real_val-pos_causal|{st}')} | {_d(r, f'real_val-pos_rel|{st}')} |")

    print("\n#### k=0 selectivity (E17), pooled k=0 caches, all test tokens (layers chosen on validation AUC "
          "against real labels; differences with 95% task-bootstrap CI, bold if the CI excludes 0)\n")
    print("| Model / benchmark | Probe | real (val-best) | token shuffle (val-best) | traj shuffle (val-best) | real − token | real − traj | "
          "real − token, same layer | real − traj, same layer | position, causal | position, rel_pos (non-causal) | "
          "real − position causal | real − position rel_pos | real, test-best layer |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for bench, pooled, name in COMBOS:
        for probe in PROBES:
            f = root / f"k0_{bench}_{pooled}_{probe}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            lv = r["val_best_layer"]
            print(f"| {name} | {probe.removeprefix('currently_')} | {_a(r, 'real_valbest')} (L{lv['real']}) | "
                  f"{_a(r, 'token_valbest')} (L{lv['token']}) | {_a(r, 'traj_valbest')} (L{lv['traj']}) | "
                  f"{_d(r, 'real-token')} | {_d(r, 'real-traj')} | {_d(r, 'real-token_same_layer')} | {_d(r, 'real-traj_same_layer')} | "
                  f"{_a(r, 'pos_causal')} | {_a(r, 'pos_rel')} | {_d(r, 'real-pos_causal')} | {_d(r, 'real-pos_rel')} | "
                  f"{_a(r, 'real_testbest')} (L{r['real_test_best_layer']}) |")


E18_STRATA = {"S0": "y_t=0", "S1": "pre-edit (y_t+1=0)", "S2": "outcome y_T=1", "S3": "pre-edit & y_T=1", "S4": "y_T=0"}


def e18_tables(root: Path = Path("reports/rerun/phase")) -> None:
    """Tables from scripts/rebuttal_edit_outcome_strata.py (E18)."""
    names = {"real_val": "real", "k0": "k=0 probe", "traj": "traj-shuffle", "pos_causal": "position causal",
             "pos_rel": "position rel_pos"}
    for max_k, strata in ((15, ["S0", "S1", "S2", "S3", "S4"]), (50, ["S0", "S1", "S3"])):
        print(f"\n#### E18, window max{max_k}: within y_t = 0, by edit turn and eventual outcome (AUC for y_(t+k); "
              "strongest control = highest point AUC among k=0 probe, traj-shuffle and both position baselines; "
              "95% task-bootstrap CI, bold if it excludes 0; ◦ = fewer than 20 test tasks with a positive token)\n")
        print("| Model / benchmark | Probe | k | stratum | tokens | positives | tasks w/ positive | real | strongest control | "
              "real − strongest | strongest causal control | real − strongest causal | oracle y_T |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for bench, pooled, name in COMBOS:
            for probe in PHASE_PROBES:
                f = root / f"e18_{bench}_{pooled}_{probe}_max{max_k}.json"
                if not f.exists():
                    continue
                for r in json.loads(f.read_text())["rows"]:
                    for st in strata:
                        c = r["counts"][st]
                        flag = "◦" if c["underpowered"] else ""
                        o = r["strongest_control"].get(st)
                        strongest = f"{names[o]} {_a(r, f'{o}|{st}')}" if o else "–"
                        oracle = _a(r, f"oracle_yT|{st}") if st in ("S0", "S1") else "–"
                        cc = [(r["auc"][f"{x}|{st}"], x) for x in ("k0", "traj", "pos_causal")
                              if r["auc"].get(f"{x}|{st}") == r["auc"].get(f"{x}|{st}")]
                        oc = max(cc)[1] if cc else None
                        causal = f"{names[oc]} {_a(r, f'{oc}|{st}')}" if oc else "–"
                        print(f"| {name} | {probe.removeprefix('currently_')} | {r['k']} | {st} {E18_STRATA[st]} | {c['n']:,} | "
                              f"{c['n_pos']:,} | {c['n_tasks_pos']}{flag} | {_a(r, f'real_val|{st}')}{flag} | {strongest} | "
                              f"{_d(r, f'real_val-{o}|{st}') if o else '–'} | {causal} | "
                              f"{_d(r, f'real_val-{oc}|{st}') if oc else '–'} | {oracle} |")

    print("\n#### E18 (b): last-turn label y_T of `correct` vs the final-patch evaluation (all trajectories of each run)\n")
    print("| Run | trajectories | y_T undefined (no labelled edit) | y_T=1, resolved | y_T=0, resolved | y_T=1, not resolved | "
          "y_T=0, not resolved | agreement (y_T defined) | last-edit label agreement | last edit issued in the last turn |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for bench, pooled, name in COMBOS:
        run = {"laguna_xs2_full_pooled": "laguna_xs2_full", "qwen36_35b_a3b_full_pooled": "qwen36_35b_a3b_full",
               "laguna_xs2_pro_full_pooled": "laguna_xs2_full", "qwen36_35b_a3b_pro_full_pooled": "qwen36_35b_a3b_full"}[pooled]
        f = root / f"e18_outcome_{bench}_{run}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        cm = r["yT_vs_outcome"]
        g = lambda k: cm.get(k, 0)
        print(f"| {name} | {r['n_trajectories']:,} | {g('yT=none,outcome=0') + g('yT=none,outcome=1'):,} | {g('yT=1,outcome=1'):,} | "
              f"{g('yT=0,outcome=1'):,} | {g('yT=1,outcome=0'):,} | {g('yT=0,outcome=0'):,} | {r['yT_agreement']['agreement']:.3f} | "
              f"{r['last_edit_agreement']['agreement']:.3f} | {r['n_last_edit_in_last_turn']:,} |")


def e11_tables(root: Path = Path("reports/rerun/lang")) -> None:
    """Tables from scripts/rebuttal_per_language.py (E11)."""
    pro = [(pooled, name) for bench, pooled, name in COMBOS if bench == "swebench_pro"]
    print("\n#### E11: k=0 probes on Pro, pooled vs language-/repo-controlled (validation-best layer as in E17; "
          "pooled AUC over all test tokens; 95% task-bootstrap CI; bold if a difference's CI excludes 0)\n")
    print("| Model | Probe | layer | pooled AUC | same-language pairs | same-repo pairs | language-only baseline | repo-only baseline | "
          "real − language-only | real − repo-only |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    ci = lambda r, k: (f"{r['auc'][k]:.3f} [{r['ci95'][k][0]:.3f}, {r['ci95'][k][1]:.3f}]"
                       if r["auc"].get(k) == r["auc"].get(k) and r["ci95"].get(k) and r["ci95"][k][0] is not None else "–")
    for pooled, name in pro:
        for probe in PROBES:
            f = root / f"lang_swebench_pro_{pooled}_{probe}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            print(f"| {name} | {probe.removeprefix('currently_')} | L{r['layer']} | {ci(r, 'real|pooled')} | {ci(r, 'real|strat_lang')} | "
                  f"{ci(r, 'real|strat_repo')} | {ci(r, 'lang_only|pooled')} | {ci(r, 'repo_only|pooled')} | "
                  f"{_d(r, 'real-lang_only')} | {_d(r, 'real-repo_only')} |")
    for probe in PROBES:
        rows = [(name, json.loads((root / f"lang_swebench_pro_{pooled}_{probe}.json").read_text()))
                for pooled, name in pro if (root / f"lang_swebench_pro_{pooled}_{probe}.json").exists()]
        if not rows:
            continue
        has_t = any(r["transfer_pooled"] for _, r in rows)
        print(f"\n#### E11: {probe.removeprefix('currently_')} on Pro by language (◦ = fewer than 20 test tasks with both classes)\n")
        print("| Model | language | AUC | tokens | positives | positive rate | test tasks | tasks w/ both classes |"
              + (" Verified-trained probe AUC |" if has_t else ""))
        print("|---|---|---|---|---|---|---|---|" + ("---|" if has_t else ""))
        for name, r in rows:
            for l, c in r["counts_by_language"].items():
                flag = "◦" if c["underpowered"] else ""
                print(f"| {name} | {l} | {ci(r, f'real|lang={l}')}{flag} | {c['n']:,} | {c['n_pos']:,} | {c['pos_rate']:.2f} | "
                      f"{c['n_tasks']} | {c['n_tasks_both_classes']} |" + (f" {ci(r, f'transfer|lang={l}')} |" if has_t else ""))
            if r["transfer_pooled"]:
                print(f"| {name} | **all (Verified-trained probe, L{r['transfer_layer']})** | pooled {ci(r, 'transfer|pooled')}; "
                      f"same-language pairs {ci(r, 'transfer|strat_lang')} | | | | | | |")
        if probe != "currently_compiles":
            continue
        print(f"\n#### E11: compiles on Pro by repository (◦ = fewer than 20 test tasks with both classes)\n")
        print("| Model | repository | AUC | tokens | positive rate | test tasks | tasks w/ both classes |")
        print("|---|---|---|---|---|---|---|")
        for name, r in rows:
            for rp, c in sorted(r["counts_by_repo"].items(), key=lambda kv: -kv[1]["n_tasks"]):
                flag = "◦" if c["underpowered"] else ""
                print(f"| {name} | {rp} | {ci(r, f'real|repo={rp}')}{flag} | {c['n']:,} | {c['pos_rate']:.2f} | {c['n_tasks']} | "
                      f"{c['n_tasks_both_classes']} |")


def e11b_tables(root: Path = Path("reports/rerun/lang")) -> None:
    """Tables from scripts/rebuttal_per_language.py on Verified (E11b, repository controls)."""
    ver = [(pooled, name) for bench, pooled, name in COMBOS if bench == "swebench"]
    ci = lambda r, k: (f"{r['auc'][k]:.3f} [{r['ci95'][k][0]:.3f}, {r['ci95'][k][1]:.3f}]"
                       if r["auc"].get(k) == r["auc"].get(k) and r["ci95"].get(k) and r["ci95"][k][0] is not None else "–")
    print("\n#### E11b: k=0 probes on Verified, pooled vs repository-controlled, and the submitted runs' repository-split AUC "
          "(task split: validation-best layer as in E17, pooled AUC over all test tokens, 95% task-bootstrap CI; "
          "legacy: stored pooled test AUC of the submitted runs on their test repositories astropy, django, sphinx)\n")
    print("| Model | Probe | layer | pooled AUC | same-repo pairs | repo-only baseline | real − repo-only | pooled − same-repo | "
          "legacy split, test-best layer | legacy split, val-best layer |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for pooled, name in ver:
        for probe in PROBES:
            f = root / f"lang_swebench_{pooled}_{probe}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            lg = r.get("legacy_split") or {}
            leg = (f"{lg['test_auc_test_best']:.3f} (L{lg['test_best_layer']})", f"{lg['test_auc_val_best']:.3f} (L{lg['val_best_layer']})") if lg else ("–", "–")
            print(f"| {name} | {probe.removeprefix('currently_')} | L{r['layer']} | {ci(r, 'real|pooled')} | {ci(r, 'real|strat_repo')} | "
                  f"{ci(r, 'repo_only|pooled')} | {_d(r, 'real-repo_only')} | {_d(r, 'real_pooled-real_strat_repo')} | {leg[0]} | {leg[1]} |")
    for probe in PROBES:
        print(f"\n#### E11b: {probe.removeprefix('currently_')} on Verified by repository (◦ = fewer than 20 test tasks with both classes)\n")
        print("| Model | repository | AUC | tokens | positives | positive rate | train positive rate | test tasks | tasks w/ both classes |")
        print("|---|---|---|---|---|---|---|---|---|")
        for pooled, name in ver:
            f = root / f"lang_swebench_{pooled}_{probe}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            for rp, c in sorted(r["counts_by_repo"].items(), key=lambda kv: -kv[1]["n_tasks"]):
                flag = "◦" if c["underpowered"] else ""
                tr = r["train_pos_rate_by_repo"].get(rp)
                print(f"| {name} | {rp} | {ci(r, f'real|repo={rp}')}{flag} | {c['n']:,} | {c['n_pos']:,} | {c['pos_rate']:.2f} | "
                      f"{'–' if tr is None else f'{tr:.2f}'} | {c['n_tasks']} | {c['n_tasks_both_classes']} |")


if __name__ == "__main__":
    print(f"† fewer than {MIN_POS} positive test tokens; ‡ test split has a single class, so AUC is undefined.")
    main_table()
    lookahead_tables()
    control_table()
    persistence_tables()
    persistence_check_tables()
    phase_tables()
    e18_tables()
    e11_tables()
    e11b_tables()
