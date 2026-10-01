# Rebuttal rerun: real agent turns, task-level splits, persistence analysis

Branch `rebuttal-rerun-turnfix` (no PR opened, by request). Runs completed 2026-10-01.

## Summary

- **Current-state probes (k=0) hold up**, and most AUCs rise slightly under the new task-level split: best-layer test AUC 0.69–0.84 across models, benchmarks and probes (see table below).
- **Most of the lookahead ("programming horizon") signal is label persistence; a real horizon signal remains only for some probes.** Labels rarely change at short horizons:

  | | flip rate, k ≤ 3 | flip rate, k = 10 | flip rate, k = 50 | persistence AUC, k = 1 |
  |---|---|---|---|---|
  | Verified | 0.3–5.6% | 1.4–14% | 3–42% | 0.85–0.93 |
  | Pro | 0.2–3.3% | 0.3–6.3% | 1.3–13% | 0.65–0.98 |

  Probe AUC minus persistence AUC (same tokens, window 50, 95% task-bootstrap CI):
  - **Probe clearly better (CI above 0):**
    - Laguna/Verified compiles, k ≥ 5 (+0.08 to +0.18);
    - Qwen/Verified reduces_failing, k = 5–25 (+0.08 to +0.11);
    - Laguna/Pro correct, k = 5–15 (+0.14 to +0.17), has_regressions, k = 3–5 (+0.12), and reduces_failing, k = 3–50 (+0.11 to +0.16);
    - Qwen/Pro reduces_failing, k = 5–50 (+0.19 to +0.23), and has_regressions at every k (+0.22 to +0.45). The has_regressions probe has the layer-0 caveat below, and positives are rare there.
  - **Probe clearly worse:**
    - compiles on Pro for both models at every k (−0.19 to −0.29);
    - Laguna/Verified has_regressions, k = 1–5 (−0.10 to −0.13), and correct at k = 1 (−0.17).
  - **Everything else:** the CI includes 0; the probe is indistinguishable from "label stays the same".
  - **After-edit targets:** best-layer AUC for k ≥ 1 has a median of 0.62 (range 0.40–0.86), with 14 of 128 cells below 0.5. Turns right after an edit are where labels change, and there the probe anticipates weakly at best.
- **The trajectory-level shuffle control is strong.** A probe trained on another trajectory's labels, which keep the trajectory-position structure but have no link to the hidden states, reaches **0.56–0.78 test AUC** for correct, has_regressions and reduces_failing (up to 0.68 for compiles on Verified). Much of the reported AUC is therefore available without reading program state from the hidden states.
- **The paper's shuffled control was uninformative:** it shuffled the test labels as well, which pins AUC at ~0.5 by construction (fixed in `fa41d05`).
- Several pipeline bugs were found and fixed along the way; see "Problems found and fixed".

## What changed relative to the paper

| | Paper | This rerun |
|---|---|---|
| Turn index per position | `int(i*K/T)` (even split) | real turn (`step_idx`, `turn_labels` from `run_attach_labels_swebench.py`) |
| Train/val/test unit | `--group-by project`: **repository** on Verified (12 groups; val = 1 repo), **base commit** on Pro (434 groups for 726 tasks) | **task** on both (`--group-by` removed, `ff6b081`); 70/15/15 of tasks |
| Hyperparameters | per-(probe, layer) W&B sweeps on the old split | new per-(probe, layer) sweeps on the task split (4 agents × 20 runs), reused for every k and control |
| Lookahead protocol | Verified and Laguna Pro: train for each k; **Qwen Pro: k=0 weights evaluated at every k**; Laguna Pro after-edit used k=0 weights | all four: train for each k; after-edit = eval of that k's weights on after-edit targets |
| Shuffled control | token shuffle of train **and test** labels | token and trajectory shuffles of train/val only, scored on real test labels |

Because the split, the hyperparameters and the turn indices all changed together, the new-vs-legacy columns below do not isolate the turn fix. A first pass with only the turn fix (repository split kept) was run and then discarded by request when the split was changed.

## Original run flags (recovered from `sacct` submit lines and W&B configs)

- All probes: `--probe-arch linear --n-bins 1`; seed 42; linear probe; `final` training.
- Caches: built 06-09 to 06-14 with the `--group-by` default `project` (default since `91522df`, 06-02); Laguna Pro lookahead caches also passed it explicitly.
- Hyperparameters: Verified (both models) lookahead and controls used `final --from-sweep <S>` with the k=0 sweep for each (probe, layer); Laguna Pro lookahead used typed-in per-(probe, layer) values (constant across k); Qwen Pro lookahead was eval-only with the k=0 weights; Qwen Pro shuffled used typed-in values and `--n-eval-bins 1`.
- After-edit runs: always eval-only (`eval --after-edit-only --edit-index-run-id <run>`), never retrained.
- Sweeps: `slurm/probe_sweep_coordinator.sh ... --n-agents 4 --count 20`.
- Label dirs: `labels/<bench>/<run>` (not `generations/.../labels`).

## Step 1: re-attached labels

All four runs pass: every `*_labels.pt` has `step_idx` and `turn_labels`, and `len(step_idx) == len(labels[probe]) ==` activation rows (0 errors). A turn's length is the number of extracted positions it contains.

| Run | files | zero-row files | turns | empty turns | trajectories with ≥1 empty turn | mean turn length | turn length median / p90 / p99 / max | turns per trajectory (median) |
|---|---|---|---|---|---|---|---|---|
| Laguna / Verified | 4,991 | 1 | 276,687 | 8,476 (3.1%) | 2,978 | 28.6 | 9 / 84 / 200 / 204 | 55 |
| Qwen / Verified | 4,998 | 1 | 226,346 | 0 | 0 | 19.0 | 8 / 50 / 126 / 200 | 39 |
| Laguna / Pro | 6,921 | 0 | 383,024 | 11,182 (2.9%) | 4,549 | 21.4 | 7 / 61 / 199 / 204 | 56 |
| Qwen / Pro | 5,804 | 0 | 327,700 | 0 | 0 | 12.4 | 7 / 19 / 131 / 201 | 54 |

The zero-row files have no activations and are skipped by `build_cache`.

## Step 2: k=0 identity check

Snapshots of the old caches (`y`, `sample_id`, `group_id`, old `step_idx`) are in `snapshots/`.

- **Turn fix only** (first rebuild, old split): `y`, `sample_id`, `group_id` and `edit_step_index.pt` identical for all 4 runs × 4 probes. `step_idx` changed for 90–95% of tokens:

  | Run | tokens | `step_idx` changed | mean change |
  |---|---|---|---|
  | Laguna / Verified | 7,908,907 | 95% | 8.7 turns |
  | Qwen / Verified | 4,297,044 | 90% | 5.5 turns |
  | Laguna / Pro | 8,181,945 | 95% | 9.3 turns |
  | Qwen / Pro | 4,078,069 | 91% | 6.4 turns |

- **Task split** (final rebuild): `y`, `sample_id` and the edit index are identical to the snapshots, and `group_id` equals the task ID everywhere (`reports/rerun/k0_check_taskgroups_*.json`).

| Run | tasks train / val / test | tokens train / val / test |
|---|---|---|
| Laguna / Verified | 350 / 75 / 75 | 5.56M / 1.17M / 1.18M |
| Qwen / Verified | 350 / 75 / 75 | 3.05M / 0.59M / 0.66M |
| Laguna / Pro | 508 / 108 / 110 | 5.58M / 1.28M / 1.32M |
| Qwen / Pro | 485 / 104 / 105 | 2.87M / 0.61M / 0.60M |

On the paper's split, Verified validation was a single repository (flask, 1.8k–3.6k tokens). In the window-50 lookahead runs it was single-class (val AUC 0.5), so early stopping had no signal there.

## Problems found and fixed

1. **Shuffled-label control shuffled the test labels** (`src/probe.py`, since `3d7b934`). Both modes now shuffle train/val only. Fix and regression test in `fa41d05`, which is also on local `main` (not pushed: no GitHub access from this node).
2. **Race in `run_final`'s results/weights merge** (since `47f4a37`). Concurrent per-layer jobs could read a half-written file or silently drop another layer. Fixed with an `fcntl` lock and atomic writes, plus a multi-process test (`05b8f24`). Two layers were lost during the turn-fix pass and repaired. Final audit: every result file has all 5 layers, legacy results included (the paper's numbers were not affected).
3. **Inconsistent split units** (repository vs base commit); now task on both (`ff6b081`).
4. **Laguna Verified after-edit lookahead in the paper is undefined**: under the old approximate turns, every evaluation bin selected by the after-edit filter held a single class. Shown as n/a‡ in the legacy columns.
5. **Qwen Pro has_regressions has very few positives** in the window-50 test set (legacy: 29 positives; flagged †).

## Other things to know

- **Layer 0 on Qwen Pro.** The best layer for compiles (0.750) and has_regressions (0.840) is layer 0, the embedding layer, where each position sees only its own token. These two labels are therefore largely predictable from token identity, i.e. from which files and outputs appear, not necessarily from computed program state. For correct, layer 0 falls to 0.637 against 0.822 at layer 30.
- **Best layer** is chosen by test AUC (the rule in `run_paper_figures.py`), so all "best layer" numbers are slightly optimistic. With the task split, validation-based selection is now possible.
- **Multi-bin evals** (transfer, bineval, after-edit) are summarised as the n_test-weighted mean of per-bin AUCs, skipping single-class bins. `run_paper_figures._weighted_mean` keeps those bins, with their placeholder AUC of 0.5.
- The persistence baseline is undefined at k=0 (the cache stores no `y_original` there); the figures note it as missing for k=0 only.
- Some per-stratum AUCs, within y_t=1 especially, rest on few flipped tokens and have wide CIs; several are below 0.5.
- Operational delays (no effect on results):
  - submission feeders crashed under the system Python 3.6, costing ~6 h on 97 jobs;
  - 4 jobs hit the 2-hour time limit and were resubmitted with 4 hours.

## Figures

- Lookahead plots, all with the persistence baseline: `paper/figures/{laguna_xs2_full_pooled, qwen36_35b_a3b_full_pooled, laguna_xs2_pro_full_pooled_pooled_pro, qwen36_35b_a3b_pro_full_pooled_pooled_pro}/<probe>_lookahead_max{50,15}[_after_edit].pdf` (64 files).
- Other figures from `slurm/figures.sh` (defaults): `paper/figures/` (summary barplot, transfer barplot, heatmaps, step_rel, bineval, after-edit).
- Example trajectories (layer 21, i.e. `--layer 20`, UMAP, after-edit, paper style):
  - `paper/figures/trajectories/final/laguna/umap_sphinx-doc__sphinx-9602.pdf`
  - `paper/figures/trajectories/final/qwen/umap_sympy__sympy-14531.pdf`
- Previous versions: `paper/figures_legacyturns/` and `*_legacyturns.{pdf,png}` next to the trajectory figures.

## Reproducing

- `scripts/rebuttal_check_labels.py` (step 1)
- `scripts/rebuttal_check_k0.py --task-groups` (step 2)
- `scripts/rebuttal_persistence_analysis.py` (step 6; one job per run × probe × window; 1,000 task-level bootstrap resamples)
- `scripts/rebuttal_report_tables.py` (the tables below)

Result directories keep the original run IDs; the paper's results are in `results/*/*_legacyturns`.

## Tables

`†` fewer than 50 positive test tokens; `‡` single-class test set, so AUC is undefined. "new / legacy": the task-split rerun vs the paper's runs.
† fewer than 50 positive test tokens; ‡ test split has a single class, so AUC is undefined.

#### Current-state probes (k=0), best layer by test AUC — task split vs paper (repository split on Verified)

| Model / benchmark | Probe | task split (new) | paper | transfer to other benchmark (new) | transfer (paper) |
|---|---|---|---|---|---|
| Laguna XS.2 / Verified | compiles | 0.691 (L10) | 0.571 (L30) | 0.538 (L30) | 0.532 (L20) |
| Laguna XS.2 / Verified | correct | 0.788 (L20) | 0.727 (L20) | 0.701 (L20) | 0.645 (L20) |
| Laguna XS.2 / Verified | has_regressions | 0.695 (L20) | 0.678 (L20) | 0.578 (L20) | 0.626 (L20) |
| Laguna XS.2 / Verified | reduces_failing | 0.773 (L20) | 0.708 (L20) | 0.670 (L20) | 0.648 (L20) |
| Qwen3.6-35B-A3B / Verified | compiles | 0.781 (L10) | 0.587 (L30) | 0.591 (L0) | 0.564 (L0) |
| Qwen3.6-35B-A3B / Verified | correct | 0.838 (L30) | 0.828 (L30) | 0.779 (L30) | 0.767 (L30) |
| Qwen3.6-35B-A3B / Verified | has_regressions | 0.772 (L20) | 0.721 (L30) | 0.779 (L20) | 0.750 (L20) |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 0.832 (L30) | 0.808 (L30) | 0.727 (L30) | 0.776 (L30) |
| Laguna XS.2 / Pro | compiles | 0.786 (L10) | 0.779 (L10) | 0.542 (L20) | 0.553 (L20) |
| Laguna XS.2 / Pro | correct | 0.794 (L20) | 0.732 (L20) | 0.708 (L20) | 0.669 (L20) |
| Laguna XS.2 / Pro | has_regressions | 0.740 (L20) | 0.745 (L30) | 0.634 (L20) | 0.620 (L20) |
| Laguna XS.2 / Pro | reduces_failing | 0.780 (L20) | 0.722 (L20) | 0.656 (L20) | 0.615 (L20) |
| Qwen3.6-35B-A3B / Pro | compiles | 0.750 (L0) | 0.737 (L39) | 0.549 (L20) | 0.531 (L20) |
| Qwen3.6-35B-A3B / Pro | correct | 0.822 (L30) | 0.832 (L30) | 0.693 (L10) | 0.781 (L20) |
| Qwen3.6-35B-A3B / Pro | has_regressions | 0.840 (L0) | 0.792 (L20) | 0.660 (L30) | 0.667 (L20) |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 0.836 (L30) | 0.841 (L20) | 0.714 (L30) | 0.771 (L30) |

#### Lookahead AUC, window max50, all target tokens — new / legacy (best layer by test AUC)

| Model / benchmark | Probe | k=0 | k=1 | k=2 | k=3 | k=5 | k=10 | k=15 | k=25 | k=50 |
|---|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | compiles | 0.883 / 0.652 | 0.885 / 0.647 | 0.881 / 0.641 | 0.869 / 0.634 | 0.850 / 0.620 | 0.815 / 0.577 | 0.804 / 0.548 | 0.784 / 0.524 | 0.790 / 0.525 |
| Laguna XS.2 / Verified | correct | 0.766 / 0.772 | 0.759 / 0.761 | 0.755 / 0.742 | 0.749 / 0.727 | 0.731 / 0.698 | 0.695 / 0.635 | 0.684 / 0.587 | 0.639 / 0.539 | 0.562 / 0.525 |
| Laguna XS.2 / Verified | has_regressions | 0.766 / 0.685 | 0.761 / 0.683 | 0.751 / 0.665 | 0.742 / 0.653 | 0.699 / 0.629 | 0.680 / 0.572 | 0.657 / 0.538 | 0.610 / 0.523 | 0.596 / 0.531 |
| Laguna XS.2 / Verified | reduces_failing | 0.848 / 0.775 | 0.840 / 0.761 | 0.828 / 0.746 | 0.815 / 0.725 | 0.783 / 0.696 | 0.732 / 0.644 | 0.701 / 0.599 | 0.631 / 0.552 | 0.546 / 0.499 |
| Qwen3.6-35B-A3B / Verified | compiles | 0.775 / 0.821 | 0.726 / 0.804 | 0.735 / 0.805 | 0.761 / 0.768 | 0.743 / 0.755 | 0.681 / 0.670 | 0.599 / 0.622 | 0.501 / 0.554 | 0.529 / 0.557 |
| Qwen3.6-35B-A3B / Verified | correct | 0.812 / 0.826 | 0.828 / 0.827 | 0.815 / 0.821 | 0.810 / 0.819 | 0.799 / 0.794 | 0.753 / 0.743 | 0.694 / 0.691 | 0.621 / 0.623 | 0.618 / 0.561 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 0.885 / 0.826 | 0.863 / 0.843 | 0.858 / 0.844 | 0.851 / 0.845 | 0.822 / 0.834 | 0.764 / 0.770 | 0.718 / 0.721 | 0.625 / 0.638 | 0.649 / 0.552 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 0.864 / 0.858 | 0.890 / 0.859 | 0.885 / 0.855 | 0.878 / 0.849 | 0.862 / 0.829 | 0.813 / 0.761 | 0.755 / 0.697 | 0.685 / 0.628 | 0.600 / 0.519 |
| Laguna XS.2 / Pro | compiles | 0.700 / 0.739 | 0.697 / 0.739 | 0.694 / 0.734 | 0.704 / 0.737 | 0.708 / 0.735 | 0.710 / 0.726 | 0.711 / 0.723 | 0.715 / 0.725 | 0.706 / 0.738 |
| Laguna XS.2 / Pro | correct | 0.839 / 0.817 | 0.842 / 0.816 | 0.807 / 0.809 | 0.788 / 0.799 | 0.778 / 0.768 | 0.719 / 0.755 | 0.683 / 0.717 | 0.622 / 0.656 | 0.605 / 0.602 |
| Laguna XS.2 / Pro | has_regressions | 0.888 / 0.810 | 0.875 / 0.801 | 0.837 / 0.804 | 0.823 / 0.796 | 0.805 / 0.766 | 0.748 / 0.687 | 0.760 / 0.645 | 0.775 / 0.609 | 0.722 / 0.590 |
| Laguna XS.2 / Pro | reduces_failing | 0.847 / 0.844 | 0.858 / 0.838 | 0.857 / 0.826 | 0.859 / 0.810 | 0.839 / 0.779 | 0.792 / 0.728 | 0.749 / 0.696 | 0.695 / 0.657 | 0.692 / 0.620 |
| Qwen3.6-35B-A3B / Pro | compiles | 0.760 / 0.760 | 0.742 / 0.758 | 0.739 / 0.757 | 0.735 / 0.756 | 0.729 / 0.754 | 0.716 / 0.747 | 0.709 / 0.742 | 0.694 / 0.736 | 0.690 / 0.725 |
| Qwen3.6-35B-A3B / Pro | correct | 0.830 / 0.847 | 0.829 / 0.853 | 0.817 / 0.858 | 0.812 / 0.837 | 0.787 / 0.806 | 0.745 / 0.753 | 0.742 / 0.718 | 0.736 / 0.666 | 0.688 / 0.623 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 0.897 / 0.979† | 0.867 / 0.978† | 0.903 / 0.973 | 0.913 / 0.971 | 0.947 / 0.950 | 0.825 / 0.935 | 0.760 / 0.813 | 0.807 / 0.761 | 0.859 / 0.730 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 0.856 / 0.961 | 0.906 / 0.961 | 0.906 / 0.959 | 0.901 / 0.955 | 0.901 / 0.942 | 0.868 / 0.883 | 0.862 / 0.823 | 0.824 / 0.790 | 0.739 / 0.785 |

#### Lookahead AUC, window max50, after-edit targets only — new / legacy (best layer by test AUC)

| Model / benchmark | Probe | k=0 | k=1 | k=2 | k=3 | k=5 | k=10 | k=15 | k=25 | k=50 |
|---|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | compiles | 0.837 / n/a‡ | 0.715 / n/a‡ | 0.740 / n/a‡ | 0.774 / n/a‡ | 0.698 / n/a‡ | 0.747 / n/a‡ | 0.708 / n/a‡ | 0.816 / n/a‡ | 0.812 / n/a‡ |
| Laguna XS.2 / Verified | correct | 0.652 / n/a‡ | 0.647 / n/a‡ | 0.573 / n/a‡ | 0.677 / n/a‡ | 0.606 / n/a‡ | 0.566 / n/a‡ | 0.592 / n/a‡ | 0.578 / n/a‡ | 0.556 / n/a‡ |
| Laguna XS.2 / Verified | has_regressions | 0.592 / n/a‡ | 0.590 / n/a‡ | 0.547 / n/a‡ | 0.591 / n/a‡ | 0.573 / n/a‡ | 0.695 / n/a‡ | 0.615 / n/a‡ | 0.660 / n/a‡ | 0.579 / n/a‡ |
| Laguna XS.2 / Verified | reduces_failing | 0.651 / n/a‡ | 0.559 / n/a‡ | 0.593 / n/a‡ | 0.620 / n/a‡ | 0.602 / n/a‡ | 0.585 / n/a‡ | 0.567 / n/a‡ | 0.533 / n/a‡ | 0.524 / n/a‡ |
| Qwen3.6-35B-A3B / Verified | compiles | 0.824 / 0.652 | 0.429 / 0.592 | 0.466 / 0.611 | 0.607 / 0.573 | 0.532 / 0.599 | 0.607 / 0.527 | 0.396 / 0.544 | 0.614 / 0.535 | 0.643 / 0.519 |
| Qwen3.6-35B-A3B / Verified | correct | 0.592 / 0.556 | 0.517 / 0.552 | 0.442 / 0.538 | 0.426 / 0.540 | 0.404 / 0.548 | 0.434 / 0.581 | 0.406 / 0.593 | 0.527 / 0.615 | 0.558 / 0.635 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 0.704 / 0.675 | 0.508 / 0.672 | 0.546 / 0.666 | 0.563 / 0.667 | 0.518 / 0.666 | 0.556 / 0.668 | 0.565 / 0.669 | 0.667 / 0.658 | 0.637 / 0.620 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 0.638 / 0.526 | 0.464 / 0.540 | 0.526 / 0.540 | 0.457 / 0.536 | 0.516 / 0.542 | 0.680 / 0.588 | 0.549 / 0.583 | 0.517 / 0.587 | 0.520 / 0.593 |
| Laguna XS.2 / Pro | compiles | 0.646 / 0.769 | 0.633 / 0.773 | 0.646 / 0.778 | 0.710 / 0.770 | 0.694 / 0.777 | 0.684 / 0.790 | 0.708 / 0.793 | 0.693 / 0.797 | 0.712 / 0.814 |
| Laguna XS.2 / Pro | correct | 0.697 / 0.666 | 0.633 / 0.654 | 0.548 / 0.654 | 0.569 / 0.662 | 0.480 / 0.662 | 0.550 / 0.597 | 0.549 / 0.583 | 0.552 / 0.571 | 0.570 / 0.515 |
| Laguna XS.2 / Pro | has_regressions | 0.751 / 0.850 | 0.745 / 0.854 | 0.638 / 0.912 | 0.641 / 0.911 | 0.740 / 0.907 | 0.716 / 0.858 | 0.770 / 0.890 | 0.826 / 0.904 | 0.775 / 0.863 |
| Laguna XS.2 / Pro | reduces_failing | 0.712 / 0.766 | 0.672 / 0.770 | 0.674 / 0.770 | 0.711 / 0.770 | 0.694 / 0.763 | 0.722 / 0.762 | 0.733 / 0.755 | 0.686 / 0.734 | 0.709 / 0.707 |
| Qwen3.6-35B-A3B / Pro | compiles | 0.750 / 0.654 | 0.681 / 0.656 | 0.720 / 0.655 | 0.708 / 0.663 | 0.683 / 0.658 | 0.738 / 0.670 | 0.772 / 0.670 | 0.721 / 0.680 | 0.712 / 0.663 |
| Qwen3.6-35B-A3B / Pro | correct | 0.700 / 0.862 | 0.642 / 0.874 | 0.466 / 0.884 | 0.532 / 0.882 | 0.431 / 0.896 | 0.484 / 0.905 | 0.526 / 0.905 | 0.612 / 0.761 | 0.627 / 0.587 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 0.538† / n/a‡ | 0.509 / n/a‡ | 0.683† / n/a‡ | 0.667† / n/a‡ | 0.860 / n/a‡ | 0.621† / n/a‡ | 0.605† / 0.758† | 0.791 / 0.680 | 0.821 / 0.809 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 0.760† / 0.927 | 0.775 / 0.926 | 0.758 / 0.931 | 0.635 / 0.932 | 0.684 / 0.928 | 0.600 / 0.928 | 0.612 / 0.889 | 0.753 / 0.814 | 0.645 / 0.798 |

#### Lookahead AUC, window max15, all target tokens — new / legacy (best layer by test AUC)

| Model / benchmark | Probe | k=0 | k=1 | k=2 | k=3 | k=5 | k=10 | k=15 |
|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | compiles | 0.691 / 0.598 | 0.675 / 0.589 | 0.672 / 0.583 | 0.667 / 0.576 | 0.654 / 0.562 | 0.667 / 0.539 | 0.674 / 0.535 |
| Laguna XS.2 / Verified | correct | 0.784 / 0.741 | 0.751 / 0.730 | 0.741 / 0.713 | 0.731 / 0.699 | 0.712 / 0.675 | 0.680 / 0.640 | 0.663 / 0.620 |
| Laguna XS.2 / Verified | has_regressions | 0.683 / 0.663 | 0.664 / 0.654 | 0.654 / 0.645 | 0.643 / 0.636 | 0.627 / 0.624 | 0.602 / 0.596 | 0.586 / 0.583 |
| Laguna XS.2 / Verified | reduces_failing | 0.792 / 0.711 | 0.759 / 0.687 | 0.744 / 0.671 | 0.730 / 0.671 | 0.705 / 0.636 | 0.664 / 0.597 | 0.635 / 0.573 |
| Qwen3.6-35B-A3B / Verified | compiles | 0.808 / 0.644 | 0.799 / 0.654 | 0.789 / 0.636 | 0.773 / 0.629 | 0.754 / 0.602 | 0.729 / 0.554 | 0.728 / 0.546 |
| Qwen3.6-35B-A3B / Verified | correct | 0.780 / 0.832 | 0.776 / 0.819 | 0.765 / 0.803 | 0.758 / 0.783 | 0.747 / 0.756 | 0.703 / 0.696 | 0.674 / 0.655 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 0.830 / 0.764 | 0.811 / 0.757 | 0.798 / 0.742 | 0.791 / 0.729 | 0.755 / 0.701 | 0.692 / 0.640 | 0.642 / 0.594 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 0.832 / 0.826 | 0.819 / 0.820 | 0.804 / 0.805 | 0.789 / 0.789 | 0.762 / 0.761 | 0.694 / 0.700 | 0.650 / 0.646 |
| Laguna XS.2 / Pro | compiles | 0.784 / 0.781 | 0.785 / 0.778 | 0.783 / 0.778 | 0.785 / 0.777 | 0.782 / 0.780 | 0.779 / 0.776 | 0.778 / 0.776 |
| Laguna XS.2 / Pro | correct | 0.803 / 0.738 | 0.791 / 0.734 | 0.791 / 0.726 | 0.787 / 0.721 | 0.778 / 0.709 | 0.762 / 0.693 | 0.744 / 0.680 |
| Laguna XS.2 / Pro | has_regressions | 0.777 / 0.746 | 0.774 / 0.746 | 0.771 / 0.745 | 0.775 / 0.741 | 0.771 / 0.722 | 0.762 / 0.709 | 0.746 / 0.700 |
| Laguna XS.2 / Pro | reduces_failing | 0.799 / 0.754 | 0.788 / 0.746 | 0.782 / 0.741 | 0.775 / 0.734 | 0.767 / 0.718 | 0.751 / 0.689 | 0.735 / 0.667 |
| Qwen3.6-35B-A3B / Pro | compiles | 0.825 / 0.699 | 0.823 / 0.698 | 0.824 / 0.697 | 0.823 / 0.695 | 0.822 / 0.692 | 0.822 / 0.686 | 0.825 / 0.684 |
| Qwen3.6-35B-A3B / Pro | correct | 0.869 / 0.797 | 0.836 / 0.796 | 0.828 / 0.791 | 0.824 / 0.782 | 0.811 / 0.759 | 0.775 / 0.699 | 0.750 / 0.664 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 0.869 / 0.817 | 0.884 / 0.813 | 0.885 / 0.807 | 0.858 / 0.802 | 0.835 / 0.789 | 0.845 / 0.733 | 0.853 / 0.689 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 0.845 / 0.836 | 0.823 / 0.833 | 0.812 / 0.828 | 0.803 / 0.824 | 0.787 / 0.810 | 0.760 / 0.749 | 0.742 / 0.692 |

#### Lookahead AUC, window max15, after-edit targets only — new / legacy (best layer by test AUC)

| Model / benchmark | Probe | k=0 | k=1 | k=2 | k=3 | k=5 | k=10 | k=15 |
|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | compiles | 0.621 / 0.553 | 0.644 / 0.549 | 0.641 / 0.553 | 0.607 / 0.550 | 0.613 / 0.550 | 0.648 / 0.549 | 0.671 / 0.550 |
| Laguna XS.2 / Verified | correct | 0.636 / 0.637 | 0.604 / 0.621 | 0.564 / 0.607 | 0.609 / 0.599 | 0.601 / 0.589 | 0.611 / 0.586 | 0.625 / 0.585 |
| Laguna XS.2 / Verified | has_regressions | 0.592 / 0.581 | 0.624 / 0.580 | 0.608 / 0.576 | 0.557 / 0.571 | 0.582 / 0.576 | 0.587 / 0.573 | 0.553 / 0.580 |
| Laguna XS.2 / Verified | reduces_failing | 0.577 / 0.573 | 0.589 / 0.560 | 0.582 / 0.550 | 0.593 / 0.545 | 0.572 / 0.538 | 0.596 / 0.537 | 0.570 / 0.527 |
| Qwen3.6-35B-A3B / Verified | compiles | 0.654 / 0.530 | 0.728 / 0.510 | 0.678 / 0.537 | 0.684 / 0.527 | 0.765 / 0.514 | 0.706 / 0.485 | 0.672 / 0.490 |
| Qwen3.6-35B-A3B / Verified | correct | 0.558 / 0.648 | 0.537 / 0.613 | 0.477 / 0.590 | 0.487 / 0.573 | 0.501 / 0.577 | 0.527 / 0.592 | 0.532 / 0.596 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 0.625 / 0.594 | 0.608 / 0.576 | 0.621 / 0.580 | 0.638 / 0.580 | 0.612 / 0.556 | 0.649 / 0.536 | 0.638 / 0.523 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 0.484 / 0.586 | 0.506 / 0.575 | 0.480 / 0.565 | 0.448 / 0.547 | 0.494 / 0.528 | 0.500 / 0.526 | 0.529 / 0.530 |
| Laguna XS.2 / Pro | compiles | 0.821 / 0.782 | 0.794 / 0.783 | 0.804 / 0.783 | 0.778 / 0.783 | 0.777 / 0.784 | 0.791 / 0.779 | 0.783 / 0.779 |
| Laguna XS.2 / Pro | correct | 0.752 / 0.634 | 0.709 / 0.631 | 0.713 / 0.628 | 0.709 / 0.626 | 0.713 / 0.621 | 0.716 / 0.608 | 0.684 / 0.597 |
| Laguna XS.2 / Pro | has_regressions | 0.677 / 0.699 | 0.733 / 0.694 | 0.729 / 0.701 | 0.721 / 0.694 | 0.723 / 0.692 | 0.744 / 0.688 | 0.757 / 0.689 |
| Laguna XS.2 / Pro | reduces_failing | 0.729 / 0.641 | 0.714 / 0.640 | 0.748 / 0.637 | 0.725 / 0.634 | 0.756 / 0.631 | 0.753 / 0.630 | 0.707 / 0.636 |
| Qwen3.6-35B-A3B / Pro | compiles | 0.827 / 0.652 | 0.811 / 0.656 | 0.845 / 0.646 | 0.816 / 0.648 | 0.830 / 0.643 | 0.846 / 0.641 | 0.841 / 0.632 |
| Qwen3.6-35B-A3B / Pro | correct | 0.662 / 0.727 | 0.667 / 0.732 | 0.688 / 0.731 | 0.677 / 0.731 | 0.679 / 0.723 | 0.667 / 0.705 | 0.647 / 0.718 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 0.825 / 0.703 | 0.830 / 0.692 | 0.804 / 0.700 | 0.854 / 0.697 | 0.796 / 0.698 | 0.831 / 0.693 | 0.820 / 0.679 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 0.598 / 0.743 | 0.650 / 0.742 | 0.652 / 0.735 | 0.663 / 0.731 | 0.638 / 0.731 | 0.675 / 0.733 | 0.675 / 0.732 |

#### Shuffled-label controls (best layer by test AUC)

| Model / benchmark | Probe | real labels | token shuffle (fixed) | trajectory shuffle (fixed) | token shuffle, legacy (test labels shuffled too) |
|---|---|---|---|---|---|
| Laguna XS.2 / Verified | compiles | 0.691 (L10) | 0.523 (L39) | 0.604 (L30) | 0.500 (L10) |
| Laguna XS.2 / Verified | correct | 0.788 (L20) | 0.518 (L39) | 0.665 (L20) | 0.501 (L10) |
| Laguna XS.2 / Verified | has_regressions | 0.695 (L20) | 0.508 (L20) | 0.631 (L10) | 0.500 (L0) |
| Laguna XS.2 / Verified | reduces_failing | 0.773 (L20) | 0.518 (L30) | 0.680 (L20) | 0.500 (L10) |
| Qwen3.6-35B-A3B / Verified | compiles | 0.781 (L10) | 0.515 (L20) | 0.683 (L10) | 0.501 (L10) |
| Qwen3.6-35B-A3B / Verified | correct | 0.838 (L30) | 0.514 (L30) | 0.776 (L30) | 0.501 (L10) |
| Qwen3.6-35B-A3B / Verified | has_regressions | 0.772 (L20) | 0.513 (L39) | 0.643 (L10) | 0.501 (L30) |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 0.832 (L30) | 0.548 (L30) | 0.778 (L20) | 0.501 (L39) |
| Laguna XS.2 / Pro | compiles | 0.786 (L10) | 0.522 (L39) | 0.479 (L30) | 0.501 (L10) |
| Laguna XS.2 / Pro | correct | 0.794 (L20) | 0.531 (L20) | 0.652 (L30) | 0.500 (L20) |
| Laguna XS.2 / Pro | has_regressions | 0.740 (L20) | 0.552 (L10) | 0.564 (L20) | 0.501 (L20) |
| Laguna XS.2 / Pro | reduces_failing | 0.780 (L20) | 0.522 (L10) | 0.632 (L20) | 0.500 (L39) |
| Qwen3.6-35B-A3B / Pro | compiles | 0.750 (L0) | 0.547 (L39) | 0.570 (L0) | 0.502 (L10) |
| Qwen3.6-35B-A3B / Pro | correct | 0.822 (L30) | 0.517 (L30) | 0.771 (L30) | 0.502 (L10) |
| Qwen3.6-35B-A3B / Pro | has_regressions | 0.840 (L0) | 0.539 (L10) | 0.693 (L10) | 0.507 (L0) |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 0.836 (L30) | 0.542 (L10) | 0.761 (L30) | 0.502 (L0) |

#### Persistence analysis, window max50 (test split; 95% CI from 1000 instance-level bootstrap resamples)

| Model / benchmark | Probe | k | layer | n_test | flip rate | pairs w/ edit between | persistence AUC | probe AUC (same tokens) | probe − persistence | probe AUC, y_t=0 | probe AUC, y_t=1 | probe AUC, edit between | persistence AUC, edit between |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | compiles | 0 | 10 | 181946 | – | – | – | 0.883 [0.773, 0.924] | – | – | – | – | – |
| Laguna XS.2 / Verified | compiles | 1 | 10 | 181946 | 0.007 | 0.094 | 0.896 [0.804, 0.923] | 0.885 [0.789, 0.921] | -0.011 [-0.090, 0.052] | 0.552 [0.275, 0.894] | 0.892 [0.814, 0.935] | 0.748 [0.613, 0.840] | 0.696 [0.549, 0.787] |
| Laguna XS.2 / Verified | compiles | 2 | 10 | 181946 | 0.011 | 0.123 | 0.853 [0.744, 0.887] | 0.881 [0.780, 0.919] | 0.028 [-0.046, 0.104] | 0.581 [0.287, 0.911] | 0.869 [0.768, 0.920] | 0.746 [0.597, 0.826] | 0.682 [0.540, 0.764] |
| Laguna XS.2 / Verified | compiles | 3 | 10 | 181946 | 0.014 | 0.159 | 0.823 [0.705, 0.860] | 0.869 [0.763, 0.912] | 0.046 [-0.029, 0.117] | 0.636 [0.319, 0.922] | 0.847 [0.730, 0.904] | 0.744 [0.584, 0.825] | 0.682 [0.547, 0.748] |
| Laguna XS.2 / Verified | compiles | 5 | 10 | 181946 | 0.021 | 0.231 | 0.769 [0.652, 0.810] | 0.850 [0.737, 0.902] | 0.081 [0.011, 0.140] | 0.736 [0.427, 0.945] | 0.818 [0.699, 0.881] | 0.747 [0.599, 0.825] | 0.659 [0.539, 0.715] |
| Laguna XS.2 / Verified | compiles | 10 | 10 | 181946 | 0.035 | 0.375 | 0.702 [0.604, 0.755] | 0.815 [0.678, 0.880] | 0.113 [0.018, 0.173] | 0.768 [0.454, 0.958] | 0.781 [0.648, 0.843] | 0.769 [0.628, 0.837] | 0.622 [0.527, 0.679] |
| Laguna XS.2 / Verified | compiles | 15 | 10 | 181946 | 0.042 | 0.483 | 0.679 [0.589, 0.734] | 0.804 [0.647, 0.881] | 0.125 [0.014, 0.180] | 0.814 [0.501, 0.965] | 0.771 [0.621, 0.845] | 0.795 [0.636, 0.867] | 0.624 [0.522, 0.686] |
| Laguna XS.2 / Verified | compiles | 25 | 10 | 181946 | 0.065 | 0.672 | 0.630 [0.552, 0.697] | 0.784 [0.642, 0.858] | 0.154 [0.052, 0.211] | 0.842 [0.549, 0.974] | 0.762 [0.627, 0.821] | 0.805 [0.669, 0.867] | 0.613 [0.516, 0.690] |
| Laguna XS.2 / Verified | compiles | 50 | 10 | 180619 | 0.084 | 0.894 | 0.607 [0.543, 0.671] | 0.790 [0.651, 0.863] | 0.182 [0.050, 0.245] | 0.829 [0.522, 0.950] | 0.780 [0.658, 0.843] | 0.808 [0.684, 0.872] | 0.603 [0.538, 0.665] |
| Laguna XS.2 / Verified | correct | 0 | 20 | 181946 | – | – | – | 0.766 [0.621, 0.896] | – | – | – | – | – |
| Laguna XS.2 / Verified | correct | 1 | 20 | 181946 | 0.012 | 0.094 | 0.933 [0.860, 0.967] | 0.759 [0.619, 0.882] | -0.174 [-0.339, -0.009] | 0.784 [0.726, 0.832] | – | 0.586 [0.470, 0.671] | 0.624 [0.536, 0.703] |
| Laguna XS.2 / Verified | correct | 2 | 20 | 181946 | 0.016 | 0.123 | 0.913 [0.834, 0.955] | 0.755 [0.623, 0.873] | -0.158 [-0.323, 0.005] | 0.767 [0.701, 0.820] | – | 0.586 [0.473, 0.677] | 0.619 [0.534, 0.690] |
| Laguna XS.2 / Verified | correct | 3 | 20 | 181946 | 0.019 | 0.159 | 0.898 [0.814, 0.945] | 0.749 [0.622, 0.863] | -0.149 [-0.316, 0.014] | 0.745 [0.678, 0.804] | – | 0.608 [0.498, 0.698] | 0.626 [0.531, 0.702] |
| Laguna XS.2 / Verified | correct | 5 | 20 | 181946 | 0.030 | 0.231 | 0.860 [0.768, 0.917] | 0.731 [0.609, 0.841] | -0.129 [-0.303, 0.039] | 0.716 [0.647, 0.781] | 0.381 [0.130, 0.641] | 0.602 [0.499, 0.690] | 0.612 [0.534, 0.675] |
| Laguna XS.2 / Verified | correct | 10 | 20 | 181946 | 0.056 | 0.375 | 0.788 [0.685, 0.866] | 0.695 [0.593, 0.787] | -0.093 [-0.260, 0.066] | 0.669 [0.602, 0.735] | 0.269 [0.186, 0.710] | 0.586 [0.495, 0.677] | 0.598 [0.532, 0.666] |
| Laguna XS.2 / Verified | correct | 15 | 20 | 181946 | 0.079 | 0.483 | 0.742 [0.644, 0.823] | 0.684 [0.602, 0.756] | -0.058 [-0.199, 0.082] | 0.654 [0.597, 0.711] | 0.288 [0.190, 0.451] | 0.613 [0.535, 0.685] | 0.604 [0.541, 0.657] |
| Laguna XS.2 / Verified | correct | 25 | 20 | 181946 | 0.121 | 0.672 | 0.689 [0.596, 0.770] | 0.639 [0.570, 0.712] | -0.050 [-0.165, 0.057] | 0.600 [0.536, 0.674] | 0.359 [0.232, 0.741] | 0.602 [0.522, 0.683] | 0.623 [0.536, 0.700] |
| Laguna XS.2 / Verified | correct | 50 | 39 | 180619 | 0.202 | 0.894 | 0.633 [0.553, 0.714] | 0.562 [0.497, 0.624] | -0.071 [-0.174, 0.030] | 0.549 [0.488, 0.605] | 0.614 [0.530, 0.687] | 0.548 [0.481, 0.616] | 0.609 [0.528, 0.695] |
| Laguna XS.2 / Verified | has_regressions | 0 | 20 | 181946 | – | – | – | 0.766 [0.696, 0.830] | – | – | – | – | – |
| Laguna XS.2 / Verified | has_regressions | 1 | 10 | 181946 | 0.018 | 0.094 | 0.893 [0.855, 0.929] | 0.761 [0.693, 0.824] | -0.132 [-0.213, -0.055] | 0.871 [0.827, 0.908] | – | 0.621 [0.529, 0.698] | 0.713 [0.625, 0.816] |
| Laguna XS.2 / Verified | has_regressions | 2 | 10 | 181946 | 0.022 | 0.123 | 0.873 [0.838, 0.902] | 0.751 [0.680, 0.813] | -0.122 [-0.202, -0.045] | 0.823 [0.757, 0.868] | – | 0.579 [0.499, 0.662] | 0.724 [0.665, 0.797] |
| Laguna XS.2 / Verified | has_regressions | 3 | 10 | 181946 | 0.028 | 0.159 | 0.848 [0.814, 0.878] | 0.742 [0.672, 0.802] | -0.106 [-0.184, -0.033] | 0.783 [0.719, 0.833] | 0.659 [0.572, 0.741] | 0.571 [0.491, 0.647] | 0.716 [0.660, 0.776] |
| Laguna XS.2 / Verified | has_regressions | 5 | 10 | 181946 | 0.045 | 0.231 | 0.796 [0.756, 0.829] | 0.699 [0.621, 0.772] | -0.097 [-0.180, -0.012] | 0.694 [0.619, 0.759] | 0.597 [0.515, 0.712] | 0.541 [0.439, 0.646] | 0.695 [0.642, 0.748] |
| Laguna XS.2 / Verified | has_regressions | 10 | 10 | 181946 | 0.077 | 0.375 | 0.728 [0.670, 0.766] | 0.680 [0.600, 0.757] | -0.048 [-0.126, 0.035] | 0.644 [0.574, 0.716] | 0.585 [0.488, 0.676] | 0.611 [0.536, 0.684] | 0.663 [0.607, 0.722] |
| Laguna XS.2 / Verified | has_regressions | 15 | 10 | 181946 | 0.107 | 0.483 | 0.686 [0.623, 0.723] | 0.657 [0.563, 0.746] | -0.029 [-0.115, 0.057] | 0.620 [0.539, 0.698] | 0.411 [0.199, 0.618] | 0.631 [0.552, 0.704] | 0.638 [0.584, 0.683] |
| Laguna XS.2 / Verified | has_regressions | 25 | 10 | 181946 | 0.139 | 0.672 | 0.651 [0.594, 0.688] | 0.610 [0.507, 0.696] | -0.041 [-0.130, 0.045] | 0.570 [0.474, 0.652] | 0.459 [0.306, 0.615] | 0.618 [0.521, 0.693] | 0.625 [0.574, 0.671] |
| Laguna XS.2 / Verified | has_regressions | 50 | 20 | 180619 | 0.170 | 0.894 | 0.613 [0.555, 0.648] | 0.596 [0.492, 0.681] | -0.017 [-0.105, 0.063] | 0.578 [0.480, 0.663] | 0.534 [0.402, 0.698] | 0.609 [0.512, 0.686] | 0.592 [0.535, 0.641] |
| Laguna XS.2 / Verified | reduces_failing | 0 | 20 | 181946 | – | – | – | 0.848 [0.800, 0.892] | – | – | – | – | – |
| Laguna XS.2 / Verified | reduces_failing | 1 | 20 | 181946 | 0.030 | 0.094 | 0.898 [0.861, 0.923] | 0.840 [0.790, 0.883] | -0.059 [-0.113, 0.000] | 0.846 [0.803, 0.883] | 0.317 [0.214, 0.470] | 0.534 [0.466, 0.602] | 0.642 [0.581, 0.687] |
| Laguna XS.2 / Verified | reduces_failing | 2 | 20 | 181946 | 0.043 | 0.123 | 0.866 [0.826, 0.895] | 0.828 [0.777, 0.875] | -0.038 [-0.097, 0.027] | 0.816 [0.767, 0.861] | 0.416 [0.232, 0.598] | 0.523 [0.445, 0.603] | 0.634 [0.576, 0.678] |
| Laguna XS.2 / Verified | reduces_failing | 3 | 20 | 181946 | 0.056 | 0.159 | 0.838 [0.794, 0.869] | 0.815 [0.761, 0.862] | -0.023 [-0.084, 0.045] | 0.786 [0.735, 0.835] | 0.415 [0.245, 0.589] | 0.525 [0.443, 0.605] | 0.635 [0.580, 0.676] |
| Laguna XS.2 / Verified | reduces_failing | 5 | 20 | 181946 | 0.083 | 0.231 | 0.791 [0.743, 0.827] | 0.783 [0.728, 0.834] | -0.008 [-0.070, 0.058] | 0.741 [0.687, 0.792] | 0.457 [0.282, 0.633] | 0.516 [0.432, 0.605] | 0.620 [0.563, 0.660] |
| Laguna XS.2 / Verified | reduces_failing | 10 | 20 | 181946 | 0.142 | 0.375 | 0.722 [0.670, 0.765] | 0.732 [0.679, 0.782] | 0.010 [-0.055, 0.074] | 0.679 [0.620, 0.730] | 0.402 [0.263, 0.518] | 0.543 [0.474, 0.614] | 0.619 [0.562, 0.665] |
| Laguna XS.2 / Verified | reduces_failing | 15 | 20 | 181946 | 0.188 | 0.483 | 0.681 [0.630, 0.726] | 0.701 [0.647, 0.754] | 0.020 [-0.046, 0.084] | 0.655 [0.599, 0.708] | 0.521 [0.337, 0.659] | 0.569 [0.504, 0.645] | 0.603 [0.550, 0.648] |
| Laguna XS.2 / Verified | reduces_failing | 25 | 20 | 181946 | 0.252 | 0.672 | 0.649 [0.604, 0.689] | 0.631 [0.576, 0.686] | -0.018 [-0.085, 0.047] | 0.589 [0.535, 0.643] | 0.471 [0.311, 0.552] | 0.569 [0.509, 0.637] | 0.617 [0.566, 0.665] |
| Laguna XS.2 / Verified | reduces_failing | 50 | 39 | 180619 | 0.353 | 0.894 | 0.612 [0.571, 0.655] | 0.546 [0.484, 0.606] | -0.066 [-0.150, 0.015] | 0.545 [0.482, 0.602] | 0.631 [0.402, 0.728] | 0.527 [0.467, 0.583] | 0.597 [0.555, 0.642] |
| Qwen3.6-35B-A3B / Verified | compiles | 0 | 20 | 60746 | – | – | – | 0.775 [0.667, 0.977] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Verified | compiles | 1 | 20 | 60746 | 0.003 | 0.063 | 0.934 [0.660, 0.964] | 0.726 [0.653, 0.963] | -0.208 [-0.297, 0.203] | 0.130 [0.038, 0.502] | 0.888 [0.831, 0.977] | 0.341 [0.200, 0.705] | 0.485 [0.440, 0.500] |
| Qwen3.6-35B-A3B / Verified | compiles | 2 | 20 | 60746 | 0.004 | 0.085 | 0.891 [0.625, 0.955] | 0.735 [0.663, 0.941] | -0.157 [-0.275, 0.234] | 0.123 [0.042, 0.522] | 0.872 [0.801, 0.945] | 0.342 [0.252, 0.585] | 0.487 [0.452, 0.500] |
| Qwen3.6-35B-A3B / Verified | compiles | 3 | 20 | 60746 | 0.005 | 0.104 | 0.851 [0.500, 0.947] | 0.761 [0.690, 0.946] | -0.090 [-0.220, 0.305] | 0.134 [0.099, 0.619] | 0.855 [0.748, 0.958] | 0.433 [0.265, 0.676] | 0.508 [0.467, 0.573] |
| Qwen3.6-35B-A3B / Verified | compiles | 5 | 20 | 60746 | 0.010 | 0.143 | 0.724 [0.499, 0.832] | 0.743 [0.652, 0.932] | 0.019 [-0.159, 0.416] | 0.136 [0.130, 0.449] | 0.784 [0.642, 0.935] | 0.454 [0.272, 0.718] | 0.539 [0.474, 0.605] |
| Qwen3.6-35B-A3B / Verified | compiles | 10 | 20 | 60746 | 0.014 | 0.242 | 0.672 [0.496, 0.798] | 0.681 [0.568, 0.867] | 0.010 [-0.207, 0.370] | 0.179 [0.146, 0.295] | 0.701 [0.479, 0.869] | 0.499 [0.358, 0.711] | 0.638 [0.484, 0.764] |
| Qwen3.6-35B-A3B / Verified | compiles | 15 | 0 | 60746 | 0.011 | 0.327 | 0.705 [0.496, 0.809] | 0.599 [0.442, 0.653] | -0.106 [-0.220, 0.142] | 0.469 [0.416, 0.653] | 0.581 [0.442, 0.649] | 0.486 [0.338, 0.550] | 0.702 [0.488, 0.808] |
| Qwen3.6-35B-A3B / Verified | compiles | 25 | 0 | 60746 | 0.017 | 0.499 | 0.640 [0.496, 0.770] | 0.501 [0.389, 0.572] | -0.139 [-0.313, 0.072] | 0.349 [0.241, 0.462] | 0.495 [0.354, 0.572] | 0.451 [0.334, 0.527] | 0.638 [0.493, 0.770] |
| Qwen3.6-35B-A3B / Verified | compiles | 50 | 20 | 60746 | 0.029 | 0.868 | 0.589 [0.496, 0.770] | 0.529 [0.312, 0.674] | -0.059 [-0.457, 0.175] | 0.159 [0.126, 0.348] | 0.580 [0.331, 0.676] | 0.522 [0.306, 0.666] | 0.588 [0.496, 0.769] |
| Qwen3.6-35B-A3B / Verified | correct | 0 | 10 | 60746 | – | – | – | 0.812 [0.751, 0.913] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Verified | correct | 1 | 10 | 60746 | 0.014 | 0.063 | 0.853 [0.723, 0.970] | 0.828 [0.751, 0.919] | -0.025 [-0.163, 0.188] | 0.907 [0.848, 0.946] | 0.141 [0.076, 0.164] | 0.488 [0.329, 0.594] | 0.502 [0.494, 0.578] |
| Qwen3.6-35B-A3B / Verified | correct | 2 | 10 | 60746 | 0.018 | 0.085 | 0.824 [0.690, 0.920] | 0.815 [0.735, 0.902] | -0.009 [-0.139, 0.190] | 0.871 [0.709, 0.925] | 0.159 [0.097, 0.179] | 0.458 [0.265, 0.568] | 0.509 [0.495, 0.575] |
| Qwen3.6-35B-A3B / Verified | correct | 3 | 10 | 60746 | 0.021 | 0.104 | 0.805 [0.669, 0.891] | 0.810 [0.728, 0.890] | 0.006 [-0.123, 0.179] | 0.848 [0.671, 0.910] | 0.215 [0.143, 0.239] | 0.468 [0.258, 0.573] | 0.517 [0.500, 0.574] |
| Qwen3.6-35B-A3B / Verified | correct | 5 | 10 | 60746 | 0.031 | 0.143 | 0.760 [0.643, 0.852] | 0.799 [0.720, 0.879] | 0.039 [-0.071, 0.187] | 0.813 [0.688, 0.895] | 0.822 [0.735, 0.853] | 0.481 [0.317, 0.603] | 0.513 [0.497, 0.564] |
| Qwen3.6-35B-A3B / Verified | correct | 10 | 10 | 60746 | 0.057 | 0.242 | 0.680 [0.588, 0.765] | 0.753 [0.671, 0.828] | 0.072 [-0.034, 0.188] | 0.736 [0.626, 0.819] | 0.792 [0.719, 0.816] | 0.500 [0.379, 0.606] | 0.527 [0.503, 0.588] |
| Qwen3.6-35B-A3B / Verified | correct | 15 | 10 | 60746 | 0.086 | 0.327 | 0.631 [0.550, 0.708] | 0.694 [0.589, 0.771] | 0.063 [-0.052, 0.168] | 0.667 [0.527, 0.749] | 0.698 [0.604, 0.725] | 0.495 [0.362, 0.586] | 0.537 [0.505, 0.595] |
| Qwen3.6-35B-A3B / Verified | correct | 25 | 10 | 60746 | 0.141 | 0.499 | 0.588 [0.520, 0.652] | 0.621 [0.521, 0.683] | 0.033 [-0.064, 0.113] | 0.593 [0.490, 0.660] | 0.672 [0.586, 0.699] | 0.506 [0.402, 0.583] | 0.534 [0.502, 0.587] |
| Qwen3.6-35B-A3B / Verified | correct | 50 | 0 | 60746 | 0.302 | 0.868 | 0.525 [0.499, 0.560] | 0.618 [0.493, 0.726] | 0.093 [-0.035, 0.212] | 0.630 [0.510, 0.736] | 0.465 [0.375, 0.776] | 0.635 [0.505, 0.745] | 0.515 [0.488, 0.548] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 0 | 20 | 60746 | – | – | – | 0.885 [0.842, 0.954] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 1 | 20 | 60746 | 0.004 | 0.063 | 0.924 [0.777, 0.976] | 0.863 [0.824, 0.947] | -0.061 [-0.130, 0.112] | 0.839 [0.785, 0.957] | – | 0.440 [0.303, 0.626] | 0.742 [0.550, 0.925] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 2 | 20 | 60746 | 0.006 | 0.085 | 0.896 [0.770, 0.953] | 0.858 [0.816, 0.943] | -0.038 [-0.109, 0.117] | 0.826 [0.758, 0.950] | – | 0.432 [0.319, 0.629] | 0.731 [0.568, 0.885] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 3 | 20 | 60746 | 0.010 | 0.104 | 0.853 [0.764, 0.900] | 0.851 [0.806, 0.938] | -0.001 [-0.065, 0.159] | 0.810 [0.749, 0.940] | – | 0.448 [0.328, 0.630] | 0.696 [0.562, 0.808] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 5 | 20 | 60746 | 0.016 | 0.143 | 0.802 [0.685, 0.838] | 0.822 [0.769, 0.924] | 0.019 [-0.046, 0.240] | 0.749 [0.670, 0.920] | 0.515 [0.475, 0.606] | 0.487 [0.366, 0.674] | 0.672 [0.530, 0.760] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 10 | 20 | 60746 | 0.028 | 0.242 | 0.729 [0.577, 0.773] | 0.764 [0.689, 0.877] | 0.034 [-0.048, 0.257] | 0.684 [0.580, 0.862] | 0.509 [0.474, 0.580] | 0.556 [0.443, 0.709] | 0.660 [0.553, 0.735] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 15 | 20 | 60746 | 0.043 | 0.327 | 0.674 [0.547, 0.739] | 0.718 [0.641, 0.817] | 0.044 [-0.051, 0.225] | 0.650 [0.537, 0.791] | 0.363 [0.330, 0.431] | 0.551 [0.476, 0.676] | 0.633 [0.547, 0.686] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 25 | 10 | 60746 | 0.088 | 0.499 | 0.595 [0.514, 0.680] | 0.625 [0.521, 0.745] | 0.031 [-0.073, 0.158] | 0.586 [0.471, 0.710] | 0.412 [0.359, 0.474] | 0.515 [0.418, 0.637] | 0.580 [0.514, 0.653] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 50 | 20 | 60746 | 0.170 | 0.868 | 0.548 [0.499, 0.642] | 0.649 [0.543, 0.714] | 0.101 [-0.056, 0.186] | 0.638 [0.504, 0.709] | 0.495 [0.422, 0.568] | 0.634 [0.526, 0.703] | 0.548 [0.499, 0.642] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 0 | 30 | 60746 | – | – | – | 0.864 [0.827, 0.909] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 1 | 10 | 60746 | 0.022 | 0.063 | 0.872 [0.805, 0.936] | 0.890 [0.833, 0.934] | 0.018 [-0.082, 0.108] | 0.918 [0.847, 0.955] | 0.238 [0.069, 0.505] | 0.420 [0.229, 0.578] | 0.715 [0.518, 0.833] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 2 | 10 | 60746 | 0.029 | 0.085 | 0.844 [0.778, 0.893] | 0.885 [0.830, 0.926] | 0.041 [-0.043, 0.124] | 0.898 [0.806, 0.934] | 0.240 [0.087, 0.467] | 0.447 [0.285, 0.567] | 0.716 [0.525, 0.831] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3 | 10 | 60746 | 0.035 | 0.104 | 0.822 [0.754, 0.870] | 0.878 [0.823, 0.917] | 0.056 [-0.027, 0.138] | 0.875 [0.778, 0.917] | 0.257 [0.095, 0.465] | 0.487 [0.327, 0.591] | 0.705 [0.529, 0.811] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 5 | 10 | 60746 | 0.051 | 0.143 | 0.776 [0.717, 0.825] | 0.862 [0.821, 0.896] | 0.086 [0.014, 0.154] | 0.844 [0.776, 0.895] | 0.701 [0.222, 0.801] | 0.530 [0.418, 0.615] | 0.664 [0.518, 0.773] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 10 | 10 | 60746 | 0.092 | 0.242 | 0.701 [0.645, 0.751] | 0.813 [0.767, 0.857] | 0.112 [0.037, 0.200] | 0.771 [0.690, 0.845] | 0.754 [0.223, 0.848] | 0.594 [0.478, 0.686] | 0.619 [0.522, 0.705] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 15 | 10 | 60746 | 0.134 | 0.327 | 0.654 [0.597, 0.709] | 0.755 [0.702, 0.804] | 0.101 [0.025, 0.181] | 0.703 [0.615, 0.778] | 0.669 [0.226, 0.741] | 0.575 [0.479, 0.650] | 0.595 [0.524, 0.667] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 25 | 10 | 60746 | 0.210 | 0.499 | 0.609 [0.554, 0.662] | 0.685 [0.619, 0.744] | 0.076 [0.002, 0.153] | 0.641 [0.560, 0.719] | 0.631 [0.226, 0.692] | 0.601 [0.496, 0.671] | 0.572 [0.517, 0.629] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 50 | 10 | 60746 | 0.419 | 0.868 | 0.540 [0.501, 0.592] | 0.600 [0.523, 0.660] | 0.060 [-0.019, 0.127] | 0.589 [0.508, 0.655] | 0.539 [0.357, 0.676] | 0.584 [0.489, 0.658] | 0.531 [0.486, 0.590] |
| Laguna XS.2 / Pro | compiles | 0 | 39 | 126174 | – | – | – | 0.700 [0.594, 0.791] | – | – | – | – | – |
| Laguna XS.2 / Pro | compiles | 1 | 39 | 126286 | 0.014 | 0.180 | 0.985 [0.972, 0.993] | 0.698 [0.591, 0.790] | -0.287 [-0.393, -0.197] | 0.484 [0.292, 0.684] | 0.681 [0.575, 0.816] | 0.640 [0.523, 0.755] | 0.916 [0.851, 0.963] |
| Laguna XS.2 / Pro | compiles | 2 | 20 | 126426 | 0.024 | 0.227 | 0.973 [0.953, 0.986] | 0.696 [0.597, 0.782] | -0.277 [-0.373, -0.191] | 0.408 [0.246, 0.575] | 0.749 [0.666, 0.831] | 0.647 [0.533, 0.746] | 0.887 [0.810, 0.942] |
| Laguna XS.2 / Pro | compiles | 3 | 39 | 126597 | 0.033 | 0.269 | 0.962 [0.934, 0.982] | 0.707 [0.604, 0.796] | -0.255 [-0.354, -0.161] | 0.434 [0.256, 0.628] | 0.758 [0.665, 0.841] | 0.642 [0.530, 0.746] | 0.870 [0.782, 0.934] |
| Laguna XS.2 / Pro | compiles | 5 | 39 | 126726 | 0.044 | 0.336 | 0.950 [0.914, 0.976] | 0.712 [0.615, 0.800] | -0.238 [-0.334, -0.145] | 0.424 [0.250, 0.611] | 0.769 [0.668, 0.849] | 0.653 [0.532, 0.763] | 0.865 [0.776, 0.932] |
| Laguna XS.2 / Pro | compiles | 10 | 20 | 127012 | 0.061 | 0.446 | 0.932 [0.889, 0.961] | 0.713 [0.625, 0.788] | -0.219 [-0.306, -0.141] | 0.507 [0.297, 0.644] | 0.742 [0.658, 0.809] | 0.656 [0.547, 0.753] | 0.858 [0.770, 0.917] |
| Laguna XS.2 / Pro | compiles | 15 | 20 | 127024 | 0.072 | 0.547 | 0.921 [0.873, 0.956] | 0.714 [0.627, 0.787] | -0.207 [-0.295, -0.127] | 0.554 [0.352, 0.680] | 0.727 [0.633, 0.806] | 0.655 [0.546, 0.752] | 0.862 [0.776, 0.920] |
| Laguna XS.2 / Pro | compiles | 25 | 20 | 126906 | 0.085 | 0.722 | 0.908 [0.850, 0.951] | 0.719 [0.624, 0.792] | -0.188 [-0.288, -0.099] | 0.618 [0.424, 0.726] | 0.721 [0.614, 0.804] | 0.675 [0.566, 0.766] | 0.872 [0.790, 0.930] |
| Laguna XS.2 / Pro | compiles | 50 | 20 | 126310 | 0.081 | 0.936 | 0.913 [0.853, 0.954] | 0.713 [0.598, 0.799] | -0.201 [-0.321, -0.088] | 0.584 [0.429, 0.696] | 0.740 [0.564, 0.842] | 0.702 [0.584, 0.792] | 0.907 [0.843, 0.950] |
| Laguna XS.2 / Pro | correct | 0 | 20 | 126174 | – | – | – | 0.839 [0.751, 0.895] | – | – | – | – | – |
| Laguna XS.2 / Pro | correct | 1 | 20 | 126286 | 0.005 | 0.180 | 0.739 [0.556, 0.861] | 0.843 [0.775, 0.893] | 0.104 [-0.072, 0.285] | 0.864 [0.819, 0.912] | 0.360 [0.069, 0.654] | 0.611 [0.512, 0.702] | 0.669 [0.499, 0.774] |
| Laguna XS.2 / Pro | correct | 2 | 20 | 126426 | 0.006 | 0.227 | 0.717 [0.563, 0.790] | 0.808 [0.718, 0.870] | 0.091 [-0.015, 0.238] | 0.803 [0.733, 0.861] | 0.401 [0.068, 0.694] | 0.553 [0.390, 0.684] | 0.679 [0.499, 0.736] |
| Laguna XS.2 / Pro | correct | 3 | 20 | 126597 | 0.006 | 0.269 | 0.702 [0.547, 0.777] | 0.790 [0.703, 0.857] | 0.088 [-0.016, 0.223] | 0.784 [0.717, 0.846] | 0.405 [0.061, 0.696] | 0.543 [0.388, 0.682] | 0.672 [0.499, 0.734] |
| Laguna XS.2 / Pro | correct | 5 | 20 | 126726 | 0.013 | 0.336 | 0.616 [0.526, 0.660] | 0.781 [0.711, 0.843] | 0.165 [0.080, 0.295] | 0.784 [0.721, 0.845] | 0.389 [0.060, 0.677] | 0.562 [0.450, 0.670] | 0.601 [0.499, 0.645] |
| Laguna XS.2 / Pro | correct | 10 | 20 | 127012 | 0.027 | 0.446 | 0.564 [0.510, 0.605] | 0.723 [0.649, 0.785] | 0.158 [0.065, 0.253] | 0.716 [0.643, 0.781] | 0.419 [0.065, 0.766] | 0.552 [0.457, 0.634] | 0.563 [0.507, 0.603] |
| Laguna XS.2 / Pro | correct | 15 | 20 | 127024 | 0.043 | 0.547 | 0.542 [0.506, 0.574] | 0.686 [0.586, 0.764] | 0.144 [0.026, 0.241] | 0.680 [0.578, 0.758] | 0.401 [0.053, 0.760] | 0.566 [0.451, 0.657] | 0.542 [0.506, 0.574] |
| Laguna XS.2 / Pro | correct | 25 | 20 | 126906 | 0.064 | 0.722 | 0.529 [0.504, 0.553] | 0.625 [0.514, 0.720] | 0.096 [-0.028, 0.200] | 0.618 [0.499, 0.717] | 0.336 [0.049, 0.646] | 0.566 [0.448, 0.663] | 0.528 [0.504, 0.553] |
| Laguna XS.2 / Pro | correct | 50 | 20 | 126310 | 0.102 | 0.936 | 0.519 [0.503, 0.537] | 0.606 [0.505, 0.699] | 0.087 [-0.019, 0.184] | 0.603 [0.501, 0.698] | 0.309 [0.070, 0.547] | 0.599 [0.496, 0.688] | 0.519 [0.503, 0.537] |
| Laguna XS.2 / Pro | has_regressions | 0 | 20 | 126174 | – | – | – | 0.888 [0.843, 0.913] | – | – | – | – | – |
| Laguna XS.2 / Pro | has_regressions | 1 | 20 | 126286 | 0.014 | 0.180 | 0.788 [0.708, 0.910] | 0.876 [0.811, 0.907] | 0.088 [-0.095, 0.181] | 0.867 [0.729, 0.905] | 0.503 [0.447, 0.556] | 0.718 [0.612, 0.783] | 0.657 [0.483, 0.843] |
| Laguna XS.2 / Pro | has_regressions | 2 | 20 | 126426 | 0.020 | 0.227 | 0.706 [0.519, 0.827] | 0.838 [0.777, 0.883] | 0.132 [-0.023, 0.322] | 0.828 [0.724, 0.876] | 0.434 [0.322, 0.480] | 0.681 [0.603, 0.762] | 0.647 [0.477, 0.792] |
| Laguna XS.2 / Pro | has_regressions | 3 | 20 | 126597 | 0.020 | 0.269 | 0.711 [0.495, 0.785] | 0.825 [0.763, 0.875] | 0.115 [0.021, 0.331] | 0.806 [0.725, 0.859] | 0.388 [0.290, 0.661] | 0.661 [0.571, 0.753] | 0.669 [0.482, 0.759] |
| Laguna XS.2 / Pro | has_regressions | 5 | 30 | 126726 | 0.022 | 0.336 | 0.686 [0.495, 0.738] | 0.808 [0.706, 0.866] | 0.122 [0.011, 0.312] | 0.795 [0.666, 0.860] | 0.490 [0.324, 0.596] | 0.695 [0.559, 0.795] | 0.659 [0.487, 0.720] |
| Laguna XS.2 / Pro | has_regressions | 10 | 30 | 127012 | 0.030 | 0.446 | 0.631 [0.495, 0.694] | 0.752 [0.626, 0.816] | 0.121 [-0.039, 0.201] | 0.738 [0.584, 0.805] | 0.502 [0.334, 0.611] | 0.663 [0.507, 0.756] | 0.624 [0.490, 0.693] |
| Laguna XS.2 / Pro | has_regressions | 15 | 39 | 127024 | 0.041 | 0.547 | 0.586 [0.495, 0.642] | 0.762 [0.520, 0.844] | 0.176 [-0.120, 0.264] | 0.777 [0.512, 0.855] | 0.470 [0.285, 0.672] | 0.733 [0.459, 0.830] | 0.582 [0.492, 0.642] |
| Laguna XS.2 / Pro | has_regressions | 25 | 39 | 126906 | 0.062 | 0.722 | 0.554 [0.495, 0.591] | 0.777 [0.478, 0.868] | 0.224 [-0.109, 0.316] | 0.796 [0.496, 0.883] | 0.454 [0.175, 0.693] | 0.765 [0.448, 0.860] | 0.551 [0.493, 0.590] |
| Laguna XS.2 / Pro | has_regressions | 50 | 39 | 126310 | 0.078 | 0.936 | 0.548 [0.495, 0.591] | 0.725 [0.534, 0.803] | 0.177 [-0.047, 0.265] | 0.741 [0.549, 0.817] | 0.545 [0.298, 0.692] | 0.722 [0.526, 0.800] | 0.547 [0.495, 0.591] |
| Laguna XS.2 / Pro | reduces_failing | 0 | 20 | 126174 | – | – | – | 0.847 [0.758, 0.905] | – | – | – | – | – |
| Laguna XS.2 / Pro | reduces_failing | 1 | 20 | 126286 | 0.014 | 0.180 | 0.823 [0.744, 0.855] | 0.859 [0.775, 0.911] | 0.036 [-0.063, 0.137] | 0.876 [0.778, 0.933] | 0.375 [0.149, 0.609] | 0.682 [0.474, 0.784] | 0.739 [0.627, 0.786] |
| Laguna XS.2 / Pro | reduces_failing | 2 | 20 | 126426 | 0.025 | 0.227 | 0.752 [0.677, 0.788] | 0.858 [0.762, 0.911] | 0.106 [-0.008, 0.181] | 0.864 [0.758, 0.926] | 0.587 [0.222, 0.735] | 0.664 [0.456, 0.762] | 0.708 [0.611, 0.742] |
| Laguna XS.2 / Pro | reduces_failing | 3 | 20 | 126597 | 0.032 | 0.269 | 0.721 [0.653, 0.764] | 0.860 [0.764, 0.913] | 0.138 [0.015, 0.208] | 0.870 [0.758, 0.931] | 0.562 [0.246, 0.693] | 0.683 [0.481, 0.781] | 0.689 [0.607, 0.716] |
| Laguna XS.2 / Pro | reduces_failing | 5 | 20 | 126726 | 0.042 | 0.336 | 0.685 [0.607, 0.730] | 0.842 [0.742, 0.901] | 0.156 [0.038, 0.226] | 0.843 [0.731, 0.911] | 0.558 [0.236, 0.699] | 0.675 [0.487, 0.783] | 0.667 [0.593, 0.697] |
| Laguna XS.2 / Pro | reduces_failing | 10 | 20 | 127012 | 0.063 | 0.446 | 0.639 [0.570, 0.678] | 0.796 [0.680, 0.868] | 0.157 [0.050, 0.225] | 0.780 [0.655, 0.859] | 0.638 [0.308, 0.745] | 0.654 [0.488, 0.759] | 0.629 [0.565, 0.660] |
| Laguna XS.2 / Pro | reduces_failing | 15 | 10 | 127024 | 0.078 | 0.547 | 0.618 [0.558, 0.647] | 0.753 [0.635, 0.830] | 0.135 [0.030, 0.205] | 0.736 [0.612, 0.820] | 0.706 [0.430, 0.801] | 0.662 [0.508, 0.765] | 0.611 [0.556, 0.637] |
| Laguna XS.2 / Pro | reduces_failing | 25 | 10 | 126906 | 0.107 | 0.722 | 0.591 [0.538, 0.623] | 0.699 [0.586, 0.786] | 0.109 [0.012, 0.183] | 0.683 [0.573, 0.773] | 0.725 [0.414, 0.825] | 0.656 [0.532, 0.752] | 0.587 [0.537, 0.617] |
| Laguna XS.2 / Pro | reduces_failing | 50 | 10 | 126310 | 0.129 | 0.936 | 0.579 [0.529, 0.613] | 0.696 [0.580, 0.780] | 0.118 [0.020, 0.196] | 0.690 [0.578, 0.772] | 0.781 [0.396, 0.873] | 0.691 [0.575, 0.775] | 0.578 [0.529, 0.613] |
| Qwen3.6-35B-A3B / Pro | compiles | 0 | 0 | 66858 | – | – | – | 0.760 [0.644, 0.859] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Pro | compiles | 1 | 0 | 66858 | 0.020 | 0.100 | 0.981 [0.958, 0.995] | 0.742 [0.626, 0.841] | -0.239 [-0.353, -0.140] | 0.334 [0.164, 0.773] | 0.375 [0.285, 0.892] | 0.683 [0.430, 0.858] | 0.808 [0.638, 0.941] |
| Qwen3.6-35B-A3B / Pro | compiles | 2 | 0 | 66858 | 0.024 | 0.125 | 0.976 [0.948, 0.994] | 0.739 [0.622, 0.838] | -0.238 [-0.352, -0.138] | 0.348 [0.180, 0.775] | 0.382 [0.272, 0.911] | 0.701 [0.431, 0.872] | 0.811 [0.628, 0.945] |
| Qwen3.6-35B-A3B / Pro | compiles | 3 | 0 | 66858 | 0.027 | 0.142 | 0.973 [0.942, 0.993] | 0.735 [0.619, 0.833] | -0.239 [-0.352, -0.139] | 0.332 [0.180, 0.778] | 0.384 [0.277, 0.915] | 0.700 [0.431, 0.881] | 0.812 [0.631, 0.945] |
| Qwen3.6-35B-A3B / Pro | compiles | 5 | 0 | 66845 | 0.032 | 0.171 | 0.969 [0.932, 0.993] | 0.729 [0.616, 0.828] | -0.239 [-0.351, -0.140] | 0.332 [0.185, 0.781] | 0.346 [0.246, 0.909] | 0.691 [0.434, 0.884] | 0.816 [0.647, 0.950] |
| Qwen3.6-35B-A3B / Pro | compiles | 10 | 0 | 66837 | 0.044 | 0.221 | 0.957 [0.908, 0.989] | 0.716 [0.605, 0.820] | -0.241 [-0.351, -0.142] | 0.333 [0.199, 0.762] | 0.340 [0.235, 0.932] | 0.676 [0.428, 0.865] | 0.805 [0.634, 0.940] |
| Qwen3.6-35B-A3B / Pro | compiles | 15 | 0 | 66838 | 0.055 | 0.266 | 0.946 [0.888, 0.984] | 0.709 [0.598, 0.815] | -0.237 [-0.342, -0.135] | 0.343 [0.207, 0.884] | 0.361 [0.254, 0.909] | 0.671 [0.442, 0.850] | 0.797 [0.629, 0.931] |
| Qwen3.6-35B-A3B / Pro | compiles | 25 | 0 | 66836 | 0.073 | 0.367 | 0.927 [0.850, 0.979] | 0.694 [0.576, 0.805] | -0.233 [-0.339, -0.127] | 0.331 [0.205, 0.672] | 0.429 [0.306, 0.864] | 0.657 [0.471, 0.819] | 0.803 [0.663, 0.937] |
| Qwen3.6-35B-A3B / Pro | compiles | 50 | 0 | 67013 | 0.112 | 0.650 | 0.887 [0.811, 0.942] | 0.690 [0.580, 0.792] | -0.197 [-0.302, -0.088] | 0.367 [0.243, 0.632] | 0.550 [0.420, 0.737] | 0.705 [0.594, 0.795] | 0.826 [0.732, 0.904] |
| Qwen3.6-35B-A3B / Pro | correct | 0 | 30 | 66858 | – | – | – | 0.830 [0.788, 0.953] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Pro | correct | 1 | 30 | 66858 | 0.010 | 0.100 | 0.902 [0.571, 0.966] | 0.829 [0.779, 0.932] | -0.073 [-0.166, 0.351] | 0.892 [0.841, 0.955] | 0.186 [0.124, 0.309] | 0.649 [0.508, 0.778] | 0.767 [0.497, 0.899] |
| Qwen3.6-35B-A3B / Pro | correct | 2 | 30 | 66858 | 0.013 | 0.125 | 0.882 [0.575, 0.946] | 0.817 [0.762, 0.930] | -0.065 [-0.160, 0.346] | 0.855 [0.782, 0.951] | 0.331 [0.166, 0.466] | 0.596 [0.474, 0.774] | 0.774 [0.500, 0.879] |
| Qwen3.6-35B-A3B / Pro | correct | 3 | 30 | 66858 | 0.015 | 0.142 | 0.869 [0.571, 0.933] | 0.812 [0.756, 0.923] | -0.057 [-0.152, 0.341] | 0.832 [0.754, 0.940] | 0.471 [0.371, 0.640] | 0.581 [0.474, 0.767] | 0.776 [0.500, 0.870] |
| Qwen3.6-35B-A3B / Pro | correct | 5 | 30 | 66845 | 0.019 | 0.171 | 0.844 [0.558, 0.911] | 0.787 [0.728, 0.911] | -0.057 [-0.154, 0.341] | 0.774 [0.673, 0.931] | 0.302 [0.119, 0.459] | 0.544 [0.444, 0.741] | 0.762 [0.500, 0.851] |
| Qwen3.6-35B-A3B / Pro | correct | 10 | 30 | 66837 | 0.027 | 0.221 | 0.804 [0.545, 0.871] | 0.745 [0.681, 0.883] | -0.059 [-0.160, 0.330] | 0.677 [0.564, 0.889] | 0.274 [0.088, 0.423] | 0.525 [0.436, 0.711] | 0.743 [0.500, 0.830] |
| Qwen3.6-35B-A3B / Pro | correct | 15 | 30 | 66838 | 0.033 | 0.266 | 0.777 [0.526, 0.854] | 0.742 [0.653, 0.842] | -0.035 [-0.132, 0.303] | 0.671 [0.567, 0.838] | 0.244 [0.107, 0.392] | 0.546 [0.435, 0.680] | 0.732 [0.500, 0.832] |
| Qwen3.6-35B-A3B / Pro | correct | 25 | 30 | 66836 | 0.051 | 0.367 | 0.723 [0.511, 0.829] | 0.736 [0.575, 0.806] | 0.013 [-0.081, 0.228] | 0.670 [0.546, 0.773] | 0.418 [0.079, 0.768] | 0.602 [0.438, 0.699] | 0.699 [0.504, 0.821] |
| Qwen3.6-35B-A3B / Pro | correct | 50 | 30 | 67013 | 0.125 | 0.650 | 0.618 [0.502, 0.738] | 0.689 [0.581, 0.770] | 0.071 [-0.036, 0.194] | 0.656 [0.563, 0.740] | 0.403 [0.145, 0.459] | 0.607 [0.499, 0.698] | 0.609 [0.501, 0.730] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 0 | 10 | 66858 | – | – | – | 0.897 [0.853, 0.934] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 1 | 10 | 66858 | 0.002 | 0.100 | 0.651 [0.649, 0.651] | 0.867 [0.811, 0.912] | 0.216 [0.161, 0.261] | 0.910 [0.864, 0.947] | 0.022 [0.022, 0.022] | 0.533 [0.437, 0.663] | 0.493 [0.471, 0.496] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 2 | 10 | 66858 | 0.003 | 0.125 | 0.554 [0.552, 0.554] | 0.903 [0.854, 0.941] | 0.349 [0.300, 0.387] | 0.916 [0.870, 0.951] | 0.125 [0.125, 0.125] | 0.602 [0.492, 0.730] | 0.493 [0.471, 0.496] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 3 | 10 | 66858 | 0.003 | 0.142 | 0.534 [0.532, 0.534] | 0.913 [0.868, 0.947] | 0.378 [0.334, 0.413] | 0.921 [0.879, 0.954] | 0.118 [0.118, 0.118] | 0.649 [0.548, 0.755] | 0.494 [0.473, 0.496] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 5 | 10 | 66845 | 0.005 | 0.171 | 0.499 [0.497, 0.499] | 0.947 [0.917, 0.970] | 0.447 [0.419, 0.472] | 0.947 [0.918, 0.971] | – | 0.776 [0.701, 0.848] | 0.495 [0.475, 0.496] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 10 | 10 | 66837 | 0.003 | 0.221 | 0.499 [0.497, 0.500] | 0.825 [0.704, 0.884] | 0.326 [0.204, 0.385] | 0.826 [0.704, 0.885] | – | 0.493 [0.320, 0.601] | 0.496 [0.484, 0.500] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 15 | 30 | 66838 | 0.003 | 0.266 | 0.499 [0.497, 0.500] | 0.760 [0.668, 0.829] | 0.260 [0.171, 0.331] | 0.761 [0.671, 0.832] | – | 0.469 [0.370, 0.581] | 0.497 [0.487, 0.500] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 25 | 30 | 66836 | 0.007 | 0.367 | 0.499 [0.497, 0.500] | 0.807 [0.680, 0.913] | 0.308 [0.182, 0.413] | 0.808 [0.681, 0.913] | – | 0.698 [0.532, 0.842] | 0.497 [0.491, 0.500] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 50 | 30 | 67013 | 0.013 | 0.650 | 0.499 [0.497, 0.500] | 0.859 [0.788, 0.940] | 0.360 [0.290, 0.440] | 0.859 [0.788, 0.940] | – | 0.853 [0.781, 0.933] | 0.499 [0.495, 0.500] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 0 | 20 | 66858 | – | – | – | 0.856 [0.740, 0.946] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 1 | 20 | 66858 | 0.013 | 0.100 | 0.753 [0.649, 0.920] | 0.906 [0.842, 0.971] | 0.154 [-0.028, 0.316] | 0.954 [0.902, 0.978] | 0.230 [0.071, 0.272] | 0.754 [0.605, 0.834] | 0.571 [0.495, 0.622] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 2 | 20 | 66858 | 0.016 | 0.125 | 0.731 [0.624, 0.906] | 0.906 [0.835, 0.972] | 0.175 [-0.023, 0.346] | 0.950 [0.894, 0.977] | 0.362 [0.139, 0.418] | 0.771 [0.612, 0.850] | 0.564 [0.495, 0.605] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 3 | 20 | 66858 | 0.017 | 0.142 | 0.722 [0.617, 0.890] | 0.901 [0.831, 0.965] | 0.178 [-0.015, 0.345] | 0.938 [0.872, 0.969] | 0.625 [0.310, 0.707] | 0.757 [0.594, 0.845] | 0.567 [0.500, 0.603] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 5 | 20 | 66845 | 0.020 | 0.171 | 0.702 [0.602, 0.866] | 0.901 [0.829, 0.961] | 0.199 [0.008, 0.356] | 0.933 [0.866, 0.965] | 0.632 [0.351, 0.711] | 0.760 [0.601, 0.848] | 0.580 [0.500, 0.632] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 10 | 20 | 66837 | 0.029 | 0.221 | 0.658 [0.569, 0.780] | 0.868 [0.798, 0.915] | 0.210 [0.044, 0.333] | 0.877 [0.777, 0.920] | 0.609 [0.411, 0.673] | 0.702 [0.548, 0.788] | 0.565 [0.500, 0.581] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 15 | 20 | 66838 | 0.036 | 0.266 | 0.635 [0.549, 0.748] | 0.862 [0.781, 0.905] | 0.227 [0.076, 0.330] | 0.860 [0.759, 0.906] | 0.569 [0.256, 0.704] | 0.713 [0.564, 0.792] | 0.557 [0.500, 0.573] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 25 | 20 | 66836 | 0.054 | 0.367 | 0.599 [0.525, 0.708] | 0.824 [0.721, 0.885] | 0.225 [0.111, 0.309] | 0.815 [0.701, 0.883] | 0.454 [0.426, 0.475] | 0.712 [0.579, 0.802] | 0.552 [0.512, 0.589] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 50 | 30 | 67013 | 0.108 | 0.650 | 0.546 [0.500, 0.625] | 0.739 [0.631, 0.851] | 0.194 [0.102, 0.305] | 0.727 [0.621, 0.844] | 0.453 [0.183, 0.521] | 0.687 [0.574, 0.805] | 0.531 [0.498, 0.587] |

#### Persistence analysis, window max15 (test split; 95% CI from 1000 instance-level bootstrap resamples)

| Model / benchmark | Probe | k | layer | n_test | flip rate | pairs w/ edit between | persistence AUC | probe AUC (same tokens) | probe − persistence | probe AUC, y_t=0 | probe AUC, y_t=1 | probe AUC, edit between | persistence AUC, edit between |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | compiles | 0 | 10 | 719160 | – | – | – | 0.691 [0.630, 0.780] | – | – | – | – | – |
| Laguna XS.2 / Verified | compiles | 1 | 30 | 719133 | 0.011 | 0.161 | 0.961 [0.916, 0.979] | 0.675 [0.590, 0.773] | -0.287 [-0.383, -0.154] | 0.421 [0.305, 0.584] | 0.729 [0.683, 0.779] | 0.625 [0.576, 0.682] | 0.784 [0.686, 0.835] |
| Laguna XS.2 / Verified | compiles | 2 | 30 | 719130 | 0.015 | 0.214 | 0.946 [0.891, 0.968] | 0.672 [0.587, 0.767] | -0.274 [-0.372, -0.134] | 0.419 [0.304, 0.579] | 0.710 [0.661, 0.762] | 0.632 [0.577, 0.689] | 0.775 [0.678, 0.831] |
| Laguna XS.2 / Verified | compiles | 3 | 30 | 719128 | 0.021 | 0.269 | 0.926 [0.871, 0.951] | 0.667 [0.580, 0.766] | -0.259 [-0.356, -0.119] | 0.440 [0.328, 0.596] | 0.656 [0.571, 0.738] | 0.619 [0.541, 0.693] | 0.756 [0.668, 0.814] |
| Laguna XS.2 / Verified | compiles | 5 | 30 | 719161 | 0.030 | 0.360 | 0.897 [0.837, 0.931] | 0.654 [0.560, 0.758] | -0.243 [-0.347, -0.093] | 0.439 [0.325, 0.592] | 0.633 [0.529, 0.731] | 0.623 [0.530, 0.705] | 0.739 [0.654, 0.805] |
| Laguna XS.2 / Verified | compiles | 10 | 10 | 719179 | 0.044 | 0.513 | 0.857 [0.779, 0.903] | 0.667 [0.588, 0.765] | -0.190 [-0.294, -0.023] | 0.502 [0.385, 0.672] | 0.675 [0.575, 0.764] | 0.667 [0.576, 0.750] | 0.726 [0.638, 0.795] |
| Laguna XS.2 / Verified | compiles | 15 | 10 | 719236 | 0.053 | 0.613 | 0.836 [0.752, 0.888] | 0.674 [0.592, 0.773] | -0.162 [-0.278, 0.011] | 0.521 [0.406, 0.687] | 0.689 [0.576, 0.780] | 0.684 [0.592, 0.769] | 0.724 [0.638, 0.798] |
| Laguna XS.2 / Verified | correct | 0 | 20 | 719160 | – | – | – | 0.784 [0.727, 0.836] | – | – | – | – | – |
| Laguna XS.2 / Verified | correct | 1 | 20 | 719133 | 0.045 | 0.161 | 0.931 [0.916, 0.944] | 0.751 [0.691, 0.803] | -0.180 [-0.244, -0.122] | 0.645 [0.597, 0.691] | 0.726 [0.662, 0.795] | 0.594 [0.528, 0.664] | 0.654 [0.595, 0.713] |
| Laguna XS.2 / Verified | correct | 2 | 20 | 719130 | 0.059 | 0.214 | 0.913 [0.895, 0.929] | 0.741 [0.682, 0.795] | -0.172 [-0.238, -0.112] | 0.646 [0.597, 0.691] | 0.723 [0.661, 0.786] | 0.589 [0.519, 0.659] | 0.667 [0.610, 0.723] |
| Laguna XS.2 / Verified | correct | 3 | 20 | 719128 | 0.072 | 0.269 | 0.897 [0.877, 0.914] | 0.731 [0.672, 0.784] | -0.166 [-0.231, -0.105] | 0.635 [0.586, 0.681] | 0.705 [0.644, 0.767] | 0.590 [0.520, 0.661] | 0.678 [0.622, 0.732] |
| Laguna XS.2 / Verified | correct | 5 | 20 | 719161 | 0.099 | 0.360 | 0.867 [0.843, 0.888] | 0.712 [0.653, 0.766] | -0.155 [-0.221, -0.093] | 0.618 [0.566, 0.669] | 0.680 [0.612, 0.742] | 0.592 [0.524, 0.662] | 0.679 [0.625, 0.733] |
| Laguna XS.2 / Verified | correct | 10 | 20 | 719179 | 0.147 | 0.513 | 0.821 [0.792, 0.847] | 0.680 [0.619, 0.738] | -0.141 [-0.212, -0.076] | 0.606 [0.547, 0.664] | 0.632 [0.556, 0.695] | 0.597 [0.531, 0.666] | 0.684 [0.630, 0.734] |
| Laguna XS.2 / Verified | correct | 15 | 20 | 719236 | 0.182 | 0.613 | 0.794 [0.762, 0.822] | 0.663 [0.599, 0.725] | -0.131 [-0.201, -0.062] | 0.604 [0.540, 0.667] | 0.612 [0.511, 0.692] | 0.602 [0.534, 0.669] | 0.685 [0.635, 0.734] |
| Laguna XS.2 / Verified | has_regressions | 0 | 20 | 719160 | – | – | – | 0.683 [0.616, 0.760] | – | – | – | – | – |
| Laguna XS.2 / Verified | has_regressions | 1 | 20 | 719133 | 0.019 | 0.161 | 0.944 [0.908, 0.963] | 0.664 [0.598, 0.741] | -0.280 [-0.352, -0.192] | 0.693 [0.644, 0.747] | 0.309 [0.234, 0.413] | 0.598 [0.525, 0.670] | 0.768 [0.694, 0.828] |
| Laguna XS.2 / Verified | has_regressions | 2 | 20 | 719130 | 0.025 | 0.214 | 0.927 [0.882, 0.951] | 0.654 [0.589, 0.726] | -0.273 [-0.345, -0.181] | 0.654 [0.602, 0.709] | 0.355 [0.283, 0.461] | 0.601 [0.532, 0.672] | 0.772 [0.699, 0.828] |
| Laguna XS.2 / Verified | has_regressions | 3 | 20 | 719128 | 0.031 | 0.269 | 0.908 [0.854, 0.937] | 0.643 [0.578, 0.717] | -0.265 [-0.342, -0.170] | 0.621 [0.568, 0.675] | 0.377 [0.310, 0.474] | 0.586 [0.520, 0.656] | 0.771 [0.702, 0.827] |
| Laguna XS.2 / Verified | has_regressions | 5 | 20 | 719161 | 0.041 | 0.360 | 0.881 [0.820, 0.913] | 0.627 [0.566, 0.698] | -0.253 [-0.329, -0.159] | 0.584 [0.526, 0.643] | 0.393 [0.330, 0.489] | 0.578 [0.514, 0.642] | 0.768 [0.700, 0.820] |
| Laguna XS.2 / Verified | has_regressions | 10 | 20 | 719179 | 0.057 | 0.513 | 0.843 [0.771, 0.880] | 0.602 [0.542, 0.673] | -0.241 [-0.315, -0.138] | 0.563 [0.501, 0.627] | 0.425 [0.356, 0.510] | 0.577 [0.523, 0.637] | 0.767 [0.700, 0.814] |
| Laguna XS.2 / Verified | has_regressions | 15 | 20 | 719236 | 0.065 | 0.613 | 0.826 [0.752, 0.862] | 0.586 [0.530, 0.653] | -0.240 [-0.311, -0.141] | 0.557 [0.496, 0.620] | 0.423 [0.358, 0.508] | 0.568 [0.519, 0.620] | 0.768 [0.698, 0.811] |
| Laguna XS.2 / Verified | reduces_failing | 0 | 20 | 719160 | – | – | – | 0.792 [0.745, 0.837] | – | – | – | – | – |
| Laguna XS.2 / Verified | reduces_failing | 1 | 20 | 719133 | 0.048 | 0.161 | 0.935 [0.919, 0.947] | 0.759 [0.707, 0.809] | -0.176 [-0.232, -0.119] | 0.692 [0.647, 0.737] | 0.609 [0.539, 0.677] | 0.581 [0.523, 0.647] | 0.691 [0.635, 0.741] |
| Laguna XS.2 / Verified | reduces_failing | 2 | 20 | 719130 | 0.063 | 0.214 | 0.916 [0.897, 0.930] | 0.744 [0.692, 0.796] | -0.171 [-0.230, -0.113] | 0.660 [0.616, 0.705] | 0.609 [0.537, 0.679] | 0.573 [0.513, 0.637] | 0.700 [0.647, 0.745] |
| Laguna XS.2 / Verified | reduces_failing | 3 | 20 | 719128 | 0.078 | 0.269 | 0.898 [0.879, 0.913] | 0.730 [0.676, 0.783] | -0.168 [-0.228, -0.110] | 0.638 [0.594, 0.682] | 0.603 [0.532, 0.671] | 0.577 [0.516, 0.639] | 0.704 [0.659, 0.744] |
| Laguna XS.2 / Verified | reduces_failing | 5 | 20 | 719161 | 0.105 | 0.360 | 0.870 [0.848, 0.886] | 0.705 [0.650, 0.760] | -0.165 [-0.226, -0.106] | 0.606 [0.562, 0.649] | 0.587 [0.515, 0.656] | 0.574 [0.515, 0.636] | 0.705 [0.660, 0.745] |
| Laguna XS.2 / Verified | reduces_failing | 10 | 20 | 719179 | 0.154 | 0.513 | 0.827 [0.800, 0.848] | 0.664 [0.608, 0.716] | -0.164 [-0.229, -0.102] | 0.572 [0.524, 0.616] | 0.588 [0.511, 0.652] | 0.580 [0.526, 0.639] | 0.705 [0.663, 0.745] |
| Laguna XS.2 / Verified | reduces_failing | 15 | 20 | 719236 | 0.190 | 0.613 | 0.801 [0.771, 0.824] | 0.635 [0.580, 0.690] | -0.166 [-0.231, -0.102] | 0.557 [0.509, 0.604] | 0.587 [0.513, 0.647] | 0.579 [0.524, 0.641] | 0.703 [0.664, 0.742] |
| Qwen3.6-35B-A3B / Verified | compiles | 0 | 10 | 353824 | – | – | – | 0.808 [0.771, 0.851] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Verified | compiles | 1 | 10 | 353824 | 0.013 | 0.094 | 0.901 [0.816, 0.953] | 0.799 [0.750, 0.854] | -0.102 [-0.192, 0.035] | 0.351 [0.222, 0.482] | 0.881 [0.799, 0.927] | 0.628 [0.473, 0.739] | 0.580 [0.539, 0.636] |
| Qwen3.6-35B-A3B / Verified | compiles | 2 | 10 | 353824 | 0.017 | 0.132 | 0.870 [0.777, 0.923] | 0.789 [0.730, 0.849] | -0.082 [-0.181, 0.060] | 0.353 [0.220, 0.517] | 0.846 [0.753, 0.897] | 0.619 [0.481, 0.713] | 0.581 [0.537, 0.627] |
| Qwen3.6-35B-A3B / Verified | compiles | 3 | 10 | 353824 | 0.022 | 0.171 | 0.842 [0.749, 0.892] | 0.773 [0.714, 0.835] | -0.069 [-0.161, 0.065] | 0.394 [0.246, 0.569] | 0.806 [0.702, 0.869] | 0.606 [0.470, 0.702] | 0.576 [0.533, 0.617] |
| Qwen3.6-35B-A3B / Verified | compiles | 5 | 10 | 353824 | 0.031 | 0.245 | 0.797 [0.705, 0.842] | 0.754 [0.685, 0.821] | -0.043 [-0.129, 0.090] | 0.430 [0.286, 0.598] | 0.767 [0.668, 0.837] | 0.619 [0.495, 0.709] | 0.571 [0.525, 0.606] |
| Qwen3.6-35B-A3B / Verified | compiles | 10 | 10 | 353824 | 0.041 | 0.410 | 0.755 [0.675, 0.796] | 0.729 [0.648, 0.803] | -0.025 [-0.132, 0.091] | 0.495 [0.354, 0.651] | 0.734 [0.639, 0.809] | 0.668 [0.555, 0.747] | 0.573 [0.536, 0.613] |
| Qwen3.6-35B-A3B / Verified | compiles | 15 | 39 | 353824 | 0.048 | 0.532 | 0.732 [0.659, 0.767] | 0.728 [0.639, 0.841] | -0.004 [-0.118, 0.154] | 0.636 [0.531, 0.832] | 0.738 [0.642, 0.829] | 0.743 [0.640, 0.836] | 0.578 [0.542, 0.615] |
| Qwen3.6-35B-A3B / Verified | correct | 0 | 30 | 353824 | – | – | – | 0.780 [0.703, 0.883] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Verified | correct | 1 | 30 | 353824 | 0.021 | 0.094 | 0.908 [0.851, 0.939] | 0.776 [0.707, 0.861] | -0.132 [-0.226, -0.004] | 0.803 [0.765, 0.836] | 0.553 [0.253, 0.790] | 0.529 [0.479, 0.575] | 0.630 [0.555, 0.697] |
| Qwen3.6-35B-A3B / Verified | correct | 2 | 30 | 353809 | 0.031 | 0.132 | 0.872 [0.805, 0.909] | 0.765 [0.698, 0.840] | -0.107 [-0.204, 0.020] | 0.761 [0.715, 0.803] | 0.583 [0.286, 0.749] | 0.507 [0.457, 0.556] | 0.617 [0.548, 0.677] |
| Qwen3.6-35B-A3B / Verified | correct | 3 | 30 | 353800 | 0.042 | 0.171 | 0.841 [0.769, 0.883] | 0.758 [0.697, 0.828] | -0.083 [-0.178, 0.042] | 0.745 [0.697, 0.788] | 0.615 [0.232, 0.738] | 0.508 [0.453, 0.562] | 0.610 [0.543, 0.670] |
| Qwen3.6-35B-A3B / Verified | correct | 5 | 30 | 353744 | 0.067 | 0.245 | 0.785 [0.710, 0.833] | 0.747 [0.692, 0.805] | -0.038 [-0.134, 0.075] | 0.722 [0.677, 0.765] | 0.648 [0.386, 0.723] | 0.512 [0.460, 0.567] | 0.604 [0.541, 0.658] |
| Qwen3.6-35B-A3B / Verified | correct | 10 | 30 | 353575 | 0.140 | 0.410 | 0.692 [0.624, 0.743] | 0.703 [0.655, 0.750] | 0.011 [-0.070, 0.098] | 0.670 [0.623, 0.717] | 0.656 [0.405, 0.739] | 0.534 [0.476, 0.593] | 0.585 [0.535, 0.631] |
| Qwen3.6-35B-A3B / Verified | correct | 15 | 30 | 353537 | 0.203 | 0.532 | 0.649 [0.589, 0.698] | 0.674 [0.624, 0.719] | 0.026 [-0.047, 0.102] | 0.648 [0.598, 0.695] | 0.587 [0.402, 0.686] | 0.562 [0.504, 0.618] | 0.572 [0.526, 0.616] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 0 | 20 | 353824 | – | – | – | 0.830 [0.781, 0.888] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 1 | 20 | 353824 | 0.016 | 0.094 | 0.855 [0.790, 0.906] | 0.811 [0.769, 0.865] | -0.043 [-0.110, 0.035] | 0.797 [0.737, 0.852] | 0.315 [0.093, 0.575] | 0.547 [0.431, 0.665] | 0.613 [0.528, 0.707] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 2 | 20 | 353809 | 0.019 | 0.132 | 0.835 [0.769, 0.886] | 0.798 [0.751, 0.856] | -0.037 [-0.104, 0.045] | 0.789 [0.729, 0.841] | 0.329 [0.130, 0.563] | 0.587 [0.479, 0.684] | 0.629 [0.536, 0.725] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 3 | 20 | 353800 | 0.023 | 0.171 | 0.812 [0.741, 0.862] | 0.791 [0.745, 0.848] | -0.021 [-0.091, 0.067] | 0.786 [0.732, 0.834] | 0.304 [0.151, 0.566] | 0.610 [0.519, 0.687] | 0.634 [0.534, 0.729] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 5 | 20 | 353744 | 0.032 | 0.245 | 0.770 [0.697, 0.819] | 0.755 [0.705, 0.813] | -0.015 [-0.089, 0.075] | 0.737 [0.684, 0.790] | 0.341 [0.231, 0.570] | 0.613 [0.537, 0.682] | 0.626 [0.538, 0.712] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 10 | 20 | 353575 | 0.053 | 0.410 | 0.700 [0.635, 0.742] | 0.692 [0.630, 0.760] | -0.008 [-0.087, 0.088] | 0.675 [0.609, 0.738] | 0.376 [0.272, 0.557] | 0.625 [0.552, 0.697] | 0.598 [0.532, 0.666] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 15 | 20 | 353537 | 0.064 | 0.532 | 0.668 [0.598, 0.711] | 0.642 [0.578, 0.718] | -0.026 [-0.105, 0.076] | 0.631 [0.567, 0.697] | 0.391 [0.285, 0.581] | 0.614 [0.547, 0.685] | 0.598 [0.539, 0.651] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 0 | 30 | 353824 | – | – | – | 0.832 [0.784, 0.883] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 1 | 30 | 353824 | 0.029 | 0.094 | 0.903 [0.855, 0.933] | 0.819 [0.772, 0.865] | -0.085 [-0.145, -0.015] | 0.807 [0.760, 0.845] | 0.387 [0.292, 0.546] | 0.489 [0.428, 0.546] | 0.639 [0.564, 0.707] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 2 | 30 | 353809 | 0.042 | 0.132 | 0.870 [0.821, 0.904] | 0.804 [0.757, 0.849] | -0.067 [-0.128, 0.005] | 0.764 [0.709, 0.808] | 0.463 [0.373, 0.600] | 0.482 [0.419, 0.538] | 0.636 [0.561, 0.702] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3 | 30 | 353800 | 0.055 | 0.171 | 0.841 [0.790, 0.876] | 0.789 [0.744, 0.835] | -0.051 [-0.113, 0.018] | 0.741 [0.693, 0.785] | 0.503 [0.398, 0.606] | 0.475 [0.418, 0.530] | 0.628 [0.561, 0.690] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 5 | 30 | 353744 | 0.084 | 0.245 | 0.789 [0.735, 0.826] | 0.762 [0.718, 0.807] | -0.027 [-0.084, 0.040] | 0.709 [0.661, 0.752] | 0.493 [0.413, 0.588] | 0.475 [0.423, 0.528] | 0.614 [0.557, 0.671] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 10 | 30 | 353575 | 0.154 | 0.410 | 0.711 [0.658, 0.755] | 0.694 [0.645, 0.741] | -0.017 [-0.076, 0.046] | 0.634 [0.586, 0.683] | 0.506 [0.397, 0.583] | 0.484 [0.427, 0.543] | 0.596 [0.550, 0.644] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 15 | 30 | 353537 | 0.206 | 0.532 | 0.675 [0.626, 0.719] | 0.650 [0.595, 0.701] | -0.025 [-0.081, 0.039] | 0.594 [0.543, 0.645] | 0.527 [0.418, 0.595] | 0.498 [0.439, 0.557] | 0.589 [0.546, 0.636] |
| Laguna XS.2 / Pro | compiles | 0 | 10 | 804894 | – | – | – | 0.784 [0.725, 0.840] | – | – | – | – | – |
| Laguna XS.2 / Pro | compiles | 1 | 10 | 804866 | 0.030 | 0.319 | 0.969 [0.957, 0.980] | 0.785 [0.726, 0.840] | -0.185 [-0.241, -0.133] | 0.542 [0.445, 0.650] | 0.754 [0.674, 0.813] | 0.794 [0.731, 0.853] | 0.905 [0.865, 0.937] |
| Laguna XS.2 / Pro | compiles | 2 | 10 | 804686 | 0.041 | 0.404 | 0.958 [0.941, 0.972] | 0.782 [0.726, 0.839] | -0.176 [-0.229, -0.124] | 0.537 [0.445, 0.636] | 0.763 [0.688, 0.818] | 0.786 [0.726, 0.845] | 0.898 [0.857, 0.932] |
| Laguna XS.2 / Pro | compiles | 3 | 10 | 804688 | 0.049 | 0.473 | 0.950 [0.929, 0.967] | 0.785 [0.726, 0.841] | -0.165 [-0.218, -0.113] | 0.551 [0.462, 0.651] | 0.756 [0.688, 0.811] | 0.785 [0.724, 0.844] | 0.896 [0.856, 0.930] |
| Laguna XS.2 / Pro | compiles | 5 | 10 | 804804 | 0.062 | 0.570 | 0.937 [0.910, 0.957] | 0.782 [0.722, 0.839] | -0.155 [-0.210, -0.103] | 0.558 [0.466, 0.663] | 0.741 [0.669, 0.802] | 0.780 [0.719, 0.840] | 0.891 [0.849, 0.924] |
| Laguna XS.2 / Pro | compiles | 10 | 10 | 804817 | 0.070 | 0.691 | 0.928 [0.897, 0.953] | 0.779 [0.715, 0.839] | -0.150 [-0.210, -0.098] | 0.577 [0.482, 0.685] | 0.721 [0.639, 0.784] | 0.778 [0.712, 0.840] | 0.897 [0.855, 0.930] |
| Laguna XS.2 / Pro | compiles | 15 | 10 | 804833 | 0.077 | 0.757 | 0.922 [0.891, 0.948] | 0.778 [0.711, 0.837] | -0.145 [-0.204, -0.091] | 0.583 [0.485, 0.694] | 0.727 [0.650, 0.790] | 0.776 [0.706, 0.841] | 0.898 [0.857, 0.931] |
| Laguna XS.2 / Pro | correct | 0 | 20 | 804894 | – | – | – | 0.803 [0.749, 0.844] | – | – | – | – | – |
| Laguna XS.2 / Pro | correct | 1 | 20 | 804866 | 0.009 | 0.319 | 0.911 [0.819, 0.948] | 0.791 [0.738, 0.824] | -0.120 [-0.161, -0.039] | 0.727 [0.672, 0.776] | 0.634 [0.552, 0.724] | 0.710 [0.634, 0.762] | 0.834 [0.650, 0.897] |
| Laguna XS.2 / Pro | correct | 2 | 20 | 804686 | 0.013 | 0.404 | 0.882 [0.763, 0.933] | 0.791 [0.727, 0.823] | -0.091 [-0.139, 0.009] | 0.727 [0.669, 0.777] | 0.640 [0.556, 0.720] | 0.707 [0.618, 0.756] | 0.814 [0.642, 0.888] |
| Laguna XS.2 / Pro | correct | 3 | 20 | 804688 | 0.017 | 0.473 | 0.855 [0.726, 0.919] | 0.787 [0.720, 0.821] | -0.068 [-0.128, 0.048] | 0.731 [0.668, 0.783] | 0.624 [0.538, 0.686] | 0.704 [0.617, 0.750] | 0.798 [0.634, 0.881] |
| Laguna XS.2 / Pro | correct | 5 | 20 | 804804 | 0.023 | 0.570 | 0.817 [0.683, 0.896] | 0.777 [0.703, 0.814] | -0.039 [-0.115, 0.085] | 0.720 [0.641, 0.779] | 0.646 [0.562, 0.720] | 0.707 [0.605, 0.753] | 0.783 [0.621, 0.873] |
| Laguna XS.2 / Pro | correct | 10 | 20 | 804817 | 0.035 | 0.691 | 0.756 [0.629, 0.852] | 0.762 [0.681, 0.802] | 0.006 [-0.086, 0.133] | 0.711 [0.627, 0.775] | 0.664 [0.564, 0.738] | 0.713 [0.623, 0.760] | 0.739 [0.592, 0.841] |
| Laguna XS.2 / Pro | correct | 15 | 20 | 804833 | 0.042 | 0.757 | 0.732 [0.610, 0.828] | 0.744 [0.656, 0.791] | 0.013 [-0.078, 0.130] | 0.689 [0.601, 0.756] | 0.676 [0.552, 0.776] | 0.710 [0.617, 0.760] | 0.719 [0.582, 0.822] |
| Laguna XS.2 / Pro | has_regressions | 0 | 30 | 804894 | – | – | – | 0.777 [0.726, 0.828] | – | – | – | – | – |
| Laguna XS.2 / Pro | has_regressions | 1 | 30 | 804866 | 0.009 | 0.319 | 0.921 [0.864, 0.939] | 0.774 [0.721, 0.834] | -0.147 [-0.199, -0.066] | 0.814 [0.764, 0.868] | 0.391 [0.287, 0.493] | 0.729 [0.668, 0.800] | 0.848 [0.767, 0.886] |
| Laguna XS.2 / Pro | has_regressions | 2 | 30 | 804686 | 0.013 | 0.404 | 0.895 [0.827, 0.916] | 0.771 [0.721, 0.825] | -0.125 [-0.181, -0.033] | 0.793 [0.745, 0.841] | 0.418 [0.320, 0.510] | 0.720 [0.660, 0.794] | 0.835 [0.737, 0.880] |
| Laguna XS.2 / Pro | has_regressions | 3 | 30 | 804688 | 0.016 | 0.473 | 0.877 [0.794, 0.903] | 0.775 [0.724, 0.829] | -0.102 [-0.162, -0.008] | 0.780 [0.734, 0.828] | 0.468 [0.373, 0.556] | 0.721 [0.658, 0.799] | 0.827 [0.726, 0.873] |
| Laguna XS.2 / Pro | has_regressions | 5 | 30 | 804804 | 0.021 | 0.570 | 0.846 [0.754, 0.877] | 0.771 [0.716, 0.827] | -0.075 [-0.142, 0.023] | 0.763 [0.716, 0.811] | 0.473 [0.371, 0.572] | 0.725 [0.663, 0.798] | 0.806 [0.700, 0.855] |
| Laguna XS.2 / Pro | has_regressions | 10 | 30 | 804817 | 0.029 | 0.691 | 0.804 [0.708, 0.847] | 0.762 [0.702, 0.824] | -0.041 [-0.127, 0.070] | 0.756 [0.699, 0.808] | 0.502 [0.382, 0.621] | 0.734 [0.666, 0.806] | 0.779 [0.667, 0.830] |
| Laguna XS.2 / Pro | has_regressions | 15 | 30 | 804833 | 0.036 | 0.757 | 0.776 [0.670, 0.827] | 0.746 [0.675, 0.816] | -0.030 [-0.131, 0.092] | 0.744 [0.684, 0.799] | 0.474 [0.348, 0.608] | 0.732 [0.659, 0.805] | 0.760 [0.641, 0.817] |
| Laguna XS.2 / Pro | reduces_failing | 0 | 20 | 804894 | – | – | – | 0.799 [0.736, 0.845] | – | – | – | – | – |
| Laguna XS.2 / Pro | reduces_failing | 1 | 20 | 804866 | 0.018 | 0.319 | 0.928 [0.897, 0.949] | 0.788 [0.729, 0.837] | -0.139 [-0.198, -0.084] | 0.761 [0.702, 0.816] | 0.505 [0.359, 0.623] | 0.686 [0.592, 0.755] | 0.854 [0.782, 0.899] |
| Laguna XS.2 / Pro | reduces_failing | 2 | 20 | 804686 | 0.025 | 0.404 | 0.903 [0.864, 0.930] | 0.782 [0.723, 0.832] | -0.121 [-0.181, -0.058] | 0.756 [0.699, 0.807] | 0.523 [0.377, 0.640] | 0.692 [0.610, 0.756] | 0.839 [0.767, 0.885] |
| Laguna XS.2 / Pro | reduces_failing | 3 | 20 | 804688 | 0.032 | 0.473 | 0.880 [0.833, 0.913] | 0.775 [0.715, 0.827] | -0.104 [-0.169, -0.038] | 0.745 [0.683, 0.799] | 0.511 [0.368, 0.625] | 0.692 [0.611, 0.757] | 0.828 [0.757, 0.875] |
| Laguna XS.2 / Pro | reduces_failing | 5 | 10 | 804804 | 0.042 | 0.570 | 0.849 [0.796, 0.887] | 0.767 [0.682, 0.837] | -0.082 [-0.161, -0.010] | 0.722 [0.634, 0.799] | 0.546 [0.385, 0.644] | 0.711 [0.604, 0.795] | 0.809 [0.738, 0.860] |
| Laguna XS.2 / Pro | reduces_failing | 10 | 10 | 804817 | 0.056 | 0.691 | 0.813 [0.760, 0.855] | 0.752 [0.669, 0.822] | -0.061 [-0.141, 0.013] | 0.695 [0.605, 0.775] | 0.579 [0.412, 0.720] | 0.713 [0.619, 0.793] | 0.786 [0.721, 0.834] |
| Laguna XS.2 / Pro | reduces_failing | 15 | 10 | 804833 | 0.063 | 0.757 | 0.795 [0.737, 0.842] | 0.735 [0.645, 0.810] | -0.060 [-0.141, 0.017] | 0.676 [0.583, 0.761] | 0.565 [0.395, 0.713] | 0.711 [0.615, 0.793] | 0.776 [0.711, 0.825] |
| Qwen3.6-35B-A3B / Pro | compiles | 0 | 0 | 360151 | – | – | – | 0.825 [0.741, 0.895] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Pro | compiles | 1 | 0 | 360199 | 0.019 | 0.279 | 0.982 [0.969, 0.992] | 0.824 [0.741, 0.891] | -0.158 [-0.238, -0.095] | 0.595 [0.355, 0.830] | 0.678 [0.530, 0.816] | 0.808 [0.687, 0.895] | 0.935 [0.890, 0.972] |
| Qwen3.6-35B-A3B / Pro | compiles | 2 | 0 | 360206 | 0.023 | 0.324 | 0.977 [0.962, 0.990] | 0.824 [0.743, 0.892] | -0.153 [-0.232, -0.091] | 0.612 [0.379, 0.835] | 0.699 [0.526, 0.823] | 0.810 [0.698, 0.894] | 0.930 [0.884, 0.969] |
| Qwen3.6-35B-A3B / Pro | compiles | 3 | 0 | 360330 | 0.026 | 0.367 | 0.975 [0.957, 0.988] | 0.823 [0.742, 0.891] | -0.152 [-0.230, -0.090] | 0.596 [0.376, 0.828] | 0.681 [0.491, 0.820] | 0.807 [0.701, 0.890] | 0.931 [0.887, 0.968] |
| Qwen3.6-35B-A3B / Pro | compiles | 5 | 0 | 360379 | 0.031 | 0.433 | 0.970 [0.953, 0.985] | 0.822 [0.743, 0.890] | -0.148 [-0.225, -0.087] | 0.586 [0.391, 0.798] | 0.702 [0.548, 0.815] | 0.807 [0.707, 0.886] | 0.931 [0.894, 0.964] |
| Qwen3.6-35B-A3B / Pro | compiles | 10 | 0 | 360478 | 0.040 | 0.545 | 0.961 [0.940, 0.979] | 0.822 [0.744, 0.889] | -0.139 [-0.213, -0.079] | 0.616 [0.447, 0.805] | 0.680 [0.549, 0.791] | 0.808 [0.716, 0.886] | 0.929 [0.893, 0.960] |
| Qwen3.6-35B-A3B / Pro | compiles | 15 | 0 | 360563 | 0.046 | 0.631 | 0.955 [0.932, 0.976] | 0.825 [0.748, 0.892] | -0.130 [-0.198, -0.071] | 0.630 [0.464, 0.809] | 0.711 [0.563, 0.816] | 0.812 [0.727, 0.889] | 0.929 [0.893, 0.962] |
| Qwen3.6-35B-A3B / Pro | correct | 0 | 20 | 360151 | – | – | – | 0.869 [0.818, 0.915] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Pro | correct | 1 | 20 | 360199 | 0.022 | 0.279 | 0.850 [0.753, 0.889] | 0.836 [0.784, 0.879] | -0.014 [-0.065, 0.065] | 0.799 [0.746, 0.854] | 0.663 [0.573, 0.814] | 0.653 [0.573, 0.757] | 0.771 [0.565, 0.840] |
| Qwen3.6-35B-A3B / Pro | correct | 2 | 20 | 360206 | 0.025 | 0.324 | 0.836 [0.733, 0.879] | 0.828 [0.776, 0.871] | -0.008 [-0.064, 0.075] | 0.791 [0.739, 0.844] | 0.660 [0.553, 0.730] | 0.655 [0.576, 0.751] | 0.769 [0.575, 0.838] |
| Qwen3.6-35B-A3B / Pro | correct | 3 | 20 | 360330 | 0.027 | 0.367 | 0.826 [0.723, 0.870] | 0.824 [0.773, 0.866] | -0.002 [-0.053, 0.082] | 0.790 [0.740, 0.838] | 0.651 [0.464, 0.720] | 0.657 [0.582, 0.745] | 0.770 [0.579, 0.836] |
| Qwen3.6-35B-A3B / Pro | correct | 5 | 20 | 360379 | 0.033 | 0.433 | 0.799 [0.686, 0.847] | 0.811 [0.758, 0.850] | 0.012 [-0.045, 0.107] | 0.776 [0.730, 0.819] | 0.572 [0.340, 0.657] | 0.665 [0.599, 0.736] | 0.752 [0.572, 0.817] |
| Qwen3.6-35B-A3B / Pro | correct | 10 | 20 | 360478 | 0.054 | 0.545 | 0.738 [0.619, 0.800] | 0.775 [0.715, 0.820] | 0.036 [-0.034, 0.132] | 0.737 [0.680, 0.785] | 0.428 [0.063, 0.530] | 0.658 [0.586, 0.724] | 0.705 [0.558, 0.780] |
| Qwen3.6-35B-A3B / Pro | correct | 15 | 20 | 360563 | 0.071 | 0.631 | 0.704 [0.592, 0.775] | 0.750 [0.685, 0.800] | 0.047 [-0.041, 0.153] | 0.716 [0.648, 0.766] | 0.456 [0.052, 0.587] | 0.660 [0.587, 0.724] | 0.678 [0.556, 0.758] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 0 | 20 | 360151 | – | – | – | 0.869 [0.728, 0.910] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 1 | 39 | 360199 | 0.005 | 0.279 | 0.758 [0.638, 0.821] | 0.884 [0.787, 0.935] | 0.127 [0.027, 0.178] | 0.880 [0.793, 0.941] | 0.613 [0.332, 0.718] | 0.841 [0.692, 0.911] | 0.678 [0.534, 0.742] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 2 | 39 | 360206 | 0.006 | 0.324 | 0.735 [0.611, 0.797] | 0.885 [0.796, 0.933] | 0.149 [0.052, 0.212] | 0.873 [0.785, 0.931] | 0.667 [0.312, 0.770] | 0.843 [0.703, 0.914] | 0.676 [0.535, 0.739] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 3 | 39 | 360330 | 0.007 | 0.367 | 0.717 [0.593, 0.777] | 0.858 [0.750, 0.921] | 0.142 [0.045, 0.193] | 0.834 [0.727, 0.911] | 0.690 [0.393, 0.799] | 0.819 [0.666, 0.901] | 0.669 [0.538, 0.725] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 5 | 30 | 360379 | 0.008 | 0.433 | 0.686 [0.547, 0.749] | 0.835 [0.735, 0.883] | 0.149 [0.081, 0.235] | 0.822 [0.737, 0.875] | 0.463 [0.213, 0.651] | 0.761 [0.625, 0.823] | 0.658 [0.521, 0.722] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 10 | 39 | 360478 | 0.011 | 0.545 | 0.631 [0.504, 0.672] | 0.845 [0.695, 0.908] | 0.214 [0.154, 0.243] | 0.821 [0.689, 0.894] | 0.794 [0.513, 0.873] | 0.830 [0.663, 0.904] | 0.619 [0.501, 0.668] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 15 | 39 | 360563 | 0.013 | 0.631 | 0.610 [0.499, 0.643] | 0.853 [0.657, 0.906] | 0.243 [0.145, 0.270] | 0.834 [0.657, 0.893] | 0.779 [0.519, 0.901] | 0.843 [0.632, 0.905] | 0.599 [0.498, 0.642] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 0 | 30 | 360151 | – | – | – | 0.845 [0.770, 0.902] | – | – | – | – | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 1 | 20 | 360199 | 0.026 | 0.279 | 0.845 [0.761, 0.882] | 0.823 [0.744, 0.884] | -0.022 [-0.111, 0.083] | 0.838 [0.774, 0.886] | 0.454 [0.332, 0.589] | 0.637 [0.505, 0.774] | 0.754 [0.582, 0.826] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 2 | 30 | 360206 | 0.029 | 0.324 | 0.829 [0.738, 0.870] | 0.812 [0.730, 0.880] | -0.017 [-0.115, 0.100] | 0.822 [0.750, 0.880] | 0.442 [0.329, 0.557] | 0.634 [0.505, 0.770] | 0.754 [0.594, 0.825] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 3 | 20 | 360330 | 0.034 | 0.367 | 0.813 [0.720, 0.858] | 0.803 [0.721, 0.870] | -0.010 [-0.109, 0.110] | 0.815 [0.746, 0.868] | 0.455 [0.362, 0.560] | 0.640 [0.516, 0.763] | 0.745 [0.593, 0.816] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 5 | 20 | 360379 | 0.044 | 0.433 | 0.780 [0.679, 0.831] | 0.787 [0.702, 0.858] | 0.007 [-0.101, 0.136] | 0.797 [0.726, 0.851] | 0.459 [0.362, 0.605] | 0.647 [0.531, 0.763] | 0.724 [0.594, 0.795] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 10 | 20 | 360478 | 0.067 | 0.545 | 0.722 [0.618, 0.787] | 0.760 [0.670, 0.836] | 0.038 [-0.079, 0.173] | 0.764 [0.691, 0.825] | 0.457 [0.217, 0.716] | 0.664 [0.552, 0.768] | 0.682 [0.561, 0.759] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 15 | 20 | 360563 | 0.084 | 0.631 | 0.694 [0.598, 0.766] | 0.742 [0.639, 0.826] | 0.048 [-0.088, 0.190] | 0.749 [0.667, 0.815] | 0.495 [0.239, 0.784] | 0.674 [0.553, 0.778] | 0.664 [0.556, 0.742] |
