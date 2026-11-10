# Candidate analyses (not yet approved for the paper)

These are results for André to review. Nothing here replaces files in `paper_rebuttal/`.

## `transfer/` (brief T-POOL)

Cross-benchmark transfer, scored as one AUC pooled over the target benchmark's task-split test tokens (y ≥ 0).

**How the transfer probe is applied** (the same as the submitted `src/probe.py::run_eval`):
- The target hidden state at layer L is centred on the source probe's training mean, with no std scaling.
- Source layer L is paired with target layer L.

**Checks** (both asserted in every job):
- The token set equals the in-dist rows.
- The binned values reproduce `paper_rebuttal/figures/transfer_table.tex`.

**Excluded tasks:** Pro test tasks with no labelled token (y ≥ 0 nowhere) contribute no tokens to either the in-dist or the transfer AUC. That is 4 of 110 for Laguna and 11 of 105 for Qwen. No Verified test task is excluded.

**Files:**
- `json/transfer_<model>_<dir>_<probe>.json`: per layer, the pooled and binned AUCs, the pooled delta, and 95% CIs (task bootstrap, 1000 resamples, paired). Also the delta between the best test layers.
- `comparison.json`: one row per model, direction and property, at the best layers on test.
- `cells.json`: every model × direction × property × layer cell, binned vs pooled.
- `transfer_table.tex` and `transfer_barplot.pdf`: the candidate artifacts, with every AUC pooled.

## `selectivity/` (request SEL-CI)

k=0 real probe vs the trajectory-shuffle and token-shuffle probes.
- Each is taken at its own best test layer. The values reproduce `paper_rebuttal/figures/auc_table.tex` (asserted).
- CIs are paired task-bootstrap intervals (1000 resamples) for real − traj and real − token.

## `lang/` (brief E11-TEST)

E11 (Pro: language and repository controls) and E11b (Verified: repository controls) rerun with the real probe at its best TEST layer. The pooled AUC reproduces Table 5 (asserted).
- **Transfer (fig C):** the Verified-trained compiles probe at its best pooled transfer layer on the Pro test tokens, from T-POOL: layer 31 for Laguna and layer 1 for Qwen. The pooled transfer AUC must reproduce T-POOL (asserted).
- **`valbest/`:** Qwen/Pro compiles and has_regressions at the validation-best layer, for side-by-side comparison. These are identical to `reports/rerun/lang/`.
- **`figures/`:**
  - The three appendix figures, with the layer in the legend: `e11_compiles_by_language`, `e11_compiles_by_repo`, `e11_compiles_transfer_by_language`.
  - Also `e11b_verified_compiles_by_repo`.

Command: `scripts/rebuttal_per_language.py --layer-rule testbest [--transfer-layer-rule testbest]`, then `scripts/rebuttal_per_language_figures.py --root paper_rebuttal_candidate/lang --out paper_rebuttal_candidate/lang/figures --show-layer`.

## `counts/` (brief E14, reviewer wokJ-Q3)

Population after filtering: tasks, trajectories and hidden states (y ≥ 0) per split. Covers the 4 runs × 3 filters (k = 0, k_max = 15, k_max = 50), plus the legacy (submitted) populations rebuilt from `snapshots/`. No probes are involved.
- **Checks:**
  - The counts equal the stored n_train / n_val / n_test.
  - The windowed test counts equal the lookahead figures' "n = … h_t" labels.
  - The legacy counts equal the stored `*_legacyturns` counts (48 of 48).
- **Splits:** like training, each population is split over its own tasks, so the windowed test sets are different task samples from the k = 0 one.
- **Files:**
  - `json/counts_<run>.json`: per filter and probe, the per-split counts, positive rates and overlap with the k = 0 test tasks. Also trajectory lengths and the legacy populations.
  - `summary.json`: compact tables and the decomposition of the submitted max50 test count.
  - `counts_table.tex`: the candidate appendix table (test split, plus train / val / test states).

Command: `sbatch -A <account> slurm/population_counts.sh --run <run> --output paper_rebuttal_candidate/counts/json/counts_<run>.json`, then `uv run python scripts/paper_rebuttal_candidate_counts.py`.
