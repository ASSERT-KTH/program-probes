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
