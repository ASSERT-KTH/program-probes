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

- **`figures_nostrat/`** (FIG-NOSTRAT): the same figures without the stratified (same-language / same-repo pairs) lines. The dashed line is labelled "AUC over all tokens". The transfer figure has no stratified line and is identical to `figures/`.

Command: `scripts/rebuttal_per_language.py --layer-rule testbest [--transfer-layer-rule testbest]`, then `scripts/rebuttal_per_language_figures.py --root paper_rebuttal_candidate/lang --out paper_rebuttal_candidate/lang/figures --show-layer` (add `--out .../figures_nostrat --no-strat` for FIG-NOSTRAT).

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

## `hparams/` (brief HP)

Probe hyperparameters of the rerun, from the 80 task-split sweeps. Metadata only.
- `json/sweeps.json`: per sweep, the coordinator and worker jobs, the W&B search space, metric and method, the best config, and all 20 trials with their mean validation AUC.
- `json/final_jobs.json`: every completed rerun training/eval job and its hyperparameter source. Fields are in `job_fields`; `by_run_family` gives the counts.
- `json/hparams_per_layer.json`: chosen config per benchmark × model × property × layer (1-indexed).
- `json/layer_variation.json`: spread of each hyperparameter over layers, and how flat the objective is in each sweep (best minus 2nd / 5th / median / worst trial).
- `hparam_table.tex`: candidate `tab:probe-hparams`, in the format of `paper/hparam_table.tex`, at the middle probed layer (21).

Commands: `uv run python scripts/rebuttal_hparam_provenance.py` (uses sacct, the job logs and W&B), then `uv run python scripts/paper_rebuttal_candidate_hparams.py`.

## `after_edit_k0/` (brief AE0; HaQJ-Q3, wokJ-1b, 1nZa-W2)

k = 0 probe AUC on the states right after an edit, before anything has run on the new program version. Eval only: the rerun's k = 0 pooled weights, the k = 0 pooled caches and their task-level test split.
- **States:** for each real edit at command N (from `edit_step_index.pt`, the same selection as the E5 `*_after_edit` runs), the assistant tokens of turn N + 1, the first turn whose label reflects the new version. Their context ends with command N and its output.
- **Filter on command N** (`scripts/after_edit_cmd_filter.py`):
  - *primary* drops the edit if the command also runs anything (interpreter, test runner, build / lint / type-check, package manager, script). It keeps Python scripts that only rewrite files: stdlib file/text imports, and no subprocess, exec, pytest, `ast.parse` or compile, judged on code tokens only.
  - *strict* also drops those editor scripts.
  - *none* is E5's selection. E5 had no command filter.
- **Probe and metric:** each property at its best all-token test layer (`tab:auc_roc`). AUC is pooled over the selected test tokens, with a 95% task-level bootstrap CI (1000 resamples). All five layers are in the JSON.
- **Checks:**
  - The all-token test AUC recomputed with the loaded weights equals the stored one (16 of 16).
  - The unfiltered selection equals the stored E5 after-edit n_test and n_pos at every layer (80 of 80).
- **Files:**
  - `json/ae0_<run>.json`: per property and layer, n edits / states / test tasks / trajectories, positive rate and AUC for each filter. Also the label-changed subset (states whose label differs from the turn before the edit), CIs at the best layer, filter counts with reasons, and example commands.
  - `summary.json`: the 16 cells and the filter counts.
  - `after_edit_table.tex`: candidate `tab:after_edit_k0`.

Commands: `sbatch -A <account> slurm/after_edit_k0.sh <run>` (laguna_verified, laguna_pro, qwen_verified, qwen_pro), then `uvx python@3.12 scripts/paper_rebuttal_candidate_after_edit.py`.

## `prompt_only/` (brief PO; HaQJ-Q2, wokJ-1b)

How much of the probes' signal a predictor that sees only the task prompt can explain. Eval of the rerun probes plus small CPU logistic regressions; no GPU.
- **Cache check** (`cache_check.json`, `scripts/rebuttal_prompt_cache_check.py`): every cached trajectory's first row is in turn 1 (one exception: one Laguna/Verified trajectory in the k=0 cache). That row is the first `content` token of turn 1 (stride 5 starts at the first assistant token).
  - Qwen: the tokens between the end of the task prompt and that token are the 4-token plain-text header `assistant:`. This is a prompt-only state.
  - Laguna: every task has the turn-1 `<think>` block before it. The gap is 25–243 tokens (median 81 on Verified, 36 on Pro), about 25 of them template. So for Laguna this state is **prompt + turn-1 reasoning**, before any action or tool output. That is more than the prompt, so it is a stronger baseline than a pure prompt-only one. A clean Laguna prompt state needs one forward pass per task prompt (not done; GPU).
- **Predictors** (`scripts/rebuttal_prompt_only.py`): one score per trajectory, given to all its tokens.
  - `prompt_hs` (primary): logistic regression on the first cached token's hidden state at the layer of the probe it is compared with. Standardised on the training trajectories.
  - `prompt_tfidf` (secondary): TF-IDF (word 1–2-grams) of the task prompt text (first user message, `scripts/rebuttal_prompt_texts.py`).
  - Training: the probe's split, population and target (train split, y ≥ 0 of that cache). Every token is weighted by one. C is chosen on pooled validation AUC (grid 1e-7..1e2 for `prompt_hs`; at a grid edge in 3 of 88 fits; `prompt_tfidf` picks the upper edge C = 100 in 28 of 88 fits, so it may be slightly under-fit).
- **Horizon** (Table 1 / E16 / E18 setup): window 15, k ∈ {1, 2, 3, 5, 10, 15}, test tokens with y_t = 0, the k-step probe at its validation-best layer. Paired 95% task bootstrap (1000) of probe − prompt-only. The E18 strata are recomputed with prompt-only added to the controls; every pre-existing AUC and count reproduces E16/E18 (asserted: counts exactly, probe and shuffle AUCs within 1e-6, refit position baselines within 1e-3; largest difference seen 7e-6).
- **k=0** (Table 6 setup): all test tokens, the k=0 probe at its test-best layer (recomputed AUC equals results.pt).
- **Files:** `json/horizon_<run>_<probe>.json`, `json/k0_<run>_<probe>.json`, `summary.json` (72 horizon cells, Table 1 recounts, 16 k=0 cells). `json_testgrid/` holds the first test job (Qwen/Verified correct, grid 1e-4..1e2), superseded.

Commands: `sbatch -A <account> slurm/prompt_only.sh {horizon,k0} --run <run> --probe <probe> --output ...`, then `uvx python@3.12 scripts/paper_rebuttal_candidate_prompt_only.py`.

## `text_baseline/` (brief E9; wokJ-1b, wokJ-Q1, 1nZa-W2)

Does a classifier on the visible text, above all the latest tool output, explain what the probes decode? CPU only.
- **Text** (`scripts/rebuttal_text_extract.py`): per trajectory, all messages and a class for every command (AE0's `classify`: `runs_project`, and `strict`), in `tmp/e9_text_<bench>_<run>.jsonl.gz` (not committed; about 0.5 GB).
- **A1** (`text_a1`, the only baseline run): TF-IDF of the most recent tool output before assistant turn t, i.e. the observation of command t − 1; empty at turn 0. Word 1–2-grams; tokens may be one character long, so "1 failed" keeps the "1"; sublinear tf, min_df 3, 200k features; no char n-grams.
  - The vectorizer is fit once per run (`features` mode, `tmp/e9_feat_*`, not committed) on the turns of tasks that are in the training split of both the k=0 cache and the window-15 cache. The two caches have different task splits (each is split over its own task set, as in training), so only 47–66% of trajectories qualify. This keeps every validation and test text out of the vocabulary and idf of either split.
  - Logistic regression per turn: the probe's split, population and target, every token weighted by one, C on pooled validation AUC (grid 1e-3..1e2; never at an edge in the horizon fits).
- **A2 (all context, last 32k characters) was not run.** Building it took 20 minutes and 275M non-zeros for the smallest run, about 10× A1, and fitting it would add roughly 10–15 core-hours for k=0 and far more for the horizon.
- **Evaluation:** k=0 as in PO (test-best layer, all test tokens); horizon as in PO (window 15, y_t = 0, validation-best layer, E18 strata), with the same E16/E18 reproduction checks.
- **Part B** (k=0): test tokens split by whether the current program version has already been run. The version is the one created by the last edit at or before the turn (`edit_step_index.pt`), or the original repository before any edit. "Already run" means some command from the edit command itself through command t − 1 ran tests, scripts, builds or interpreters (AE0's `runs_project`; the `strict` variant also counts Python editor scripts).
  - The `*_post_edit` subsets drop the phase before the first edit (the original repository, also the start of the trajectory), which is almost single-class for correct and reduces_failing. `pre_first_edit` is that phase.
- **Files:** `json/horizon_*.json`, `json/k0_*.json` (with part B), `summary.json` (also the Table 1 recount with PO, E9 and both). `json_partb_v1/` holds the first k=0 run, before the post-edit subsets were added; its other numbers are identical (checked).

Commands: `sbatch ... slurm/text_baseline.sh features --run <run> --kinds text_a1` per run, then `sbatch ... slurm/text_baseline.sh {horizon,k0} --run <run> --probe <probe> --output ...`, then `uvx python@3.12 scripts/paper_rebuttal_candidate_text_baseline.py`.

## `fig3/` (brief FIG3)

`auc_summary_barplot.pdf` redrawn (`scripts/paper_rebuttal_figures.py --fig3 paper_rebuttal_candidate/fig3`, compute node) with the token shuffle as a black-filled circle (same size as the trajectory-shuffle circle, zorder 6) instead of a black tick; legend handle to match. Same data, size, colours and legend order as `paper_rebuttal/figures/figures/auc_summary_barplot.pdf`. Where token and trajectory shuffle are within 0.01 AUC, the token circle would be drawn smaller (`overlaps.json` lists those bars): none are. The closest pair, Laguna-XS.2 (Pro) regression, is about 0.012 apart (circles touch, both visible).

## `edit_indexed/` (brief EI; HaQJ-W1c, HaQJ-Q3)

Edit-indexed horizon: from the hidden state at turn t, predict the label after the m-th future edit (m = 1, 2, 3; edit written in a turn > t). CPU only.
- **Edits and targets:** `edit_step_index.pt` (AE0's real edits). Outcome of edit e = turn label of turn N_e + 1 (`outputs/<bench>/<run>/<sid>_labels.pt`). A token whose e_m has an undefined outcome is excluded for that m (e_m is not renumbered). Cache labels were checked against the turn labels (0 mismatches).
- **Tokens:** pooled k = 0 cache, its task split (Table 6 test tasks), y_t ≥ 0, no window exclusion.
- **Probe:** linear, trained as `src/probe.py` (pooled, mean-centred, Adam, CE, early stop on validation loss), only at the k = 0 probe's validation-best layer (E17 `val_best_layer`), with that layer's k = 0 sweep hyperparameters; weight decay chosen on validation AUC from {k = 0 value, 1e-2, 1e-1}.
- **Baselines** (fit on train tokens with the same target): persistence (A only), k = 0 probe (Table 6 layer), `pos_causal` and `pos_rel` (E16 splines), `edits_so_far`, `observable` (step, edits so far, turns since the last edit), `traj` (trajectory shuffle at the selected layer and weight decay), `prompt_hs` (PO, refit), `yT` (oracle).
- **Populations:** A all tokens with Target_m; B y_t = 0; C y_t = 0 and y_T = 1. Paired task bootstrap, 1000 resamples.
- **Files:** `json/ei_<run>_<property>.json`, `summary.json` (headline counts per m and population, and every cell). `json_timing/` holds the cost-estimate run (Qwen/Verified correct, layer 21, k = 0 weight decay only), not used in the summary.

Commands: `sbatch ... slurm/edit_indexed.sh --run <run> --probe <property> --k0-val-layer --output ...`, then `uvx python@3.12 scripts/paper_rebuttal_candidate_edit_indexed.py`.

## `table1_k1/` (brief T1K1)

Table 1 at k = 1, window 15, strata S0 (y_t = 0) and S2 (y_t = 0 and y_T = 1), 12 settings (4 runs x correct, reduces_failing, has_regressions). E18 and PO stopped S1-S4 at k >= 2. Same setup as E18/PO: k = 1 probe at its validation-best layer, controls k0, traj, pos_causal, pos_rel, PO's prompt_hs and prompt_tfidf (refit on y_{t+1} at that layer), oracle_yT (defined where y_T >= 0), and the paired task bootstrap (1000, seed 42). Each file has one row in the PO/E18 format (`auc`, `ci95` with `real_val-<control>|S0/S2`, `counts`). `real_val-oracle_yT|<stratum>` is an extra key; E18 has no such paired difference. Check (asserted, passed in all 12): every S0 AUC, CI and count equals PO's k = 1 row (max abs diff 0), and the pre-existing scorers match E16.
- **Files:** `json/k1_<run>_<property>.json`.

Command: `sbatch ... slurm/table1_k1.sh --run <run> --probe <property> --output paper_rebuttal_candidate/table1_k1/json/k1_<run>_<property>.json` (`scripts/rebuttal_table1_k1.py`; about 3 min per job on 2 cores).
