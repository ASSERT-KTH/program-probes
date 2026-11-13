# Rebuttal rerun: real agent turns, task-level splits, persistence analysis

Branch `rebuttal-rerun-turnfix` (PR #20). Runs completed 2026-10-01; persistence-baseline check added 2026-11-09.

## Summary

- **Current-state probes (k=0) hold up**, and most AUCs rise slightly under the new task-level split: best-layer test AUC 0.69–0.84 across models, benchmarks and probes (see table below).
- **Most of the lookahead ("programming horizon") signal is label persistence.** A signal beyond persistence appears for some probes in the window-50 subset but, except for Qwen/Pro has_regressions, does not replicate at window 15. Labels rarely change at short horizons:

  | | flip rate, k ≤ 3 | flip rate, k = 10 | flip rate, k = 50 | persistence AUC, k = 1 |
  |---|---|---|---|---|
  | Verified | 0.3–5.6% | 1.4–14% | 3–42% | 0.85–0.93 |
  | Pro | 0.2–3.3% | 0.3–6.3% | 1.3–13% | 0.65–0.98 |

  Probe AUC minus persistence AUC (same tokens, 95% task-bootstrap CI). **The answer depends on the window, and the window-50 advantages on Pro do not replicate at window 15** (see "Persistence-baseline check" below):

  **Window 15** (tokens with at least 15 later turns):
  - **Probe clearly better:** only Qwen/Pro has_regressions, at every k (+0.13 to +0.24); layer-0 caveat below.
  - **Probe clearly worse:**
    - Laguna/Verified, all four probes at nearly every k (−0.13 to −0.29);
    - compiles on Pro for both models at every k (−0.13 to −0.18);
    - Laguna/Pro correct, has_regressions and reduces_failing at short k (−0.12 to −0.15 at k = 1);
    - Qwen/Verified reduces_failing at k = 1 (−0.08), and correct at k = 1 (−0.13, CI upper bound −0.00).
  - **Everything else:** the CI includes 0.

  **Window 50** (tokens with at least 50 later turns):
  - **Probe clearly better (CI above 0):**
    - Laguna/Verified compiles, k ≥ 5 (+0.08 to +0.18);
    - Qwen/Verified reduces_failing, k = 5–25 (+0.08 to +0.11);
    - Laguna/Pro correct, k = 5–15 (+0.14 to +0.17), has_regressions, k = 3–5 (+0.12), and reduces_failing, k = 3–50 (+0.11 to +0.16);
    - Qwen/Pro reduces_failing, k = 5–50 (+0.19 to +0.23), and has_regressions at every k (+0.22 to +0.45). The has_regressions probe has the layer-0 caveat below, and positives are rare there.
  - **Probe clearly worse:**
    - compiles on Pro for both models at every k (−0.19 to −0.29);
    - Laguna/Verified has_regressions, k = 1–5 (−0.10 to −0.13), and correct at k = 1 (−0.17).
  - **Everything else:** the CI includes 0; the probe is indistinguishable from "label stays the same".
  - On Pro, window 50 leaves very few positive turns: Laguna/Pro correct has 33 positive test turns (559 tokens) at k = 1. Its persistence AUC is therefore low and noisy (0.739, CI [0.556, 0.861]) and the window-50 Pro advantages should not be claimed on their own.
  - **After-edit targets:** best-layer AUC for k ≥ 1 has a median of 0.62 (range 0.40–0.86), with 14 of 128 cells below 0.5. Turns right after an edit are where labels change, and there the probe anticipates weakly at best.
- **Phase/position confound (E2, E16, E17; rerun, layers chosen on validation).** Within tokens whose current label is 0 (y_t = 0), where persistence has no information, the k-trained probe beats every control at window 15. CIs exclude 0 for the k=0 probe in 72/72 cells, the trajectory-shuffle probe in 70/72, a causal position baseline (turn index, token index) in 62/72, and a non-causal `rel_pos` baseline in 66/72. Over all tokens, the current-state (k=0) probe used as a forecast is nearly as good: the k-probe is ahead in only 33/72 cells, by at most 0.13. At k=0, trajectory phase explains much of the AUC but not all of it: real − trajectory shuffle is +0.05 to +0.32 (CI > 0 in 15 of 16 probe × run cells; not Qwen/Verified compiles), and `rel_pos` alone reaches 0.50–0.78. See "Phase-confound analyses".
- **Edit-turn and outcome confounds (E18; rerun).** Within y_t = 0, the eventual outcome alone (the last-turn label y_T used as a score) reaches 0.68–0.96 AUC for y_{t+k}, above the probe in 57/60 window-15 cells. Holding the outcome fixed (y_T = 1) and excluding the turn that writes the label-changing edit (S3), the probe beats the strongest control in only 6 of 35 adequately powered cells (all reduces_failing on Verified) and the strongest causal control in 19/35; it is never significantly worse. Most of the within-y_t=0 signal is therefore *whether* the trajectory ends up passing plus the edit turn itself, with a small *when* component. See "Edit-turn and outcome confounds (E18)".
- **Pro well-formedness is largely language/repository identity (E11; rerun).** On Pro, scoring each token by its language's training positive rate matches the compiles probe (Laguna 0.778 vs 0.786; Qwen 0.756 vs 0.720), and the repository rate beats it (0.833, 0.819). Counting only same-language pairs, the probe drops to 0.646 [0.584, 0.700] (Laguna) and 0.578 [0.512, 0.635] (Qwen). The Verified-trained probe is at chance on every Pro language, Python included (0.47–0.48), so the transfer drop is not explained by language. Full correctness is unaffected by language (stratified 0.785/0.811 ≈ pooled). See "Per-language and per-repo well-formedness on Pro (E11)".
- **The trajectory-level shuffle control is strong.** A probe trained on another trajectory's labels, which keep the trajectory-position structure but have no link to the hidden states, reaches **0.56–0.78 test AUC** for correct, has_regressions and reduces_failing (up to 0.68 for compiles on Verified). Much of the reported AUC is therefore available without reading program state from the hidden states.
- **The paper's shuffled control was uninformative:** it shuffled the test labels as well, which pins AUC at ~0.5 by construction (fixed in `fa41d05`).
- Several pipeline bugs were found and fixed along the way; see "Problems found and fixed".

## What changed relative to the paper

| | Paper | This rerun |
|---|---|---|
| Turn index per position | `int(i*K/T)` (even split) | real turn (`step_idx`, `turn_labels` from `run_attach_labels_swebench.py`) |
| Train/val/test unit | `--group-by project`: **repository** on Verified (12 groups; val = 1 repo), **base commit** on Pro (434 groups for 726 tasks) | **task** on both (`--group-by` removed, `ff6b081`); 70/15/15 of tasks |
| Hyperparameters | per-(probe, layer) W&B sweeps on the old split | new per-(probe, layer) sweeps on the task split (20 random-search trials each, run as 4 agents × 5), reused for every k and control |
| Lookahead protocol | Verified and Laguna Pro: train for each k; **Qwen Pro: k=0 weights evaluated at every k**; Laguna Pro after-edit used k=0 weights | all four: train for each k; after-edit = eval of that k's weights on after-edit targets |
| Shuffled control | token shuffle of train **and test** labels | token and trajectory shuffles of train/val only, scored on real test labels |

Because the split, the hyperparameters and the turn indices all changed together, the new-vs-legacy columns below do not isolate the turn fix. A first pass with only the turn fix (repository split kept) was run and then discarded by request when the split was changed.

## Submitted (legacy) run flags (recovered from `sacct` submit lines and W&B configs)

Everything in this section describes the submitted runs (`*_legacyturns`). The rerun keeps `--probe-arch linear --n-bins 1`, seed 42 and `final` training, with the changes in the table above.

- All probes: `--probe-arch linear --n-bins 1`; seed 42; linear probe; `final` training.
- Caches: built 06-09 to 06-14 with the `--group-by` default `project` (default since `91522df`, 06-02); Laguna Pro lookahead caches also passed it explicitly.
- Hyperparameters: Verified (both models) lookahead and controls used `final --from-sweep <S>` with the k=0 sweep for each (probe, layer); Laguna Pro lookahead used typed-in per-(probe, layer) values (constant across k); Qwen Pro lookahead was eval-only with the k=0 weights; Qwen Pro shuffled used typed-in values and `--n-eval-bins 1`.
- After-edit runs: always eval-only (`eval --after-edit-only --edit-index-run-id <run>`), never retrained. The rerun does the same.
- Sweeps: `slurm/probe_sweep_coordinator.sh ... --n-agents 4 --count 20`, i.e. 20 trials per sweep in total (`--count` is split across the agents). A few June sweeps ran 24–30 trials.
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

In the submitted runs (paper's split), Verified validation was a single repository (flask, 1.8k–3.6k tokens). In the window-50 lookahead runs it was single-class (val AUC 0.5), so early stopping had no signal there.

## Problems found and fixed

1. **Shuffled-label control shuffled the test labels** (`src/probe.py`, since `3d7b934`). Both modes now shuffle train/val only. Fix and regression test in `fa41d05` (rerun only; the submitted runs used the buggy control), part of PR #20.
2. **Race in `run_final`'s results/weights merge** (since `47f4a37`). Concurrent per-layer jobs could read a half-written file or silently drop another layer. Fixed with an `fcntl` lock and atomic writes, plus a multi-process test (`05b8f24`). Two layers were lost during the rerun's turn-fix pass and repaired. Final audit: every result file has all 5 layers, legacy results included (the paper's numbers were not affected).
3. **Inconsistent split units** (repository vs base commit); now task on both (`ff6b081`).
4. **Laguna Verified after-edit lookahead in the paper is undefined**: under the old approximate turns, every evaluation bin selected by the after-edit filter held a single class. Shown as n/a‡ in the legacy columns.
5. **Qwen Pro has_regressions has very few positives** in the window-50 test set (legacy: 29 positives; flagged †).
6. **Deferred (note for a future rebuttal): the Qwen activations were extracted from text without the chat template or reasoning.** This affects the submitted runs and the rerun alike, and was not fixed by decision. `src/agents/trajectory.py::_tokenize_trajectory` catches any exception from `apply_chat_template` and falls back to `"\n".join(f"{role}: {content}")`. Every Qwen trajectory was tokenized this way (4,998/4,998 Verified, 5,804/5,804 Pro; 0 of the Laguna trajectories), and `run_extract_swebench.py` runs the model on these stored `token_ids`. So the Qwen hidden states come from re-encoding the trajectory as plain `system: … user: … assistant: …` text.
   - **Reasoning.** Leaving the reasoning out is defensible as a design choice: the probe then sees only actions and observations, not a written-out plan. It has to be stated as such. Qwen/Pro trajectories store `reasoning_content`; Qwen/Verified ones do not at the top level, because that run (2026-05-21) predates `93b950d` (2026-05-27), which started saving it. The reasoning is still in each message's `extra.response`. Thinking was enabled for both runs (same config, `--reasoning-parser qwen3`).
   - **Chat template.** This is the part that was not a choice: Qwen never saw its transcripts in this plain `role:` format. The effect is unmeasured.
   - **Asymmetry with Laguna.** Laguna's extraction uses its chat template and has its `<think>` reasoning in context for every turn (1,075/1,075 assistant messages in 20 sampled trajectories). Laguna-vs-Qwen differences are therefore not like-for-like.
   - **Likely cause (not confirmed by running it).** Every trajectory ends with a `{"role": "exit"}` message, and Qwen's `chat_template.jinja` raises `'Unexpected message role.'` for roles other than system, user, assistant and tool. Two further points if re-extracting with the template:
     - The template applies `|trim` to content. `_build_segments` matches the untrimmed `content` with `text.find` and silently skips misses, so most Qwen contents (which start with `\n\n`) would lose their segment unless matching is fixed.
     - The template keeps `<think>` only for assistant turns after the last non-`<tool_response>` user message, and mini-swe-agent observations are plain user messages. So the default rendering drops all past reasoning; `preserve_thinking=True` keeps it.
   - **Options and cost estimates** (from `sacct`).
     - Disclose in the methods: no compute.
     - Subset check, re-extracting the test split plus about a third of tasks with the template and comparing k=0 probes on the same tasks under both renderings: about 100 A100 GPU-hours.
     - Full Qwen re-extraction: the original runs used about 111 GPU-hours (Verified) and 101 (Pro). The downstream rerun of the Qwen jobs used about 3,100 CPU core-hours, including the discarded repository-split pass.
   - A diagnostic script, `scripts/rebuttal_qwen_template_check.py`, is in the working tree. It is uncommitted and has not completed a run.

## Persistence-baseline check

A review comment asked whether the low persistence AUC on Pro at k = 1 (0.74 for Laguna/Pro correct, 0.75 for Qwen/Pro reduces_failing, against 0.93 on Verified) means the two compared labels disagree. `y_original` is the per-position `label[t]`; the target is `turn_label[step_idx[t] + k]` (`src/build_cache.py`).

**No bug found.**
- `label[t] == turn_label[step_idx[t]]` holds for all 24.5M extracted positions in every label file, for all 4 runs × 4 probes (0 mismatches, 0 length mismatches). Both come from `edit_labels_by_turn` over the same edits.
- In every window-50 shift cache (k = 1…50), `y_original` and `y` equal the k = 0 cache's label for the source and target turn wherever that turn has extracted tokens: 0 mismatches over 153M lookups (38M per probe). `target_step_idx − step_idx = k`, the token set equals k = 0's, labels are constant within a turn, and `group_id` is the task id.
- Labels do **not** flip more on Pro: the fraction of trajectories with any change in the correct label is 12–15% on Pro against 59% on Verified. Reverts (0→1→0) are not more common on Pro.

**Why the token-level persistence AUC is low.** With hard 0/1 predictions, persistence AUC is balanced accuracy, (TPR + TNR) / 2. Almost every flip is 0→1, so TNR ≈ 1 and the AUC is set by TPR, i.e. by the number of 0→1 flip tokens relative to positive tokens. Two things make that ratio unfavourable:
1. **Few positives in the window.** Window 50 keeps a token only if 50 later turns exist, which removes most turns after a task is solved. Laguna/Pro correct keeps 33 positive turns (559 tokens) out of 11,369 test turns at k = 1, against 537 tokens that flip 0→1: TPR 0.478, AUC 0.739.
2. **Token weighting.** The turn before a flip is the turn that writes the fix; flipped turns hold about 30–100 tokens against 8–21 for an average turn.

Counted per turn, persistence at k = 1 is 0.88–1.00 at window 50 and 0.95–1.00 at window 15 everywhere (table "Persistence at k=1 … tokens vs turns" below).

**Consequence for the claims.** With so few positives, window-50 persistence on Pro is noisy, and the probe's apparent advantage there (Laguna/Pro correct, has_regressions, reduces_failing; Qwen/Pro reduces_failing) does not replicate at window 15, where persistence wins at short k or the CI includes 0. Only Qwen/Pro has_regressions beats persistence in both windows.

The fraction of test tokens with y_{t+k} ≠ y_t for each k, and the fraction of trajectories with at least one flip, are in the table "Fraction of test tokens with y_{t+k} ≠ y_t" below. Scripts: `scripts/rebuttal_check_persistence_labels.py` (`slurm/check_persistence_labels.sh`) and `scripts/rebuttal_persistence_turn_level.py` (`slurm/persistence_turn_level.sh`); outputs in `reports/rerun/persistence_label_check_*.json` and `reports/rerun/persistence_turn_level_*.json`.

## Phase-confound analyses (E2, E16, E17)

All rerun. Script `scripts/rebuttal_phase_confound.py` (`slurm/phase_confound.sh`, submitted by `scripts/rebuttal_submit_phase_confound.py`); outputs in `reports/rerun/phase/*.json`; tables at the end of this report. Test tokens are those of `rebuttal_persistence_analysis.py` (test split, y_{t+k} ≥ 0; strata need y_t ≥ 0). CIs: 1,000 task-level bootstrap resamples; differences are paired (same resample for both terms).

**Layer rule.** Every new number uses the layer with the best **validation** AUC. The real probe is also shown at its test-best layer, to line up with the earlier tables; the two layers coincide in 108 of 168 lookahead cells.

**Scorers for y_{t+k}** (lookahead, windows 15 and 50, probes correct / reduces_failing / has_regressions):
- *real*: the probe trained on y_{t+k}.
- *k=0 probe* (E2): the current-state probe of the same window (`<run>_shift0_max{K}`) at its own validation-best layer, used unchanged as a forecast of y_{t+k}.
- *traj-shuffle* (E16 ii): a probe trained exactly like the real one (same cache, same sweep hyperparameters, same layer) but on trajectory-shuffled labels (`--shuffle-labels --shuffle-unit trajectory`): each training/validation trajectory gets another trajectory's y_{t+k} sequence, resampled by relative position. It can learn only what the hidden states say about position or phase. 168 new runs, `<run>_shift{k}_max{K}_shuffled_traj`, trained at the real probe's validation-best layer (`scripts/rebuttal_submit_traj_shuffle_lookahead.py`, which copies each real probe's own submit line).
- *position, causal* (E16 iii a): logistic regression on cubic-spline features of log(1 + turn index) and log(1 + token index within the trajectory), fit on training tokens with target y_{t+k}. It uses only information available at time t.
- *position, `rel_pos`* (E16 iii b): the same on `rel_pos` = token index / trajectory length. **Non-causal** (it uses the final length), so a deliberately strong upper bound.

**Results, window 15** (72 cells = 4 runs × 3 probes × 6 k):

| stratum | real − k=0 probe | real − traj-shuffle | real − position causal | real − position `rel_pos` |
|---|---|---|---|---|
| y_t = 0 | CI > 0 in 72/72 (+0.06 to +0.26) | 70/72 (−0.04 to +0.30) | 62/72 (+0.02 to +0.33) | 66/72 (+0.05 to +0.36) |
| all tokens | 33/72 (−0.02 to +0.13) | 66/72 (−0.01 to +0.27) | 58/72 (+0.01 to +0.23) | 58/72 (+0.03 to +0.22) |
| y_t = 1 | 0/72; **CI < 0 in 38/72** (−0.21 to +0.09) | 9/72 > 0, 10/72 < 0 | 5/72 > 0, 8/72 < 0 | 5/72 > 0, 8/72 < 0 |

No difference has a CI entirely below 0 except within y_t = 1. Window 50 points the same way, with more noise (y_t = 0: CI > 0 in 68/96 vs the k=0 probe, 76/96 vs traj-shuffle, 60/96 vs position causal, 71/96 vs `rel_pos`).

Reading:
- **Within y_t = 0 the signal is not phase or position.** Real AUC is 0.56–0.87. The traj-shuffle probe and the position baselines give 0.32–0.77 there (medians 0.52–0.54). Cells where a control is not significantly below the real probe: traj-shuffle for Qwen/Verified has_regressions at k = 10, 15; causal position for Laguna/Verified correct (k = 10, 15) and reduces_failing (k = 15), Qwen/Verified correct (k = 15), Laguna/Pro correct (k = 15) and Qwen/Pro has_regressions (k = 1–10); `rel_pos` for Qwen/Verified correct and reduces_failing (k = 15) and has_regressions (k = 2–10).
- **The current-state probe is a strong overall forecast** (E2): over all tokens the k-trained probe adds at most 0.13, and its gain is not significant in 39/72 cells. Its gain is concentrated in y_t = 0. Within y_t = 1 the k=0 probe is better (38/72 cells), i.e. its score tells which currently-positive states will stay positive better than the k-probe does. This stratum is underpowered: its events are 1→0 changes, which are few at window 15 (Qwen: 155–2,960 tokens per cell, e.g. 155–840 for Qwen/Pro has_regressions; Laguna: 1,211–18,127), so its AUCs rest on few trajectories.
- **Within y_t = 0 on Laguna/Verified, the k=0 probe is below 0.5** (0.40–0.44 for correct and reduces_failing): among tokens that are currently failing, a higher current-state score goes with *not* becoming correct within k turns.
- **Remaining confound.** The traj-shuffle probe and the position baselines control for where in the trajectory a token is, not for which trajectory it is in. Within y_t = 0, a probe that recognises trajectories that will eventually succeed (e.g. easier tasks) scores above 0.5 without any turn-specific foresight. These analyses do not separate the two; E18 below does.

**k = 0 selectivity (E17)**, pooled k=0 caches, all test tokens, 4 runs × 4 probes. Each probe's layer is chosen on validation AUC against the **real** validation labels, which gives each control its best layer (the shuffled runs' stored `val_auc` is against shuffled labels).
- Token shuffle: 0.49–0.55; real − token shuffle +0.17 to +0.33, CI > 0 everywhere.
- Trajectory shuffle: 0.47–0.78; real − traj shuffle +0.05 to +0.32, CI > 0 in 15/16 (not Qwen/Verified compiles, +0.05 [−0.03, +0.14]). At the real probe's layer the gap is +0.05 to +0.32, CI > 0 in 16/16. The smallest gaps are on Qwen: correct +0.06 (Verified) and +0.05 (Pro), reduces_failing +0.05 and +0.07.
- Position baselines at k = 0: causal 0.52–0.76, `rel_pos` 0.50–0.78. Real − `rel_pos` is +0.05 to +0.28 (CI > 0 in 15/16); real − causal is +0.03 to +0.24 (CI > 0 in 11/16; not Laguna/Verified has_regressions and compiles, Qwen/Verified has_regressions and compiles, Laguna/Pro has_regressions).
- So at k = 0 a large part of the AUC (Qwen correct: about 0.77 of 0.82–0.84) is available from trajectory phase alone, and the hidden-state-specific part is 0.05–0.15 for correct and reduces_failing.

## Edit-turn and outcome confounds (E18)

All rerun, eval-only. Script `scripts/rebuttal_edit_outcome_strata.py` (`slurm/edit_outcome_strata.sh`, submitted by `scripts/rebuttal_submit_phase_confound.py --modes e18 e18_outcome`); outputs `reports/rerun/phase/e18_*.json`; tables at the end of this report. It reuses `rebuttal_phase_confound.py` unchanged: same test tokens, the same probes at the same (validation-best) layers, the same position baselines, the same 1,000-task bootstrap with paired differences. S0 reproduces every E16 within-y_t=0 AUC exactly (checked in the script). The token sets of all shift caches of a window are checked to be identical (sample ids, turn indices).

**What a turn is.** The extracted positions of assistant turn t are the tokens of that assistant message's visible `content`, i.e. the action (here a `mswea_bash_command` block, sometimes with a line of text). Segment boundaries come from locating `content` in the rendered text (`src/agents/trajectory.py`), and only assistant segments are extracted (`src/tasks/swe_bench_extract.py`). Not extracted, but in the context the hidden states attend to:
- the observation of action t−1 (the user message with `<returncode>` and `<output>`), which comes right before turn t;
- for Laguna, the turn's own reasoning (`<think>…</think>`), which the chat template puts before `content`.

An edit with `cmd_idx` N is the command in turn N's content; its effect is in the labels from turn N+1 on. So at k = 1 within y_t = 0 the positives are exactly the tokens of the command that makes the label-changing edit.

**Strata, all within y_t = 0** (y_t = `y_original`):
- S0: all (= E16).
- S1, pre-edit: y_{t+1} = 0 (shift-1 label of the same token). This removes the tokens of the turn whose command flips the label; the token can still see the previous observation and, for Laguna, its own reasoning.
- S2, outcome: y_T = 1, where y_T is the label at the trajectory's last assistant turn (`turn_labels[probe][-1]`).
- S3 = S1 ∩ S2: fixed eventual outcome, label-changing edit not in the current turn. An AUC above the controls here means the probe knows *when*, not just *whether*.
- S4: y_T = 0 (positives are later changes that revert before the end; few).
- Oracle: y_T itself as a score for y_{t+k} (S0, S1; constant in S2–S4). It bounds how much eventual outcome alone could explain.

A cell is underpowered (◦) if fewer than 20 test tasks have a positive token in it.

**Results, window 15** (60 cells per stratum = 4 runs × 3 probes × k ∈ {2, 3, 5, 10, 15}; counts over adequately powered cells; "strongest" = highest point AUC among the k=0 probe, traj-shuffle, causal position and `rel_pos`; "strongest causal" drops the non-causal `rel_pos`):

| stratum | powered cells | real AUC | real − strongest: CI > 0 / CI < 0 | real − strongest causal: CI > 0 / CI < 0 |
|---|---|---|---|---|
| S1 pre-edit | 49 | 0.52–0.74 | 31 / 0 | 35 / 0 |
| S2 y_T = 1 | 35 | 0.59–0.82 | 15 / 0 | 24 / 0 |
| S3 pre-edit, y_T = 1 | 35 | 0.55–0.71 | 6 / 0 | 19 / 0 |
| S4 y_T = 0 | 5 | 0.54–0.65 | 4 / 0 | 4 / 0 |

- **Eventual outcome explains more than the probe achieves.** The oracle y_T reaches 0.68–0.96 within y_t = 0 and beats the probe in 57/60 cells (probe 0.56–0.85).
- **The edit turn matters.** Removing it (S1) lowers the probe's AUC, e.g. Laguna/Verified correct 0.65 → 0.54 at k = 2, and the advantage over controls shrinks, but it stays significant in 31/49 cells.
- **With the outcome fixed and the edit turn removed (S3), little "when" signal remains.** The probe beats every control in 6/35 cells: reduces_failing on Laguna/Verified (k = 3, 5, 15) and Qwen/Verified (k = 2, 3, 5). Against causal controls only it wins 19/35. The strongest control in S3 is usually `rel_pos`, which uses the trajectory's final length.
- Pro is the least powered: Pro has_regressions never reaches 20 tasks with a positive token in S2/S3 (2–12), and Pro correct only barely (17–25); Qwen/Pro correct already has only 19 such tasks in S0 at k = 2.
- Window 50 is mostly underpowered (S3: 16/84 cells powered) and points the same way (S3: 5 cells CI > 0, 1 cell CI < 0).

**(b) y_T versus the final-patch evaluation** (`correct`, every trajectory of each run). y_T is undefined for trajectories without a labelled edit (316–913 per run, almost all unresolved). Where it is defined it agrees with the final-patch evaluation (`metadata.outcome`) in 97.3–99.4% of trajectories (Laguna/Verified 97.7%, Qwen/Verified 97.3%, Laguna/Pro 99.4%, Qwen/Pro 98.8%). The label of the last edit, which also counts an edit issued in the last turn (50–375 trajectories per run), agrees in 97.4–99.6%. The residual disagreement is between the labeller's per-edit test run and the final evaluation.

## Per-language and per-repo well-formedness on Pro (E11)

All rerun, eval-only. Script `scripts/rebuttal_per_language.py` (`slurm/per_language.sh`), figures `scripts/rebuttal_per_language_figures.py`; outputs `reports/rerun/lang/*.json`, figures `reports/rerun/figures/e11_*.{pdf,png}`; tables at the end of this report.

**Setup.**
- **Probe and layer:** the k=0 probe of the rerun (task split) at its validation-best layer, chosen against real validation labels as in E17 (Laguna/Pro compiles L10, Qwen/Pro compiles L39).
- **Test tokens and AUC:** E17's. Every test token with y ≥ 0, one pooled AUC, no position bins; the script checks that the pooled AUC equals E17's.
- **Language and repository:** `repo_language` (go, python, js, ts) and `repo` of `ScaleAI/SWE-bench_Pro`, joined on the instance id.
- **CIs:** 1,000 task-level bootstrap resamples, paired differences.
- **Stratified AUC:** counts only positive/negative pairs from the same language (resp. repository).
- **Language-only (repo-only) baseline:** scores each test token by the training-split positive rate of its language (repo). Every test repository occurs in training, so the language fallback was never needed.
- **Power:** a cell is underpowered if fewer than 20 test tasks have both classes. Every per-language and per-repository compiles cell is underpowered (at most 19), so the stratified AUC is the main number.

**Results for well-formedness (compiles).**

| | Laguna/Pro | Qwen/Pro |
|---|---|---|
| pooled AUC | 0.786 [0.725, 0.843] | 0.720 [0.651, 0.788] |
| same-language pairs only | 0.646 [0.584, 0.700] | 0.578 [0.512, 0.635] |
| same-repo pairs only | 0.649 [0.569, 0.714] | 0.568 [0.486, 0.632] |
| language-only baseline | 0.778 [0.688, 0.856] | 0.756 [0.656, 0.854] |
| repo-only baseline | 0.833 [0.757, 0.897] | 0.819 [0.737, 0.891] |
| real − language-only | +0.008 [−0.04, +0.06] | −0.037 [−0.10, +0.02] |
| pooled − same-language | +0.140 [+0.07, +0.21] | +0.141 [+0.07, +0.21] |
| Verified-trained probe on Pro (pooled; same-language) | 0.516; 0.459 | 0.505; 0.480 |

- **The reviewer's concern holds for Pro well-formedness.**
  - Positive rates differ sharply by language: training rate 0.24–0.28 for go and 0.74–0.78 for python and js; test-set rates per repository range from 0.00 to 1.00.
  - The pooled AUC is no better than a score that only knows the language, and worse than one that only knows the repository.
  - Within a language the probe still separates the classes (same-language AUC above 0.5 for both models), but at 0.58–0.65.
- **Transfer.** The Verified-trained probe (Python only, at its Verified validation-best layer, with its Verified training mean) is at chance on Pro in every language, Python included (Laguna 0.47 [0.38, 0.61], Qwen 0.48 [0.43, 0.62]). So the Verified→Pro drop is not a Python-vs-other-language effect: the Verified probe does not carry over even to Pro's Python repositories. These are pooled token AUCs; the paper's transfer numbers average position bins.
- **Other properties** (JSON and tables only):
  - Full correctness does not depend on language: same-language AUC 0.785 (Laguna) and 0.811 (Qwen) against pooled 0.794 and 0.822, and the language-only baseline is near 0.5.
  - Reduces_failing loses 0.04–0.08 when stratified, and the probe beats the language-only baseline.
  - Has_regressions behaves like compiles: the language-only baseline matches or beats the probe. On Qwen/Pro all has_regressions positives are Python tokens, which likely explains why its best layer is layer 0 (see "Layer 0 on Qwen Pro").

### E11b: repository controls on Verified

All rerun, eval-only, the same script and setup as E11 with `--bench swebench`.
- **Repository:** the `repo` field of `princeton-nlp/SWE-bench_Verified`; Verified is Python-only, so only the repository controls apply.
- **Outputs:** `reports/rerun/lang/lang_swebench_*.json`; figure `reports/rerun/figures/e11b_verified_compiles_by_repo.{pdf,png}`. No figure is drawn for the other properties, which show no repository effect.
- **Coverage:** all 11 test repositories occur in training (flask has no test task), so no fallback was needed.
- **Power:** no repository has 20 test tasks with both classes, for any property.

**Legacy context.** The submitted runs' repository split used seed 42 over 12 repositories:
- train: matplotlib, seaborn, requests, xarray, pylint, pytest, scikit-learn, sympy;
- validation: flask (1 task);
- test: astropy (22 tasks), django (231) and sphinx (44).

Their stored pooled test AUC (same AUC definition) therefore measures transfer to unseen repositories on a different, django-dominated test set. It is shown next to the task-split numbers, not as a paired comparison.

| | pooled (task split) | same-repo pairs | repo-only baseline | real − repo-only | legacy split (test-best layer) |
|---|---|---|---|---|---|
| Laguna compiles | 0.691 | 0.709 | 0.692 | −0.001 [−0.14, +0.16] | 0.571 |
| Qwen compiles | 0.736 | 0.733 | 0.785 | −0.050 [−0.19, +0.14] | 0.587 |
| Laguna correct / reduces_failing / has_regressions | 0.788 / 0.773 / 0.695 | 0.752 / 0.765 / 0.694 | 0.574 / 0.450 / 0.507 | +0.22 / +0.32 / +0.19 (all CI > 0) | 0.727 / 0.708 / 0.678 |
| Qwen correct / reduces_failing / has_regressions | 0.838 / 0.832 / 0.770 | 0.833 / 0.830 / 0.759 | 0.565 / 0.448 / 0.521 | +0.27 / +0.38 / +0.25 (all CI > 0) | 0.828 / 0.808 / 0.721 |

- **Well-formedness.**
  - Training positive rates differ by repository: lowest for pylint (0.41 Laguna / 0.64 Qwen), 0.74–1.00 for the others. So a score that knows only the repository already reaches the probe's pooled AUC (0.69/0.79).
  - Unlike on Pro, the probe does not depend on that. Its AUC over same-repository pairs equals its pooled AUC (pooled − same-repo −0.02 and +0.00, CIs around 0).
  - So repository base rates do not explain the task-split number; the probe ranks well within repositories.
  - The jump from the legacy split (+0.12/+0.15) is the largest of the four properties. The task split trains on tasks from the test repositories (django among them), which the legacy split never did. These controls do not separate "repository-specific features learned in training" from "a generally better probe": the within-repository AUC rules out base rates only.
- **Other properties.**
  - The probe beats the repo-only baseline by 0.19–0.38 (CIs > 0), and stratifying by repository changes the AUC by at most 0.04.
  - The legacy-to-task-split change is smaller (+0.01 to +0.07; Qwen has_regressions' legacy validation set was single-class, so its legacy layer choice is arbitrary).

## Candidate analyses for the revision (not yet in `paper_rebuttal/`)

**k=0 selectivity at the best test layers (SEL-CI)**
- Setup: the real probe and each shuffle control are each taken at their own best test layer, as in Table 5 and Figure 3. The selected AUCs reproduce `paper_rebuttal/figures/auc_table.tex`.
- Result: real − trajectory shuffle is +0.05 to +0.31, and real − token shuffle is +0.17 to +0.32. Both CIs (paired task bootstrap, 1000 resamples) exclude 0 in all 16 run × property cells.
- Output: `paper_rebuttal_candidate/selectivity/` (`scripts/rebuttal_selectivity_testbest.py`).

**Pooled transfer AUC (T-POOL)**
- What changed: transfer probes were re-scored with one pooled AUC over the target's task-split test tokens. The token set matches the in-dist rows (asserted). The binned values reproduce the current `tab:transfer` (asserted).
- How the probe is applied: the target hidden state is centred on the source training mean, with no std scaling. Source layer L is applied to target layer L. This is the same as the submitted `run_eval`.
- Pooled minus binned: −0.028 to +0.093 per cell (median +0.025). Pooled is lower in 13 of 80 cells, mostly compiles and Qwen Verified→Pro has_regressions.
- Transfer − in-dist, both pooled: the CI is below 0 in 70 of 80 cells and never above 0.
  - At the best test layers, correct transfers almost fully: Verified→Pro −0.016 (Laguna) and −0.001 (Qwen), both CIs including 0.
  - Compiles does not transfer: −0.16 to −0.25.
- Output: `paper_rebuttal_candidate/transfer/` (`scripts/rebuttal_transfer_pooled.py`, `scripts/paper_rebuttal_candidate_transfer.py`).

**E11/E11b at the best test layers (E11-TEST)**
- Setup: the same analysis as E11/E11b, with the real probe at its best test layer, which reproduces Table 5 (asserted). Laguna's layers are unchanged, so its numbers equal E11's.
- Qwen/Pro compiles and has_regressions peak at layer 1. There, the same-language AUC is 0.539 [0.475, 0.593] and 0.623 [0.593, 0.681], against 0.578 and 0.711 at the validation-best layers (40 and 21). The language-only scores (0.756 / 0.831) match the pooled AUCs (0.750 / 0.840).
  - So the layer-1 peaks are mostly the task's language, and more so than at the deeper layers.
  - For Qwen/Pro has_regressions, every test task with both classes is a Python task.
- Verified (repository controls): same-repo AUC stays within 0.04 of pooled for every property.
- Output: `paper_rebuttal_candidate/lang/`.

**Population counts after filtering (E14, wokJ-Q3)**
- Counting only. Kept states are those with a defined label (y ≥ 0). All counts reproduce the stored n_train / n_val / n_test of the pooled runs, and the windowed test counts equal the lookahead figures' "n = … h_t" (asserted).
- Each population is split over its own tasks, as `run_final` does. A window that removes every trajectory of a task changes the 70/15/15 split, so the k_max = 50 test set shares only 9–18 tasks with the k = 0 test set (14 of 63 for Laguna Verified).
- k_max = 50 keeps 31–53% of all trajectories (Qwen Verified 31%, Laguna Verified 51%). Turns ≥ 50: 57% / 60% (Laguna V / P) and 40% / 55% (Qwen V / P), which matches wokJ-2b's 40–60%.
- Submitted Fig 6 (1.1M vs 145k): the legacy populations were rebuilt from `snapshots/` with the legacy window rule and match every stored legacy count (48 of 48). Laguna Verified max50 test: 248 tasks, 1,535 trajectories, 1,062,176 states. Laguna Pro: 53 tasks, 302 trajectories, 144,545 states. The 7.3× ratio is 5.2× more trajectories in test (the repository split put astropy / django / sphinx, 297 of 500 tasks, in test) × 1.45× more states per kept trajectory × 0.97 from the window. The window keeps a similar share on both benchmarks (52% vs 53% of test trajectories).
- The legacy Laguna Verified max50 validation set was 1 trajectory and 25 states, since the repository split put a 1-task repository in validation.
- Output: `paper_rebuttal_candidate/counts/` (`scripts/rebuttal_population_counts.py`, `scripts/paper_rebuttal_candidate_counts.py`).

**Probe hyperparameters (HP)**
- The 80 task-split sweeps (4 runs × 4 properties × 5 layers) each ran 20 random-search trials, all finished (checked on W&B).
  - Search space: lr and weight decay log-uniform in [1e-4, 1e-1] and [1e-5, 1e-1], batch size {256, 512, 1024}, patience {10, 25, 50}.
  - Objective: maximise `mean_val_auc`. With one pooled bin this is the validation AUC over all validation tokens.
  - Not swept: Adam, unweighted cross-entropy, at most 1000 epochs with early stopping on validation loss.
- Every completed rerun training job (5,768) used `final --from-sweep` with the sweep of its own benchmark, model, property and layer, and logged that sweep's best config (checked). This covers k=0, both lookahead windows, and the token and trajectory shuffles.
  - All transfer, after-edit, bin and step evaluations are eval-only runs of these weights.
  - The only exceptions are the `*_shuffled_v2` jobs of 2026-09-28: old-split sweeps, and typed-in values for Qwen Pro. They are from the discarded first pass, and no results directory remains.
  - The position baselines (E16) are sklearn logistic regressions with default settings, not swept.
- Chosen configs vary a lot across layers (lr up to 2.4 decades, weight decay up to 3.4), but the objective is flat near the top. The best trial beats the 5th best by a median of 0.007 validation AUC (IQR 0.003–0.014) and the median trial by 0.035.
- Output: `paper_rebuttal_candidate/hparams/` (`scripts/rebuttal_hparam_provenance.py`, `scripts/paper_rebuttal_candidate_hparams.py`).

**k=0 AUC right after an edit, before any execution (AE0; HaQJ-Q3, wokJ-1b, 1nZa-W2)**
- **States:** the assistant tokens of turn N+1 after each real edit at command N (E5's selection). The k=0 pooled weights are taken at the `tab:auc_roc` best layer, on the task-split test set.
- **Filter:** edits whose command also ran something are dropped, keeping Python scripts that only rewrite files (`scripts/after_edit_cmd_filter.py`). E5 had no such filter.
  - Of all edits, primary drops 3.3% (Laguna Verified), 1.2% (Qwen Verified), 3.6% (Laguna Pro) and 2.7% (Qwen Pro). In the test split: 3.5%, 0.5%, 4.2% and 1.8%.
  - Strict, which also drops the editor scripts, drops 73%, 41%, 46% and 36%.
- **Checks:** the recomputed all-token AUCs equal the stored ones, and the unfiltered selection equals E5's stored n_test and n_pos (asserted / checked).
- **Result:** after-edit AUC is 0.58–0.81, against 0.69–0.84 on all tokens.
  - Verified: 0.58–0.65, which is 0.05–0.26 below the all-token AUC.
  - Pro: 0.66–0.81, which is −0.16 to +0.02.
  - The 95% CI lies above 0.5 in 14 of 16 cells. The two exceptions are has_regressions on Verified (Laguna [0.49, 0.64], Qwen [0.49, 0.70]).
  - The filter barely matters: primary and no filter differ by at most 0.021 in any cell.
- **Caveat (label-changed subset):** on states where the edit changed the label relative to the turn before, AUC is below 0.5 in 12 of 16 cells (0.22–0.74).
  - There the probe tends to report the pre-edit status.
  - So most of the after-edit signal comes from edits that leave the label unchanged (persistence). The probe does not anticipate the effect of a fresh edit before it is executed.
- Output: `paper_rebuttal_candidate/after_edit_k0/` (`scripts/rebuttal_after_edit_k0.py`, `scripts/after_edit_cmd_filter.py`, `scripts/paper_rebuttal_candidate_after_edit.py`, `slurm/after_edit_k0.sh`).

**Prompt-only baseline (PO; HaQJ-Q2, wokJ-1b)**
- **Predictor:** logistic regression from one vector per trajectory to the probe's own target, trained on the probe's population, C chosen on validation. Every token of the trajectory gets that score.
  - Primary: the hidden state of the first cached token (first `content` token of turn 1) at the probe's layer. For Qwen its context is the prompt only. For Laguna it also includes the turn-1 `<think>` block (median 81 tokens on Verified, 36 on Pro), so it is prompt + opening reasoning, a stronger baseline than prompt-only.
  - Secondary: TF-IDF of the task prompt text.
- **Horizon, within y_t = 0, window 15 (72 settings):** prompt-only (hidden state) reaches 0.45–0.80 (Laguna/Verified 0.51–0.60, Qwen/Verified 0.47–0.61, Laguna/Pro 0.60–0.75, Qwen/Pro 0.45–0.80); the probe 0.56–0.87.
  - Probe − prompt-only: CI > 0 in 26/72, CI < 0 in 0/72. By run: Qwen/Verified 14/18, Qwen/Pro 7/18, Laguna/Verified 5/18, Laguna/Pro 0/18.
  - TF-IDF variant: CI > 0 in 18/72, CI < 0 in 2/72 (Laguna/Pro reduces_failing, k = 10, 15).
- **Table 1 recount (E18 strata, k ≥ 2, 60 settings; powered settings in brackets).** Old → with prompt-only (hidden state) added:
  - "beats all baselines": S0 45 → 20 (38 → 18 of 49), S1 38 → 8 (31 → 6 of 49), S2 17 → 17 (15 → 15 of 35), S3 6 → 6 (6 → 6 of 35), S4 23 → 13 (4 → 0 of 5).
  - "beats baselines available at step t": S0 49 → 23 (42 → 21), S1 42 → 8 (35 → 6), S2 27 → 25 (24 → 22), S3 21 → 19 (19 → 17), S4 23 → 13 (4 → 0).
  - With the TF-IDF variant also added: all baselines S0 10, S1 3, S2 17, S3 6, S4 0; at step t S0 10, S1 3, S2 25, S3 19, S4 0 (over 60).
  - CIs below 0: at most 1 of 60 per stratum with the old or the old + prompt-only sets (unpowered cells only), up to 5 of 60 with TF-IDF also added.
  - Reading: the prompt explains most of what the probe adds over the old controls in S0 and S1, where the eventual outcome varies. Once the outcome is fixed (S2, S3) it adds almost nothing. This matches E18: the y_t = 0 signal is mostly *whether* the trajectory ends up passing, which the task (and for Laguna the opening reasoning) partly predicts.
- **k = 0 (Table 6 layer, all test tokens):** prompt-only (hidden state) 0.48–0.77. The probe is ahead by +0.02 to +0.35, with CI > 0 in 11 of 16 cells; not in Qwen/Verified compiles or Laguna/Pro compiles, correct, reduces_failing, or Qwen/Pro compiles.
  - TF-IDF of the prompt reaches 0.51–0.86 and is above the probe in 4 Pro cells (Laguna compiles 0.83 vs 0.79 and has_regressions 0.83 vs 0.74; Qwen compiles 0.78 vs 0.75 and has_regressions 0.86 vs 0.84), none significantly. The probe − TF-IDF CI is > 0 in 5 of 16 cells (correct and reduces_failing on Verified, Qwen/Pro correct). Compare E11: on Pro, the repository alone predicts compiles with 0.83.
- Output: `paper_rebuttal_candidate/prompt_only/` (`scripts/rebuttal_prompt_only.py`, `scripts/rebuttal_prompt_texts.py`, `scripts/rebuttal_prompt_cache_check.py`, `scripts/paper_rebuttal_candidate_prompt_only.py`, `slurm/prompt_only.sh`).

**Text-only baseline and AUC relative to tool outputs (E9; wokJ-1b, wokJ-Q1, 1nZa-W2)**
- **Baseline:** TF-IDF (word 1–2-grams) + logistic regression on the most recent tool output before each assistant turn, trained like the probe it is compared with (split, population, target; C on validation). Every token of the turn gets the turn's score. The all-context variant (A2) was not run: it costs about 10× A1.
- **k = 0 (Table 6 layer):** text 0.55–0.81, probe 0.69–0.84. Probe − text CI > 0 in 12 of 16 cells (+0.05 to +0.18).
  - Not significant: Laguna/Verified compiles (text 0.703 vs probe 0.691), Laguna/Pro has_regressions (0.801 vs 0.740), Qwen/Pro compiles (0.727 vs 0.750) and has_regressions (0.807 vs 0.840).
- **Horizon, y_t = 0, window 15 (72 settings):** text 0.46–0.88; probe − text CI > 0 in 51/72, CI < 0 in 0/72 (by run: Laguna/Verified 15, Qwen/Verified 16, Laguna/Pro 8, Qwen/Pro 12 of 18).
- **Table 1 recount (E18 strata, k ≥ 2, 60 settings, powered in brackets).** Each row reads old / + prompt-only (PO) / + text (E9) / + both.
  - "beats all baselines":
    - S0: 45 [38/49] / 20 [18] / 32 [31] / 18 [17]
    - S1: 38 [31/49] / 8 [6] / 18 [17] / 5 [4]
    - S2: 17 [15/35] in all four
    - S3: 6 [6/35] in all four
    - S4: 23 [4/5] / 13 [0] / 6 [0] / 5 [0]
  - "beats baselines available at step t":
    - S0: 49 [42] / 23 [21] / 35 [34] / 20 [19]
    - S1: 42 [35] / 8 [6] / 20 [19] / 5 [4]
    - S2: 27 [24] / 25 [22] / 27 [24] / 25 [22]
    - S3: 21 [19] / 19 [17] / 21 [19] / 19 [17]
    - S4: 23 [4] / 13 [0] / 6 [0] / 5 [0]
  - With both added, at most 1 of 60 settings per stratum has a CI below 0.
  - The latest tool output removes less of the probe's lead than the prompt does. Neither changes S2 or S3 (fixed outcome), where the strongest control remains the position baselines.
- **Part B (k = 0, current program version not yet run vs already run; AE0 command rule):**
  - **Shares of test tokens:**
    - Not yet run: 17–24% on Verified and 54–64% on Pro.
    - After the first edit only: not yet run 10–38%, already run 27–63%, before the first edit 21–43%.
  - **The probe is not better once the version has been run.** Over all test tokens, its AUC on already-run minus not-yet-run states is −0.15 to +0.02 (CI < 0 in 10/16 cells, 14/16 with the strict rule; never > 0).
    - That comparison mixes in the pre-first-edit phase, where correct and reduces_failing are almost single-class. After the first edit only, the difference is −0.15 to +0.10 (CI < 0 in 3, CI > 0 in 2 of 16). It is negative mainly on Pro: Qwen correct −0.15, has_regressions −0.07; Laguna reduces_failing −0.09.
  - **Probe vs text:**
    - Probe − text, over all tokens: CI > 0 in 12/16 cells for not yet run and 9/16 for already run.
    - After the first edit only: 8/16 and 7/16. One cell has the text significantly better: Laguna/Pro has_regressions, already run (−0.11 [−0.22, −0.01]).
  - Reading: the probes do not depend on having seen the newest program version run, and they stay ahead of the text of the latest tool output in most cells either way.
- Output: `paper_rebuttal_candidate/text_baseline/` (`scripts/rebuttal_text_baseline.py`, `scripts/rebuttal_text_extract.py`, `scripts/paper_rebuttal_candidate_text_baseline.py`, `slurm/text_baseline.sh`).

**Edit-indexed horizon (EI; HaQJ-W1c, HaQJ-Q3)**
- **Question:** from the hidden state at turn t, can a probe predict the label after the m-th future edit (m = 1, 2, 3; edit written in a turn after t, so the state precedes the target edit)? Edits and outcomes as in AE0 (outcome = label of turn N_e + 1). Tokens: pooled k = 0 caches, Table 6 test tasks, y_t ≥ 0.
- **Probe:** new linear probes trained as in `src/probe.py`, at the k = 0 probe's validation-best layer only (5 layers did not fit the budget), with that layer's k = 0 sweep hyperparameters. Weight decay (the analogue of C) chosen on validation from {k = 0 value, 1e-2, 1e-1}. With the k = 0 value alone the first test probes early-stopped after 1–2 epochs; the grid picked a larger value in 10 of 36 fits.
- **Baselines:** persistence y_t, the k = 0 probe (Table 6 layer), E16 position splines (causal, relative), edits so far, observable features (step, edits so far, turns since last edit), trajectory shuffle, prompt-only (PO), y_T oracle. Populations: A all, B y_t = 0, C y_t = 0 and y_T = 1. Properties: correct, reduces_failing, has_regressions (compiles not run).
- **Result: the probe never beats all step-t baselines in A or C, and in B only in 1 of 12 (m = 1) and 2 of 12 (m = 3) settings.**
  - A: probe 0.54–0.75. Persistence is the strongest baseline in 8 of 12 (m = 1, 2) and is significantly better than the probe in 3–4 of 12 per m (all Laguna/Verified and one Qwen/Verified); the probe is never significantly better than persistence.
  - B: probe 0.51–0.75, ahead of the k = 0 probe (CI > 0) in 10, 9 and 8 of 12 for m = 1, 2, 3, and of the trajectory shuffle in 9, 8, 7. The strongest baseline is mostly prompt-only or causal position, which the probe does not beat.
  - C (eventually solved, not yet holding; does the probe know which edit fixes it?): probe 0.37–0.66, at chance; it beats no baseline in more than 2 of 12 settings per m.
  - Target edits change the label in 1–35% of cases (A); median distance to e_m: 3–9 turns (m = 1), 7–16 (m = 2), 11–23 (m = 3).
- **Cost:** 48 CPU core-hours (estimate was ~21; the Laguna jobs took 2–2.8 h each), over the ~20 budget.
- Output: `paper_rebuttal_candidate/edit_indexed/` (`scripts/rebuttal_edit_indexed.py`, `scripts/paper_rebuttal_candidate_edit_indexed.py`, `slurm/edit_indexed.sh`).

**Table 1 at k = 1 (T1K1)**
- Table 1 now shows only S0 and S2, so k = 1 no longer has to be excluded (only S1 needed k >= 2). The k = 1 S0/S2 cells were computed with the E18/PO setup: same probe layer, controls, prompt_hs refit, oracle_yT, paired bootstrap.
- S0 reproduces PO's k = 1 numbers exactly (AUCs, CIs, counts; asserted) and E16.
- Cost: about 1.1 CPU core-hours (12 jobs, 2 cores, about 3 min each). No summary counts; the table is computed from the JSON.
- Output: `paper_rebuttal_candidate/table1_k1/` (`scripts/rebuttal_table1_k1.py`, `slurm/table1_k1.sh`).

## Other things to know

- **What "after-edit" means** (rerun). A real edit (`cmd_idx` = N ≥ 0) is issued during assistant turn N, and its effect appears in the labels from turn N+1 on (`edit_labels_by_turn`). `edit_step_index.pt` stores N+1 for each edit (the baseline edit is excluded). The after-edit filter keeps a token only if its **target** turn (`target_step_idx`, the turn its label refers to) is one of these turns. That is exactly the turn after each edit, never later turns. The probes and the persistence baseline use the same filter.
  - At lookahead k, the source tokens come from turn N+1−k.
  - Every extracted assistant token of turn N+1 is kept, not only the first.
  - At k = 0, assistant message N+1 comes after the tool output of the edit command (and any test run), so those hidden states can already attend to the edit's observed result.
  - An edit that leaves a label unchanged still contributes its turn N+1, and so does a turn N+1 that is itself an edit turn.

- **Layer 0 on Qwen Pro** (rerun). The best layer for compiles (0.750) and has_regressions (0.840) is code layer 0. Code layer L is `out.hidden_states[L + 1]` in every extractor (`src/models/*.py`), so layer 0 is the output of the first transformer block, not the embedding layer. After a single block the representation is shallow and mostly lexical, so these two labels are largely predictable from which tokens are present (files, outputs), not necessarily from computed program state. For correct, layer 0 falls to 0.637 against 0.822 at layer 30. E11 points the same way: on Qwen/Pro every has_regressions positive is a Python token, and a language-only score already reaches 0.831; Pro compiles is likewise matched by language-only (0.756).
- **Best layer** is chosen by test AUC (the rule in `run_paper_figures.py`, used for the submitted runs and for every rerun table except the phase-confound section), so those "best layer" numbers are slightly optimistic. The phase-confound section selects layers on validation AUC.
- **Multi-bin evals** (transfer, bineval, after-edit) are summarised as the n_test-weighted mean of per-bin AUCs, skipping single-class bins. `run_paper_figures._weighted_mean` keeps those bins, with their placeholder AUC of 0.5. `scripts/paper_rebuttal_figures.py` (the revision's figures in `paper_rebuttal/`) drops them. In the inputs to the paper's figures, the only rerun runs with a single-class bin are Qwen's Verified→Pro transfer evals (one bin of 10, for correct, reduces_failing and has_regressions), which move up by 0.003 to 0.028. Every pooled run and every stored persistence baseline of the rerun has one bin, so none of them change.
- **Transfer deltas mix two AUC definitions.** In `tab:transfer` the transfer AUC is the mean over 10 relative-position bins, while the in-distribution reference is one pooled AUC. On the same test tokens, the in-distribution probe's bin average (`<pooled>_bineval`) is 0.00 to 0.07 below its pooled AUC at the best layer, so the table's deltas overstate the transfer loss by that much.
- The persistence baseline (rerun) is undefined at k=0 (the cache stores no `y_original` there); the figures note it as missing for k=0 only.
- Some per-stratum AUCs, within y_t=1 especially, rest on few flipped tokens and have wide CIs; several are below 0.5.
- Operational delays in the rerun (no effect on results):
  - submission feeders crashed under the system Python 3.6, costing ~6 h on 97 jobs;
  - 4 jobs hit the 2-hour time limit and were resubmitted with 4 hours.

## Figures

- Rerun lookahead plots, all with the persistence baseline: `paper/figures/{laguna_xs2_full_pooled, qwen36_35b_a3b_full_pooled, laguna_xs2_pro_full_pooled_pooled_pro, qwen36_35b_a3b_pro_full_pooled_pooled_pro}/<probe>_lookahead_max{50,15}[_after_edit].pdf` (64 files).
- Other figures from `slurm/figures.sh` (defaults): `paper/figures/` (summary barplot, transfer barplot, heatmaps, step_rel, bineval, after-edit).
- Example trajectories (layer 21, i.e. `--layer 20`, UMAP, after-edit, paper style):
  - `paper/figures/trajectories/final/laguna/umap_sphinx-doc__sphinx-9602.pdf`
  - `paper/figures/trajectories/final/qwen/umap_sympy__sympy-14531.pdf`
- Submitted (legacy) versions: `paper/figures_legacyturns/` and `*_legacyturns.{pdf,png}` next to the trajectory figures.

## Reproducing

- `scripts/rebuttal_check_labels.py` (step 1)
- `scripts/rebuttal_check_k0.py --task-groups` (step 2)
- `scripts/rebuttal_persistence_analysis.py` (step 6; one job per run × probe × window; 1,000 task-level bootstrap resamples)
- `scripts/rebuttal_check_persistence_labels.py` and `scripts/rebuttal_persistence_turn_level.py` (persistence-baseline check)
- `scripts/rebuttal_submit_traj_shuffle_lookahead.py`, `scripts/rebuttal_phase_confound.py`, `scripts/rebuttal_submit_phase_confound.py` (phase-confound analyses)
- `scripts/rebuttal_edit_outcome_strata.py` (E18)
- `scripts/rebuttal_per_language.py`, `scripts/rebuttal_per_language_figures.py` (E11)
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

#### Persistence analysis, window max50 (test split; 95% CI from 1000 task-level bootstrap resamples)

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

#### Persistence analysis, window max15 (test split; 95% CI from 1000 task-level bootstrap resamples)

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

#### Label consistency (all label files and all window-50 shift caches)

| Model / benchmark | Probe | positions checked | `label[t] != turn_label[step_idx[t]]` | cache `y_original` / `y` mismatches vs k=0 turn labels (k=1…50) | trajectories with any label change | label changes / trajectory | changes that revert |
|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | compiles | 7,908,907 | 0 | 0 of 12,889,734 | 0.115 | 0.17 | 0.339 |
| Laguna XS.2 / Verified | correct | 7,908,907 | 0 | 0 of 12,889,734 | 0.586 | 0.68 | 0.140 |
| Laguna XS.2 / Verified | has_regressions | 7,908,907 | 0 | 0 of 12,889,734 | 0.139 | 0.28 | 0.496 |
| Laguna XS.2 / Verified | reduces_failing | 7,908,907 | 0 | 0 of 12,889,734 | 0.624 | 0.77 | 0.195 |
| Qwen3.6-35B-A3B / Verified | compiles | 4,297,044 | 0 | 0 of 7,372,077 | 0.102 | 0.14 | 0.275 |
| Qwen3.6-35B-A3B / Verified | correct | 4,297,044 | 0 | 0 of 7,372,077 | 0.593 | 0.62 | 0.043 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 4,297,044 | 0 | 0 of 7,372,077 | 0.110 | 0.18 | 0.390 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 4,297,044 | 0 | 0 of 7,372,077 | 0.619 | 0.67 | 0.083 |
| Laguna XS.2 / Pro | compiles | 8,181,945 | 0 | 0 of 9,422,640 | 0.174 | 0.38 | 0.541 |
| Laguna XS.2 / Pro | correct | 8,181,945 | 0 | 0 of 9,422,640 | 0.125 | 0.16 | 0.232 |
| Laguna XS.2 / Pro | has_regressions | 8,181,945 | 0 | 0 of 9,422,640 | 0.068 | 0.13 | 0.465 |
| Laguna XS.2 / Pro | reduces_failing | 8,181,945 | 0 | 0 of 9,422,640 | 0.195 | 0.31 | 0.380 |
| Qwen3.6-35B-A3B / Pro | compiles | 4,078,069 | 0 | 0 of 8,590,213 | 0.110 | 0.21 | 0.471 |
| Qwen3.6-35B-A3B / Pro | correct | 4,078,069 | 0 | 0 of 8,590,213 | 0.154 | 0.18 | 0.124 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 4,078,069 | 0 | 0 of 8,590,213 | 0.046 | 0.07 | 0.351 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 4,078,069 | 0 | 0 of 8,590,213 | 0.166 | 0.22 | 0.232 |

#### Fraction of test tokens with y_{t+k} ≠ y_t, window max50 (fraction of test trajectories with at least one such token in parentheses)

| Model / benchmark | Probe | k=1 | k=2 | k=3 | k=5 | k=10 | k=15 | k=25 | k=50 |
|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | compiles | 0.007 (0.04) | 0.011 (0.04) | 0.014 (0.04) | 0.021 (0.05) | 0.035 (0.07) | 0.042 (0.10) | 0.065 (0.13) | 0.084 (0.13) |
| Laguna XS.2 / Verified | correct | 0.012 (0.08) | 0.016 (0.08) | 0.019 (0.09) | 0.030 (0.11) | 0.056 (0.17) | 0.079 (0.23) | 0.121 (0.33) | 0.202 (0.39) |
| Laguna XS.2 / Verified | has_regressions | 0.018 (0.08) | 0.022 (0.09) | 0.028 (0.10) | 0.045 (0.11) | 0.077 (0.14) | 0.107 (0.17) | 0.139 (0.21) | 0.170 (0.22) |
| Laguna XS.2 / Verified | reduces_failing | 0.030 (0.17) | 0.043 (0.18) | 0.056 (0.18) | 0.083 (0.21) | 0.142 (0.29) | 0.188 (0.37) | 0.252 (0.51) | 0.353 (0.59) |
| Qwen3.6-35B-A3B / Verified | compiles | 0.003 (0.02) | 0.004 (0.02) | 0.005 (0.02) | 0.010 (0.03) | 0.014 (0.03) | 0.011 (0.04) | 0.017 (0.06) | 0.029 (0.07) |
| Qwen3.6-35B-A3B / Verified | correct | 0.014 (0.06) | 0.018 (0.07) | 0.021 (0.07) | 0.031 (0.08) | 0.057 (0.09) | 0.086 (0.15) | 0.141 (0.25) | 0.302 (0.51) |
| Qwen3.6-35B-A3B / Verified | has_regressions | 0.004 (0.03) | 0.006 (0.03) | 0.010 (0.04) | 0.016 (0.05) | 0.028 (0.07) | 0.043 (0.10) | 0.088 (0.15) | 0.170 (0.22) |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 0.022 (0.11) | 0.029 (0.11) | 0.035 (0.11) | 0.051 (0.13) | 0.092 (0.18) | 0.134 (0.24) | 0.210 (0.32) | 0.419 (0.55) |
| Laguna XS.2 / Pro | compiles | 0.014 (0.04) | 0.024 (0.05) | 0.033 (0.06) | 0.044 (0.06) | 0.061 (0.09) | 0.072 (0.12) | 0.085 (0.16) | 0.081 (0.13) |
| Laguna XS.2 / Pro | correct | 0.005 (0.02) | 0.006 (0.02) | 0.006 (0.02) | 0.013 (0.03) | 0.027 (0.05) | 0.043 (0.06) | 0.064 (0.09) | 0.102 (0.13) |
| Laguna XS.2 / Pro | has_regressions | 0.014 (0.02) | 0.020 (0.02) | 0.020 (0.02) | 0.022 (0.03) | 0.030 (0.04) | 0.041 (0.05) | 0.062 (0.06) | 0.078 (0.07) |
| Laguna XS.2 / Pro | reduces_failing | 0.014 (0.04) | 0.025 (0.05) | 0.032 (0.05) | 0.042 (0.06) | 0.063 (0.10) | 0.078 (0.13) | 0.107 (0.17) | 0.129 (0.18) |
| Qwen3.6-35B-A3B / Pro | compiles | 0.020 (0.03) | 0.024 (0.04) | 0.027 (0.04) | 0.032 (0.04) | 0.044 (0.06) | 0.055 (0.07) | 0.073 (0.11) | 0.112 (0.20) |
| Qwen3.6-35B-A3B / Pro | correct | 0.010 (0.05) | 0.013 (0.05) | 0.015 (0.05) | 0.019 (0.05) | 0.027 (0.06) | 0.033 (0.08) | 0.051 (0.12) | 0.125 (0.23) |
| Qwen3.6-35B-A3B / Pro | has_regressions | 0.002 (0.00) | 0.003 (0.00) | 0.003 (0.00) | 0.005 (0.00) | 0.003 (0.01) | 0.003 (0.01) | 0.007 (0.02) | 0.013 (0.03) |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 0.013 (0.02) | 0.016 (0.03) | 0.017 (0.03) | 0.020 (0.03) | 0.029 (0.04) | 0.036 (0.06) | 0.054 (0.09) | 0.108 (0.19) |

#### Persistence at k=1, window max50, test split: tokens vs turns

| Model / benchmark | Probe | tokens y_t=1 | tokens y_{t+1}=1 | tokens 0→1 | tokens 1→0 | token TPR | token TNR | token persistence AUC | turns | turns y_t=1 | flipped turns | tokens / turn (all, flipped) | turn persistence AUC | turn persistence AUC, max15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | compiles | 177,511 | 176,567 | 172 | 1,116 | 0.999 | 0.793 | 0.896 | 8,523 | 8,418 | 18 | 21.3, 71.6 | 0.936 | 0.977 |
| Laguna XS.2 / Verified | correct | 13,882 | 16,037 | 2,155 | 0 | 0.866 | 1.000 | 0.933 | 8,523 | 485 | 30 | 21.3, 71.8 | 0.971 | 0.968 |
| Laguna XS.2 / Verified | has_regressions | 11,799 | 15,012 | 3,213 | 0 | 0.786 | 1.000 | 0.893 | 8,523 | 362 | 32 | 21.3, 100.4 | 0.959 | 0.975 |
| Laguna XS.2 / Verified | reduces_failing | 21,243 | 26,382 | 5,342 | 203 | 0.798 | 0.999 | 0.898 | 8,523 | 610 | 69 | 21.3, 80.4 | 0.951 | 0.969 |
| Qwen3.6-35B-A3B / Verified | compiles | 60,213 | 60,266 | 115 | 62 | 0.998 | 0.871 | 0.934 | 4,972 | 4,949 | 6 | 12.2, 29.5 | 0.920 | 0.961 |
| Qwen3.6-35B-A3B / Verified | correct | 2,020 | 2,845 | 836 | 11 | 0.706 | 1.000 | 0.853 | 4,972 | 133 | 17 | 12.2, 49.8 | 0.946 | 0.954 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 1,456 | 1,719 | 263 | 0 | 0.847 | 1.000 | 0.924 | 4,972 | 116 | 8 | 12.2, 32.9 | 0.968 | 0.966 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3,835 | 5,124 | 1,310 | 21 | 0.744 | 1.000 | 0.872 | 4,972 | 198 | 28 | 12.2, 47.5 | 0.941 | 0.959 |
| Laguna XS.2 / Pro | compiles | 76,793 | 76,276 | 621 | 1,138 | 0.992 | 0.977 | 0.985 | 11,369 | 6,527 | 42 | 11.1, 41.9 | 0.996 | 0.990 |
| Laguna XS.2 / Pro | correct | 559 | 1,029 | 537 | 67 | 0.478 | 0.999 | 0.739 | 11,369 | 33 | 12 | 11.1, 50.3 | 0.884 | 0.956 |
| Laguna XS.2 / Pro | has_regressions | 2,167 | 2,633 | 1,102 | 636 | 0.581 | 0.995 | 0.788 | 11,369 | 70 | 18 | 11.1, 96.6 | 0.921 | 0.966 |
| Laguna XS.2 / Pro | reduces_failing | 3,249 | 4,898 | 1,733 | 84 | 0.646 | 0.999 | 0.823 | 11,369 | 147 | 30 | 11.1, 60.6 | 0.925 | 0.959 |
| Qwen3.6-35B-A3B / Pro | compiles | 38,528 | 38,852 | 814 | 490 | 0.979 | 0.983 | 0.981 | 8,339 | 4,965 | 16 | 8.0, 81.5 | 0.998 | 0.996 |
| Qwen3.6-35B-A3B / Pro | correct | 2,737 | 3,376 | 662 | 23 | 0.804 | 1.000 | 0.902 | 8,339 | 313 | 20 | 8.0, 34.2 | 0.973 | 0.958 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 122 | 99 | 69 | 92 | 0.303 | 0.999 | 0.651 | 8,339 | 5 | 2 | 8.0, 80.5 | 0.900 | 0.949 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 907 | 1,762 | 871 | 16 | 0.506 | 1.000 | 0.753 | 8,339 | 71 | 10 | 8.0, 88.7 | 0.943 | 0.956 |

#### Phase confound, window max15, within y_t = 0: AUC for y_(t+k) (test split; differences with 95% task-bootstrap CI, bold if the CI excludes 0)

| Model / benchmark | Probe | k | n (pos) | real, val-best layer | real, test-best layer | k=0 probe (E2) | traj-shuffle | position, causal | position, rel_pos (non-causal) | real − k=0 probe | real − traj-shuffle | real − position causal | real − position rel_pos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | correct | 1 | 522,140 (28,807) | 0.645 (L20) | 0.645 (L20) | 0.398 (L20) | 0.472 | 0.468 | 0.497 | **+0.247 [+0.21, +0.29]** | **+0.173 [+0.08, +0.26]** | **+0.178 [+0.08, +0.29]** | **+0.149 [+0.06, +0.23]** |
| Laguna XS.2 / Verified | correct | 2 | 522,132 (38,097) | 0.646 (L20) | 0.646 (L20) | 0.407 (L20) | 0.457 | 0.483 | 0.491 | **+0.239 [+0.20, +0.28]** | **+0.189 [+0.10, +0.28]** | **+0.163 [+0.07, +0.26]** | **+0.155 [+0.07, +0.24]** |
| Laguna XS.2 / Verified | correct | 3 | 522,125 (47,141) | 0.635 (L20) | 0.635 (L20) | 0.419 (L20) | 0.427 | 0.498 | 0.482 | **+0.216 [+0.18, +0.25]** | **+0.208 [+0.12, +0.30]** | **+0.137 [+0.05, +0.23]** | **+0.153 [+0.07, +0.24]** |
| Laguna XS.2 / Verified | correct | 5 | 522,125 (64,820) | 0.618 (L20) | 0.618 (L20) | 0.422 (L20) | 0.421 | 0.513 | 0.464 | **+0.196 [+0.15, +0.24]** | **+0.198 [+0.11, +0.30]** | **+0.105 [+0.03, +0.19]** | **+0.154 [+0.07, +0.25]** |
| Laguna XS.2 / Verified | correct | 10 | 522,143 (97,600) | 0.606 (L20) | 0.606 (L20) | 0.427 (L20) | 0.399 | 0.551 | 0.456 | **+0.179 [+0.13, +0.23]** | **+0.207 [+0.11, +0.31]** | +0.054 [-0.03, +0.14] | **+0.150 [+0.05, +0.25]** |
| Laguna XS.2 / Verified | correct | 15 | 522,200 (122,830) | 0.604 (L20) | 0.604 (L20) | 0.425 (L20) | 0.401 | 0.583 | 0.458 | **+0.178 [+0.12, +0.23]** | **+0.203 [+0.10, +0.31]** | +0.021 [-0.06, +0.11] | **+0.146 [+0.05, +0.26]** |
| Laguna XS.2 / Verified | reduces_failing | 1 | 498,561 (29,668) | 0.692 (L20) | 0.692 (L20) | 0.444 (L20) | 0.469 | 0.437 | 0.476 | **+0.247 [+0.21, +0.29]** | **+0.223 [+0.16, +0.29]** | **+0.255 [+0.17, +0.34]** | **+0.215 [+0.15, +0.28]** |
| Laguna XS.2 / Verified | reduces_failing | 2 | 498,553 (40,409) | 0.660 (L20) | 0.660 (L20) | 0.438 (L20) | 0.455 | 0.443 | 0.471 | **+0.223 [+0.19, +0.26]** | **+0.206 [+0.14, +0.27]** | **+0.217 [+0.14, +0.30]** | **+0.190 [+0.12, +0.26]** |
| Laguna XS.2 / Verified | reduces_failing | 3 | 498,546 (50,878) | 0.638 (L20) | 0.638 (L20) | 0.438 (L20) | 0.447 | 0.459 | 0.462 | **+0.200 [+0.17, +0.23]** | **+0.191 [+0.13, +0.25]** | **+0.179 [+0.11, +0.25]** | **+0.176 [+0.11, +0.24]** |
| Laguna XS.2 / Verified | reduces_failing | 5 | 498,546 (68,856) | 0.606 (L20) | 0.606 (L20) | 0.425 (L20) | 0.431 | 0.476 | 0.439 | **+0.181 [+0.15, +0.21]** | **+0.176 [+0.11, +0.24]** | **+0.131 [+0.06, +0.20]** | **+0.168 [+0.10, +0.24]** |
| Laguna XS.2 / Verified | reduces_failing | 10 | 498,564 (102,039) | 0.572 (L20) | 0.572 (L20) | 0.413 (L20) | 0.409 | 0.499 | 0.430 | **+0.159 [+0.12, +0.20]** | **+0.164 [+0.10, +0.23]** | **+0.073 [+0.00, +0.15]** | **+0.143 [+0.07, +0.22]** |
| Laguna XS.2 / Verified | reduces_failing | 15 | 498,621 (128,607) | 0.557 (L20) | 0.557 (L20) | 0.407 (L20) | 0.419 | 0.540 | 0.435 | **+0.150 [+0.10, +0.20]** | **+0.138 [+0.07, +0.21]** | +0.017 [-0.05, +0.09] | **+0.122 [+0.05, +0.21]** |
| Laguna XS.2 / Verified | has_regressions | 1 | 640,088 (8,622) | 0.693 (L20) | 0.693 (L20) | 0.545 (L20) | 0.530 | 0.467 | 0.435 | **+0.149 [+0.11, +0.17]** | **+0.163 [+0.09, +0.23]** | **+0.227 [+0.16, +0.29]** | **+0.259 [+0.18, +0.33]** |
| Laguna XS.2 / Verified | has_regressions | 2 | 640,080 (11,431) | 0.654 (L20) | 0.654 (L20) | 0.535 (L20) | 0.533 | 0.464 | 0.427 | **+0.120 [+0.10, +0.14]** | **+0.121 [+0.06, +0.18]** | **+0.190 [+0.13, +0.26]** | **+0.227 [+0.15, +0.30]** |
| Laguna XS.2 / Verified | has_regressions | 3 | 640,073 (14,687) | 0.621 (L20) | 0.621 (L20) | 0.516 (L20) | 0.499 | 0.443 | 0.401 | **+0.105 [+0.08, +0.12]** | **+0.121 [+0.06, +0.18]** | **+0.177 [+0.11, +0.24]** | **+0.219 [+0.13, +0.30]** |
| Laguna XS.2 / Verified | has_regressions | 5 | 640,073 (19,751) | 0.584 (L20) | 0.584 (L20) | 0.489 (L20) | 0.500 | 0.413 | 0.363 | **+0.095 [+0.07, +0.12]** | **+0.084 [+0.02, +0.15]** | **+0.171 [+0.12, +0.23]** | **+0.221 [+0.14, +0.29]** |
| Laguna XS.2 / Verified | has_regressions | 10 | 640,091 (26,671) | 0.563 (L20) | 0.563 (L20) | 0.467 (L20) | 0.441 | 0.377 | 0.335 | **+0.096 [+0.06, +0.12]** | **+0.122 [+0.05, +0.19]** | **+0.187 [+0.14, +0.23]** | **+0.228 [+0.14, +0.30]** |
| Laguna XS.2 / Verified | has_regressions | 15 | 640,148 (28,471) | 0.557 (L20) | 0.557 (L20) | 0.449 (L20) | 0.438 | 0.350 | 0.318 | **+0.109 [+0.07, +0.14]** | **+0.119 [+0.04, +0.20]** | **+0.208 [+0.15, +0.26]** | **+0.240 [+0.14, +0.32]** |
| Qwen3.6-35B-A3B / Verified | correct | 1 | 322,297 (7,015) | 0.803 (L30) | 0.803 (L30) | 0.540 (L30) | 0.694 | 0.492 | 0.616 | **+0.263 [+0.23, +0.29]** | **+0.109 [+0.05, +0.17]** | **+0.311 [+0.19, +0.43]** | **+0.187 [+0.07, +0.30]** |
| Qwen3.6-35B-A3B / Verified | correct | 2 | 322,282 (10,668) | 0.761 (L30) | 0.761 (L30) | 0.514 (L30) | 0.626 | 0.490 | 0.593 | **+0.247 [+0.21, +0.28]** | **+0.135 [+0.07, +0.20]** | **+0.270 [+0.17, +0.38]** | **+0.167 [+0.06, +0.27]** |
| Qwen3.6-35B-A3B / Verified | correct | 3 | 322,273 (14,364) | 0.745 (L30) | 0.745 (L30) | 0.509 (L30) | 0.593 | 0.494 | 0.591 | **+0.236 [+0.19, +0.28]** | **+0.151 [+0.09, +0.22]** | **+0.250 [+0.15, +0.34]** | **+0.154 [+0.05, +0.25]** |
| Qwen3.6-35B-A3B / Verified | correct | 5 | 322,217 (23,111) | 0.722 (L30) | 0.722 (L30) | 0.491 (L30) | 0.574 | 0.517 | 0.583 | **+0.232 [+0.19, +0.27]** | **+0.148 [+0.08, +0.22]** | **+0.206 [+0.12, +0.29]** | **+0.140 [+0.05, +0.23]** |
| Qwen3.6-35B-A3B / Verified | correct | 10 | 322,048 (48,825) | 0.670 (L30) | 0.670 (L30) | 0.484 (L30) | 0.483 | 0.571 | 0.560 | **+0.187 [+0.14, +0.24]** | **+0.187 [+0.09, +0.28]** | **+0.099 [+0.03, +0.17]** | **+0.111 [+0.01, +0.21]** |
| Qwen3.6-35B-A3B / Verified | correct | 15 | 322,010 (70,815) | 0.648 (L30) | 0.648 (L30) | 0.473 (L30) | 0.467 | 0.590 | 0.552 | **+0.176 [+0.12, +0.23]** | **+0.182 [+0.09, +0.28]** | +0.058 [-0.01, +0.12] | +0.096 [-0.00, +0.20] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 1 | 312,595 (9,574) | 0.798 (L20) | 0.807 (L30) | 0.591 (L20) | 0.665 | 0.469 | 0.614 | **+0.207 [+0.18, +0.23]** | **+0.133 [+0.08, +0.18]** | **+0.329 [+0.23, +0.42]** | **+0.184 [+0.10, +0.27]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 2 | 312,580 (13,867) | 0.764 (L20) | 0.764 (L30) | 0.546 (L20) | 0.621 | 0.451 | 0.586 | **+0.218 [+0.19, +0.24]** | **+0.143 [+0.09, +0.19]** | **+0.313 [+0.23, +0.40]** | **+0.178 [+0.10, +0.26]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3 | 312,571 (18,412) | 0.741 (L30) | 0.741 (L30) | 0.520 (L20) | 0.607 | 0.460 | 0.578 | **+0.222 [+0.19, +0.25]** | **+0.134 [+0.08, +0.18]** | **+0.282 [+0.20, +0.36]** | **+0.163 [+0.09, +0.24]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 5 | 312,515 (28,326) | 0.709 (L30) | 0.709 (L30) | 0.496 (L20) | 0.589 | 0.459 | 0.567 | **+0.213 [+0.18, +0.25]** | **+0.120 [+0.07, +0.18]** | **+0.250 [+0.17, +0.32]** | **+0.141 [+0.07, +0.22]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 10 | 312,346 (52,786) | 0.634 (L30) | 0.634 (L30) | 0.468 (L20) | 0.484 | 0.463 | 0.542 | **+0.166 [+0.13, +0.21]** | **+0.150 [+0.08, +0.22]** | **+0.172 [+0.10, +0.24]** | **+0.092 [+0.01, +0.18]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 15 | 312,308 (71,216) | 0.594 (L30) | 0.594 (L30) | 0.448 (L20) | 0.462 | 0.465 | 0.529 | **+0.146 [+0.11, +0.19]** | **+0.132 [+0.06, +0.22]** | **+0.129 [+0.06, +0.20]** | +0.065 [-0.02, +0.16] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 1 | 340,499 (5,387) | 0.797 (L20) | 0.797 (L20) | 0.626 (L30) | 0.695 | 0.502 | 0.662 | **+0.172 [+0.13, +0.20]** | **+0.103 [+0.06, +0.14]** | **+0.295 [+0.20, +0.38]** | **+0.135 [+0.04, +0.24]** |
| Qwen3.6-35B-A3B / Verified | has_regressions | 2 | 340,484 (6,410) | 0.742 (L30) | 0.789 (L20) | 0.616 (L30) | 0.581 | 0.494 | 0.655 | **+0.126 [+0.09, +0.15]** | **+0.161 [+0.10, +0.22]** | **+0.248 [+0.17, +0.33]** | +0.087 [-0.01, +0.19] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 3 | 340,475 (7,785) | 0.739 (L30) | 0.786 (L20) | 0.620 (L30) | 0.582 | 0.493 | 0.647 | **+0.119 [+0.08, +0.14]** | **+0.157 [+0.10, +0.22]** | **+0.246 [+0.16, +0.32]** | +0.092 [-0.01, +0.20] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 5 | 340,419 (10,516) | 0.687 (L30) | 0.737 (L20) | 0.592 (L30) | 0.525 | 0.463 | 0.609 | **+0.095 [+0.06, +0.13]** | **+0.162 [+0.08, +0.24]** | **+0.224 [+0.15, +0.29]** | +0.078 [-0.03, +0.18] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 10 | 340,250 (16,611) | 0.619 (L30) | 0.675 (L20) | 0.562 (L30) | 0.622 | 0.435 | 0.573 | **+0.057 [+0.02, +0.09]** | -0.003 [-0.07, +0.08] | **+0.184 [+0.11, +0.26]** | +0.046 [-0.06, +0.15] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 15 | 340,212 (19,628) | 0.631 (L20) | 0.631 (L20) | 0.541 (L30) | 0.666 | 0.402 | 0.542 | **+0.090 [+0.06, +0.13]** | -0.035 [-0.11, +0.05] | **+0.229 [+0.15, +0.31]** | **+0.089 [+0.00, +0.17]** |
| Laguna XS.2 / Pro | correct | 1 | 774,424 (6,163) | 0.727 (L20) | 0.727 (L20) | 0.612 (L20) | 0.499 | 0.604 | 0.590 | **+0.115 [+0.06, +0.17]** | **+0.228 [+0.16, +0.29]** | **+0.124 [+0.02, +0.24]** | **+0.138 [+0.06, +0.21]** |
| Laguna XS.2 / Pro | correct | 2 | 774,234 (8,609) | 0.727 (L20) | 0.727 (L20) | 0.604 (L20) | 0.513 | 0.614 | 0.583 | **+0.123 [+0.05, +0.19]** | **+0.214 [+0.12, +0.29]** | **+0.113 [+0.01, +0.24]** | **+0.144 [+0.07, +0.21]** |
| Laguna XS.2 / Pro | correct | 3 | 774,049 (11,258) | 0.731 (L20) | 0.731 (L20) | 0.608 (L20) | 0.518 | 0.615 | 0.568 | **+0.124 [+0.05, +0.20]** | **+0.214 [+0.12, +0.29]** | **+0.116 [+0.02, +0.24]** | **+0.163 [+0.09, +0.23]** |
| Laguna XS.2 / Pro | correct | 5 | 774,148 (15,717) | 0.720 (L20) | 0.720 (L20) | 0.587 (L20) | 0.506 | 0.621 | 0.526 | **+0.132 [+0.07, +0.20]** | **+0.214 [+0.09, +0.30]** | **+0.099 [+0.01, +0.22]** | **+0.194 [+0.11, +0.25]** |
| Laguna XS.2 / Pro | correct | 10 | 774,154 (25,523) | 0.711 (L20) | 0.711 (L20) | 0.567 (L20) | 0.481 | 0.601 | 0.503 | **+0.144 [+0.08, +0.21]** | **+0.230 [+0.12, +0.32]** | **+0.109 [+0.01, +0.21]** | **+0.208 [+0.10, +0.29]** |
| Laguna XS.2 / Pro | correct | 15 | 774,170 (31,461) | 0.689 (L20) | 0.689 (L20) | 0.551 (L20) | 0.520 | 0.595 | 0.503 | **+0.138 [+0.06, +0.21]** | **+0.169 [+0.05, +0.27]** | +0.094 [-0.01, +0.20] | **+0.186 [+0.06, +0.28]** |
| Laguna XS.2 / Pro | reduces_failing | 1 | 732,420 (11,307) | 0.761 (L20) | 0.761 (L20) | 0.671 (L20) | 0.499 | 0.546 | 0.449 | **+0.090 [+0.06, +0.12]** | **+0.262 [+0.19, +0.33]** | **+0.215 [+0.15, +0.30]** | **+0.311 [+0.24, +0.39]** |
| Laguna XS.2 / Pro | reduces_failing | 2 | 732,224 (15,810) | 0.756 (L20) | 0.756 (L20) | 0.659 (L20) | 0.507 | 0.574 | 0.444 | **+0.097 [+0.06, +0.14]** | **+0.249 [+0.17, +0.32]** | **+0.182 [+0.11, +0.27]** | **+0.312 [+0.24, +0.40]** |
| Laguna XS.2 / Pro | reduces_failing | 3 | 732,038 (20,356) | 0.729 (L10) | 0.745 (L20) | 0.632 (L20) | 0.466 | 0.570 | 0.430 | **+0.097 [+0.05, +0.14]** | **+0.263 [+0.17, +0.35]** | **+0.159 [+0.07, +0.26]** | **+0.299 [+0.21, +0.40]** |
| Laguna XS.2 / Pro | reduces_failing | 5 | 732,138 (26,885) | 0.722 (L10) | 0.722 (L10) | 0.611 (L20) | 0.496 | 0.583 | 0.414 | **+0.111 [+0.07, +0.16]** | **+0.226 [+0.12, +0.32]** | **+0.139 [+0.05, +0.25]** | **+0.308 [+0.21, +0.42]** |
| Laguna XS.2 / Pro | reduces_failing | 10 | 732,143 (36,120) | 0.695 (L10) | 0.695 (L10) | 0.555 (L20) | 0.465 | 0.572 | 0.370 | **+0.140 [+0.09, +0.19]** | **+0.230 [+0.12, +0.34]** | **+0.123 [+0.03, +0.25]** | **+0.326 [+0.21, +0.44]** |
| Laguna XS.2 / Pro | reduces_failing | 15 | 732,153 (40,499) | 0.676 (L10) | 0.676 (L10) | 0.524 (L20) | 0.497 | 0.542 | 0.368 | **+0.152 [+0.10, +0.21]** | **+0.179 [+0.07, +0.28]** | **+0.134 [+0.03, +0.28]** | **+0.308 [+0.19, +0.42]** |
| Laguna XS.2 / Pro | has_regressions | 1 | 769,276 (6,315) | 0.814 (L30) | 0.814 (L30) | 0.690 (L20) | 0.537 | 0.579 | 0.478 | **+0.124 [+0.09, +0.15]** | **+0.277 [+0.23, +0.32]** | **+0.235 [+0.14, +0.33]** | **+0.336 [+0.19, +0.43]** |
| Laguna XS.2 / Pro | has_regressions | 2 | 769,086 (8,813) | 0.793 (L30) | 0.793 (L30) | 0.665 (L20) | 0.540 | 0.581 | 0.437 | **+0.128 [+0.09, +0.17]** | **+0.253 [+0.21, +0.29]** | **+0.212 [+0.12, +0.33]** | **+0.356 [+0.24, +0.44]** |
| Laguna XS.2 / Pro | has_regressions | 3 | 768,901 (10,658) | 0.780 (L30) | 0.780 (L30) | 0.661 (L20) | 0.525 | 0.563 | 0.433 | **+0.119 [+0.09, +0.15]** | **+0.255 [+0.21, +0.30]** | **+0.217 [+0.13, +0.32]** | **+0.347 [+0.22, +0.44]** |
| Laguna XS.2 / Pro | has_regressions | 5 | 769,000 (14,209) | 0.763 (L30) | 0.763 (L30) | 0.646 (L20) | 0.489 | 0.527 | 0.412 | **+0.117 [+0.08, +0.15]** | **+0.274 [+0.21, +0.32]** | **+0.236 [+0.15, +0.35]** | **+0.351 [+0.24, +0.44]** |
| Laguna XS.2 / Pro | has_regressions | 10 | 769,006 (20,188) | 0.756 (L30) | 0.756 (L30) | 0.641 (L20) | 0.516 | 0.520 | 0.445 | **+0.114 [+0.07, +0.16]** | **+0.240 [+0.16, +0.30]** | **+0.236 [+0.13, +0.37]** | **+0.311 [+0.18, +0.42]** |
| Laguna XS.2 / Pro | has_regressions | 15 | 769,022 (25,079) | 0.744 (L30) | 0.744 (L30) | 0.626 (L20) | 0.500 | 0.519 | 0.460 | **+0.118 [+0.08, +0.17]** | **+0.244 [+0.15, +0.33]** | **+0.225 [+0.12, +0.35]** | **+0.284 [+0.17, +0.39]** |
| Qwen3.6-35B-A3B / Pro | correct | 1 | 342,282 (7,328) | 0.800 (L30) | 0.799 (L20) | 0.672 (L30) | 0.599 | 0.633 | 0.522 | **+0.128 [+0.08, +0.17]** | **+0.202 [+0.16, +0.24]** | **+0.167 [+0.08, +0.24]** | **+0.279 [+0.16, +0.37]** |
| Qwen3.6-35B-A3B / Pro | correct | 2 | 342,270 (8,380) | 0.793 (L30) | 0.791 (L20) | 0.673 (L30) | 0.622 | 0.624 | 0.525 | **+0.121 [+0.08, +0.16]** | **+0.172 [+0.11, +0.24]** | **+0.169 [+0.08, +0.24]** | **+0.268 [+0.14, +0.36]** |
| Qwen3.6-35B-A3B / Pro | correct | 3 | 342,315 (9,277) | 0.789 (L30) | 0.790 (L20) | 0.664 (L30) | 0.601 | 0.620 | 0.523 | **+0.125 [+0.08, +0.16]** | **+0.188 [+0.10, +0.29]** | **+0.169 [+0.08, +0.24]** | **+0.266 [+0.14, +0.35]** |
| Qwen3.6-35B-A3B / Pro | correct | 5 | 342,321 (11,665) | 0.773 (L30) | 0.776 (L20) | 0.665 (L30) | 0.580 | 0.607 | 0.534 | **+0.108 [+0.06, +0.15]** | **+0.192 [+0.10, +0.27]** | **+0.166 [+0.08, +0.24]** | **+0.238 [+0.11, +0.34]** |
| Qwen3.6-35B-A3B / Pro | correct | 10 | 342,315 (18,853) | 0.731 (L30) | 0.737 (L20) | 0.653 (L30) | 0.568 | 0.577 | 0.536 | **+0.078 [+0.03, +0.12]** | **+0.163 [+0.08, +0.24]** | **+0.154 [+0.08, +0.22]** | **+0.196 [+0.05, +0.32]** |
| Qwen3.6-35B-A3B / Pro | correct | 15 | 342,285 (24,884) | 0.711 (L30) | 0.716 (L20) | 0.638 (L30) | 0.579 | 0.575 | 0.541 | **+0.073 [+0.02, +0.13]** | **+0.133 [+0.04, +0.23]** | **+0.136 [+0.07, +0.20]** | **+0.171 [+0.03, +0.31]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 1 | 340,086 (8,642) | 0.838 (L20) | 0.838 (L20) | 0.745 (L20) | 0.640 | 0.650 | 0.552 | **+0.092 [+0.08, +0.11]** | **+0.197 [+0.12, +0.27]** | **+0.188 [+0.11, +0.26]** | **+0.285 [+0.16, +0.39]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 2 | 340,074 (9,897) | 0.828 (L20) | 0.822 (L30) | 0.735 (L20) | 0.653 | 0.643 | 0.553 | **+0.093 [+0.08, +0.11]** | **+0.175 [+0.10, +0.24]** | **+0.185 [+0.11, +0.26]** | **+0.275 [+0.15, +0.38]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 3 | 340,119 (11,322) | 0.815 (L20) | 0.815 (L20) | 0.718 (L20) | 0.658 | 0.631 | 0.555 | **+0.097 [+0.08, +0.11]** | **+0.158 [+0.08, +0.25]** | **+0.185 [+0.11, +0.26]** | **+0.260 [+0.13, +0.37]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 5 | 340,125 (14,938) | 0.797 (L20) | 0.797 (L20) | 0.702 (L20) | 0.576 | 0.618 | 0.550 | **+0.094 [+0.07, +0.12]** | **+0.221 [+0.11, +0.30]** | **+0.178 [+0.10, +0.25]** | **+0.247 [+0.13, +0.34]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 10 | 340,119 (22,769) | 0.764 (L20) | 0.764 (L20) | 0.670 (L20) | 0.566 | 0.586 | 0.524 | **+0.094 [+0.05, +0.13]** | **+0.198 [+0.10, +0.28]** | **+0.178 [+0.11, +0.24]** | **+0.240 [+0.14, +0.33]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 15 | 340,089 (28,729) | 0.732 (L39) | 0.749 (L20) | 0.653 (L20) | 0.545 | 0.575 | 0.528 | **+0.078 [+0.02, +0.14]** | **+0.186 [+0.09, +0.27]** | **+0.157 [+0.09, +0.22]** | **+0.204 [+0.09, +0.31]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 1 | 358,139 (1,676) | 0.865 (L30) | 0.880 (L39) | 0.782 (L30) | 0.567 | 0.755 | 0.710 | **+0.084 [+0.06, +0.10]** | **+0.299 [+0.22, +0.35]** | +0.111 [-0.00, +0.37] | **+0.156 [+0.01, +0.31]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 2 | 358,127 (1,975) | 0.853 (L30) | 0.873 (L39) | 0.772 (L30) | 0.562 | 0.774 | 0.694 | **+0.081 [+0.05, +0.10]** | **+0.292 [+0.25, +0.35]** | +0.079 [-0.04, +0.32] | **+0.159 [+0.08, +0.31]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 3 | 358,172 (2,160) | 0.834 (L30) | 0.834 (L39) | 0.757 (L30) | 0.616 | 0.773 | 0.687 | **+0.077 [+0.03, +0.11]** | **+0.219 [+0.19, +0.25]** | +0.062 [-0.06, +0.31] | **+0.148 [+0.05, +0.31]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 5 | 358,178 (2,591) | 0.822 (L30) | 0.822 (L30) | 0.727 (L30) | 0.583 | 0.765 | 0.678 | **+0.095 [+0.04, +0.14]** | **+0.239 [+0.09, +0.36]** | +0.058 [-0.07, +0.30] | **+0.144 [+0.03, +0.25]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 10 | 358,172 (3,417) | 0.784 (L30) | 0.821 (L39) | 0.687 (L30) | 0.557 | 0.743 | 0.671 | **+0.096 [+0.03, +0.15]** | **+0.227 [+0.15, +0.28]** | +0.040 [-0.09, +0.30] | **+0.112 [+0.01, +0.21]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 15 | 358,142 (3,854) | 0.814 (L0) | 0.834 (L39) | 0.643 (L30) | 0.576 | 0.709 | 0.660 | **+0.171 [+0.08, +0.23]** | **+0.238 [+0.16, +0.33]** | **+0.104 [+0.02, +0.32]** | **+0.154 [+0.09, +0.26]** |

#### Phase confound, window max15, all test tokens: AUC for y_(t+k) (test split; differences with 95% task-bootstrap CI, bold if the CI excludes 0)

| Model / benchmark | Probe | k | n (pos) | real, val-best layer | real, test-best layer | k=0 probe (E2) | traj-shuffle | position, causal | position, rel_pos (non-causal) | real − k=0 probe | real − traj-shuffle | real − position causal | real − position rel_pos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | correct | 1 | 719,133 | 0.751 (L20) | 0.751 (L20) | 0.733 (L20) | 0.638 | 0.599 | 0.648 | **+0.018 [+0.00, +0.04]** | **+0.113 [+0.04, +0.19]** | **+0.152 [+0.05, +0.25]** | **+0.103 [+0.04, +0.17]** |
| Laguna XS.2 / Verified | correct | 2 | 719,130 | 0.741 (L20) | 0.741 (L20) | 0.720 (L20) | 0.619 | 0.589 | 0.639 | **+0.021 [+0.00, +0.04]** | **+0.122 [+0.04, +0.21]** | **+0.152 [+0.05, +0.25]** | **+0.102 [+0.04, +0.17]** |
| Laguna XS.2 / Verified | correct | 3 | 719,128 | 0.731 (L20) | 0.731 (L20) | 0.709 (L20) | 0.602 | 0.580 | 0.629 | **+0.022 [+0.00, +0.05]** | **+0.129 [+0.05, +0.22]** | **+0.152 [+0.06, +0.24]** | **+0.102 [+0.04, +0.17]** |
| Laguna XS.2 / Verified | correct | 5 | 719,161 | 0.712 (L20) | 0.712 (L20) | 0.688 (L20) | 0.576 | 0.564 | 0.609 | +0.024 [-0.00, +0.05] | **+0.136 [+0.05, +0.23]** | **+0.148 [+0.06, +0.23]** | **+0.103 [+0.03, +0.18]** |
| Laguna XS.2 / Verified | correct | 10 | 719,179 | 0.680 (L20) | 0.680 (L20) | 0.654 (L20) | 0.501 | 0.552 | 0.581 | +0.025 [-0.01, +0.06] | **+0.179 [+0.08, +0.27]** | **+0.128 [+0.05, +0.21]** | **+0.099 [+0.02, +0.18]** |
| Laguna XS.2 / Verified | correct | 15 | 719,236 | 0.663 (L20) | 0.663 (L20) | 0.633 (L20) | 0.464 | 0.548 | 0.564 | +0.030 [-0.01, +0.07] | **+0.199 [+0.10, +0.30]** | **+0.115 [+0.03, +0.19]** | **+0.099 [+0.02, +0.18]** |
| Laguna XS.2 / Verified | reduces_failing | 1 | 719,133 | 0.759 (L20) | 0.759 (L20) | 0.747 (L20) | 0.667 | 0.633 | 0.654 | +0.012 [-0.01, +0.03] | **+0.091 [+0.02, +0.17]** | **+0.126 [+0.03, +0.23]** | **+0.105 [+0.05, +0.17]** |
| Laguna XS.2 / Verified | reduces_failing | 2 | 719,130 | 0.744 (L20) | 0.744 (L20) | 0.731 (L20) | 0.654 | 0.623 | 0.644 | +0.013 [-0.01, +0.04] | **+0.091 [+0.02, +0.17]** | **+0.121 [+0.03, +0.23]** | **+0.100 [+0.04, +0.16]** |
| Laguna XS.2 / Verified | reduces_failing | 3 | 719,128 | 0.730 (L20) | 0.730 (L20) | 0.717 (L20) | 0.639 | 0.614 | 0.632 | +0.013 [-0.01, +0.04] | **+0.092 [+0.02, +0.17]** | **+0.117 [+0.02, +0.22]** | **+0.099 [+0.04, +0.16]** |
| Laguna XS.2 / Verified | reduces_failing | 5 | 719,161 | 0.705 (L20) | 0.705 (L20) | 0.693 (L20) | 0.614 | 0.598 | 0.610 | +0.012 [-0.01, +0.04] | **+0.090 [+0.02, +0.17]** | **+0.107 [+0.02, +0.21]** | **+0.095 [+0.03, +0.16]** |
| Laguna XS.2 / Verified | reduces_failing | 10 | 719,179 | 0.664 (L20) | 0.664 (L20) | 0.655 (L20) | 0.553 | 0.578 | 0.579 | +0.009 [-0.02, +0.04] | **+0.110 [+0.04, +0.19]** | **+0.085 [+0.00, +0.18]** | **+0.085 [+0.03, +0.15]** |
| Laguna XS.2 / Verified | reduces_failing | 15 | 719,236 | 0.635 (L20) | 0.635 (L20) | 0.630 (L20) | 0.522 | 0.576 | 0.561 | +0.006 [-0.03, +0.04] | **+0.113 [+0.04, +0.19]** | +0.059 [-0.02, +0.15] | **+0.074 [+0.02, +0.14]** |
| Laguna XS.2 / Verified | has_regressions | 1 | 719,133 | 0.664 (L20) | 0.664 (L20) | 0.661 (L20) | 0.609 | 0.656 | 0.614 | +0.003 [-0.01, +0.02] | **+0.055 [+0.01, +0.10]** | +0.008 [-0.05, +0.10] | +0.050 [-0.02, +0.14] |
| Laguna XS.2 / Verified | has_regressions | 2 | 719,130 | 0.654 (L20) | 0.654 (L20) | 0.655 (L20) | 0.596 | 0.646 | 0.604 | -0.001 [-0.01, +0.01] | **+0.058 [+0.01, +0.11]** | +0.008 [-0.05, +0.09] | +0.050 [-0.01, +0.14] |
| Laguna XS.2 / Verified | has_regressions | 3 | 719,128 | 0.643 (L20) | 0.643 (L20) | 0.646 (L20) | 0.588 | 0.633 | 0.590 | -0.003 [-0.01, +0.02] | **+0.055 [+0.01, +0.11]** | +0.010 [-0.05, +0.09] | +0.053 [-0.01, +0.14] |
| Laguna XS.2 / Verified | has_regressions | 5 | 719,161 | 0.627 (L20) | 0.627 (L20) | 0.629 (L20) | 0.587 | 0.610 | 0.567 | -0.002 [-0.02, +0.02] | +0.040 [-0.01, +0.09] | +0.017 [-0.04, +0.09] | +0.061 [-0.00, +0.14] |
| Laguna XS.2 / Verified | has_regressions | 10 | 719,179 | 0.602 (L20) | 0.602 (L20) | 0.608 (L20) | 0.509 | 0.576 | 0.533 | -0.006 [-0.02, +0.02] | **+0.093 [+0.03, +0.16]** | +0.026 [-0.03, +0.10] | **+0.069 [+0.01, +0.15]** |
| Laguna XS.2 / Verified | has_regressions | 15 | 719,236 | 0.586 (L20) | 0.586 (L20) | 0.593 (L20) | 0.502 | 0.556 | 0.516 | -0.007 [-0.04, +0.02] | **+0.084 [+0.02, +0.15]** | +0.031 [-0.02, +0.10] | **+0.070 [+0.01, +0.14]** |
| Qwen3.6-35B-A3B / Verified | correct | 1 | 353,824 | 0.776 (L30) | 0.776 (L30) | 0.737 (L30) | 0.708 | 0.631 | 0.702 | **+0.038 [+0.01, +0.08]** | **+0.068 [+0.02, +0.14]** | **+0.145 [+0.04, +0.30]** | **+0.074 [+0.02, +0.15]** |
| Qwen3.6-35B-A3B / Verified | correct | 2 | 353,809 | 0.765 (L30) | 0.765 (L30) | 0.713 (L30) | 0.692 | 0.624 | 0.689 | **+0.051 [+0.02, +0.10]** | **+0.073 [+0.02, +0.15]** | **+0.140 [+0.05, +0.28]** | **+0.076 [+0.02, +0.15]** |
| Qwen3.6-35B-A3B / Verified | correct | 3 | 353,800 | 0.758 (L30) | 0.758 (L30) | 0.696 (L30) | 0.677 | 0.616 | 0.681 | **+0.062 [+0.03, +0.11]** | **+0.081 [+0.02, +0.17]** | **+0.142 [+0.05, +0.27]** | **+0.077 [+0.02, +0.16]** |
| Qwen3.6-35B-A3B / Verified | correct | 5 | 353,744 | 0.747 (L30) | 0.747 (L30) | 0.657 (L30) | 0.662 | 0.606 | 0.664 | **+0.090 [+0.05, +0.14]** | **+0.085 [+0.02, +0.17]** | **+0.141 [+0.05, +0.24]** | **+0.083 [+0.02, +0.16]** |
| Qwen3.6-35B-A3B / Verified | correct | 10 | 353,575 | 0.703 (L30) | 0.703 (L30) | 0.599 (L30) | 0.560 | 0.607 | 0.624 | **+0.104 [+0.06, +0.15]** | **+0.143 [+0.06, +0.24]** | **+0.096 [+0.02, +0.18]** | **+0.079 [+0.00, +0.16]** |
| Qwen3.6-35B-A3B / Verified | correct | 15 | 353,537 | 0.674 (L30) | 0.674 (L30) | 0.564 (L30) | 0.519 | 0.605 | 0.602 | **+0.110 [+0.07, +0.16]** | **+0.155 [+0.07, +0.25]** | **+0.069 [+0.00, +0.14]** | +0.073 [-0.01, +0.16] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 1 | 353,824 | 0.810 (L20) | 0.819 (L30) | 0.778 (L20) | 0.733 | 0.605 | 0.722 | **+0.032 [+0.01, +0.05]** | **+0.077 [+0.02, +0.14]** | **+0.204 [+0.12, +0.31]** | **+0.088 [+0.03, +0.15]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 2 | 353,809 | 0.797 (L20) | 0.804 (L30) | 0.751 (L20) | 0.713 | 0.588 | 0.706 | **+0.046 [+0.03, +0.07]** | **+0.084 [+0.03, +0.15]** | **+0.209 [+0.12, +0.32]** | **+0.091 [+0.03, +0.16]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3 | 353,800 | 0.789 (L30) | 0.789 (L30) | 0.726 (L20) | 0.703 | 0.580 | 0.695 | **+0.063 [+0.04, +0.09]** | **+0.086 [+0.03, +0.14]** | **+0.209 [+0.13, +0.31]** | **+0.094 [+0.03, +0.16]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 5 | 353,744 | 0.762 (L30) | 0.762 (L30) | 0.684 (L20) | 0.685 | 0.560 | 0.674 | **+0.078 [+0.05, +0.11]** | **+0.077 [+0.02, +0.14]** | **+0.202 [+0.12, +0.29]** | **+0.088 [+0.02, +0.16]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 10 | 353,575 | 0.694 (L30) | 0.694 (L30) | 0.616 (L20) | 0.565 | 0.535 | 0.631 | **+0.078 [+0.05, +0.12]** | **+0.129 [+0.07, +0.19]** | **+0.159 [+0.08, +0.24]** | +0.063 [-0.01, +0.14] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 15 | 353,537 | 0.650 (L30) | 0.650 (L30) | 0.576 (L20) | 0.529 | 0.522 | 0.604 | **+0.073 [+0.04, +0.12]** | **+0.120 [+0.05, +0.19]** | **+0.128 [+0.06, +0.20]** | +0.046 [-0.03, +0.13] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 1 | 353,824 | 0.811 (L20) | 0.811 (L20) | 0.762 (L30) | 0.718 | 0.593 | 0.742 | **+0.049 [+0.03, +0.07]** | **+0.094 [+0.05, +0.13]** | **+0.218 [+0.15, +0.29]** | +0.069 [-0.00, +0.14] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 2 | 353,809 | 0.775 (L30) | 0.798 (L20) | 0.751 (L30) | 0.638 | 0.578 | 0.735 | **+0.024 [+0.00, +0.05]** | **+0.137 [+0.05, +0.22]** | **+0.197 [+0.13, +0.27]** | +0.040 [-0.03, +0.11] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 3 | 353,800 | 0.765 (L30) | 0.791 (L20) | 0.742 (L30) | 0.627 | 0.565 | 0.722 | +0.022 [-0.00, +0.06] | **+0.137 [+0.06, +0.22]** | **+0.199 [+0.13, +0.27]** | +0.043 [-0.03, +0.12] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 5 | 353,744 | 0.718 (L30) | 0.755 (L20) | 0.712 (L30) | 0.561 | 0.525 | 0.687 | +0.007 [-0.02, +0.05] | **+0.157 [+0.05, +0.26]** | **+0.193 [+0.12, +0.27]** | +0.031 [-0.04, +0.11] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 10 | 353,575 | 0.645 (L30) | 0.692 (L20) | 0.660 (L30) | 0.637 | 0.461 | 0.616 | -0.016 [-0.05, +0.04] | +0.008 [-0.07, +0.11] | **+0.184 [+0.10, +0.27]** | +0.029 [-0.05, +0.12] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 15 | 353,537 | 0.642 (L20) | 0.642 (L20) | 0.629 (L30) | 0.655 | 0.412 | 0.577 | +0.013 [-0.02, +0.06] | -0.013 [-0.08, +0.08] | **+0.230 [+0.15, +0.31]** | +0.065 [-0.02, +0.15] |
| Laguna XS.2 / Pro | correct | 1 | 804,866 | 0.791 (L20) | 0.791 (L20) | 0.772 (L20) | 0.599 | 0.677 | 0.656 | **+0.018 [+0.00, +0.05]** | **+0.192 [+0.13, +0.24]** | **+0.114 [+0.06, +0.21]** | **+0.135 [+0.03, +0.19]** |
| Laguna XS.2 / Pro | correct | 2 | 804,686 | 0.791 (L20) | 0.791 (L20) | 0.761 (L20) | 0.601 | 0.671 | 0.650 | **+0.030 [+0.01, +0.07]** | **+0.190 [+0.13, +0.23]** | **+0.120 [+0.06, +0.21]** | **+0.141 [+0.03, +0.19]** |
| Laguna XS.2 / Pro | correct | 3 | 804,688 | 0.787 (L20) | 0.787 (L20) | 0.750 (L20) | 0.596 | 0.664 | 0.643 | **+0.037 [+0.01, +0.08]** | **+0.191 [+0.13, +0.24]** | **+0.123 [+0.06, +0.21]** | **+0.144 [+0.05, +0.20]** |
| Laguna XS.2 / Pro | correct | 5 | 804,804 | 0.778 (L20) | 0.778 (L20) | 0.728 (L20) | 0.607 | 0.656 | 0.619 | **+0.049 [+0.02, +0.10]** | **+0.171 [+0.10, +0.24]** | **+0.121 [+0.06, +0.21]** | **+0.158 [+0.07, +0.21]** |
| Laguna XS.2 / Pro | correct | 10 | 804,817 | 0.762 (L20) | 0.762 (L20) | 0.693 (L20) | 0.575 | 0.634 | 0.591 | **+0.068 [+0.03, +0.13]** | **+0.186 [+0.11, +0.26]** | **+0.128 [+0.06, +0.20]** | **+0.171 [+0.08, +0.23]** |
| Laguna XS.2 / Pro | correct | 15 | 804,833 | 0.744 (L20) | 0.744 (L20) | 0.672 (L20) | 0.537 | 0.622 | 0.582 | **+0.072 [+0.03, +0.14]** | **+0.207 [+0.09, +0.28]** | **+0.122 [+0.04, +0.19]** | **+0.163 [+0.06, +0.23]** |
| Laguna XS.2 / Pro | reduces_failing | 1 | 804,866 | 0.788 (L20) | 0.788 (L20) | 0.782 (L20) | 0.610 | 0.654 | 0.601 | +0.006 [-0.01, +0.02] | **+0.178 [+0.09, +0.25]** | **+0.134 [+0.07, +0.22]** | **+0.187 [+0.12, +0.25]** |
| Laguna XS.2 / Pro | reduces_failing | 2 | 804,686 | 0.782 (L20) | 0.782 (L20) | 0.775 (L20) | 0.594 | 0.646 | 0.593 | +0.008 [-0.01, +0.03] | **+0.188 [+0.09, +0.27]** | **+0.136 [+0.07, +0.22]** | **+0.189 [+0.13, +0.25]** |
| Laguna XS.2 / Pro | reduces_failing | 3 | 804,688 | 0.775 (L10) | 0.775 (L20) | 0.760 (L20) | 0.548 | 0.634 | 0.582 | +0.014 [-0.02, +0.05] | **+0.227 [+0.13, +0.32]** | **+0.141 [+0.05, +0.23]** | **+0.193 [+0.11, +0.27]** |
| Laguna XS.2 / Pro | reduces_failing | 5 | 804,804 | 0.767 (L10) | 0.767 (L10) | 0.746 (L20) | 0.582 | 0.628 | 0.565 | +0.021 [-0.02, +0.06] | **+0.184 [+0.09, +0.27]** | **+0.139 [+0.04, +0.23]** | **+0.202 [+0.12, +0.27]** |
| Laguna XS.2 / Pro | reduces_failing | 10 | 804,817 | 0.751 (L10) | 0.751 (L10) | 0.715 (L20) | 0.542 | 0.604 | 0.533 | +0.037 [-0.01, +0.08] | **+0.210 [+0.11, +0.30]** | **+0.147 [+0.04, +0.25]** | **+0.219 [+0.14, +0.29]** |
| Laguna XS.2 / Pro | reduces_failing | 15 | 804,833 | 0.735 (L10) | 0.735 (L10) | 0.692 (L20) | 0.568 | 0.580 | 0.523 | +0.043 [-0.01, +0.09] | **+0.167 [+0.07, +0.26]** | **+0.155 [+0.04, +0.26]** | **+0.212 [+0.13, +0.29]** |
| Laguna XS.2 / Pro | has_regressions | 1 | 804,866 | 0.774 (L30) | 0.774 (L30) | 0.757 (L20) | 0.540 | 0.693 | 0.621 | +0.017 [-0.01, +0.07] | **+0.235 [+0.17, +0.29]** | **+0.082 [+0.01, +0.22]** | **+0.153 [+0.10, +0.20]** |
| Laguna XS.2 / Pro | has_regressions | 2 | 804,686 | 0.771 (L30) | 0.771 (L30) | 0.748 (L20) | 0.540 | 0.687 | 0.604 | +0.022 [-0.01, +0.08] | **+0.230 [+0.16, +0.30]** | **+0.084 [+0.01, +0.23]** | **+0.167 [+0.12, +0.22]** |
| Laguna XS.2 / Pro | has_regressions | 3 | 804,688 | 0.775 (L30) | 0.775 (L30) | 0.744 (L20) | 0.533 | 0.678 | 0.596 | **+0.031 [+0.00, +0.08]** | **+0.242 [+0.17, +0.30]** | **+0.097 [+0.02, +0.24]** | **+0.179 [+0.13, +0.23]** |
| Laguna XS.2 / Pro | has_regressions | 5 | 804,804 | 0.771 (L30) | 0.771 (L30) | 0.732 (L20) | 0.548 | 0.662 | 0.577 | **+0.038 [+0.01, +0.09]** | **+0.222 [+0.16, +0.27]** | **+0.109 [+0.03, +0.26]** | **+0.194 [+0.14, +0.24]** |
| Laguna XS.2 / Pro | has_regressions | 10 | 804,817 | 0.762 (L30) | 0.762 (L30) | 0.723 (L20) | 0.544 | 0.647 | 0.562 | **+0.039 [+0.01, +0.09]** | **+0.218 [+0.14, +0.27]** | **+0.115 [+0.02, +0.28]** | **+0.200 [+0.15, +0.26]** |
| Laguna XS.2 / Pro | has_regressions | 15 | 804,833 | 0.746 (L30) | 0.746 (L30) | 0.707 (L20) | 0.479 | 0.642 | 0.552 | **+0.038 [+0.00, +0.10]** | **+0.266 [+0.18, +0.34]** | +0.103 [-0.00, +0.28] | **+0.194 [+0.13, +0.26]** |
| Qwen3.6-35B-A3B / Pro | correct | 1 | 360,199 | 0.833 (L30) | 0.836 (L20) | 0.808 (L30) | 0.717 | 0.703 | 0.681 | +0.026 [-0.00, +0.06] | **+0.117 [+0.08, +0.15]** | **+0.130 [+0.06, +0.20]** | **+0.153 [+0.06, +0.22]** |
| Qwen3.6-35B-A3B / Pro | correct | 2 | 360,206 | 0.826 (L30) | 0.828 (L20) | 0.803 (L30) | 0.699 | 0.692 | 0.675 | +0.023 [-0.00, +0.06] | **+0.127 [+0.05, +0.19]** | **+0.134 [+0.05, +0.21]** | **+0.151 [+0.06, +0.22]** |
| Qwen3.6-35B-A3B / Pro | correct | 3 | 360,330 | 0.819 (L30) | 0.824 (L20) | 0.795 (L30) | 0.699 | 0.684 | 0.669 | +0.023 [-0.01, +0.06] | **+0.120 [+0.03, +0.22]** | **+0.135 [+0.05, +0.21]** | **+0.150 [+0.06, +0.22]** |
| Qwen3.6-35B-A3B / Pro | correct | 5 | 360,379 | 0.805 (L30) | 0.811 (L20) | 0.784 (L30) | 0.638 | 0.668 | 0.662 | +0.021 [-0.01, +0.06] | **+0.167 [+0.11, +0.22]** | **+0.138 [+0.06, +0.21]** | **+0.144 [+0.05, +0.21]** |
| Qwen3.6-35B-A3B / Pro | correct | 10 | 360,478 | 0.769 (L30) | 0.775 (L20) | 0.754 (L30) | 0.618 | 0.622 | 0.637 | +0.015 [-0.01, +0.05] | **+0.151 [+0.07, +0.22]** | **+0.147 [+0.07, +0.21]** | **+0.132 [+0.02, +0.22]** |
| Qwen3.6-35B-A3B / Pro | correct | 15 | 360,563 | 0.740 (L30) | 0.750 (L20) | 0.732 (L30) | 0.597 | 0.600 | 0.624 | +0.008 [-0.03, +0.06] | **+0.142 [+0.06, +0.21]** | **+0.140 [+0.07, +0.20]** | **+0.116 [+0.00, +0.22]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 1 | 360,199 | 0.823 (L20) | 0.823 (L20) | 0.815 (L20) | 0.733 | 0.755 | 0.685 | +0.008 [-0.01, +0.04] | +0.090 [-0.01, +0.20] | +0.068 [-0.00, +0.15] | **+0.138 [+0.04, +0.22]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 2 | 360,206 | 0.812 (L20) | 0.812 (L30) | 0.808 (L20) | 0.717 | 0.744 | 0.677 | +0.003 [-0.02, +0.04] | **+0.095 [+0.01, +0.18]** | **+0.068 [+0.00, +0.15]** | **+0.135 [+0.03, +0.22]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 3 | 360,330 | 0.803 (L20) | 0.803 (L20) | 0.798 (L20) | 0.736 | 0.729 | 0.668 | +0.005 [-0.02, +0.05] | +0.067 [-0.04, +0.19] | **+0.074 [+0.01, +0.15]** | **+0.135 [+0.03, +0.22]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 5 | 360,379 | 0.787 (L20) | 0.787 (L20) | 0.783 (L20) | 0.629 | 0.705 | 0.647 | +0.003 [-0.03, +0.06] | **+0.158 [+0.05, +0.25]** | **+0.082 [+0.02, +0.16]** | **+0.140 [+0.04, +0.23]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 10 | 360,478 | 0.760 (L20) | 0.760 (L20) | 0.751 (L20) | 0.611 | 0.648 | 0.599 | +0.009 [-0.04, +0.07] | **+0.149 [+0.06, +0.23]** | **+0.112 [+0.06, +0.18]** | **+0.161 [+0.06, +0.25]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 15 | 360,563 | 0.711 (L39) | 0.742 (L20) | 0.732 (L20) | 0.558 | 0.617 | 0.577 | -0.021 [-0.11, +0.08] | +0.153 [-0.01, +0.28] | **+0.094 [+0.02, +0.17]** | +0.134 [-0.00, +0.25] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 1 | 360,199 | 0.866 (L30) | 0.884 (L39) | 0.823 (L30) | 0.602 | 0.767 | 0.716 | **+0.044 [+0.02, +0.07]** | **+0.264 [+0.21, +0.31]** | +0.100 [-0.02, +0.27] | **+0.151 [+0.06, +0.22]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 2 | 360,206 | 0.862 (L30) | 0.885 (L39) | 0.814 (L30) | 0.596 | 0.777 | 0.711 | **+0.048 [+0.03, +0.07]** | **+0.266 [+0.21, +0.33]** | +0.085 [-0.04, +0.27] | **+0.151 [+0.08, +0.22]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 3 | 360,330 | 0.846 (L30) | 0.858 (L39) | 0.801 (L30) | 0.645 | 0.779 | 0.698 | **+0.045 [+0.02, +0.07]** | **+0.201 [+0.17, +0.25]** | +0.067 [-0.07, +0.26] | **+0.148 [+0.07, +0.25]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 5 | 360,379 | 0.835 (L30) | 0.835 (L30) | 0.775 (L30) | 0.610 | 0.775 | 0.693 | **+0.060 [+0.02, +0.09]** | **+0.225 [+0.10, +0.35]** | +0.061 [-0.09, +0.27] | **+0.142 [+0.04, +0.23]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 10 | 360,478 | 0.806 (L30) | 0.845 (L39) | 0.731 (L30) | 0.585 | 0.764 | 0.673 | **+0.076 [+0.02, +0.11]** | **+0.221 [+0.13, +0.27]** | +0.043 [-0.10, +0.30] | **+0.133 [+0.02, +0.20]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 15 | 360,563 | 0.819 (L0) | 0.853 (L39) | 0.688 (L30) | 0.588 | 0.735 | 0.647 | **+0.130 [+0.06, +0.22]** | **+0.231 [+0.16, +0.32]** | **+0.083 [+0.00, +0.32]** | **+0.172 [+0.10, +0.26]** |

#### Phase confound, window max15, within y_t = 1: AUC for y_(t+k) (test split; differences with 95% task-bootstrap CI, bold if the CI excludes 0)

| Model / benchmark | Probe | k | n (pos) | real, val-best layer | real, test-best layer | k=0 probe (E2) | traj-shuffle | position, causal | position, rel_pos (non-causal) | real − k=0 probe | real − traj-shuffle | real − position causal | real − position rel_pos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | correct | 1 | 196,941 (193,052) | 0.726 (L20) | 0.726 (L20) | 0.819 (L20) | 0.431 | 0.490 | 0.543 | **-0.093 [-0.14, -0.07]** | **+0.295 [+0.15, +0.40]** | **+0.237 [+0.11, +0.37]** | **+0.183 [+0.09, +0.36]** |
| Laguna XS.2 / Verified | correct | 2 | 196,941 (192,718) | 0.723 (L20) | 0.723 (L20) | 0.822 (L20) | 0.444 | 0.516 | 0.530 | **-0.099 [-0.14, -0.08]** | **+0.279 [+0.17, +0.37]** | **+0.207 [+0.09, +0.35]** | **+0.193 [+0.09, +0.36]** |
| Laguna XS.2 / Verified | correct | 3 | 196,941 (192,393) | 0.705 (L20) | 0.705 (L20) | 0.785 (L20) | 0.474 | 0.545 | 0.474 | **-0.080 [-0.11, -0.06]** | **+0.231 [+0.13, +0.32]** | **+0.160 [+0.04, +0.32]** | **+0.231 [+0.13, +0.39]** |
| Laguna XS.2 / Verified | correct | 5 | 196,941 (190,746) | 0.680 (L20) | 0.680 (L20) | 0.750 (L20) | 0.482 | 0.549 | 0.465 | **-0.070 [-0.11, -0.04]** | **+0.198 [+0.10, +0.29]** | **+0.131 [+0.01, +0.27]** | **+0.215 [+0.11, +0.33]** |
| Laguna XS.2 / Verified | correct | 10 | 196,941 (189,153) | 0.632 (L20) | 0.632 (L20) | 0.681 (L20) | 0.467 | 0.550 | 0.491 | **-0.050 [-0.10, -0.01]** | **+0.164 [+0.05, +0.25]** | +0.082 [-0.02, +0.19] | **+0.140 [+0.04, +0.24]** |
| Laguna XS.2 / Verified | correct | 15 | 196,941 (189,160) | 0.612 (L20) | 0.612 (L20) | 0.661 (L20) | 0.468 | 0.565 | 0.532 | **-0.049 [-0.08, -0.01]** | **+0.144 [+0.03, +0.25]** | +0.047 [-0.04, +0.12] | +0.080 [-0.03, +0.19] |
| Laguna XS.2 / Verified | reduces_failing | 1 | 220,520 (215,837) | 0.609 (L20) | 0.609 (L20) | 0.725 (L20) | 0.462 | 0.450 | 0.577 | **-0.116 [-0.17, -0.09]** | +0.146 [-0.02, +0.28] | +0.159 [-0.02, +0.30] | +0.032 [-0.06, +0.17] |
| Laguna XS.2 / Verified | reduces_failing | 2 | 220,520 (215,522) | 0.609 (L20) | 0.609 (L20) | 0.724 (L20) | 0.479 | 0.492 | 0.569 | **-0.115 [-0.16, -0.09]** | +0.130 [-0.00, +0.25] | +0.116 [-0.04, +0.25] | +0.040 [-0.05, +0.17] |
| Laguna XS.2 / Verified | reduces_failing | 3 | 220,520 (215,100) | 0.603 (L20) | 0.603 (L20) | 0.701 (L20) | 0.475 | 0.520 | 0.541 | **-0.098 [-0.14, -0.07]** | **+0.128 [+0.00, +0.24]** | +0.084 [-0.06, +0.22] | +0.062 [-0.04, +0.19] |
| Laguna XS.2 / Verified | reduces_failing | 5 | 220,520 (213,531) | 0.587 (L20) | 0.587 (L20) | 0.690 (L20) | 0.510 | 0.545 | 0.553 | **-0.103 [-0.15, -0.07]** | +0.077 [-0.04, +0.19] | +0.042 [-0.09, +0.17] | +0.034 [-0.09, +0.16] |
| Laguna XS.2 / Verified | reduces_failing | 10 | 220,520 (212,172) | 0.588 (L20) | 0.588 (L20) | 0.664 (L20) | 0.502 | 0.605 | 0.557 | **-0.076 [-0.15, -0.02]** | +0.086 [-0.03, +0.19] | -0.017 [-0.12, +0.11] | +0.031 [-0.11, +0.17] |
| Laguna XS.2 / Verified | reduces_failing | 15 | 220,520 (212,577) | 0.587 (L20) | 0.587 (L20) | 0.664 (L20) | 0.498 | 0.586 | 0.554 | **-0.078 [-0.14, -0.03]** | +0.089 [-0.02, +0.18] | +0.001 [-0.10, +0.11] | +0.033 [-0.10, +0.17] |
| Laguna XS.2 / Verified | has_regressions | 1 | 78,993 (74,129) | 0.309 (L20) | 0.309 (L20) | 0.378 (L20) | 0.455 | 0.625 | 0.493 | **-0.070 [-0.09, -0.05]** | **-0.146 [-0.20, -0.09]** | **-0.316 [-0.42, -0.18]** | **-0.184 [-0.28, -0.04]** |
| Laguna XS.2 / Verified | has_regressions | 2 | 78,993 (72,438) | 0.355 (L20) | 0.355 (L20) | 0.404 (L20) | 0.462 | 0.576 | 0.461 | **-0.049 [-0.06, -0.03]** | **-0.107 [-0.16, -0.05]** | **-0.221 [-0.33, -0.09]** | -0.106 [-0.20, +0.03] |
| Laguna XS.2 / Verified | has_regressions | 3 | 78,993 (71,256) | 0.377 (L20) | 0.377 (L20) | 0.416 (L20) | 0.488 | 0.580 | 0.453 | **-0.039 [-0.06, -0.02]** | **-0.111 [-0.17, -0.05]** | **-0.203 [-0.32, -0.06]** | -0.075 [-0.17, +0.06] |
| Laguna XS.2 / Verified | has_regressions | 5 | 78,993 (68,903) | 0.393 (L20) | 0.393 (L20) | 0.426 (L20) | 0.503 | 0.584 | 0.458 | **-0.033 [-0.06, -0.01]** | **-0.109 [-0.17, -0.05]** | **-0.190 [-0.31, -0.06]** | -0.064 [-0.17, +0.06] |
| Laguna XS.2 / Verified | has_regressions | 10 | 78,993 (64,883) | 0.425 (L20) | 0.425 (L20) | 0.451 (L20) | 0.535 | 0.567 | 0.427 | -0.026 [-0.08, +0.01] | **-0.110 [-0.23, -0.02]** | **-0.142 [-0.29, -0.00]** | -0.003 [-0.11, +0.12] |
| Laguna XS.2 / Verified | has_regressions | 15 | 78,993 (60,866) | 0.423 (L20) | 0.423 (L20) | 0.449 (L20) | 0.541 | 0.539 | 0.430 | -0.026 [-0.09, +0.02] | **-0.117 [-0.23, -0.02]** | -0.116 [-0.27, +0.01] | -0.007 [-0.09, +0.08] |
| Qwen3.6-35B-A3B / Verified | correct | 1 | 31,527 (31,264) | 0.553 (L30) | 0.553 (L30) | 0.758 (L30) | 0.463 | 0.587 | 0.789 | **-0.206 [-0.26, -0.14]** | +0.090 [-0.02, +0.26] | -0.034 [-0.31, +0.10] | -0.236 [-0.36, +0.01] |
| Qwen3.6-35B-A3B / Verified | correct | 2 | 31,527 (31,175) | 0.583 (L30) | 0.583 (L30) | 0.758 (L30) | 0.481 | 0.584 | 0.756 | **-0.175 [-0.21, -0.14]** | +0.102 [-0.04, +0.23] | -0.001 [-0.36, +0.13] | -0.173 [-0.33, +0.02] |
| Qwen3.6-35B-A3B / Verified | correct | 3 | 31,527 (31,036) | 0.615 (L30) | 0.615 (L30) | 0.757 (L30) | 0.522 | 0.532 | 0.700 | **-0.141 [-0.19, -0.10]** | +0.093 [-0.14, +0.21] | +0.083 [-0.43, +0.20] | -0.084 [-0.24, +0.03] |
| Qwen3.6-35B-A3B / Verified | correct | 5 | 31,527 (30,899) | 0.648 (L30) | 0.648 (L30) | 0.752 (L30) | 0.573 | 0.584 | 0.711 | -0.104 [-0.24, +0.04] | +0.075 [-0.19, +0.15] | +0.065 [-0.37, +0.20] | -0.062 [-0.33, +0.11] |
| Qwen3.6-35B-A3B / Verified | correct | 10 | 31,527 (30,752) | 0.656 (L30) | 0.656 (L30) | 0.748 (L30) | 0.489 | 0.646 | 0.601 | -0.092 [-0.22, +0.02] | +0.167 [-0.16, +0.29] | +0.010 [-0.41, +0.18] | +0.055 [-0.24, +0.34] |
| Qwen3.6-35B-A3B / Verified | correct | 15 | 31,527 (30,542) | 0.587 (L30) | 0.587 (L30) | 0.653 (L30) | 0.426 | 0.519 | 0.572 | -0.066 [-0.35, +0.09] | +0.161 [-0.23, +0.28] | +0.068 [-0.35, +0.24] | +0.014 [-0.33, +0.11] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 1 | 41,229 (40,578) | 0.410 (L20) | 0.387 (L30) | 0.621 (L20) | 0.371 | 0.559 | 0.575 | **-0.211 [-0.25, -0.18]** | +0.039 [-0.15, +0.19] | -0.149 [-0.35, +0.06] | **-0.165 [-0.37, -0.00]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 2 | 41,229 (40,296) | 0.465 (L20) | 0.463 (L30) | 0.635 (L20) | 0.373 | 0.578 | 0.557 | **-0.170 [-0.22, -0.14]** | +0.092 [-0.09, +0.25] | -0.113 [-0.29, +0.09] | -0.092 [-0.32, +0.06] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3 | 41,229 (40,053) | 0.503 (L30) | 0.503 (L30) | 0.634 (L20) | 0.394 | 0.571 | 0.579 | **-0.131 [-0.24, -0.08]** | +0.109 [-0.13, +0.27] | -0.068 [-0.24, +0.10] | -0.076 [-0.31, +0.08] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 5 | 41,229 (39,676) | 0.493 (L30) | 0.493 (L30) | 0.625 (L20) | 0.444 | 0.581 | 0.626 | **-0.132 [-0.21, -0.10]** | +0.049 [-0.17, +0.19] | -0.088 [-0.27, +0.09] | **-0.133 [-0.29, -0.02]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 10 | 41,229 (39,536) | 0.506 (L30) | 0.506 (L30) | 0.641 (L20) | 0.453 | 0.614 | 0.612 | **-0.135 [-0.23, -0.09]** | +0.053 [-0.15, +0.19] | -0.107 [-0.32, +0.09] | -0.105 [-0.32, +0.03] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 15 | 41,229 (39,524) | 0.527 (L30) | 0.527 (L30) | 0.649 (L20) | 0.496 | 0.616 | 0.578 | **-0.123 [-0.22, -0.06]** | +0.031 [-0.17, +0.18] | -0.089 [-0.30, +0.09] | -0.051 [-0.38, +0.12] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 1 | 13,325 (13,163) | 0.315 (L20) | 0.315 (L20) | 0.460 (L30) | 0.359 | 0.419 | 0.540 | **-0.145 [-0.27, -0.08]** | -0.044 [-0.32, +0.08] | -0.104 [-0.49, +0.19] | -0.225 [-0.56, +0.05] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 2 | 13,325 (13,104) | 0.341 (L30) | 0.329 (L20) | 0.460 (L30) | 0.408 | 0.398 | 0.561 | **-0.120 [-0.20, -0.02]** | -0.067 [-0.27, +0.14] | -0.057 [-0.33, +0.22] | -0.221 [-0.46, +0.06] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 3 | 13,325 (12,941) | 0.325 (L30) | 0.304 (L20) | 0.383 (L30) | 0.447 | 0.400 | 0.495 | -0.058 [-0.16, +0.01] | -0.122 [-0.29, +0.07] | -0.075 [-0.33, +0.19] | -0.170 [-0.37, +0.08] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 5 | 13,325 (12,485) | 0.330 (L30) | 0.341 (L20) | 0.377 (L30) | 0.416 | 0.308 | 0.513 | -0.047 [-0.13, +0.01] | -0.086 [-0.26, +0.11] | +0.022 [-0.17, +0.23] | -0.184 [-0.37, +0.17] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 10 | 13,325 (11,337) | 0.351 (L30) | 0.376 (L20) | 0.399 (L30) | 0.519 | 0.266 | 0.538 | -0.048 [-0.07, +0.01] | -0.168 [-0.28, +0.05] | +0.085 [-0.08, +0.20] | -0.187 [-0.40, +0.16] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 15 | 13,325 (10,365) | 0.391 (L20) | 0.391 (L20) | 0.407 (L30) | 0.543 | 0.210 | 0.456 | -0.016 [-0.07, +0.05] | -0.152 [-0.26, +0.07] | **+0.181 [+0.04, +0.33]** | -0.065 [-0.23, +0.13] |
| Laguna XS.2 / Pro | correct | 1 | 30,234 (28,784) | 0.634 (L20) | 0.634 (L20) | 0.658 (L20) | 0.429 | 0.415 | 0.513 | -0.024 [-0.12, +0.02] | **+0.205 [+0.11, +0.30]** | +0.219 [-0.00, +0.38] | +0.121 [-0.07, +0.25] |
| Laguna XS.2 / Pro | correct | 2 | 30,234 (28,179) | 0.640 (L20) | 0.640 (L20) | 0.663 (L20) | 0.423 | 0.435 | 0.530 | -0.023 [-0.11, +0.00] | **+0.217 [+0.08, +0.27]** | +0.204 [-0.01, +0.37] | +0.110 [-0.08, +0.24] |
| Laguna XS.2 / Pro | correct | 3 | 30,234 (27,918) | 0.624 (L20) | 0.624 (L20) | 0.633 (L20) | 0.442 | 0.440 | 0.565 | -0.009 [-0.13, +0.04] | +0.182 [-0.03, +0.24] | +0.184 [-0.07, +0.35] | +0.059 [-0.16, +0.19] |
| Laguna XS.2 / Pro | correct | 5 | 30,234 (27,564) | 0.646 (L20) | 0.646 (L20) | 0.645 (L20) | 0.417 | 0.391 | 0.585 | +0.002 [-0.10, +0.05] | +0.230 [-0.05, +0.36] | +0.255 [-0.05, +0.44] | +0.061 [-0.18, +0.19] |
| Laguna XS.2 / Pro | correct | 10 | 30,234 (27,234) | 0.664 (L20) | 0.664 (L20) | 0.683 (L20) | 0.497 | 0.453 | 0.584 | -0.019 [-0.15, +0.06] | +0.166 [-0.14, +0.37] | +0.211 [-0.15, +0.44] | +0.080 [-0.09, +0.17] |
| Laguna XS.2 / Pro | correct | 15 | 30,234 (27,553) | 0.676 (L20) | 0.676 (L20) | 0.675 (L20) | 0.499 | 0.470 | 0.579 | +0.001 [-0.13, +0.09] | +0.178 [-0.05, +0.25] | +0.207 [-0.24, +0.51] | +0.097 [-0.10, +0.26] |
| Laguna XS.2 / Pro | reduces_failing | 1 | 72,238 (69,101) | 0.505 (L20) | 0.505 (L20) | 0.534 (L20) | 0.493 | 0.540 | 0.504 | -0.029 [-0.08, +0.01] | +0.011 [-0.16, +0.15] | -0.035 [-0.19, +0.13] | +0.001 [-0.21, +0.19] |
| Laguna XS.2 / Pro | reduces_failing | 2 | 72,244 (68,105) | 0.523 (L20) | 0.523 (L20) | 0.556 (L20) | 0.495 | 0.542 | 0.533 | -0.033 [-0.09, +0.01] | +0.028 [-0.15, +0.17] | -0.019 [-0.18, +0.13] | -0.009 [-0.19, +0.16] |
| Laguna XS.2 / Pro | reduces_failing | 3 | 72,245 (66,922) | 0.547 (L10) | 0.511 (L20) | 0.526 (L20) | 0.489 | 0.511 | 0.516 | +0.021 [-0.06, +0.07] | +0.058 [-0.11, +0.20] | +0.036 [-0.15, +0.18] | +0.031 [-0.14, +0.15] |
| Laguna XS.2 / Pro | reduces_failing | 5 | 72,244 (65,280) | 0.546 (L10) | 0.546 (L10) | 0.562 (L20) | 0.488 | 0.515 | 0.552 | -0.017 [-0.12, +0.05] | +0.057 [-0.17, +0.21] | +0.031 [-0.17, +0.18] | -0.006 [-0.20, +0.15] |
| Laguna XS.2 / Pro | reduces_failing | 10 | 72,245 (63,509) | 0.579 (L10) | 0.579 (L10) | 0.613 (L20) | 0.466 | 0.478 | 0.509 | -0.034 [-0.14, +0.05] | +0.113 [-0.16, +0.35] | +0.101 [-0.11, +0.26] | +0.070 [-0.15, +0.27] |
| Laguna XS.2 / Pro | reduces_failing | 15 | 72,251 (61,965) | 0.565 (L10) | 0.565 (L10) | 0.577 (L20) | 0.451 | 0.466 | 0.496 | -0.012 [-0.13, +0.08] | +0.115 [-0.09, +0.30] | +0.099 [-0.07, +0.24] | +0.070 [-0.11, +0.21] |
| Laguna XS.2 / Pro | has_regressions | 1 | 35,382 (34,171) | 0.391 (L30) | 0.391 (L30) | 0.442 (L20) | 0.506 | 0.665 | 0.418 | -0.051 [-0.12, +0.04] | **-0.115 [-0.27, -0.02]** | **-0.274 [-0.46, -0.03]** | -0.027 [-0.19, +0.11] |
| Laguna XS.2 / Pro | has_regressions | 2 | 35,382 (33,702) | 0.418 (L30) | 0.418 (L30) | 0.473 (L20) | 0.507 | 0.669 | 0.456 | -0.055 [-0.13, +0.03] | **-0.089 [-0.20, -0.02]** | -0.251 [-0.45, +0.01] | -0.038 [-0.20, +0.12] |
| Laguna XS.2 / Pro | has_regressions | 3 | 35,382 (33,163) | 0.468 (L30) | 0.468 (L30) | 0.497 (L20) | 0.530 | 0.657 | 0.483 | -0.028 [-0.09, +0.06] | -0.062 [-0.17, +0.02] | -0.189 [-0.39, +0.07] | -0.015 [-0.18, +0.18] |
| Laguna XS.2 / Pro | has_regressions | 5 | 35,382 (32,483) | 0.473 (L30) | 0.473 (L30) | 0.493 (L20) | 0.446 | 0.643 | 0.499 | -0.019 [-0.07, +0.06] | +0.027 [-0.06, +0.09] | -0.170 [-0.38, +0.10] | -0.025 [-0.22, +0.19] |
| Laguna XS.2 / Pro | has_regressions | 10 | 35,382 (31,863) | 0.502 (L30) | 0.502 (L30) | 0.549 (L20) | 0.422 | 0.684 | 0.462 | -0.047 [-0.10, +0.04] | +0.080 [-0.01, +0.14] | -0.182 [-0.40, +0.10] | +0.040 [-0.14, +0.23] |
| Laguna XS.2 / Pro | has_regressions | 15 | 35,382 (31,489) | 0.474 (L30) | 0.474 (L30) | 0.522 (L20) | 0.362 | 0.659 | 0.494 | -0.048 [-0.11, +0.06] | +0.111 [-0.01, +0.20] | -0.186 [-0.40, +0.13] | -0.020 [-0.23, +0.18] |
| Qwen3.6-35B-A3B / Pro | correct | 1 | 17,795 (17,269) | 0.652 (L30) | 0.663 (L20) | 0.743 (L30) | 0.665 | 0.428 | 0.754 | **-0.091 [-0.11, -0.01]** | -0.013 [-0.16, +0.06] | +0.224 [-0.05, +0.47] | -0.102 [-0.35, +0.07] |
| Qwen3.6-35B-A3B / Pro | correct | 2 | 17,795 (17,279) | 0.649 (L30) | 0.660 (L20) | 0.752 (L30) | 0.648 | 0.427 | 0.737 | **-0.102 [-0.13, -0.06]** | +0.002 [-0.09, +0.20] | +0.222 [-0.17, +0.45] | -0.088 [-0.36, +0.09] |
| Qwen3.6-35B-A3B / Pro | correct | 3 | 17,795 (17,404) | 0.662 (L30) | 0.651 (L20) | 0.748 (L30) | 0.465 | 0.429 | 0.735 | **-0.086 [-0.36, -0.01]** | +0.197 [-0.14, +0.33] | +0.233 [-0.34, +0.43] | -0.073 [-0.46, +0.19] |
| Qwen3.6-35B-A3B / Pro | correct | 5 | 17,795 (17,475) | 0.566 (L30) | 0.572 (L20) | 0.654 (L30) | 0.659 | 0.408 | 0.728 | **-0.088 [-0.30, -0.01]** | **-0.093 [-0.31, -0.03]** | +0.157 [-0.61, +0.36] | -0.162 [-0.41, +0.04] |
| Qwen3.6-35B-A3B / Pro | correct | 10 | 17,795 (17,294) | 0.435 (L30) | 0.428 (L20) | 0.573 (L30) | 0.490 | 0.282 | 0.668 | **-0.138 [-0.16, -0.02]** | -0.055 [-0.45, +0.07] | +0.153 [-0.51, +0.28] | **-0.234 [-0.57, -0.07]** |
| Qwen3.6-35B-A3B / Pro | correct | 15 | 17,795 (17,218) | 0.465 (L30) | 0.456 (L20) | 0.617 (L30) | 0.470 | 0.287 | 0.640 | **-0.152 [-0.18, -0.03]** | -0.006 [-0.48, +0.09] | +0.178 [-0.40, +0.31] | -0.175 [-0.55, +0.09] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 1 | 19,991 (19,410) | 0.454 (L20) | 0.454 (L20) | 0.561 (L20) | 0.584 | 0.760 | 0.669 | **-0.106 [-0.19, -0.03]** | -0.130 [-0.27, +0.04] | -0.306 [-0.49, +0.02] | **-0.215 [-0.35, -0.03]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 2 | 19,991 (19,273) | 0.438 (L20) | 0.442 (L30) | 0.572 (L20) | 0.596 | 0.654 | 0.739 | **-0.134 [-0.21, -0.06]** | **-0.158 [-0.29, -0.02]** | -0.216 [-0.45, +0.06] | **-0.301 [-0.41, -0.15]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 3 | 19,991 (19,210) | 0.455 (L20) | 0.455 (L20) | 0.558 (L20) | 0.494 | 0.589 | 0.712 | **-0.103 [-0.19, -0.00]** | -0.038 [-0.18, +0.13] | -0.134 [-0.38, +0.14] | **-0.257 [-0.37, -0.09]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 5 | 19,991 (19,149) | 0.459 (L20) | 0.459 (L20) | 0.550 (L20) | 0.455 | 0.527 | 0.595 | -0.091 [-0.18, +0.01] | +0.004 [-0.15, +0.24] | -0.068 [-0.37, +0.23] | -0.136 [-0.29, +0.10] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 10 | 19,991 (18,581) | 0.457 (L20) | 0.457 (L20) | 0.581 (L20) | 0.385 | 0.411 | 0.515 | **-0.124 [-0.20, -0.03]** | +0.071 [-0.20, +0.28] | +0.046 [-0.28, +0.25] | -0.059 [-0.29, +0.15] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 15 | 19,991 (18,602) | 0.470 (L39) | 0.495 (L20) | 0.603 (L20) | 0.464 | 0.408 | 0.511 | -0.133 [-0.24, +0.01] | +0.006 [-0.28, +0.31] | +0.062 [-0.24, +0.28] | -0.041 [-0.25, +0.18] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 1 | 1,938 (1,783) | 0.412 (L30) | 0.613 (L39) | 0.456 (L30) | 0.408 | 0.712 | 0.509 | **-0.044 [-0.09, -0.01]** | +0.005 [-0.10, +0.10] | **-0.300 [-0.55, -0.04]** | **-0.096 [-0.48, -0.02]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 2 | 1,938 (1,758) | 0.470 (L30) | 0.667 (L39) | 0.473 (L30) | 0.391 | 0.741 | 0.513 | -0.002 [-0.03, +0.12] | +0.080 [-0.18, +0.17] | -0.270 [-0.71, +0.01] | -0.043 [-0.62, +0.04] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 3 | 1,938 (1,658) | 0.504 (L30) | 0.690 (L39) | 0.448 (L30) | 0.563 | 0.762 | 0.427 | +0.056 [-0.08, +0.14] | -0.058 [-0.20, +0.29] | -0.257 [-0.66, +0.09] | +0.077 [-0.57, +0.15] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 5 | 1,938 (1,543) | 0.463 (L30) | 0.463 (L30) | 0.467 (L30) | 0.473 | 0.788 | 0.413 | -0.003 [-0.10, +0.10] | -0.010 [-0.20, +0.17] | **-0.325 [-0.74, -0.05]** | +0.050 [-0.49, +0.17] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 10 | 1,938 (1,228) | 0.595 (L30) | 0.794 (L39) | 0.505 (L30) | 0.686 | 0.790 | 0.422 | +0.090 [-0.01, +0.13] | -0.091 [-0.61, +0.25] | -0.195 [-0.77, +0.49] | +0.173 [-0.36, +0.46] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 15 | 1,938 (1,098) | 0.531 (L0) | 0.779 (L39) | 0.505 (L30) | 0.575 | 0.856 | 0.368 | +0.026 [-0.16, +0.34] | -0.044 [-0.24, +0.44] | -0.325 [-0.69, +0.47] | +0.163 [-0.40, +0.64] |

#### Phase confound, window max50, within y_t = 0: AUC for y_(t+k) (test split; differences with 95% task-bootstrap CI, bold if the CI excludes 0)

| Model / benchmark | Probe | k | n (pos) | real, val-best layer | real, test-best layer | k=0 probe (E2) | traj-shuffle | position, causal | position, rel_pos (non-causal) | real − k=0 probe | real − traj-shuffle | real − position causal | real − position rel_pos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | correct | 1 | 168,064 (2,155) | 0.784 (L20) | 0.784 (L20) | 0.469 (L20) | 0.655 | 0.396 | 0.434 | **+0.315 [+0.28, +0.35]** | **+0.130 [+0.04, +0.21]** | **+0.389 [+0.18, +0.55]** | **+0.350 [+0.14, +0.50]** |
| Laguna XS.2 / Verified | correct | 2 | 168,064 (2,905) | 0.767 (L20) | 0.767 (L20) | 0.492 (L20) | 0.622 | 0.421 | 0.451 | **+0.275 [+0.22, +0.32]** | **+0.145 [+0.07, +0.21]** | **+0.346 [+0.17, +0.51]** | **+0.316 [+0.12, +0.47]** |
| Laguna XS.2 / Verified | correct | 3 | 168,064 (3,544) | 0.745 (L20) | 0.745 (L20) | 0.497 (L20) | 0.557 | 0.425 | 0.443 | **+0.248 [+0.19, +0.31]** | **+0.188 [+0.10, +0.28]** | **+0.320 [+0.15, +0.49]** | **+0.303 [+0.12, +0.45]** |
| Laguna XS.2 / Verified | correct | 5 | 168,064 (5,309) | 0.716 (L20) | 0.716 (L20) | 0.508 (L20) | 0.546 | 0.440 | 0.438 | **+0.209 [+0.14, +0.28]** | **+0.170 [+0.09, +0.25]** | **+0.276 [+0.13, +0.45]** | **+0.279 [+0.12, +0.42]** |
| Laguna XS.2 / Verified | correct | 10 | 168,064 (9,232) | 0.669 (L20) | 0.669 (L20) | 0.491 (L20) | 0.497 | 0.410 | 0.434 | **+0.178 [+0.10, +0.25]** | **+0.172 [+0.09, +0.26]** | **+0.259 [+0.13, +0.41]** | **+0.235 [+0.09, +0.38]** |
| Laguna XS.2 / Verified | correct | 15 | 168,064 (13,585) | 0.654 (L20) | 0.654 (L20) | 0.493 (L20) | 0.512 | 0.423 | 0.440 | **+0.162 [+0.10, +0.23]** | **+0.142 [+0.07, +0.22]** | **+0.232 [+0.13, +0.35]** | **+0.215 [+0.10, +0.34]** |
| Laguna XS.2 / Verified | correct | 25 | 168,064 (21,412) | 0.600 (L20) | 0.600 (L20) | 0.465 (L20) | 0.572 | 0.465 | 0.457 | **+0.135 [+0.07, +0.21]** | +0.028 [-0.05, +0.12] | **+0.136 [+0.05, +0.23]** | **+0.143 [+0.05, +0.26]** |
| Laguna XS.2 / Verified | correct | 50 | 166,737 (36,008) | 0.504 (L10) | 0.549 (L39) | 0.405 (L20) | 0.494 | 0.562 | 0.441 | **+0.099 [+0.02, +0.18]** | +0.010 [-0.08, +0.12] | -0.059 [-0.14, +0.03] | +0.063 [-0.03, +0.19] |
| Laguna XS.2 / Verified | reduces_failing | 1 | 160,703 (5,342) | 0.846 (L20) | 0.846 (L20) | 0.611 (L20) | 0.637 | 0.567 | 0.547 | **+0.235 [+0.20, +0.27]** | **+0.209 [+0.15, +0.28]** | **+0.279 [+0.17, +0.41]** | **+0.298 [+0.20, +0.41]** |
| Laguna XS.2 / Verified | reduces_failing | 2 | 160,703 (7,607) | 0.816 (L20) | 0.816 (L20) | 0.607 (L20) | 0.650 | 0.583 | 0.551 | **+0.209 [+0.17, +0.25]** | **+0.166 [+0.11, +0.23]** | **+0.233 [+0.11, +0.37]** | **+0.265 [+0.17, +0.37]** |
| Laguna XS.2 / Verified | reduces_failing | 3 | 160,703 (9,948) | 0.786 (L20) | 0.786 (L20) | 0.604 (L20) | 0.631 | 0.600 | 0.548 | **+0.182 [+0.14, +0.23]** | **+0.155 [+0.09, +0.23]** | **+0.187 [+0.06, +0.33]** | **+0.238 [+0.14, +0.35]** |
| Laguna XS.2 / Verified | reduces_failing | 5 | 160,703 (14,597) | 0.741 (L20) | 0.741 (L20) | 0.601 (L20) | 0.630 | 0.606 | 0.527 | **+0.140 [+0.10, +0.19]** | **+0.111 [+0.05, +0.19]** | **+0.135 [+0.01, +0.28]** | **+0.214 [+0.11, +0.32]** |
| Laguna XS.2 / Verified | reduces_failing | 10 | 160,703 (25,183) | 0.679 (L20) | 0.679 (L20) | 0.573 (L20) | 0.603 | 0.560 | 0.478 | **+0.106 [+0.06, +0.15]** | **+0.076 [+0.02, +0.15]** | **+0.119 [+0.00, +0.24]** | **+0.201 [+0.10, +0.30]** |
| Laguna XS.2 / Verified | reduces_failing | 15 | 160,703 (32,506) | 0.655 (L20) | 0.655 (L20) | 0.557 (L20) | 0.539 | 0.524 | 0.464 | **+0.098 [+0.05, +0.14]** | **+0.116 [+0.05, +0.19]** | **+0.130 [+0.01, +0.25]** | **+0.191 [+0.10, +0.29]** |
| Laguna XS.2 / Verified | reduces_failing | 25 | 160,703 (44,518) | 0.589 (L20) | 0.589 (L20) | 0.526 (L20) | 0.603 | 0.471 | 0.446 | **+0.062 [+0.02, +0.10]** | -0.014 [-0.07, +0.05] | **+0.118 [+0.02, +0.21]** | **+0.142 [+0.07, +0.22]** |
| Laguna XS.2 / Verified | reduces_failing | 50 | 159,376 (62,266) | 0.545 (L39) | 0.545 (L39) | 0.465 (L20) | 0.449 | 0.519 | 0.453 | +0.080 [-0.00, +0.16] | **+0.095 [+0.02, +0.17]** | +0.026 [-0.05, +0.10] | **+0.091 [+0.02, +0.16]** |
| Laguna XS.2 / Verified | has_regressions | 1 | 170,147 (3,213) | 0.871 (L10) | 0.871 (L10) | 0.758 (L10) | 0.708 | 0.618 | 0.631 | **+0.113 [+0.07, +0.15]** | **+0.164 [+0.12, +0.21]** | **+0.254 [+0.17, +0.38]** | **+0.240 [+0.17, +0.34]** |
| Laguna XS.2 / Verified | has_regressions | 2 | 170,147 (4,014) | 0.823 (L10) | 0.823 (L10) | 0.724 (L10) | 0.693 | 0.606 | 0.609 | **+0.099 [+0.06, +0.13]** | **+0.131 [+0.09, +0.17]** | **+0.218 [+0.13, +0.35]** | **+0.214 [+0.15, +0.30]** |
| Laguna XS.2 / Verified | has_regressions | 3 | 170,147 (5,148) | 0.783 (L10) | 0.783 (L10) | 0.702 (L10) | 0.679 | 0.595 | 0.591 | **+0.081 [+0.05, +0.11]** | **+0.104 [+0.06, +0.15]** | **+0.189 [+0.10, +0.32]** | **+0.193 [+0.13, +0.28]** |
| Laguna XS.2 / Verified | has_regressions | 5 | 170,147 (8,078) | 0.694 (L10) | 0.694 (L10) | 0.661 (L10) | 0.628 | 0.579 | 0.570 | +0.032 [-0.02, +0.09] | **+0.066 [+0.00, +0.13]** | **+0.115 [+0.02, +0.25]** | **+0.124 [+0.04, +0.23]** |
| Laguna XS.2 / Verified | has_regressions | 10 | 170,147 (13,943) | 0.644 (L10) | 0.644 (L10) | 0.624 (L10) | 0.588 | 0.538 | 0.527 | +0.020 [-0.06, +0.09] | +0.056 [-0.06, +0.15] | **+0.107 [+0.03, +0.23]** | **+0.118 [+0.02, +0.24]** |
| Laguna XS.2 / Verified | has_regressions | 15 | 170,147 (18,994) | 0.620 (L10) | 0.620 (L10) | 0.604 (L10) | 0.539 | 0.528 | 0.526 | +0.015 [-0.09, +0.10] | +0.081 [-0.04, +0.17] | +0.092 [-0.01, +0.21] | **+0.094 [+0.00, +0.21]** |
| Laguna XS.2 / Verified | has_regressions | 25 | 170,147 (24,272) | 0.570 (L10) | 0.570 (L10) | 0.528 (L10) | 0.548 | 0.461 | 0.434 | +0.042 [-0.06, +0.12] | +0.022 [-0.08, +0.12] | **+0.109 [+0.03, +0.20]** | **+0.136 [+0.05, +0.22]** |
| Laguna XS.2 / Verified | has_regressions | 50 | 169,767 (28,941) | 0.563 (L10) | 0.578 (L20) | 0.505 (L10) | 0.404 | 0.445 | 0.425 | +0.058 [-0.05, +0.15] | +0.159 [-0.03, +0.33] | +0.118 [-0.01, +0.26] | **+0.138 [+0.04, +0.22]** |
| Qwen3.6-35B-A3B / Verified | correct | 1 | 58,726 (836) | 0.881 (L20) | 0.907 (L10) | 0.788 (L30) | 0.450 | 0.754 | 0.717 | **+0.093 [+0.06, +0.15]** | **+0.431 [+0.32, +0.60]** | **+0.127 [+0.01, +0.33]** | **+0.164 [+0.05, +0.28]** |
| Qwen3.6-35B-A3B / Verified | correct | 2 | 58,726 (1,085) | 0.833 (L20) | 0.871 (L10) | 0.717 (L30) | 0.456 | 0.744 | 0.696 | **+0.116 [+0.07, +0.19]** | **+0.378 [+0.12, +0.52]** | +0.089 [-0.04, +0.20] | +0.137 [-0.06, +0.24] |
| Qwen3.6-35B-A3B / Verified | correct | 3 | 58,726 (1,291) | 0.832 (L30) | 0.848 (L10) | 0.698 (L30) | 0.482 | 0.730 | 0.691 | **+0.134 [+0.08, +0.21]** | +0.350 [-0.00, +0.53] | +0.102 [-0.03, +0.21] | +0.141 [-0.11, +0.26] |
| Qwen3.6-35B-A3B / Verified | correct | 5 | 58,726 (1,782) | 0.784 (L30) | 0.813 (L10) | 0.637 (L30) | 0.440 | 0.694 | 0.634 | **+0.147 [+0.07, +0.22]** | **+0.344 [+0.07, +0.47]** | +0.091 [-0.02, +0.23] | +0.151 [-0.07, +0.27] |
| Qwen3.6-35B-A3B / Verified | correct | 10 | 58,726 (3,329) | 0.679 (L30) | 0.736 (L10) | 0.577 (L30) | 0.429 | 0.651 | 0.546 | **+0.102 [+0.04, +0.18]** | **+0.250 [+0.09, +0.36]** | +0.028 [-0.12, +0.23] | +0.133 [-0.03, +0.25] |
| Qwen3.6-35B-A3B / Verified | correct | 15 | 58,726 (5,010) | 0.626 (L30) | 0.667 (L10) | 0.564 (L30) | 0.455 | 0.566 | 0.541 | **+0.062 [+0.01, +0.14]** | **+0.171 [+0.05, +0.25]** | +0.060 [-0.11, +0.23] | +0.085 [-0.08, +0.17] |
| Qwen3.6-35B-A3B / Verified | correct | 25 | 58,726 (8,350) | 0.550 (L20) | 0.593 (L10) | 0.520 (L30) | 0.497 | 0.469 | 0.512 | +0.030 [-0.02, +0.10] | +0.052 [-0.07, +0.15] | +0.081 [-0.05, +0.19] | +0.038 [-0.08, +0.12] |
| Qwen3.6-35B-A3B / Verified | correct | 50 | 58,726 (17,606) | 0.630 (L0) | 0.630 (L0) | 0.475 (L30) | 0.493 | 0.517 | 0.511 | +0.155 [-0.00, +0.32] | +0.136 [-0.01, +0.27] | +0.113 [-0.02, +0.22] | +0.118 [-0.02, +0.24] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 1 | 56,911 (1,310) | 0.902 (L30) | 0.918 (L10) | 0.788 (L30) | 0.403 | 0.691 | 0.631 | **+0.114 [+0.05, +0.17]** | **+0.499 [+0.37, +0.72]** | **+0.211 [+0.08, +0.39]** | **+0.271 [+0.15, +0.45]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 2 | 56,911 (1,717) | 0.898 (L10) | 0.898 (L10) | 0.730 (L30) | 0.459 | 0.675 | 0.617 | **+0.168 [+0.08, +0.25]** | **+0.439 [+0.30, +0.64]** | **+0.223 [+0.08, +0.35]** | **+0.281 [+0.13, +0.42]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3 | 56,911 (2,099) | 0.875 (L10) | 0.875 (L10) | 0.695 (L30) | 0.531 | 0.659 | 0.606 | **+0.181 [+0.09, +0.26]** | **+0.344 [+0.19, +0.56]** | **+0.217 [+0.08, +0.34]** | **+0.270 [+0.12, +0.41]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 5 | 56,911 (2,987) | 0.844 (L10) | 0.844 (L10) | 0.659 (L30) | 0.509 | 0.652 | 0.578 | **+0.185 [+0.09, +0.26]** | **+0.335 [+0.17, +0.56]** | **+0.192 [+0.06, +0.30]** | **+0.265 [+0.10, +0.40]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 10 | 56,911 (5,432) | 0.771 (L10) | 0.771 (L10) | 0.607 (L30) | 0.518 | 0.602 | 0.539 | **+0.165 [+0.10, +0.23]** | **+0.254 [+0.08, +0.45]** | **+0.169 [+0.05, +0.26]** | **+0.232 [+0.09, +0.35]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 15 | 56,911 (7,944) | 0.703 (L10) | 0.703 (L10) | 0.575 (L30) | 0.509 | 0.575 | 0.543 | **+0.128 [+0.07, +0.19]** | **+0.194 [+0.02, +0.39]** | **+0.128 [+0.00, +0.24]** | **+0.160 [+0.00, +0.28]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 25 | 56,911 (12,561) | 0.635 (L30) | 0.641 (L10) | 0.560 (L30) | 0.530 | 0.564 | 0.538 | **+0.075 [+0.02, +0.13]** | **+0.105 [+0.01, +0.20]** | +0.071 [-0.01, +0.15] | **+0.096 [+0.01, +0.18]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 50 | 56,911 (24,595) | 0.551 (L20) | 0.589 (L10) | 0.520 (L30) | 0.523 | 0.550 | 0.530 | +0.031 [-0.06, +0.12] | +0.028 [-0.05, +0.11] | +0.001 [-0.05, +0.06] | +0.021 [-0.05, +0.09] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 1 | 59,290 (263) | 0.839 (L20) | 0.839 (L20) | 0.716 (L20) | 0.562 | 0.454 | 0.650 | **+0.123 [+0.02, +0.19]** | **+0.277 [+0.11, +0.39]** | **+0.385 [+0.14, +0.53]** | **+0.189 [+0.08, +0.28]** |
| Qwen3.6-35B-A3B / Verified | has_regressions | 2 | 59,290 (381) | 0.826 (L20) | 0.826 (L20) | 0.680 (L20) | 0.526 | 0.453 | 0.616 | **+0.146 [+0.00, +0.20]** | **+0.300 [+0.11, +0.37]** | **+0.372 [+0.13, +0.50]** | **+0.210 [+0.05, +0.31]** |
| Qwen3.6-35B-A3B / Verified | has_regressions | 3 | 59,290 (609) | 0.810 (L20) | 0.810 (L20) | 0.661 (L20) | 0.506 | 0.443 | 0.589 | **+0.148 [+0.02, +0.20]** | **+0.304 [+0.12, +0.39]** | **+0.367 [+0.14, +0.50]** | **+0.220 [+0.07, +0.33]** |
| Qwen3.6-35B-A3B / Verified | has_regressions | 5 | 59,290 (895) | 0.749 (L20) | 0.749 (L20) | 0.640 (L20) | 0.469 | 0.411 | 0.554 | **+0.109 [+0.04, +0.16]** | **+0.281 [+0.12, +0.39]** | **+0.339 [+0.14, +0.47]** | **+0.196 [+0.06, +0.31]** |
| Qwen3.6-35B-A3B / Verified | has_regressions | 10 | 59,290 (1,639) | 0.692 (L10) | 0.684 (L20) | 0.588 (L20) | 0.516 | 0.457 | 0.545 | **+0.104 [+0.02, +0.18]** | **+0.176 [+0.01, +0.30]** | **+0.234 [+0.04, +0.40]** | +0.146 [-0.05, +0.30] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 15 | 59,290 (2,562) | 0.650 (L20) | 0.650 (L20) | 0.561 (L20) | 0.553 | 0.476 | 0.532 | **+0.089 [+0.02, +0.17]** | +0.097 [-0.02, +0.20] | +0.174 [-0.01, +0.38] | +0.118 [-0.03, +0.28] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 25 | 59,290 (5,137) | 0.584 (L20) | 0.586 (L10) | 0.530 (L20) | 0.547 | 0.508 | 0.567 | +0.054 [-0.01, +0.15] | +0.037 [-0.03, +0.12] | +0.076 [-0.04, +0.25] | +0.018 [-0.15, +0.23] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 50 | 59,290 (10,044) | 0.637 (L0) | 0.638 (L20) | 0.519 (L20) | 0.574 | 0.575 | 0.504 | **+0.118 [+0.01, +0.20]** | +0.063 [-0.04, +0.13] | +0.062 [-0.00, +0.15] | **+0.133 [+0.02, +0.22]** |
| Laguna XS.2 / Pro | correct | 1 | 125,593 (537) | 0.864 (L20) | 0.864 (L20) | 0.774 (L20) | 0.246 | 0.641 | 0.723 | **+0.090 [+0.04, +0.12]** | **+0.617 [+0.49, +0.71]** | **+0.223 [+0.11, +0.36]** | +0.141 [-0.01, +0.34] |
| Laguna XS.2 / Pro | correct | 2 | 125,585 (638) | 0.803 (L20) | 0.803 (L20) | 0.739 (L20) | 0.298 | 0.665 | 0.797 | **+0.064 [+0.00, +0.11]** | **+0.505 [+0.40, +0.60]** | **+0.138 [+0.07, +0.22]** | +0.007 [-0.09, +0.13] |
| Laguna XS.2 / Pro | correct | 3 | 125,581 (724) | 0.784 (L20) | 0.784 (L20) | 0.731 (L20) | 0.341 | 0.647 | 0.797 | +0.053 [-0.01, +0.09] | **+0.444 [+0.34, +0.55]** | **+0.138 [+0.07, +0.21]** | -0.013 [-0.11, +0.10] |
| Laguna XS.2 / Pro | correct | 5 | 125,565 (1,623) | 0.784 (L20) | 0.784 (L20) | 0.741 (L20) | 0.352 | 0.670 | 0.813 | +0.043 [-0.01, +0.09] | **+0.432 [+0.34, +0.52]** | **+0.114 [+0.05, +0.18]** | -0.029 [-0.10, +0.03] |
| Laguna XS.2 / Pro | correct | 10 | 125,543 (3,320) | 0.716 (L20) | 0.716 (L20) | 0.712 (L20) | 0.418 | 0.653 | 0.775 | +0.003 [-0.06, +0.07] | **+0.298 [+0.21, +0.38]** | +0.062 [-0.02, +0.14] | -0.059 [-0.16, +0.04] |
| Laguna XS.2 / Pro | correct | 15 | 125,509 (5,292) | 0.680 (L20) | 0.680 (L20) | 0.675 (L20) | 0.512 | 0.595 | 0.702 | +0.005 [-0.07, +0.07] | **+0.168 [+0.07, +0.27]** | **+0.084 [+0.02, +0.15]** | -0.023 [-0.12, +0.06] |
| Laguna XS.2 / Pro | correct | 25 | 125,298 (8,022) | 0.618 (L20) | 0.618 (L20) | 0.590 (L20) | 0.531 | 0.544 | 0.604 | +0.028 [-0.06, +0.10] | +0.087 [-0.03, +0.20] | +0.074 [-0.00, +0.14] | +0.014 [-0.10, +0.11] |
| Laguna XS.2 / Pro | correct | 50 | 124,588 (12,656) | 0.603 (L20) | 0.603 (L20) | 0.537 (L20) | 0.527 | 0.523 | 0.532 | +0.066 [-0.01, +0.14] | +0.076 [-0.04, +0.19] | +0.080 [-0.02, +0.17] | +0.071 [-0.03, +0.16] |
| Laguna XS.2 / Pro | reduces_failing | 1 | 122,903 (1,733) | 0.876 (L20) | 0.876 (L20) | 0.820 (L20) | 0.469 | 0.726 | 0.730 | **+0.056 [+0.02, +0.12]** | **+0.407 [+0.27, +0.50]** | **+0.150 [+0.07, +0.25]** | **+0.145 [+0.07, +0.22]** |
| Laguna XS.2 / Pro | reduces_failing | 2 | 122,895 (2,931) | 0.864 (L20) | 0.864 (L20) | 0.814 (L20) | 0.496 | 0.735 | 0.739 | **+0.050 [+0.02, +0.12]** | **+0.368 [+0.22, +0.45]** | **+0.129 [+0.06, +0.21]** | **+0.125 [+0.01, +0.19]** |
| Laguna XS.2 / Pro | reduces_failing | 3 | 122,891 (3,722) | 0.870 (L20) | 0.870 (L20) | 0.808 (L20) | 0.465 | 0.713 | 0.695 | **+0.062 [+0.03, +0.14]** | **+0.404 [+0.27, +0.47]** | **+0.156 [+0.09, +0.23]** | **+0.174 [+0.03, +0.25]** |
| Laguna XS.2 / Pro | reduces_failing | 5 | 122,875 (4,951) | 0.843 (L20) | 0.843 (L20) | 0.779 (L20) | 0.544 | 0.684 | 0.670 | **+0.064 [+0.03, +0.14]** | **+0.299 [+0.18, +0.37]** | **+0.158 [+0.10, +0.23]** | **+0.173 [+0.02, +0.26]** |
| Laguna XS.2 / Pro | reduces_failing | 10 | 122,853 (7,704) | 0.774 (L30) | 0.780 (L20) | 0.722 (L20) | 0.596 | 0.626 | 0.615 | **+0.052 [+0.01, +0.13]** | +0.178 [-0.00, +0.29] | **+0.148 [+0.08, +0.22]** | **+0.159 [+0.01, +0.26]** |
| Laguna XS.2 / Pro | reduces_failing | 15 | 122,819 (9,565) | 0.732 (L30) | 0.736 (L10) | 0.660 (L20) | 0.535 | 0.560 | 0.566 | **+0.071 [+0.02, +0.15]** | **+0.196 [+0.02, +0.31]** | **+0.171 [+0.10, +0.23]** | **+0.166 [+0.02, +0.26]** |
| Laguna XS.2 / Pro | reduces_failing | 25 | 122,608 (13,231) | 0.683 (L10) | 0.683 (L10) | 0.578 (L20) | 0.512 | 0.516 | 0.491 | **+0.105 [+0.04, +0.18]** | **+0.171 [+0.07, +0.25]** | **+0.167 [+0.09, +0.24]** | **+0.192 [+0.08, +0.27]** |
| Laguna XS.2 / Pro | reduces_failing | 50 | 121,898 (15,890) | 0.690 (L10) | 0.690 (L10) | 0.542 (L20) | 0.542 | 0.499 | 0.443 | **+0.147 [+0.07, +0.24]** | **+0.147 [+0.07, +0.23]** | **+0.190 [+0.10, +0.28]** | **+0.247 [+0.13, +0.33]** |
| Laguna XS.2 / Pro | has_regressions | 1 | 123,985 (1,102) | 0.867 (L20) | 0.867 (L20) | 0.782 (L20) | 0.357 | 0.751 | 0.573 | **+0.085 [+0.06, +0.13]** | **+0.509 [+0.37, +0.56]** | +0.115 [-0.12, +0.20] | **+0.294 [+0.05, +0.37]** |
| Laguna XS.2 / Pro | has_regressions | 2 | 123,977 (1,449) | 0.730 (L39) | 0.828 (L20) | 0.742 (L20) | 0.361 | 0.709 | 0.532 | -0.011 [-0.08, +0.07] | **+0.369 [+0.20, +0.46]** | +0.021 [-0.20, +0.14] | **+0.198 [+0.04, +0.28]** |
| Laguna XS.2 / Pro | has_regressions | 3 | 123,973 (1,338) | 0.778 (L10) | 0.806 (L20) | 0.694 (L20) | 0.358 | 0.712 | 0.567 | +0.084 [-0.00, +0.14] | **+0.421 [+0.31, +0.51]** | +0.067 [-0.11, +0.18] | **+0.211 [+0.09, +0.29]** |
| Laguna XS.2 / Pro | has_regressions | 5 | 123,957 (1,626) | 0.786 (L10) | 0.795 (L30) | 0.661 (L20) | 0.357 | 0.657 | 0.518 | **+0.125 [+0.02, +0.18]** | **+0.429 [+0.27, +0.53]** | +0.129 [-0.07, +0.26] | **+0.268 [+0.15, +0.37]** |
| Laguna XS.2 / Pro | has_regressions | 10 | 123,935 (2,571) | 0.680 (L10) | 0.738 (L30) | 0.612 (L20) | 0.318 | 0.727 | 0.570 | **+0.068 [+0.01, +0.13]** | **+0.362 [+0.20, +0.45]** | -0.047 [-0.13, +0.27] | **+0.110 [+0.01, +0.30]** |
| Laguna XS.2 / Pro | has_regressions | 15 | 123,901 (3,871) | 0.682 (L10) | 0.777 (L39) | 0.582 (L20) | 0.419 | 0.752 | 0.570 | +0.099 [-0.01, +0.15] | +0.263 [-0.09, +0.39] | -0.070 [-0.15, +0.20] | **+0.111 [+0.03, +0.27]** |
| Laguna XS.2 / Pro | has_regressions | 25 | 123,690 (6,495) | 0.699 (L10) | 0.796 (L39) | 0.570 (L20) | 0.359 | 0.780 | 0.538 | +0.129 [-0.02, +0.19] | **+0.340 [+0.06, +0.51]** | -0.081 [-0.17, +0.17] | **+0.162 [+0.09, +0.24]** |
| Laguna XS.2 / Pro | has_regressions | 50 | 122,980 (8,615) | 0.694 (L10) | 0.741 (L39) | 0.566 (L20) | 0.403 | 0.753 | 0.568 | **+0.128 [+0.03, +0.21]** | **+0.291 [+0.04, +0.40]** | -0.060 [-0.17, +0.15] | **+0.126 [+0.02, +0.25]** |
| Qwen3.6-35B-A3B / Pro | correct | 1 | 64,121 (662) | 0.826 (L0) | 0.892 (L30) | 0.818 (L0) | 0.553 | 0.554 | 0.641 | +0.008 [-0.01, +0.02] | **+0.273 [+0.11, +0.40]** | +0.272 [-0.02, +0.52] | **+0.185 [+0.02, +0.36]** |
| Qwen3.6-35B-A3B / Pro | correct | 2 | 64,121 (837) | 0.765 (L0) | 0.855 (L30) | 0.758 (L0) | 0.501 | 0.506 | 0.588 | +0.006 [-0.01, +0.02] | **+0.264 [+0.11, +0.42]** | +0.259 [-0.03, +0.47] | **+0.177 [+0.01, +0.35]** |
| Qwen3.6-35B-A3B / Pro | correct | 3 | 64,121 (968) | 0.724 (L0) | 0.832 (L30) | 0.717 (L0) | 0.481 | 0.480 | 0.563 | +0.007 [-0.01, +0.02] | **+0.242 [+0.07, +0.42]** | +0.244 [-0.03, +0.43] | **+0.161 [+0.00, +0.35]** |
| Qwen3.6-35B-A3B / Pro | correct | 5 | 64,108 (1,224) | 0.657 (L0) | 0.774 (L30) | 0.660 (L0) | 0.452 | 0.465 | 0.521 | -0.003 [-0.02, +0.02] | **+0.205 [+0.04, +0.38]** | +0.192 [-0.04, +0.38] | +0.136 [-0.03, +0.38] |
| Qwen3.6-35B-A3B / Pro | correct | 10 | 64,100 (1,746) | 0.671 (L20) | 0.677 (L30) | 0.597 (L0) | 0.445 | 0.442 | 0.469 | +0.074 [-0.02, +0.15] | **+0.226 [+0.13, +0.33]** | +0.229 [-0.07, +0.40] | +0.203 [-0.02, +0.40] |
| Qwen3.6-35B-A3B / Pro | correct | 15 | 64,101 (2,180) | 0.664 (L20) | 0.671 (L30) | 0.567 (L0) | 0.441 | 0.450 | 0.468 | +0.097 [-0.00, +0.20] | +0.224 [-0.01, +0.47] | +0.214 [-0.04, +0.39] | +0.197 [-0.01, +0.38] |
| Qwen3.6-35B-A3B / Pro | correct | 25 | 64,099 (3,370) | 0.656 (L20) | 0.670 (L30) | 0.498 (L0) | 0.442 | 0.447 | 0.459 | **+0.158 [+0.03, +0.25]** | **+0.214 [+0.05, +0.37]** | **+0.209 [+0.00, +0.37]** | **+0.197 [+0.03, +0.36]** |
| Qwen3.6-35B-A3B / Pro | correct | 50 | 64,079 (8,216) | 0.656 (L30) | 0.656 (L30) | 0.492 (L0) | 0.475 | 0.457 | 0.431 | **+0.164 [+0.08, +0.24]** | **+0.181 [+0.06, +0.31]** | **+0.199 [+0.06, +0.32]** | **+0.225 [+0.10, +0.36]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 1 | 65,951 (871) | 0.921 (L10) | 0.954 (L20) | 0.839 (L10) | 0.245 | 0.659 | 0.723 | **+0.082 [+0.04, +0.22]** | **+0.676 [+0.50, +0.78]** | **+0.261 [+0.07, +0.51]** | **+0.198 [+0.05, +0.36]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 2 | 65,951 (1,033) | 0.909 (L10) | 0.950 (L20) | 0.827 (L10) | 0.332 | 0.672 | 0.716 | **+0.082 [+0.06, +0.18]** | **+0.577 [+0.38, +0.68]** | **+0.237 [+0.07, +0.46]** | **+0.193 [+0.04, +0.33]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 3 | 65,951 (1,116) | 0.894 (L10) | 0.938 (L20) | 0.821 (L10) | 0.367 | 0.675 | 0.681 | **+0.073 [+0.05, +0.15]** | **+0.526 [+0.30, +0.60]** | **+0.219 [+0.06, +0.42]** | **+0.213 [+0.11, +0.32]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 5 | 65,938 (1,317) | 0.887 (L10) | 0.933 (L20) | 0.790 (L10) | 0.426 | 0.681 | 0.692 | **+0.097 [+0.05, +0.16]** | **+0.461 [+0.22, +0.54]** | **+0.206 [+0.08, +0.35]** | **+0.195 [+0.10, +0.35]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 10 | 65,930 (1,928) | 0.877 (L20) | 0.877 (L20) | 0.753 (L10) | 0.656 | 0.668 | 0.657 | **+0.123 [+0.07, +0.19]** | **+0.220 [+0.13, +0.31]** | **+0.209 [+0.05, +0.41]** | **+0.220 [+0.10, +0.40]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 15 | 65,931 (2,412) | 0.860 (L20) | 0.860 (L20) | 0.721 (L10) | 0.332 | 0.635 | 0.624 | **+0.139 [+0.07, +0.22]** | **+0.529 [+0.30, +0.64]** | **+0.225 [+0.06, +0.39]** | **+0.237 [+0.12, +0.42]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 25 | 65,929 (3,601) | 0.815 (L20) | 0.815 (L20) | 0.673 (L10) | 0.536 | 0.605 | 0.608 | **+0.142 [+0.06, +0.24]** | **+0.279 [+0.08, +0.49]** | **+0.210 [+0.04, +0.36]** | **+0.207 [+0.08, +0.36]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 50 | 65,909 (7,029) | 0.706 (L0) | 0.727 (L30) | 0.553 (L10) | 0.427 | 0.436 | 0.565 | +0.152 [-0.03, +0.34] | **+0.278 [+0.02, +0.55]** | **+0.270 [+0.09, +0.48]** | **+0.141 [+0.00, +0.31]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 1 | 66,736 (69) | 0.796 (L30) | 0.910 (L10) | 0.897 (L10) | 0.318 | 0.973 | 0.875 | **-0.101 [-0.12, -0.09]** | **+0.478 [+0.38, +0.57]** | **-0.178 [-0.23, -0.13]** | **-0.079 [-0.15, -0.01]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 2 | 66,736 (113) | 0.916 (L10) | 0.916 (L10) | 0.894 (L10) | 0.305 | 0.939 | 0.851 | **+0.022 [+0.01, +0.04]** | **+0.611 [+0.52, +0.69]** | -0.022 [-0.07, +0.03] | **+0.065 [+0.00, +0.12]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 3 | 66,736 (119) | 0.921 (L10) | 0.921 (L10) | 0.887 (L10) | 0.426 | 0.943 | 0.848 | **+0.034 [+0.02, +0.05]** | **+0.495 [+0.41, +0.57]** | -0.022 [-0.07, +0.03] | **+0.073 [+0.01, +0.13]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 5 | 66,723 (243) | 0.947 (L10) | 0.947 (L10) | 0.912 (L10) | 0.497 | 0.854 | 0.764 | **+0.035 [+0.02, +0.05]** | **+0.450 [+0.38, +0.52]** | **+0.093 [+0.04, +0.15]** | **+0.184 [+0.12, +0.24]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 10 | 66,715 (56) | 0.826 (L10) | 0.826 (L10) | 0.673 (L10) | 0.810 | 0.560 | 0.535 | **+0.153 [+0.13, +0.18]** | +0.016 [-0.08, +0.10] | **+0.266 [+0.17, +0.38]** | **+0.291 [+0.00, +0.38]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 15 | 66,716 (89) | 0.711 (L10) | 0.761 (L30) | 0.579 (L10) | 0.556 | 0.518 | 0.483 | **+0.131 [+0.10, +0.16]** | **+0.155 [+0.10, +0.19]** | **+0.193 [+0.08, +0.28]** | +0.227 [-0.06, +0.38] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 25 | 66,714 (320) | 0.754 (L10) | 0.808 (L30) | 0.654 (L10) | 0.357 | 0.426 | 0.503 | **+0.100 [+0.01, +0.19]** | **+0.397 [+0.07, +0.72]** | **+0.328 [+0.18, +0.44]** | **+0.251 [+0.06, +0.46]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 50 | 66,694 (759) | 0.841 (L20) | 0.859 (L30) | 0.565 (L10) | 0.456 | 0.553 | 0.664 | **+0.275 [+0.18, +0.53]** | **+0.385 [+0.29, +0.55]** | **+0.288 [+0.21, +0.41]** | **+0.177 [+0.09, +0.47]** |

#### Phase confound, window max50, all test tokens: AUC for y_(t+k) (test split; differences with 95% task-bootstrap CI, bold if the CI excludes 0)

| Model / benchmark | Probe | k | n (pos) | real, val-best layer | real, test-best layer | k=0 probe (E2) | traj-shuffle | position, causal | position, rel_pos (non-causal) | real − k=0 probe | real − traj-shuffle | real − position causal | real − position rel_pos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | correct | 1 | 181,946 | 0.759 (L20) | 0.759 (L20) | 0.725 (L20) | 0.659 | 0.593 | 0.619 | **+0.034 [+0.00, +0.08]** | **+0.100 [+0.05, +0.17]** | **+0.166 [+0.04, +0.38]** | **+0.140 [+0.02, +0.30]** |
| Laguna XS.2 / Verified | correct | 2 | 181,946 | 0.755 (L20) | 0.755 (L20) | 0.718 (L20) | 0.642 | 0.587 | 0.613 | **+0.037 [+0.01, +0.09]** | **+0.114 [+0.07, +0.17]** | **+0.168 [+0.05, +0.38]** | **+0.142 [+0.03, +0.29]** |
| Laguna XS.2 / Verified | correct | 3 | 181,946 | 0.749 (L20) | 0.749 (L20) | 0.711 (L20) | 0.637 | 0.581 | 0.605 | **+0.039 [+0.01, +0.09]** | **+0.112 [+0.06, +0.19]** | **+0.169 [+0.05, +0.37]** | **+0.144 [+0.03, +0.29]** |
| Laguna XS.2 / Verified | correct | 5 | 181,946 | 0.731 (L20) | 0.731 (L20) | 0.692 (L20) | 0.591 | 0.570 | 0.586 | **+0.039 [+0.00, +0.10]** | **+0.140 [+0.09, +0.21]** | **+0.161 [+0.03, +0.36]** | **+0.145 [+0.03, +0.29]** |
| Laguna XS.2 / Verified | correct | 10 | 181,946 | 0.695 (L20) | 0.695 (L20) | 0.641 (L20) | 0.535 | 0.515 | 0.542 | **+0.055 [+0.01, +0.12]** | **+0.160 [+0.10, +0.23]** | **+0.180 [+0.05, +0.35]** | **+0.154 [+0.03, +0.28]** |
| Laguna XS.2 / Verified | correct | 15 | 181,946 | 0.684 (L20) | 0.684 (L20) | 0.619 (L20) | 0.522 | 0.504 | 0.526 | **+0.066 [+0.02, +0.13]** | **+0.162 [+0.09, +0.23]** | **+0.181 [+0.07, +0.31]** | **+0.159 [+0.05, +0.27]** |
| Laguna XS.2 / Verified | correct | 25 | 181,946 | 0.639 (L20) | 0.639 (L20) | 0.575 (L20) | 0.541 | 0.521 | 0.503 | **+0.064 [+0.01, +0.13]** | **+0.099 [+0.00, +0.19]** | **+0.118 [+0.04, +0.21]** | **+0.137 [+0.05, +0.23]** |
| Laguna XS.2 / Verified | correct | 50 | 180,619 | 0.512 (L10) | 0.562 (L39) | 0.496 (L20) | 0.488 | 0.545 | 0.489 | +0.017 [-0.07, +0.11] | +0.024 [-0.07, +0.12] | -0.033 [-0.11, +0.05] | +0.023 [-0.06, +0.13] |
| Laguna XS.2 / Verified | reduces_failing | 1 | 181,946 | 0.840 (L20) | 0.840 (L20) | 0.800 (L20) | 0.734 | 0.688 | 0.687 | **+0.039 [+0.02, +0.06]** | **+0.106 [+0.06, +0.17]** | **+0.151 [+0.05, +0.27]** | **+0.153 [+0.08, +0.24]** |
| Laguna XS.2 / Verified | reduces_failing | 2 | 181,946 | 0.828 (L20) | 0.828 (L20) | 0.785 (L20) | 0.723 | 0.689 | 0.678 | **+0.044 [+0.02, +0.07]** | **+0.105 [+0.06, +0.17]** | **+0.140 [+0.03, +0.27]** | **+0.151 [+0.08, +0.24]** |
| Laguna XS.2 / Verified | reduces_failing | 3 | 181,946 | 0.815 (L20) | 0.815 (L20) | 0.770 (L20) | 0.709 | 0.688 | 0.667 | **+0.044 [+0.02, +0.07]** | **+0.106 [+0.05, +0.17]** | **+0.126 [+0.02, +0.26]** | **+0.147 [+0.07, +0.23]** |
| Laguna XS.2 / Verified | reduces_failing | 5 | 181,946 | 0.783 (L20) | 0.783 (L20) | 0.747 (L20) | 0.700 | 0.679 | 0.642 | **+0.036 [+0.01, +0.07]** | **+0.083 [+0.02, +0.15]** | +0.104 [-0.01, +0.24] | **+0.141 [+0.06, +0.23]** |
| Laguna XS.2 / Verified | reduces_failing | 10 | 181,946 | 0.732 (L20) | 0.732 (L20) | 0.695 (L20) | 0.651 | 0.621 | 0.585 | **+0.037 [+0.01, +0.07]** | **+0.081 [+0.02, +0.15]** | +0.111 [-0.01, +0.23] | **+0.147 [+0.07, +0.22]** |
| Laguna XS.2 / Verified | reduces_failing | 15 | 181,946 | 0.701 (L20) | 0.701 (L20) | 0.663 (L20) | 0.572 | 0.576 | 0.558 | **+0.037 [+0.00, +0.08]** | **+0.128 [+0.06, +0.20]** | **+0.124 [+0.01, +0.24]** | **+0.142 [+0.06, +0.23]** |
| Laguna XS.2 / Verified | reduces_failing | 25 | 181,946 | 0.631 (L20) | 0.631 (L20) | 0.621 (L20) | 0.628 | 0.530 | 0.525 | +0.009 [-0.03, +0.05] | +0.002 [-0.05, +0.06] | +0.101 [-0.00, +0.21] | **+0.105 [+0.04, +0.18]** |
| Laguna XS.2 / Verified | reduces_failing | 50 | 180,619 | 0.546 (L39) | 0.546 (L39) | 0.550 (L20) | 0.463 | 0.510 | 0.503 | -0.004 [-0.09, +0.08] | **+0.083 [+0.01, +0.16]** | +0.036 [-0.03, +0.12] | +0.043 [-0.03, +0.11] |
| Laguna XS.2 / Verified | has_regressions | 1 | 181,946 | 0.761 (L10) | 0.761 (L10) | 0.752 (L10) | 0.691 | 0.715 | 0.751 | +0.009 [-0.02, +0.04] | **+0.070 [+0.01, +0.13]** | +0.045 [-0.02, +0.15] | +0.009 [-0.06, +0.09] |
| Laguna XS.2 / Verified | has_regressions | 2 | 181,946 | 0.751 (L10) | 0.751 (L10) | 0.744 (L10) | 0.685 | 0.699 | 0.740 | +0.008 [-0.02, +0.04] | **+0.067 [+0.01, +0.13]** | +0.052 [-0.02, +0.17] | +0.012 [-0.06, +0.10] |
| Laguna XS.2 / Verified | has_regressions | 3 | 181,946 | 0.742 (L10) | 0.742 (L10) | 0.736 (L10) | 0.678 | 0.689 | 0.725 | +0.005 [-0.02, +0.04] | **+0.064 [+0.00, +0.12]** | +0.053 [-0.03, +0.18] | +0.016 [-0.05, +0.10] |
| Laguna XS.2 / Verified | has_regressions | 5 | 181,946 | 0.699 (L10) | 0.699 (L10) | 0.715 (L10) | 0.645 | 0.673 | 0.697 | -0.015 [-0.06, +0.04] | +0.054 [-0.01, +0.12] | +0.027 [-0.07, +0.16] | +0.003 [-0.07, +0.10] |
| Laguna XS.2 / Verified | has_regressions | 10 | 181,946 | 0.680 (L10) | 0.680 (L10) | 0.683 (L10) | 0.617 | 0.618 | 0.644 | -0.003 [-0.07, +0.06] | +0.063 [-0.04, +0.15] | +0.061 [-0.01, +0.16] | +0.036 [-0.06, +0.14] |
| Laguna XS.2 / Verified | has_regressions | 15 | 181,946 | 0.657 (L10) | 0.657 (L10) | 0.659 (L10) | 0.569 | 0.595 | 0.608 | -0.002 [-0.08, +0.08] | +0.088 [-0.02, +0.17] | +0.061 [-0.02, +0.16] | +0.049 [-0.04, +0.16] |
| Laguna XS.2 / Verified | has_regressions | 25 | 181,946 | 0.610 (L10) | 0.610 (L10) | 0.592 (L10) | 0.582 | 0.504 | 0.533 | +0.018 [-0.07, +0.09] | +0.027 [-0.07, +0.13] | **+0.106 [+0.04, +0.19]** | +0.076 [-0.01, +0.16] |
| Laguna XS.2 / Verified | has_regressions | 50 | 180,619 | 0.594 (L10) | 0.596 (L20) | 0.555 (L10) | 0.406 | 0.473 | 0.495 | +0.039 [-0.06, +0.12] | +0.188 [-0.02, +0.37] | +0.121 [-0.00, +0.24] | **+0.099 [+0.01, +0.18]** |
| Qwen3.6-35B-A3B / Verified | correct | 1 | 60,746 | 0.806 (L20) | 0.828 (L10) | 0.804 (L30) | 0.474 | 0.772 | 0.745 | +0.002 [-0.03, +0.05] | **+0.331 [+0.26, +0.48]** | +0.034 [-0.11, +0.28] | +0.060 [-0.05, +0.21] |
| Qwen3.6-35B-A3B / Verified | correct | 2 | 60,746 | 0.784 (L20) | 0.815 (L10) | 0.778 (L30) | 0.474 | 0.768 | 0.737 | +0.007 [-0.03, +0.07] | **+0.310 [+0.22, +0.42]** | +0.016 [-0.14, +0.26] | +0.047 [-0.07, +0.17] |
| Qwen3.6-35B-A3B / Verified | correct | 3 | 60,746 | 0.791 (L30) | 0.810 (L10) | 0.767 (L30) | 0.452 | 0.759 | 0.732 | +0.024 [-0.00, +0.08] | **+0.339 [+0.21, +0.44]** | +0.032 [-0.12, +0.26] | +0.059 [-0.06, +0.19] |
| Qwen3.6-35B-A3B / Verified | correct | 5 | 60,746 | 0.771 (L30) | 0.799 (L10) | 0.734 (L30) | 0.423 | 0.738 | 0.697 | +0.037 [-0.00, +0.10] | **+0.348 [+0.20, +0.45]** | +0.033 [-0.12, +0.28] | +0.074 [-0.04, +0.19] |
| Qwen3.6-35B-A3B / Verified | correct | 10 | 60,746 | 0.701 (L30) | 0.753 (L10) | 0.668 (L30) | 0.449 | 0.691 | 0.622 | +0.033 [-0.01, +0.11] | **+0.252 [+0.13, +0.36]** | +0.010 [-0.16, +0.27] | +0.079 [-0.03, +0.18] |
| Qwen3.6-35B-A3B / Verified | correct | 15 | 60,746 | 0.653 (L30) | 0.694 (L10) | 0.634 (L30) | 0.473 | 0.616 | 0.600 | +0.019 [-0.02, +0.10] | **+0.180 [+0.09, +0.25]** | +0.037 [-0.14, +0.25] | +0.054 [-0.06, +0.13] |
| Qwen3.6-35B-A3B / Verified | correct | 25 | 60,746 | 0.578 (L20) | 0.621 (L10) | 0.574 (L30) | 0.525 | 0.516 | 0.557 | +0.005 [-0.05, +0.07] | +0.054 [-0.05, +0.14] | +0.063 [-0.07, +0.18] | +0.022 [-0.08, +0.10] |
| Qwen3.6-35B-A3B / Verified | correct | 50 | 60,746 | 0.618 (L0) | 0.618 (L0) | 0.492 (L30) | 0.495 | 0.518 | 0.513 | +0.126 [-0.03, +0.29] | +0.123 [-0.02, +0.26] | +0.100 [-0.03, +0.21] | +0.105 [-0.03, +0.23] |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 1 | 60,746 | 0.885 (L30) | 0.890 (L10) | 0.848 (L30) | 0.468 | 0.750 | 0.715 | **+0.037 [+0.01, +0.07]** | **+0.417 [+0.33, +0.52]** | **+0.135 [+0.02, +0.25]** | **+0.170 [+0.07, +0.27]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 2 | 60,746 | 0.885 (L10) | 0.885 (L10) | 0.826 (L30) | 0.517 | 0.743 | 0.704 | **+0.060 [+0.00, +0.10]** | **+0.368 [+0.27, +0.48]** | **+0.143 [+0.03, +0.24]** | **+0.181 [+0.07, +0.28]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3 | 60,746 | 0.878 (L10) | 0.878 (L10) | 0.807 (L30) | 0.569 | 0.733 | 0.695 | **+0.070 [+0.01, +0.12]** | **+0.309 [+0.19, +0.45]** | **+0.145 [+0.03, +0.24]** | **+0.183 [+0.08, +0.27]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 5 | 60,746 | 0.862 (L10) | 0.862 (L10) | 0.778 (L30) | 0.545 | 0.720 | 0.670 | **+0.084 [+0.02, +0.13]** | **+0.317 [+0.19, +0.48]** | **+0.143 [+0.03, +0.24]** | **+0.193 [+0.07, +0.28]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 10 | 60,746 | 0.813 (L10) | 0.813 (L10) | 0.715 (L30) | 0.550 | 0.668 | 0.620 | **+0.098 [+0.05, +0.14]** | **+0.263 [+0.13, +0.43]** | **+0.145 [+0.04, +0.23]** | **+0.194 [+0.09, +0.28]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 15 | 60,746 | 0.755 (L10) | 0.755 (L10) | 0.667 (L30) | 0.521 | 0.635 | 0.605 | **+0.088 [+0.04, +0.13]** | **+0.234 [+0.10, +0.39]** | **+0.120 [+0.01, +0.20]** | **+0.150 [+0.04, +0.24]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 25 | 60,746 | 0.672 (L30) | 0.685 (L10) | 0.630 (L30) | 0.543 | 0.610 | 0.584 | +0.042 [-0.00, +0.09] | **+0.128 [+0.04, +0.21]** | +0.062 [-0.01, +0.13] | **+0.087 [+0.01, +0.16]** |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 50 | 60,746 | 0.561 (L20) | 0.600 (L10) | 0.548 (L30) | 0.532 | 0.546 | 0.537 | +0.013 [-0.06, +0.08] | +0.028 [-0.05, +0.11] | +0.014 [-0.04, +0.07] | +0.024 [-0.05, +0.09] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 1 | 60,746 | 0.863 (L20) | 0.863 (L20) | 0.859 (L20) | 0.671 | 0.659 | 0.826 | +0.003 [-0.03, +0.07] | **+0.192 [+0.10, +0.24]** | **+0.204 [+0.02, +0.34]** | +0.037 [-0.04, +0.12] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 2 | 60,746 | 0.858 (L20) | 0.858 (L20) | 0.843 (L20) | 0.666 | 0.651 | 0.808 | +0.015 [-0.02, +0.07] | **+0.192 [+0.09, +0.24]** | **+0.207 [+0.03, +0.35]** | +0.050 [-0.04, +0.14] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 3 | 60,746 | 0.851 (L20) | 0.851 (L20) | 0.820 (L20) | 0.651 | 0.624 | 0.779 | +0.032 [-0.00, +0.07] | **+0.201 [+0.09, +0.25]** | **+0.227 [+0.05, +0.36]** | +0.072 [-0.02, +0.15] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 5 | 60,746 | 0.822 (L20) | 0.822 (L20) | 0.793 (L20) | 0.612 | 0.577 | 0.733 | +0.029 [-0.00, +0.08] | **+0.209 [+0.11, +0.29]** | **+0.245 [+0.08, +0.37]** | **+0.089 [+0.01, +0.20]** |
| Qwen3.6-35B-A3B / Verified | has_regressions | 10 | 60,746 | 0.757 (L10) | 0.764 (L20) | 0.725 (L20) | 0.599 | 0.565 | 0.687 | +0.031 [-0.01, +0.10] | **+0.157 [+0.02, +0.24]** | **+0.192 [+0.04, +0.32]** | +0.070 [-0.05, +0.21] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 15 | 60,746 | 0.718 (L20) | 0.718 (L20) | 0.674 (L20) | 0.613 | 0.552 | 0.644 | **+0.044 [+0.00, +0.09]** | +0.105 [-0.01, +0.17] | **+0.166 [+0.01, +0.31]** | +0.074 [-0.04, +0.20] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 25 | 60,746 | 0.625 (L20) | 0.625 (L10) | 0.597 (L20) | 0.571 | 0.543 | 0.614 | +0.028 [-0.02, +0.09] | +0.054 [-0.02, +0.13] | +0.082 [-0.03, +0.22] | +0.011 [-0.15, +0.17] |
| Qwen3.6-35B-A3B / Verified | has_regressions | 50 | 60,746 | 0.635 (L0) | 0.649 (L20) | 0.554 (L20) | 0.569 | 0.590 | 0.540 | +0.081 [-0.04, +0.18] | +0.066 [-0.03, +0.13] | +0.046 [-0.03, +0.12] | +0.095 [-0.04, +0.19] |
| Laguna XS.2 / Pro | correct | 1 | 126,286 | 0.842 (L20) | 0.842 (L20) | 0.804 (L20) | 0.315 | 0.707 | 0.826 | +0.038 [-0.01, +0.09] | **+0.527 [+0.37, +0.65]** | **+0.135 [+0.03, +0.26]** | +0.016 [-0.11, +0.18] |
| Laguna XS.2 / Pro | correct | 2 | 126,426 | 0.807 (L20) | 0.807 (L20) | 0.781 (L20) | 0.334 | 0.714 | 0.858 | +0.026 [-0.03, +0.08] | **+0.474 [+0.34, +0.60]** | **+0.094 [+0.01, +0.18]** | -0.050 [-0.15, +0.07] |
| Laguna XS.2 / Pro | correct | 3 | 126,597 | 0.788 (L20) | 0.788 (L20) | 0.773 (L20) | 0.351 | 0.690 | 0.853 | +0.015 [-0.05, +0.07] | **+0.437 [+0.31, +0.57]** | **+0.098 [+0.01, +0.18]** | -0.065 [-0.17, +0.04] |
| Laguna XS.2 / Pro | correct | 5 | 126,726 | 0.778 (L20) | 0.778 (L20) | 0.762 (L20) | 0.375 | 0.688 | 0.841 | +0.016 [-0.04, +0.06] | **+0.403 [+0.30, +0.51]** | **+0.090 [+0.02, +0.15]** | -0.063 [-0.13, +0.00] |
| Laguna XS.2 / Pro | correct | 10 | 127,012 | 0.719 (L20) | 0.719 (L20) | 0.727 (L20) | 0.429 | 0.666 | 0.795 | -0.008 [-0.07, +0.05] | **+0.290 [+0.19, +0.38]** | +0.054 [-0.04, +0.13] | -0.075 [-0.17, +0.01] |
| Laguna XS.2 / Pro | correct | 15 | 127,024 | 0.683 (L20) | 0.683 (L20) | 0.687 (L20) | 0.521 | 0.607 | 0.721 | -0.004 [-0.08, +0.06] | **+0.162 [+0.06, +0.27]** | +0.076 [-0.00, +0.14] | -0.038 [-0.15, +0.05] |
| Laguna XS.2 / Pro | correct | 25 | 126,906 | 0.622 (L20) | 0.622 (L20) | 0.603 (L20) | 0.529 | 0.555 | 0.622 | +0.020 [-0.07, +0.10] | +0.093 [-0.03, +0.20] | +0.067 [-0.01, +0.14] | +0.001 [-0.12, +0.10] |
| Laguna XS.2 / Pro | correct | 50 | 126,310 | 0.605 (L20) | 0.605 (L20) | 0.547 (L20) | 0.527 | 0.530 | 0.546 | +0.058 [-0.02, +0.14] | +0.078 [-0.04, +0.19] | +0.076 [-0.03, +0.16] | +0.059 [-0.04, +0.15] |
| Laguna XS.2 / Pro | reduces_failing | 1 | 126,286 | 0.858 (L20) | 0.858 (L20) | 0.839 (L20) | 0.542 | 0.761 | 0.808 | +0.019 [-0.00, +0.05] | **+0.316 [+0.21, +0.39]** | **+0.097 [+0.03, +0.18]** | +0.050 [-0.04, +0.12] |
| Laguna XS.2 / Pro | reduces_failing | 2 | 126,426 | 0.857 (L20) | 0.857 (L20) | 0.835 (L20) | 0.547 | 0.758 | 0.791 | **+0.022 [+0.00, +0.06]** | **+0.310 [+0.20, +0.38]** | **+0.099 [+0.04, +0.17]** | +0.066 [-0.04, +0.12] |
| Laguna XS.2 / Pro | reduces_failing | 3 | 126,597 | 0.859 (L20) | 0.859 (L20) | 0.829 (L20) | 0.509 | 0.743 | 0.758 | **+0.029 [+0.00, +0.08]** | **+0.350 [+0.25, +0.41]** | **+0.116 [+0.06, +0.18]** | +0.101 [-0.01, +0.16] |
| Laguna XS.2 / Pro | reduces_failing | 5 | 126,726 | 0.839 (L20) | 0.839 (L20) | 0.807 (L20) | 0.564 | 0.722 | 0.733 | **+0.032 [+0.00, +0.09]** | **+0.275 [+0.17, +0.34]** | **+0.117 [+0.06, +0.18]** | +0.106 [-0.02, +0.18] |
| Laguna XS.2 / Pro | reduces_failing | 10 | 127,012 | 0.784 (L30) | 0.792 (L20) | 0.759 (L20) | 0.607 | 0.664 | 0.676 | +0.024 [-0.02, +0.09] | +0.177 [-0.01, +0.28] | **+0.119 [+0.05, +0.18]** | +0.107 [-0.03, +0.19] |
| Laguna XS.2 / Pro | reduces_failing | 15 | 127,024 | 0.745 (L30) | 0.749 (L10) | 0.706 (L20) | 0.533 | 0.606 | 0.628 | +0.039 [-0.01, +0.12] | **+0.212 [+0.03, +0.33]** | **+0.139 [+0.07, +0.20]** | +0.118 [-0.01, +0.20] |
| Laguna XS.2 / Pro | reduces_failing | 25 | 126,906 | 0.695 (L10) | 0.695 (L10) | 0.627 (L20) | 0.518 | 0.556 | 0.542 | +0.067 [-0.00, +0.15] | **+0.176 [+0.07, +0.26]** | **+0.139 [+0.06, +0.21]** | **+0.152 [+0.05, +0.23]** |
| Laguna XS.2 / Pro | reduces_failing | 50 | 126,310 | 0.692 (L10) | 0.692 (L10) | 0.590 (L20) | 0.547 | 0.534 | 0.493 | **+0.102 [+0.02, +0.21]** | **+0.145 [+0.06, +0.22]** | **+0.158 [+0.07, +0.25]** | **+0.199 [+0.09, +0.29]** |
| Laguna XS.2 / Pro | has_regressions | 1 | 126,286 | 0.875 (L20) | 0.875 (L20) | 0.839 (L20) | 0.429 | 0.773 | 0.674 | **+0.036 [+0.01, +0.05]** | **+0.446 [+0.31, +0.54]** | +0.102 [-0.00, +0.20] | +0.201 [-0.03, +0.34] |
| Laguna XS.2 / Pro | has_regressions | 2 | 126,426 | 0.737 (L39) | 0.837 (L20) | 0.791 (L20) | 0.409 | 0.738 | 0.630 | -0.054 [-0.15, +0.01] | **+0.328 [+0.19, +0.43]** | -0.001 [-0.16, +0.16] | +0.107 [-0.03, +0.26] |
| Laguna XS.2 / Pro | has_regressions | 3 | 126,597 | 0.793 (L10) | 0.823 (L20) | 0.764 (L20) | 0.404 | 0.740 | 0.657 | +0.029 [-0.07, +0.10] | **+0.389 [+0.30, +0.47]** | +0.053 [-0.09, +0.19] | **+0.137 [+0.02, +0.27]** |
| Laguna XS.2 / Pro | has_regressions | 5 | 126,726 | 0.795 (L10) | 0.805 (L30) | 0.734 (L20) | 0.392 | 0.702 | 0.613 | +0.061 [-0.05, +0.16] | **+0.402 [+0.27, +0.50]** | +0.092 [-0.06, +0.26] | **+0.181 [+0.07, +0.35]** |
| Laguna XS.2 / Pro | has_regressions | 10 | 127,012 | 0.705 (L10) | 0.748 (L30) | 0.675 (L20) | 0.337 | 0.747 | 0.621 | +0.030 [-0.06, +0.11] | **+0.368 [+0.21, +0.46]** | -0.042 [-0.12, +0.26] | +0.084 [-0.00, +0.28] |
| Laguna XS.2 / Pro | has_regressions | 15 | 127,024 | 0.693 (L10) | 0.760 (L39) | 0.627 (L20) | 0.459 | 0.765 | 0.600 | +0.067 [-0.06, +0.12] | +0.235 [-0.10, +0.37] | -0.072 [-0.14, +0.19] | **+0.093 [+0.01, +0.25]** |
| Laguna XS.2 / Pro | has_regressions | 25 | 126,906 | 0.696 (L10) | 0.775 (L39) | 0.598 (L20) | 0.401 | 0.785 | 0.553 | +0.098 [-0.06, +0.16] | **+0.295 [+0.06, +0.50]** | -0.089 [-0.18, +0.17] | **+0.143 [+0.07, +0.22]** |
| Laguna XS.2 / Pro | has_regressions | 50 | 126,310 | 0.681 (L10) | 0.722 (L39) | 0.591 (L20) | 0.435 | 0.746 | 0.550 | +0.090 [-0.03, +0.18] | **+0.246 [+0.02, +0.35]** | -0.065 [-0.17, +0.15] | **+0.131 [+0.03, +0.26]** |
| Qwen3.6-35B-A3B / Pro | correct | 1 | 66,858 | 0.615 (L0) | 0.829 (L30) | 0.624 (L0) | 0.462 | 0.673 | 0.731 | -0.010 [-0.02, +0.01] | **+0.153 [+0.10, +0.31]** | -0.058 [-0.16, +0.33] | -0.116 [-0.22, +0.20] |
| Qwen3.6-35B-A3B / Pro | correct | 2 | 66,858 | 0.610 (L0) | 0.817 (L30) | 0.621 (L0) | 0.450 | 0.658 | 0.714 | -0.011 [-0.02, +0.01] | **+0.160 [+0.10, +0.33]** | -0.048 [-0.15, +0.31] | -0.104 [-0.21, +0.20] |
| Qwen3.6-35B-A3B / Pro | correct | 3 | 66,858 | 0.606 (L0) | 0.812 (L30) | 0.615 (L0) | 0.461 | 0.646 | 0.704 | -0.010 [-0.02, +0.02] | **+0.145 [+0.07, +0.33]** | -0.041 [-0.14, +0.30] | -0.098 [-0.20, +0.19] |
| Qwen3.6-35B-A3B / Pro | correct | 5 | 66,845 | 0.585 (L0) | 0.787 (L30) | 0.603 (L0) | 0.448 | 0.630 | 0.680 | -0.017 [-0.03, +0.02] | **+0.138 [+0.06, +0.31]** | -0.045 [-0.15, +0.28] | -0.095 [-0.20, +0.21] |
| Qwen3.6-35B-A3B / Pro | correct | 10 | 66,837 | 0.737 (L20) | 0.745 (L30) | 0.585 (L0) | 0.442 | 0.595 | 0.640 | +0.152 [-0.00, +0.22] | **+0.295 [+0.21, +0.36]** | +0.142 [-0.06, +0.29] | +0.097 [-0.03, +0.24] |
| Qwen3.6-35B-A3B / Pro | correct | 15 | 66,838 | 0.734 (L20) | 0.742 (L30) | 0.572 (L0) | 0.451 | 0.580 | 0.625 | **+0.162 [+0.01, +0.24]** | **+0.283 [+0.05, +0.47]** | +0.154 [-0.04, +0.28] | +0.109 [-0.02, +0.24] |
| Qwen3.6-35B-A3B / Pro | correct | 25 | 66,836 | 0.728 (L20) | 0.736 (L30) | 0.532 (L0) | 0.422 | 0.532 | 0.581 | **+0.196 [+0.03, +0.29]** | **+0.306 [+0.07, +0.41]** | +0.196 [-0.01, +0.32] | **+0.147 [+0.01, +0.24]** |
| Qwen3.6-35B-A3B / Pro | correct | 50 | 67,013 | 0.688 (L30) | 0.688 (L30) | 0.510 (L0) | 0.451 | 0.478 | 0.497 | **+0.179 [+0.07, +0.26]** | **+0.237 [+0.08, +0.37]** | **+0.210 [+0.07, +0.33]** | **+0.192 [+0.09, +0.29]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 1 | 66,858 | 0.886 (L10) | 0.906 (L20) | 0.845 (L10) | 0.302 | 0.787 | 0.782 | **+0.040 [+0.03, +0.07]** | **+0.584 [+0.48, +0.65]** | +0.098 [-0.04, +0.26] | **+0.103 [+0.02, +0.15]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 2 | 66,858 | 0.877 (L10) | 0.906 (L20) | 0.838 (L10) | 0.362 | 0.782 | 0.766 | **+0.038 [+0.01, +0.07]** | **+0.515 [+0.39, +0.57]** | +0.095 [-0.06, +0.24] | **+0.111 [+0.02, +0.16]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 3 | 66,858 | 0.872 (L10) | 0.901 (L20) | 0.835 (L10) | 0.390 | 0.780 | 0.759 | **+0.036 [+0.01, +0.07]** | **+0.482 [+0.35, +0.54]** | +0.092 [-0.05, +0.23] | **+0.113 [+0.03, +0.17]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 5 | 66,845 | 0.870 (L10) | 0.901 (L20) | 0.816 (L10) | 0.428 | 0.778 | 0.729 | **+0.055 [+0.02, +0.10]** | **+0.443 [+0.30, +0.50]** | +0.093 [-0.03, +0.20] | **+0.141 [+0.06, +0.21]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 10 | 66,837 | 0.868 (L20) | 0.868 (L20) | 0.786 (L10) | 0.660 | 0.749 | 0.682 | **+0.082 [+0.04, +0.13]** | **+0.208 [+0.14, +0.28]** | **+0.119 [+0.01, +0.25]** | **+0.186 [+0.10, +0.27]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 15 | 66,838 | 0.862 (L20) | 0.862 (L20) | 0.757 (L10) | 0.318 | 0.712 | 0.637 | **+0.104 [+0.05, +0.16]** | **+0.544 [+0.35, +0.63]** | **+0.150 [+0.03, +0.26]** | **+0.224 [+0.12, +0.32]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 25 | 66,836 | 0.824 (L20) | 0.824 (L20) | 0.710 (L10) | 0.529 | 0.665 | 0.556 | **+0.115 [+0.05, +0.18]** | **+0.295 [+0.12, +0.46]** | **+0.159 [+0.02, +0.27]** | **+0.268 [+0.12, +0.40]** |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 50 | 67,013 | 0.719 (L0) | 0.739 (L30) | 0.578 (L10) | 0.417 | 0.474 | 0.526 | +0.141 [-0.03, +0.31] | **+0.301 [+0.05, +0.56]** | **+0.245 [+0.09, +0.45]** | **+0.192 [+0.04, +0.38]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 1 | 66,858 | 0.748 (L30) | 0.867 (L10) | 0.865 (L10) | 0.444 | 0.978 | 0.887 | **-0.117 [-0.13, -0.10]** | **+0.304 [+0.21, +0.39]** | **-0.230 [-0.29, -0.18]** | **-0.139 [-0.21, -0.07]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 2 | 66,858 | 0.903 (L10) | 0.903 (L10) | 0.885 (L10) | 0.346 | 0.942 | 0.857 | **+0.018 [+0.01, +0.03]** | **+0.557 [+0.46, +0.64]** | -0.039 [-0.09, +0.01] | +0.047 [-0.02, +0.11] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 3 | 66,858 | 0.913 (L10) | 0.913 (L10) | 0.880 (L10) | 0.453 | 0.945 | 0.851 | **+0.032 [+0.02, +0.05]** | **+0.459 [+0.37, +0.53]** | -0.032 [-0.08, +0.02] | **+0.062 [+0.00, +0.12]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 5 | 66,845 | 0.947 (L10) | 0.947 (L10) | 0.911 (L10) | 0.497 | 0.852 | 0.762 | **+0.035 [+0.02, +0.05]** | **+0.450 [+0.38, +0.52]** | **+0.094 [+0.04, +0.15]** | **+0.184 [+0.12, +0.24]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 10 | 66,837 | 0.825 (L10) | 0.825 (L10) | 0.671 (L10) | 0.810 | 0.559 | 0.534 | **+0.153 [+0.13, +0.18]** | +0.015 [-0.08, +0.10] | **+0.266 [+0.17, +0.38]** | **+0.291 [+0.00, +0.38]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 15 | 66,838 | 0.710 (L10) | 0.760 (L30) | 0.578 (L10) | 0.555 | 0.517 | 0.482 | **+0.131 [+0.10, +0.16]** | **+0.155 [+0.10, +0.19]** | **+0.193 [+0.08, +0.28]** | +0.227 [-0.06, +0.38] |
| Qwen3.6-35B-A3B / Pro | has_regressions | 25 | 66,836 | 0.753 (L10) | 0.807 (L30) | 0.653 (L10) | 0.357 | 0.425 | 0.502 | **+0.100 [+0.01, +0.19]** | **+0.396 [+0.07, +0.72]** | **+0.328 [+0.18, +0.44]** | **+0.251 [+0.06, +0.46]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | 50 | 67,013 | 0.840 (L20) | 0.859 (L30) | 0.565 (L10) | 0.456 | 0.553 | 0.664 | **+0.276 [+0.18, +0.53]** | **+0.385 [+0.29, +0.55]** | **+0.288 [+0.21, +0.40]** | **+0.176 [+0.09, +0.47]** |

#### k=0 selectivity (E17), pooled k=0 caches, all test tokens (layers chosen on validation AUC against real labels; differences with 95% task-bootstrap CI, bold if the CI excludes 0)

| Model / benchmark | Probe | real (val-best) | token shuffle (val-best) | traj shuffle (val-best) | real − token | real − traj | real − token, same layer | real − traj, same layer | position, causal | position, rel_pos (non-causal) | real − position causal | real − position rel_pos | real, test-best layer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | compiles | 0.691 (L10) | 0.498 (L30) | 0.604 (L30) | **+0.193 [+0.11, +0.29]** | **+0.087 [+0.01, +0.18]** | **+0.177 [+0.11, +0.25]** | **+0.103 [+0.03, +0.19]** | 0.660 | 0.616 | +0.031 [-0.05, +0.15] | **+0.075 [+0.02, +0.13]** | 0.691 (L10) |
| Laguna XS.2 / Verified | correct | 0.788 (L20) | 0.518 (L39) | 0.656 (L30) | **+0.271 [+0.22, +0.32]** | **+0.132 [+0.07, +0.20]** | **+0.276 [+0.23, +0.33]** | **+0.123 [+0.06, +0.19]** | 0.612 | 0.690 | **+0.177 [+0.11, +0.24]** | **+0.098 [+0.05, +0.15]** | 0.788 (L20) |
| Laguna XS.2 / Verified | has_regressions | 0.695 (L20) | 0.508 (L20) | 0.631 (L10) | **+0.188 [+0.12, +0.25]** | **+0.064 [+0.00, +0.14]** | **+0.188 [+0.12, +0.25]** | **+0.131 [+0.07, +0.20]** | 0.658 | 0.579 | +0.038 [-0.01, +0.10] | **+0.116 [+0.06, +0.17]** | 0.695 (L20) |
| Laguna XS.2 / Verified | reduces_failing | 0.773 (L20) | 0.518 (L30) | 0.680 (L20) | **+0.255 [+0.21, +0.30]** | **+0.094 [+0.03, +0.16]** | **+0.282 [+0.22, +0.34]** | **+0.094 [+0.03, +0.16]** | 0.653 | 0.698 | **+0.120 [+0.05, +0.19]** | **+0.075 [+0.03, +0.12]** | 0.773 (L20) |
| Qwen3.6-35B-A3B / Verified | compiles | 0.736 (L39) | 0.508 (L30) | 0.683 (L10) | **+0.228 [+0.14, +0.33]** | +0.052 [-0.03, +0.14] | **+0.228 [+0.15, +0.32]** | **+0.144 [+0.05, +0.25]** | 0.664 | 0.683 | +0.072 [-0.01, +0.21] | +0.053 [-0.02, +0.14] | 0.781 (L10) |
| Qwen3.6-35B-A3B / Verified | correct | 0.838 (L30) | 0.513 (L39) | 0.776 (L30) | **+0.325 [+0.28, +0.37]** | **+0.062 [+0.02, +0.11]** | **+0.324 [+0.28, +0.36]** | **+0.062 [+0.02, +0.11]** | 0.595 | 0.771 | **+0.243 [+0.18, +0.31]** | **+0.066 [+0.03, +0.10]** | 0.838 (L30) |
| Qwen3.6-35B-A3B / Verified | has_regressions | 0.770 (L30) | 0.513 (L39) | 0.607 (L30) | **+0.257 [+0.21, +0.32]** | **+0.163 [+0.11, +0.21]** | **+0.264 [+0.22, +0.31]** | **+0.163 [+0.11, +0.21]** | 0.683 | 0.669 | +0.086 [-0.00, +0.21] | **+0.101 [+0.04, +0.15]** | 0.772 (L20) |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 0.832 (L30) | 0.548 (L30) | 0.778 (L20) | **+0.285 [+0.25, +0.32]** | **+0.054 [+0.02, +0.09]** | **+0.285 [+0.25, +0.32]** | **+0.055 [+0.02, +0.09]** | 0.619 | 0.775 | **+0.213 [+0.15, +0.28]** | **+0.057 [+0.02, +0.09]** | 0.832 (L30) |
| Laguna XS.2 / Pro | compiles | 0.786 (L10) | 0.495 (L20) | 0.466 (L39) | **+0.291 [+0.22, +0.35]** | **+0.320 [+0.24, +0.39]** | **+0.297 [+0.23, +0.36]** | **+0.322 [+0.25, +0.39]** | 0.550 | 0.510 | **+0.236 [+0.16, +0.31]** | **+0.276 [+0.21, +0.34]** | 0.786 (L10) |
| Laguna XS.2 / Pro | correct | 0.794 (L20) | 0.494 (L30) | 0.649 (L20) | **+0.300 [+0.24, +0.34]** | **+0.145 [+0.09, +0.20]** | **+0.263 [+0.21, +0.32]** | **+0.145 [+0.09, +0.20]** | 0.601 | 0.713 | **+0.193 [+0.11, +0.28]** | **+0.081 [+0.02, +0.14]** | 0.794 (L20) |
| Laguna XS.2 / Pro | has_regressions | 0.740 (L20) | 0.518 (L20) | 0.564 (L20) | **+0.222 [+0.16, +0.27]** | **+0.176 [+0.13, +0.23]** | **+0.222 [+0.16, +0.27]** | **+0.176 [+0.13, +0.23]** | 0.685 | 0.612 | +0.054 [-0.04, +0.15] | **+0.127 [+0.06, +0.19]** | 0.740 (L20) |
| Laguna XS.2 / Pro | reduces_failing | 0.780 (L20) | 0.508 (L0) | 0.632 (L20) | **+0.273 [+0.21, +0.33]** | **+0.148 [+0.09, +0.22]** | **+0.319 [+0.26, +0.38]** | **+0.148 [+0.09, +0.22]** | 0.662 | 0.627 | **+0.118 [+0.04, +0.20]** | **+0.153 [+0.10, +0.21]** | 0.780 (L20) |
| Qwen3.6-35B-A3B / Pro | compiles | 0.720 (L39) | 0.547 (L39) | 0.557 (L30) | **+0.172 [+0.11, +0.23]** | **+0.163 [+0.10, +0.24]** | **+0.172 [+0.11, +0.23]** | **+0.220 [+0.15, +0.29]** | 0.522 | 0.497 | **+0.198 [+0.09, +0.30]** | **+0.223 [+0.16, +0.29]** | 0.750 (L0) |
| Qwen3.6-35B-A3B / Pro | correct | 0.822 (L30) | 0.513 (L10) | 0.771 (L30) | **+0.308 [+0.27, +0.34]** | **+0.051 [+0.02, +0.08]** | **+0.305 [+0.27, +0.35]** | **+0.051 [+0.02, +0.08]** | 0.692 | 0.776 | **+0.130 [+0.05, +0.22]** | **+0.046 [+0.01, +0.08]** | 0.822 (L30) |
| Qwen3.6-35B-A3B / Pro | has_regressions | 0.818 (L20) | 0.539 (L10) | 0.693 (L10) | **+0.279 [+0.22, +0.33]** | **+0.125 [+0.08, +0.23]** | **+0.296 [+0.23, +0.37]** | **+0.182 [+0.14, +0.26]** | 0.741 | 0.560 | **+0.077 [+0.02, +0.16]** | **+0.259 [+0.19, +0.33]** | 0.840 (L0) |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 0.830 (L20) | 0.507 (L20) | 0.761 (L30) | **+0.323 [+0.27, +0.36]** | **+0.069 [+0.03, +0.10]** | **+0.323 [+0.27, +0.36]** | **+0.074 [+0.04, +0.10]** | 0.762 | 0.718 | **+0.068 [+0.01, +0.13]** | **+0.112 [+0.06, +0.15]** | 0.836 (L30) |

#### E18, window max15: within y_t = 0, by edit turn and eventual outcome (AUC for y_(t+k); strongest control = highest point AUC among k=0 probe, traj-shuffle and both position baselines; 95% task-bootstrap CI, bold if it excludes 0; ◦ = fewer than 20 test tasks with a positive token)

| Model / benchmark | Probe | k | stratum | tokens | positives | tasks w/ positive | real | strongest control | real − strongest | strongest causal control | real − strongest causal | oracle y_T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | correct | 2 | S0 y_t=0 | 522,132 | 38,097 | 58 | 0.646 | position rel_pos 0.491 | **+0.155 [+0.07, +0.24]** | position causal 0.483 | **+0.163 [+0.07, +0.26]** | 0.813 |
| Laguna XS.2 / Verified | correct | 2 | S1 pre-edit (y_t+1=0) | 493,325 | 9,666 | 57 | 0.544 | position rel_pos 0.472 | +0.072 [-0.02, +0.16] | position causal 0.464 | +0.081 [-0.01, +0.18] | 0.816 |
| Laguna XS.2 / Verified | correct | 2 | S2 outcome y_T=1 | 180,839 | 35,335 | 56 | 0.719 | position rel_pos 0.631 | **+0.087 [+0.02, +0.15]** | traj-shuffle 0.608 | **+0.111 [+0.04, +0.16]** | – |
| Laguna XS.2 / Verified | correct | 2 | S3 pre-edit & y_T=1 | 154,403 | 9,011 | 55 | 0.621 | position rel_pos 0.617 | +0.004 [-0.07, +0.06] | position causal 0.554 | +0.067 [-0.02, +0.16] | – |
| Laguna XS.2 / Verified | correct | 2 | S4 y_T=0 | 341,293 | 2,762 | 10◦ | 0.587◦ | traj-shuffle 0.606 | -0.019 [-0.13, +0.17] | traj-shuffle 0.606 | -0.019 [-0.13, +0.17] | – |
| Laguna XS.2 / Verified | correct | 3 | S0 y_t=0 | 522,125 | 47,141 | 58 | 0.635 | position causal 0.498 | **+0.137 [+0.05, +0.23]** | position causal 0.498 | **+0.137 [+0.05, +0.23]** | 0.827 |
| Laguna XS.2 / Verified | correct | 3 | S1 pre-edit (y_t+1=0) | 493,318 | 19,324 | 58 | 0.555 | position causal 0.477 | **+0.079 [+0.00, +0.17]** | position causal 0.477 | **+0.079 [+0.00, +0.17]** | 0.829 |
| Laguna XS.2 / Verified | correct | 3 | S2 outcome y_T=1 | 180,832 | 44,416 | 56 | 0.709 | position rel_pos 0.630 | **+0.080 [+0.01, +0.14]** | traj-shuffle 0.585 | **+0.124 [+0.06, +0.18]** | – |
| Laguna XS.2 / Verified | correct | 3 | S3 pre-edit & y_T=1 | 154,396 | 18,271 | 56 | 0.633 | position rel_pos 0.614 | +0.019 [-0.04, +0.08] | traj-shuffle 0.560 | **+0.073 [+0.01, +0.13]** | – |
| Laguna XS.2 / Verified | correct | 3 | S4 y_T=0 | 341,293 | 2,725 | 10◦ | 0.593◦ | position causal 0.546 | +0.047 [-0.17, +0.26] | position causal 0.546 | +0.047 [-0.17, +0.26] | – |
| Laguna XS.2 / Verified | correct | 5 | S0 y_t=0 | 522,125 | 64,820 | 58 | 0.618 | position causal 0.513 | **+0.105 [+0.03, +0.19]** | position causal 0.513 | **+0.105 [+0.03, +0.19]** | 0.841 |
| Laguna XS.2 / Verified | correct | 5 | S1 pre-edit (y_t+1=0) | 493,239 | 37,039 | 58 | 0.554 | position causal 0.482 | +0.073 [-0.00, +0.16] | position causal 0.482 | +0.073 [-0.00, +0.16] | 0.844 |
| Laguna XS.2 / Verified | correct | 5 | S2 outcome y_T=1 | 180,832 | 61,219 | 56 | 0.682 | position rel_pos 0.617 | +0.065 [-0.00, +0.12] | traj-shuffle 0.583 | **+0.099 [+0.03, +0.15]** | – |
| Laguna XS.2 / Verified | correct | 5 | S3 pre-edit & y_T=1 | 154,317 | 35,180 | 56 | 0.621 | position rel_pos 0.594 | +0.027 [-0.04, +0.09] | traj-shuffle 0.553 | **+0.068 [+0.00, +0.13]** | – |
| Laguna XS.2 / Verified | correct | 5 | S4 y_T=0 | 341,293 | 3,601 | 10◦ | 0.594◦ | traj-shuffle 0.519 | +0.076 [-0.06, +0.23] | traj-shuffle 0.519 | +0.076 [-0.06, +0.23] | – |
| Laguna XS.2 / Verified | correct | 10 | S0 y_t=0 | 522,143 | 97,600 | 58 | 0.606 | position causal 0.551 | +0.054 [-0.03, +0.14] | position causal 0.551 | +0.054 [-0.03, +0.14] | 0.870 |
| Laguna XS.2 / Verified | correct | 10 | S1 pre-edit (y_t+1=0) | 493,257 | 70,035 | 58 | 0.569 | position causal 0.519 | +0.050 [-0.03, +0.14] | position causal 0.519 | +0.050 [-0.03, +0.14] | 0.872 |
| Laguna XS.2 / Verified | correct | 10 | S2 outcome y_T=1 | 180,850 | 92,535 | 56 | 0.648 | position rel_pos 0.623 | +0.026 [-0.05, +0.10] | k=0 probe 0.552 | **+0.096 [+0.06, +0.14]** | – |
| Laguna XS.2 / Verified | correct | 10 | S3 pre-edit & y_T=1 | 154,335 | 66,639 | 56 | 0.613 | position rel_pos 0.605 | +0.008 [-0.07, +0.07] | k=0 probe 0.564 | **+0.049 [+0.02, +0.08]** | – |
| Laguna XS.2 / Verified | correct | 10 | S4 y_T=0 | 341,293 | 5,065 | 10◦ | 0.584◦ | position causal 0.532 | +0.052 [-0.07, +0.21] | position causal 0.532 | +0.052 [-0.07, +0.21] | – |
| Laguna XS.2 / Verified | correct | 15 | S0 y_t=0 | 522,200 | 122,830 | 59 | 0.604 | position causal 0.583 | +0.021 [-0.06, +0.11] | position causal 0.583 | +0.021 [-0.06, +0.11] | 0.896 |
| Laguna XS.2 / Verified | correct | 15 | S1 pre-edit (y_t+1=0) | 493,314 | 95,381 | 59 | 0.576 | position causal 0.560 | +0.015 [-0.07, +0.11] | position causal 0.560 | +0.015 [-0.07, +0.11] | 0.898 |
| Laguna XS.2 / Verified | correct | 15 | S2 outcome y_T=1 | 180,907 | 116,982 | 57 | 0.639 | position rel_pos 0.636 | +0.002 [-0.08, +0.08] | k=0 probe 0.575 | **+0.064 [+0.03, +0.11]** | – |
| Laguna XS.2 / Verified | correct | 15 | S3 pre-edit & y_T=1 | 154,392 | 91,094 | 57 | 0.612 | position rel_pos 0.620 | -0.008 [-0.09, +0.07] | k=0 probe 0.582 | +0.030 [-0.00, +0.07] | – |
| Laguna XS.2 / Verified | correct | 15 | S4 y_T=0 | 341,293 | 5,848 | 10◦ | 0.598◦ | k=0 probe 0.538 | +0.060 [-0.01, +0.22] | k=0 probe 0.538 | +0.060 [-0.01, +0.22] | – |
| Laguna XS.2 / Verified | reduces_failing | 2 | S0 y_t=0 | 498,553 | 40,409 | 62 | 0.660 | position rel_pos 0.471 | **+0.190 [+0.12, +0.26]** | traj-shuffle 0.455 | **+0.206 [+0.14, +0.27]** | 0.781 |
| Laguna XS.2 / Verified | reduces_failing | 2 | S1 pre-edit (y_t+1=0) | 468,885 | 11,199 | 62 | 0.539 | position rel_pos 0.450 | **+0.089 [+0.01, +0.17]** | position causal 0.443 | **+0.096 [+0.01, +0.18]** | 0.784 |
| Laguna XS.2 / Verified | reduces_failing | 2 | S2 outcome y_T=1 | 193,726 | 36,588 | 55 | 0.776 | position rel_pos 0.597 | **+0.179 [+0.10, +0.25]** | traj-shuffle 0.557 | **+0.219 [+0.15, +0.29]** | – |
| Laguna XS.2 / Verified | reduces_failing | 2 | S3 pre-edit & y_T=1 | 167,237 | 10,206 | 55 | 0.659 | position rel_pos 0.587 | +0.072 [-0.01, +0.14] | k=0 probe 0.548 | **+0.112 [+0.07, +0.16]** | – |
| Laguna XS.2 / Verified | reduces_failing | 2 | S4 y_T=0 | 304,827 | 3,821 | 15◦ | 0.611◦ | position causal 0.607 | +0.004 [-0.14, +0.19] | position causal 0.607 | +0.004 [-0.14, +0.19] | – |
| Laguna XS.2 / Verified | reduces_failing | 3 | S0 y_t=0 | 498,546 | 50,878 | 62 | 0.638 | position rel_pos 0.462 | **+0.176 [+0.11, +0.24]** | position causal 0.459 | **+0.179 [+0.11, +0.25]** | 0.801 |
| Laguna XS.2 / Verified | reduces_failing | 3 | S1 pre-edit (y_t+1=0) | 468,878 | 22,673 | 62 | 0.544 | position causal 0.468 | **+0.076 [+0.01, +0.15]** | position causal 0.468 | **+0.076 [+0.01, +0.15]** | 0.801 |
| Laguna XS.2 / Verified | reduces_failing | 3 | S2 outcome y_T=1 | 193,719 | 47,260 | 55 | 0.759 | position rel_pos 0.590 | **+0.169 [+0.10, +0.23]** | k=0 probe 0.561 | **+0.198 [+0.16, +0.24]** | – |
| Laguna XS.2 / Verified | reduces_failing | 3 | S3 pre-edit & y_T=1 | 167,230 | 21,057 | 55 | 0.669 | position rel_pos 0.573 | **+0.096 [+0.02, +0.16]** | k=0 probe 0.564 | **+0.105 [+0.07, +0.14]** | – |
| Laguna XS.2 / Verified | reduces_failing | 3 | S4 y_T=0 | 304,827 | 3,618 | 15◦ | 0.576◦ | position causal 0.616 | -0.041 [-0.18, +0.11] | position causal 0.616 | -0.041 [-0.18, +0.11] | – |
| Laguna XS.2 / Verified | reduces_failing | 5 | S0 y_t=0 | 498,546 | 68,856 | 62 | 0.606 | position causal 0.476 | **+0.131 [+0.06, +0.20]** | position causal 0.476 | **+0.131 [+0.06, +0.20]** | 0.820 |
| Laguna XS.2 / Verified | reduces_failing | 5 | S1 pre-edit (y_t+1=0) | 468,799 | 41,124 | 62 | 0.527 | position causal 0.465 | +0.062 [-0.00, +0.13] | position causal 0.465 | +0.062 [-0.00, +0.13] | 0.821 |
| Laguna XS.2 / Verified | reduces_failing | 5 | S2 outcome y_T=1 | 193,719 | 64,750 | 55 | 0.727 | position rel_pos 0.570 | **+0.157 [+0.08, +0.23]** | k=0 probe 0.554 | **+0.174 [+0.13, +0.22]** | – |
| Laguna XS.2 / Verified | reduces_failing | 5 | S3 pre-edit & y_T=1 | 167,151 | 38,754 | 55 | 0.654 | k=0 probe 0.550 | **+0.104 [+0.07, +0.14]** | k=0 probe 0.550 | **+0.104 [+0.07, +0.14]** | – |
| Laguna XS.2 / Verified | reduces_failing | 5 | S4 y_T=0 | 304,827 | 4,106 | 15◦ | 0.544◦ | position causal 0.572 | -0.028 [-0.17, +0.09] | position causal 0.572 | -0.028 [-0.17, +0.09] | – |
| Laguna XS.2 / Verified | reduces_failing | 10 | S0 y_t=0 | 498,564 | 102,039 | 62 | 0.572 | position causal 0.499 | **+0.073 [+0.00, +0.15]** | position causal 0.499 | **+0.073 [+0.00, +0.15]** | 0.858 |
| Laguna XS.2 / Verified | reduces_failing | 10 | S1 pre-edit (y_t+1=0) | 468,817 | 74,688 | 62 | 0.523 | position causal 0.482 | +0.041 [-0.03, +0.11] | position causal 0.482 | +0.041 [-0.03, +0.11] | 0.860 |
| Laguna XS.2 / Verified | reduces_failing | 10 | S2 outcome y_T=1 | 193,737 | 97,750 | 56 | 0.685 | position rel_pos 0.572 | **+0.113 [+0.03, +0.20]** | k=0 probe 0.560 | **+0.126 [+0.07, +0.18]** | – |
| Laguna XS.2 / Verified | reduces_failing | 10 | S3 pre-edit & y_T=1 | 167,169 | 71,792 | 56 | 0.638 | position rel_pos 0.557 | +0.082 [-0.00, +0.17] | k=0 probe 0.555 | **+0.083 [+0.03, +0.14]** | – |
| Laguna XS.2 / Verified | reduces_failing | 10 | S4 y_T=0 | 304,827 | 4,289 | 15◦ | 0.533◦ | position causal 0.513 | +0.020 [-0.10, +0.18] | position causal 0.513 | +0.020 [-0.10, +0.18] | – |
| Laguna XS.2 / Verified | reduces_failing | 15 | S0 y_t=0 | 498,621 | 128,607 | 62 | 0.557 | position causal 0.540 | +0.017 [-0.05, +0.09] | position causal 0.540 | +0.017 [-0.05, +0.09] | 0.888 |
| Laguna XS.2 / Verified | reduces_failing | 15 | S1 pre-edit (y_t+1=0) | 468,874 | 101,304 | 62 | 0.520 | position causal 0.517 | +0.003 [-0.07, +0.08] | position causal 0.517 | +0.003 [-0.07, +0.08] | 0.890 |
| Laguna XS.2 / Verified | reduces_failing | 15 | S2 outcome y_T=1 | 193,794 | 124,057 | 56 | 0.673 | position rel_pos 0.590 | +0.083 [-0.01, +0.18] | k=0 probe 0.581 | **+0.093 [+0.03, +0.16]** | – |
| Laguna XS.2 / Verified | reduces_failing | 15 | S3 pre-edit & y_T=1 | 167,226 | 98,149 | 56 | 0.639 | k=0 probe 0.578 | **+0.060 [+0.00, +0.12]** | k=0 probe 0.578 | **+0.060 [+0.00, +0.12]** | – |
| Laguna XS.2 / Verified | reduces_failing | 15 | S4 y_T=0 | 304,827 | 4,550 | 15◦ | 0.520◦ | position causal 0.517 | +0.003 [-0.10, +0.16] | position causal 0.517 | +0.003 [-0.10, +0.16] | – |
| Laguna XS.2 / Verified | has_regressions | 2 | S0 y_t=0 | 640,080 | 11,431 | 29 | 0.654 | k=0 probe 0.535 | **+0.120 [+0.10, +0.14]** | k=0 probe 0.535 | **+0.120 [+0.10, +0.14]** | 0.688 |
| Laguna XS.2 / Verified | has_regressions | 2 | S1 pre-edit (y_t+1=0) | 631,458 | 2,809 | 29 | 0.570 | k=0 probe 0.504 | **+0.066 [+0.05, +0.09]** | k=0 probe 0.504 | **+0.066 [+0.05, +0.09]** | 0.665 |
| Laguna XS.2 / Verified | has_regressions | 2 | S2 outcome y_T=1 | 27,876 | 4,722 | 14◦ | 0.748◦ | position causal 0.692 | +0.055 [-0.02, +0.13] | position causal 0.692 | +0.055 [-0.02, +0.13] | – |
| Laguna XS.2 / Verified | has_regressions | 2 | S3 pre-edit & y_T=1 | 24,182 | 1,028 | 14◦ | 0.641◦ | position causal 0.652 | -0.011 [-0.12, +0.12] | position causal 0.652 | -0.011 [-0.12, +0.12] | – |
| Laguna XS.2 / Verified | has_regressions | 2 | S4 y_T=0 | 612,204 | 6,709 | 23 | 0.648 | k=0 probe 0.532 | **+0.116 [+0.09, +0.14]** | k=0 probe 0.532 | **+0.116 [+0.09, +0.14]** | – |
| Laguna XS.2 / Verified | has_regressions | 3 | S0 y_t=0 | 640,073 | 14,687 | 31 | 0.621 | k=0 probe 0.516 | **+0.105 [+0.08, +0.12]** | k=0 probe 0.516 | **+0.105 [+0.08, +0.12]** | 0.682 |
| Laguna XS.2 / Verified | has_regressions | 3 | S1 pre-edit (y_t+1=0) | 631,451 | 6,576 | 31 | 0.550 | k=0 probe 0.487 | **+0.063 [+0.05, +0.08]** | k=0 probe 0.487 | **+0.063 [+0.05, +0.08]** | 0.660 |
| Laguna XS.2 / Verified | has_regressions | 3 | S2 outcome y_T=1 | 27,876 | 5,869 | 15◦ | 0.733◦ | position causal 0.680 | +0.053 [-0.02, +0.14] | position causal 0.680 | +0.053 [-0.02, +0.14] | – |
| Laguna XS.2 / Verified | has_regressions | 3 | S3 pre-edit & y_T=1 | 24,182 | 2,338 | 15◦ | 0.663◦ | position causal 0.650 | +0.013 [-0.08, +0.12] | position causal 0.650 | +0.013 [-0.08, +0.12] | – |
| Laguna XS.2 / Verified | has_regressions | 3 | S4 y_T=0 | 612,197 | 8,818 | 25 | 0.608 | k=0 probe 0.504 | **+0.103 [+0.07, +0.13]** | k=0 probe 0.504 | **+0.103 [+0.07, +0.13]** | – |
| Laguna XS.2 / Verified | has_regressions | 5 | S0 y_t=0 | 640,073 | 19,751 | 32 | 0.584 | traj-shuffle 0.500 | **+0.084 [+0.02, +0.15]** | traj-shuffle 0.500 | **+0.084 [+0.02, +0.15]** | 0.689 |
| Laguna XS.2 / Verified | has_regressions | 5 | S1 pre-edit (y_t+1=0) | 631,372 | 12,313 | 32 | 0.534 | k=0 probe 0.465 | **+0.069 [+0.05, +0.09]** | k=0 probe 0.465 | **+0.069 [+0.05, +0.09]** | 0.670 |
| Laguna XS.2 / Verified | has_regressions | 5 | S2 outcome y_T=1 | 27,876 | 8,107 | 15◦ | 0.707◦ | position causal 0.673 | +0.034 [-0.04, +0.13] | position causal 0.673 | +0.034 [-0.04, +0.13] | – |
| Laguna XS.2 / Verified | has_regressions | 5 | S3 pre-edit & y_T=1 | 24,182 | 4,576 | 15◦ | 0.659◦ | position causal 0.647 | +0.012 [-0.07, +0.12] | position causal 0.647 | +0.012 [-0.07, +0.12] | – |
| Laguna XS.2 / Verified | has_regressions | 5 | S4 y_T=0 | 612,197 | 11,644 | 26 | 0.564 | k=0 probe 0.471 | **+0.093 [+0.06, +0.12]** | k=0 probe 0.471 | **+0.093 [+0.06, +0.12]** | – |
| Laguna XS.2 / Verified | has_regressions | 10 | S0 y_t=0 | 640,091 | 26,671 | 31 | 0.563 | k=0 probe 0.467 | **+0.096 [+0.06, +0.12]** | k=0 probe 0.467 | **+0.096 [+0.06, +0.12]** | 0.722 |
| Laguna XS.2 / Verified | has_regressions | 10 | S1 pre-edit (y_t+1=0) | 631,390 | 20,089 | 31 | 0.528 | k=0 probe 0.448 | **+0.080 [+0.05, +0.10]** | k=0 probe 0.448 | **+0.080 [+0.05, +0.10]** | 0.707 |
| Laguna XS.2 / Verified | has_regressions | 10 | S2 outcome y_T=1 | 27,876 | 12,497 | 15◦ | 0.650◦ | position causal 0.655 | -0.005 [-0.09, +0.10] | position causal 0.655 | -0.005 [-0.09, +0.10] | – |
| Laguna XS.2 / Verified | has_regressions | 10 | S3 pre-edit & y_T=1 | 24,182 | 8,803 | 15◦ | 0.602◦ | position causal 0.621 | -0.019 [-0.10, +0.09] | position causal 0.621 | -0.019 [-0.10, +0.09] | – |
| Laguna XS.2 / Verified | has_regressions | 10 | S4 y_T=0 | 612,215 | 14,174 | 25 | 0.556 | k=0 probe 0.464 | **+0.092 [+0.05, +0.12]** | k=0 probe 0.464 | **+0.092 [+0.05, +0.12]** | – |
| Laguna XS.2 / Verified | has_regressions | 15 | S0 y_t=0 | 640,148 | 28,471 | 28 | 0.557 | k=0 probe 0.449 | **+0.109 [+0.07, +0.14]** | k=0 probe 0.449 | **+0.109 [+0.07, +0.14]** | 0.772 |
| Laguna XS.2 / Verified | has_regressions | 15 | S1 pre-edit (y_t+1=0) | 631,447 | 22,851 | 28 | 0.536 | traj-shuffle 0.438 | **+0.099 [+0.02, +0.18]** | traj-shuffle 0.438 | **+0.099 [+0.02, +0.18]** | 0.760 |
| Laguna XS.2 / Verified | has_regressions | 15 | S2 outcome y_T=1 | 27,876 | 16,018 | 16◦ | 0.628◦ | position causal 0.657 | -0.029 [-0.12, +0.07] | position causal 0.657 | -0.029 [-0.12, +0.07] | – |
| Laguna XS.2 / Verified | has_regressions | 15 | S3 pre-edit & y_T=1 | 24,182 | 12,324 | 16◦ | 0.595◦ | position causal 0.623 | -0.028 [-0.13, +0.07] | position causal 0.623 | -0.028 [-0.13, +0.07] | – |
| Laguna XS.2 / Verified | has_regressions | 15 | S4 y_T=0 | 612,272 | 12,453 | 22 | 0.545 | traj-shuffle 0.454 | +0.091 [-0.01, +0.19] | traj-shuffle 0.454 | +0.091 [-0.01, +0.19] | – |
| Qwen3.6-35B-A3B / Verified | correct | 2 | S0 y_t=0 | 322,282 | 10,668 | 40 | 0.761 | traj-shuffle 0.626 | **+0.135 [+0.07, +0.20]** | traj-shuffle 0.626 | **+0.135 [+0.07, +0.20]** | 0.783 |
| Qwen3.6-35B-A3B / Verified | correct | 2 | S1 pre-edit (y_t+1=0) | 315,267 | 3,653 | 40 | 0.636 | traj-shuffle 0.550 | **+0.086 [+0.02, +0.15]** | traj-shuffle 0.550 | **+0.086 [+0.02, +0.15]** | 0.787 |
| Qwen3.6-35B-A3B / Verified | correct | 2 | S2 outcome y_T=1 | 134,686 | 10,305 | 40 | 0.799 | traj-shuffle 0.690 | **+0.108 [+0.06, +0.16]** | traj-shuffle 0.690 | **+0.108 [+0.06, +0.16]** | – |
| Qwen3.6-35B-A3B / Verified | correct | 2 | S3 pre-edit & y_T=1 | 127,937 | 3,556 | 40 | 0.679 | position rel_pos 0.625 | +0.055 [-0.05, +0.14] | traj-shuffle 0.613 | **+0.066 [+0.01, +0.12]** | – |
| Qwen3.6-35B-A3B / Verified | correct | 2 | S4 y_T=0 | 187,324 | 363 | 3◦ | 0.715◦ | position causal 0.580 | +0.136 [-0.11, +0.25] | position causal 0.580 | +0.136 [-0.11, +0.25] | – |
| Qwen3.6-35B-A3B / Verified | correct | 3 | S0 y_t=0 | 322,273 | 14,364 | 42 | 0.745 | traj-shuffle 0.593 | **+0.151 [+0.09, +0.22]** | traj-shuffle 0.593 | **+0.151 [+0.09, +0.22]** | 0.787 |
| Qwen3.6-35B-A3B / Verified | correct | 3 | S1 pre-edit (y_t+1=0) | 315,258 | 7,357 | 42 | 0.653 | position rel_pos 0.565 | +0.087 [-0.00, +0.18] | traj-shuffle 0.538 | **+0.115 [+0.04, +0.19]** | 0.790 |
| Qwen3.6-35B-A3B / Verified | correct | 3 | S2 outcome y_T=1 | 134,686 | 13,898 | 42 | 0.779 | position rel_pos 0.669 | **+0.110 [+0.02, +0.20]** | traj-shuffle 0.659 | **+0.120 [+0.07, +0.17]** | – |
| Qwen3.6-35B-A3B / Verified | correct | 3 | S3 pre-edit & y_T=1 | 127,937 | 7,157 | 42 | 0.692 | position rel_pos 0.640 | +0.052 [-0.02, +0.14] | traj-shuffle 0.601 | **+0.091 [+0.04, +0.14]** | – |
| Qwen3.6-35B-A3B / Verified | correct | 3 | S4 y_T=0 | 187,324 | 466 | 3◦ | 0.698◦ | position causal 0.642 | +0.057 [-0.17, +0.18] | position causal 0.642 | +0.057 [-0.17, +0.18] | – |
| Qwen3.6-35B-A3B / Verified | correct | 5 | S0 y_t=0 | 322,217 | 23,111 | 47 | 0.722 | position rel_pos 0.583 | **+0.140 [+0.05, +0.23]** | traj-shuffle 0.574 | **+0.148 [+0.08, +0.22]** | 0.797 |
| Qwen3.6-35B-A3B / Verified | correct | 5 | S1 pre-edit (y_t+1=0) | 315,202 | 16,096 | 47 | 0.664 | position rel_pos 0.567 | **+0.097 [+0.01, +0.18]** | traj-shuffle 0.540 | **+0.124 [+0.06, +0.19]** | 0.799 |
| Qwen3.6-35B-A3B / Verified | correct | 5 | S2 outcome y_T=1 | 134,686 | 22,420 | 47 | 0.753 | position rel_pos 0.669 | **+0.084 [+0.01, +0.17]** | traj-shuffle 0.645 | **+0.108 [+0.06, +0.15]** | – |
| Qwen3.6-35B-A3B / Verified | correct | 5 | S3 pre-edit & y_T=1 | 127,937 | 15,671 | 47 | 0.697 | position rel_pos 0.652 | +0.045 [-0.02, +0.12] | traj-shuffle 0.610 | **+0.087 [+0.04, +0.13]** | – |
| Qwen3.6-35B-A3B / Verified | correct | 5 | S4 y_T=0 | 187,324 | 691 | 3◦ | 0.677◦ | position causal 0.702 | -0.025 [-0.17, +0.09] | position causal 0.702 | -0.025 [-0.17, +0.09] | – |
| Qwen3.6-35B-A3B / Verified | correct | 10 | S0 y_t=0 | 322,048 | 48,825 | 50 | 0.670 | position causal 0.571 | **+0.099 [+0.03, +0.17]** | position causal 0.571 | **+0.099 [+0.03, +0.17]** | 0.831 |
| Qwen3.6-35B-A3B / Verified | correct | 10 | S1 pre-edit (y_t+1=0) | 315,033 | 41,977 | 50 | 0.641 | position causal 0.565 | **+0.076 [+0.00, +0.15]** | position causal 0.565 | **+0.076 [+0.00, +0.15]** | 0.830 |
| Qwen3.6-35B-A3B / Verified | correct | 10 | S2 outcome y_T=1 | 134,686 | 47,807 | 50 | 0.699 | position rel_pos 0.677 | +0.021 [-0.05, +0.10] | position causal 0.583 | **+0.116 [+0.04, +0.20]** | – |
| Qwen3.6-35B-A3B / Verified | correct | 10 | S3 pre-edit & y_T=1 | 127,937 | 41,087 | 50 | 0.670 | position rel_pos 0.667 | +0.003 [-0.06, +0.08] | position causal 0.576 | **+0.094 [+0.02, +0.17]** | – |
| Qwen3.6-35B-A3B / Verified | correct | 10 | S4 y_T=0 | 187,324 | 1,018 | 3◦ | 0.599◦ | position causal 0.733 | **-0.135 [-0.21, -0.05]** | position causal 0.733 | **-0.135 [-0.21, -0.05]** | – |
| Qwen3.6-35B-A3B / Verified | correct | 15 | S0 y_t=0 | 322,010 | 70,815 | 51 | 0.648 | position causal 0.590 | +0.058 [-0.01, +0.12] | position causal 0.590 | +0.058 [-0.01, +0.12] | 0.865 |
| Qwen3.6-35B-A3B / Verified | correct | 15 | S1 pre-edit (y_t+1=0) | 314,995 | 63,953 | 51 | 0.629 | position causal 0.587 | +0.042 [-0.02, +0.10] | position causal 0.587 | +0.042 [-0.02, +0.10] | 0.865 |
| Qwen3.6-35B-A3B / Verified | correct | 15 | S2 outcome y_T=1 | 134,686 | 69,913 | 51 | 0.672 | position rel_pos 0.694 | -0.022 [-0.09, +0.06] | position causal 0.585 | **+0.087 [+0.01, +0.16]** | – |
| Qwen3.6-35B-A3B / Verified | correct | 15 | S3 pre-edit & y_T=1 | 127,937 | 63,205 | 51 | 0.653 | position rel_pos 0.686 | -0.033 [-0.10, +0.05] | position causal 0.582 | +0.071 [-0.00, +0.15] | – |
| Qwen3.6-35B-A3B / Verified | correct | 15 | S4 y_T=0 | 187,324 | 902 | 3◦ | 0.659◦ | position causal 0.777 | -0.118 [-0.24, +0.02] | position causal 0.777 | -0.118 [-0.24, +0.02] | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 2 | S0 y_t=0 | 312,580 | 13,867 | 44 | 0.764 | traj-shuffle 0.621 | **+0.143 [+0.09, +0.19]** | traj-shuffle 0.621 | **+0.143 [+0.09, +0.19]** | 0.770 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 2 | S1 pre-edit (y_t+1=0) | 303,006 | 4,293 | 44 | 0.615 | traj-shuffle 0.531 | **+0.084 [+0.03, +0.13]** | traj-shuffle 0.531 | **+0.084 [+0.03, +0.13]** | 0.775 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 2 | S2 outcome y_T=1 | 136,499 | 13,212 | 41 | 0.825 | traj-shuffle 0.678 | **+0.147 [+0.10, +0.19]** | traj-shuffle 0.678 | **+0.147 [+0.10, +0.19]** | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 2 | S3 pre-edit & y_T=1 | 127,425 | 4,138 | 41 | 0.689 | position rel_pos 0.598 | **+0.091 [+0.03, +0.15]** | traj-shuffle 0.592 | **+0.098 [+0.05, +0.15]** | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 2 | S4 y_T=0 | 175,809 | 655 | 8◦ | 0.750◦ | traj-shuffle 0.656 | **+0.094 [+0.00, +0.21]** | traj-shuffle 0.656 | **+0.094 [+0.00, +0.21]** | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3 | S0 y_t=0 | 312,571 | 18,412 | 47 | 0.741 | traj-shuffle 0.607 | **+0.134 [+0.08, +0.18]** | traj-shuffle 0.607 | **+0.134 [+0.08, +0.18]** | 0.779 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3 | S1 pre-edit (y_t+1=0) | 302,997 | 8,957 | 47 | 0.625 | traj-shuffle 0.545 | **+0.081 [+0.02, +0.14]** | traj-shuffle 0.545 | **+0.081 [+0.02, +0.14]** | 0.780 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3 | S2 outcome y_T=1 | 136,499 | 17,704 | 44 | 0.803 | traj-shuffle 0.665 | **+0.138 [+0.09, +0.18]** | traj-shuffle 0.665 | **+0.138 [+0.09, +0.18]** | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3 | S3 pre-edit & y_T=1 | 127,425 | 8,630 | 44 | 0.697 | position rel_pos 0.621 | **+0.076 [+0.02, +0.13]** | traj-shuffle 0.603 | **+0.094 [+0.05, +0.14]** | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3 | S4 y_T=0 | 175,809 | 708 | 8◦ | 0.720◦ | traj-shuffle 0.626 | +0.094 [-0.03, +0.24] | traj-shuffle 0.626 | +0.094 [-0.03, +0.24] | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 5 | S0 y_t=0 | 312,515 | 28,326 | 52 | 0.709 | traj-shuffle 0.589 | **+0.120 [+0.07, +0.18]** | traj-shuffle 0.589 | **+0.120 [+0.07, +0.18]** | 0.790 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 5 | S1 pre-edit (y_t+1=0) | 302,941 | 18,991 | 52 | 0.632 | traj-shuffle 0.548 | **+0.084 [+0.03, +0.15]** | traj-shuffle 0.548 | **+0.084 [+0.03, +0.15]** | 0.788 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 5 | S2 outcome y_T=1 | 136,499 | 27,316 | 49 | 0.775 | position rel_pos 0.658 | **+0.117 [+0.05, +0.18]** | traj-shuffle 0.650 | **+0.126 [+0.08, +0.17]** | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 5 | S3 pre-edit & y_T=1 | 127,425 | 18,250 | 49 | 0.706 | position rel_pos 0.632 | **+0.074 [+0.01, +0.13]** | traj-shuffle 0.608 | **+0.098 [+0.06, +0.14]** | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 5 | S4 y_T=0 | 175,809 | 1,010 | 8◦ | 0.689◦ | traj-shuffle 0.606 | +0.083 [-0.10, +0.24] | traj-shuffle 0.606 | +0.083 [-0.10, +0.24] | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 10 | S0 y_t=0 | 312,346 | 52,786 | 54 | 0.634 | position rel_pos 0.542 | **+0.092 [+0.01, +0.18]** | traj-shuffle 0.484 | **+0.150 [+0.08, +0.22]** | 0.821 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 10 | S1 pre-edit (y_t+1=0) | 302,772 | 43,729 | 54 | 0.587 | position rel_pos 0.528 | +0.060 [-0.02, +0.15] | traj-shuffle 0.462 | **+0.125 [+0.06, +0.20]** | 0.819 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 10 | S2 outcome y_T=1 | 136,499 | 51,222 | 51 | 0.710 | position rel_pos 0.660 | +0.050 [-0.02, +0.12] | k=0 probe 0.558 | **+0.152 [+0.11, +0.20]** | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 10 | S3 pre-edit & y_T=1 | 127,425 | 42,307 | 51 | 0.666 | position rel_pos 0.644 | +0.023 [-0.05, +0.09] | k=0 probe 0.534 | **+0.132 [+0.09, +0.18]** | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 10 | S4 y_T=0 | 175,809 | 1,564 | 8◦ | 0.629◦ | position causal 0.555 | +0.074 [-0.10, +0.24] | position causal 0.555 | +0.074 [-0.10, +0.24] | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 15 | S0 y_t=0 | 312,308 | 71,216 | 54 | 0.594 | position rel_pos 0.529 | +0.065 [-0.02, +0.16] | position causal 0.465 | **+0.129 [+0.06, +0.20]** | 0.852 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 15 | S1 pre-edit (y_t+1=0) | 302,734 | 62,224 | 54 | 0.560 | position rel_pos 0.516 | +0.044 [-0.04, +0.14] | position causal 0.453 | **+0.107 [+0.04, +0.18]** | 0.851 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 15 | S2 outcome y_T=1 | 136,499 | 69,847 | 51 | 0.676 | position rel_pos 0.667 | +0.009 [-0.06, +0.09] | k=0 probe 0.549 | **+0.127 [+0.09, +0.18]** | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 15 | S3 pre-edit & y_T=1 | 127,425 | 60,942 | 51 | 0.645 | position rel_pos 0.654 | -0.009 [-0.08, +0.07] | k=0 probe 0.531 | **+0.114 [+0.07, +0.16]** | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 15 | S4 y_T=0 | 175,809 | 1,369 | 8◦ | 0.564◦ | position causal 0.599 | -0.035 [-0.37, +0.26] | position causal 0.599 | -0.035 [-0.37, +0.26] | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 2 | S0 y_t=0 | 340,484 | 6,410 | 20 | 0.742 | position rel_pos 0.655 | +0.087 [-0.01, +0.19] | k=0 probe 0.616 | **+0.126 [+0.09, +0.15]** | 0.743 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 2 | S1 pre-edit (y_t+1=0) | 335,097 | 1,023 | 20 | 0.646 | position rel_pos 0.616 | +0.030 [-0.04, +0.12] | traj-shuffle 0.567 | **+0.079 [+0.00, +0.16]** | 0.748 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 2 | S2 outcome y_T=1 | 31,617 | 3,655 | 13◦ | 0.782◦ | position rel_pos 0.790 | -0.008 [-0.11, +0.12] | position causal 0.661 | +0.120 [-0.03, +0.23] | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 2 | S3 pre-edit & y_T=1 | 28,556 | 594 | 13◦ | 0.702◦ | position rel_pos 0.758 | -0.056 [-0.17, +0.05] | position causal 0.627 | +0.075 [-0.08, +0.16] | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 2 | S4 y_T=0 | 308,595 | 2,755 | 12◦ | 0.774◦ | k=0 probe 0.649 | **+0.125 [+0.08, +0.15]** | k=0 probe 0.649 | **+0.125 [+0.08, +0.15]** | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 3 | S0 y_t=0 | 340,475 | 7,785 | 22 | 0.739 | position rel_pos 0.647 | +0.092 [-0.01, +0.20] | k=0 probe 0.620 | **+0.119 [+0.08, +0.14]** | 0.740 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 3 | S1 pre-edit (y_t+1=0) | 335,088 | 2,398 | 22 | 0.689 | k=0 probe 0.605 | **+0.084 [+0.05, +0.12]** | k=0 probe 0.605 | **+0.084 [+0.05, +0.12]** | 0.732 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 3 | S2 outcome y_T=1 | 31,617 | 4,370 | 13◦ | 0.780◦ | position rel_pos 0.787 | -0.008 [-0.11, +0.11] | position causal 0.662 | +0.117 [-0.03, +0.21] | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 3 | S3 pre-edit & y_T=1 | 28,556 | 1,309 | 13◦ | 0.748◦ | position rel_pos 0.752 | -0.004 [-0.13, +0.08] | k=0 probe 0.660 | **+0.088 [+0.03, +0.14]** | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 3 | S4 y_T=0 | 308,595 | 3,415 | 14◦ | 0.770◦ | k=0 probe 0.643 | **+0.127 [+0.08, +0.15]** | k=0 probe 0.643 | **+0.127 [+0.08, +0.15]** | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 5 | S0 y_t=0 | 340,419 | 10,516 | 22 | 0.687 | position rel_pos 0.609 | +0.078 [-0.03, +0.18] | k=0 probe 0.592 | **+0.095 [+0.06, +0.13]** | 0.744 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 5 | S1 pre-edit (y_t+1=0) | 335,032 | 5,270 | 22 | 0.644 | k=0 probe 0.557 | **+0.087 [+0.05, +0.13]** | k=0 probe 0.557 | **+0.087 [+0.05, +0.13]** | 0.735 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 5 | S2 outcome y_T=1 | 31,617 | 5,953 | 13◦ | 0.733◦ | position rel_pos 0.746 | -0.013 [-0.13, +0.11] | k=0 probe 0.653 | **+0.080 [+0.04, +0.12]** | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 5 | S3 pre-edit & y_T=1 | 28,556 | 2,892 | 13◦ | 0.711◦ | position rel_pos 0.684 | +0.027 [-0.11, +0.11] | k=0 probe 0.639 | **+0.072 [+0.02, +0.12]** | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 5 | S4 y_T=0 | 308,595 | 4,563 | 14◦ | 0.706◦ | k=0 probe 0.597 | **+0.109 [+0.07, +0.14]** | k=0 probe 0.597 | **+0.109 [+0.07, +0.14]** | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 10 | S0 y_t=0 | 340,250 | 16,611 | 23 | 0.619 | traj-shuffle 0.622 | -0.003 [-0.07, +0.08] | traj-shuffle 0.622 | -0.003 [-0.07, +0.08] | 0.783 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 10 | S1 pre-edit (y_t+1=0) | 334,863 | 11,890 | 23 | 0.594 | traj-shuffle 0.629 | -0.035 [-0.11, +0.05] | traj-shuffle 0.629 | -0.035 [-0.11, +0.05] | 0.782 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 10 | S2 outcome y_T=1 | 31,617 | 10,483 | 13◦ | 0.660◦ | position rel_pos 0.672 | -0.012 [-0.12, +0.09] | k=0 probe 0.633 | +0.027 [-0.01, +0.07] | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 10 | S3 pre-edit & y_T=1 | 28,556 | 7,491 | 13◦ | 0.642◦ | position rel_pos 0.632 | +0.010 [-0.11, +0.10] | k=0 probe 0.614 | +0.028 [-0.01, +0.07] | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 10 | S4 y_T=0 | 308,595 | 6,128 | 15◦ | 0.649◦ | traj-shuffle 0.606 | +0.044 [-0.02, +0.10] | traj-shuffle 0.606 | +0.044 [-0.02, +0.10] | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 15 | S0 y_t=0 | 340,212 | 19,628 | 22 | 0.631 | traj-shuffle 0.666 | -0.035 [-0.11, +0.05] | traj-shuffle 0.666 | -0.035 [-0.11, +0.05] | 0.830 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 15 | S1 pre-edit (y_t+1=0) | 334,825 | 16,000 | 22 | 0.618 | traj-shuffle 0.659 | -0.041 [-0.13, +0.05] | traj-shuffle 0.659 | -0.041 [-0.13, +0.05] | 0.818 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 15 | S2 outcome y_T=1 | 31,617 | 14,037 | 13◦ | 0.666◦ | position rel_pos 0.652 | +0.014 [-0.07, +0.09] | traj-shuffle 0.638 | +0.028 [-0.09, +0.18] | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 15 | S3 pre-edit & y_T=1 | 28,556 | 11,045 | 13◦ | 0.650◦ | position rel_pos 0.631 | +0.020 [-0.07, +0.09] | traj-shuffle 0.630 | +0.020 [-0.10, +0.17] | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 15 | S4 y_T=0 | 308,595 | 5,591 | 14◦ | 0.650◦ | traj-shuffle 0.638 | +0.012 [-0.07, +0.12] | traj-shuffle 0.638 | +0.012 [-0.07, +0.12] | – |
| Laguna XS.2 / Pro | correct | 2 | S0 y_t=0 | 774,234 | 8,609 | 21 | 0.727 | position causal 0.614 | **+0.113 [+0.01, +0.24]** | position causal 0.614 | **+0.113 [+0.01, +0.24]** | 0.922 |
| Laguna XS.2 / Pro | correct | 2 | S1 pre-edit (y_t+1=0) | 768,058 | 2,684 | 20 | 0.676 | position causal 0.635 | +0.042 [-0.04, +0.18] | position causal 0.635 | +0.042 [-0.04, +0.18] | 0.912 |
| Laguna XS.2 / Pro | correct | 2 | S2 outcome y_T=1 | 61,204 | 7,862 | 19◦ | 0.690◦ | position rel_pos 0.697 | -0.007 [-0.10, +0.08] | position causal 0.606 | +0.084 [-0.01, +0.16] | – |
| Laguna XS.2 / Pro | correct | 2 | S3 pre-edit & y_T=1 | 55,518 | 2,397 | 19◦ | 0.640◦ | position rel_pos 0.701 | -0.061 [-0.16, +0.02] | position causal 0.631 | +0.009 [-0.07, +0.10] | – |
| Laguna XS.2 / Pro | correct | 2 | S4 y_T=0 | 712,268 | 747 | 4◦ | 0.706◦ | position causal 0.662 | +0.044 [-0.12, +0.47] | position causal 0.662 | +0.044 [-0.12, +0.47] | – |
| Laguna XS.2 / Pro | correct | 3 | S0 y_t=0 | 774,049 | 11,258 | 21 | 0.731 | position causal 0.615 | **+0.116 [+0.02, +0.24]** | position causal 0.615 | **+0.116 [+0.02, +0.24]** | 0.922 |
| Laguna XS.2 / Pro | correct | 3 | S1 pre-edit (y_t+1=0) | 767,873 | 5,414 | 21 | 0.692 | position causal 0.612 | +0.080 [-0.01, +0.21] | position causal 0.612 | +0.080 [-0.01, +0.21] | 0.917 |
| Laguna XS.2 / Pro | correct | 3 | S2 outcome y_T=1 | 61,204 | 10,258 | 19◦ | 0.685◦ | position rel_pos 0.686 | -0.001 [-0.11, +0.09] | traj-shuffle 0.607 | +0.079 [-0.01, +0.16] | – |
| Laguna XS.2 / Pro | correct | 3 | S3 pre-edit & y_T=1 | 55,518 | 4,874 | 19◦ | 0.642◦ | position rel_pos 0.666 | -0.023 [-0.15, +0.07] | traj-shuffle 0.600 | +0.043 [-0.06, +0.14] | – |
| Laguna XS.2 / Pro | correct | 3 | S4 y_T=0 | 712,091 | 1,000 | 4◦ | 0.739◦ | position causal 0.733 | +0.007 [-0.11, +0.47] | position causal 0.733 | +0.007 [-0.11, +0.47] | – |
| Laguna XS.2 / Pro | correct | 5 | S0 y_t=0 | 774,148 | 15,717 | 21 | 0.720 | position causal 0.621 | **+0.099 [+0.01, +0.22]** | position causal 0.621 | **+0.099 [+0.01, +0.22]** | 0.920 |
| Laguna XS.2 / Pro | correct | 5 | S1 pre-edit (y_t+1=0) | 767,774 | 10,624 | 20 | 0.687 | position causal 0.618 | +0.068 [-0.01, +0.18] | position causal 0.618 | +0.068 [-0.01, +0.18] | 0.919 |
| Laguna XS.2 / Pro | correct | 5 | S2 outcome y_T=1 | 61,204 | 14,172 | 19◦ | 0.662◦ | position rel_pos 0.647 | +0.014 [-0.08, +0.10] | position causal 0.594 | +0.068 [-0.03, +0.16] | – |
| Laguna XS.2 / Pro | correct | 5 | S3 pre-edit & y_T=1 | 55,518 | 9,539 | 18◦ | 0.627◦ | position rel_pos 0.627 | -0.000 [-0.09, +0.08] | position causal 0.592 | +0.035 [-0.05, +0.12] | – |
| Laguna XS.2 / Pro | correct | 5 | S4 y_T=0 | 712,257 | 1,545 | 4◦ | 0.758◦ | position causal 0.757 | +0.001 [-0.07, +0.57] | position causal 0.757 | +0.001 [-0.07, +0.57] | – |
| Laguna XS.2 / Pro | correct | 10 | S0 y_t=0 | 774,154 | 25,523 | 21 | 0.711 | position causal 0.601 | **+0.109 [+0.01, +0.21]** | position causal 0.601 | **+0.109 [+0.01, +0.21]** | 0.938 |
| Laguna XS.2 / Pro | correct | 10 | S1 pre-edit (y_t+1=0) | 767,755 | 20,331 | 21 | 0.687 | position causal 0.588 | **+0.099 [+0.00, +0.20]** | position causal 0.588 | **+0.099 [+0.00, +0.20]** | 0.936 |
| Laguna XS.2 / Pro | correct | 10 | S2 outcome y_T=1 | 61,204 | 23,618 | 20 | 0.651 | position rel_pos 0.640 | +0.011 [-0.09, +0.11] | position causal 0.585 | +0.066 [-0.01, +0.13] | – |
| Laguna XS.2 / Pro | correct | 10 | S3 pre-edit & y_T=1 | 55,518 | 18,725 | 20 | 0.627 | position rel_pos 0.620 | +0.006 [-0.10, +0.11] | position causal 0.575 | +0.052 [-0.02, +0.12] | – |
| Laguna XS.2 / Pro | correct | 10 | S4 y_T=0 | 712,341 | 1,905 | 5◦ | 0.720◦ | position causal 0.645 | **+0.075 [+0.02, +0.39]** | position causal 0.645 | **+0.075 [+0.02, +0.39]** | – |
| Laguna XS.2 / Pro | correct | 15 | S0 y_t=0 | 774,170 | 31,461 | 25 | 0.689 | position causal 0.595 | +0.094 [-0.01, +0.20] | position causal 0.595 | +0.094 [-0.01, +0.20] | 0.959 |
| Laguna XS.2 / Pro | correct | 15 | S1 pre-edit (y_t+1=0) | 767,771 | 26,274 | 25 | 0.668 | position causal 0.583 | +0.085 [-0.03, +0.20] | position causal 0.583 | +0.085 [-0.03, +0.20] | 0.955 |
| Laguna XS.2 / Pro | correct | 15 | S2 outcome y_T=1 | 61,204 | 30,190 | 24 | 0.617 | position rel_pos 0.652 | -0.035 [-0.15, +0.08] | position causal 0.576 | +0.041 [-0.03, +0.11] | – |
| Laguna XS.2 / Pro | correct | 15 | S3 pre-edit & y_T=1 | 55,518 | 25,003 | 24 | 0.595 | position rel_pos 0.639 | -0.044 [-0.15, +0.07] | position causal 0.565 | +0.030 [-0.04, +0.10] | – |
| Laguna XS.2 / Pro | correct | 15 | S4 y_T=0 | 712,427 | 1,271 | 5◦ | 0.631◦ | traj-shuffle 0.571 | +0.060 [-0.14, +0.23] | traj-shuffle 0.571 | +0.060 [-0.14, +0.23] | – |
| Laguna XS.2 / Pro | reduces_failing | 2 | S0 y_t=0 | 732,224 | 15,810 | 29 | 0.756 | k=0 probe 0.659 | **+0.097 [+0.06, +0.14]** | k=0 probe 0.659 | **+0.097 [+0.06, +0.14]** | 0.803 |
| Laguna XS.2 / Pro | reduces_failing | 2 | S1 pre-edit (y_t+1=0) | 720,917 | 4,627 | 28 | 0.708 | k=0 probe 0.633 | **+0.075 [+0.03, +0.12]** | k=0 probe 0.633 | **+0.075 [+0.03, +0.12]** | 0.780 |
| Laguna XS.2 / Pro | reduces_failing | 2 | S2 outcome y_T=1 | 53,346 | 10,529 | 21 | 0.775 | k=0 probe 0.705 | **+0.070 [+0.03, +0.11]** | k=0 probe 0.705 | **+0.070 [+0.03, +0.11]** | – |
| Laguna XS.2 / Pro | reduces_failing | 2 | S3 pre-edit & y_T=1 | 45,556 | 2,863 | 21 | 0.702 | position causal 0.665 | +0.038 [-0.03, +0.11] | position causal 0.665 | +0.038 [-0.03, +0.11] | – |
| Laguna XS.2 / Pro | reduces_failing | 2 | S4 y_T=0 | 678,116 | 5,281 | 16◦ | 0.751◦ | k=0 probe 0.696 | **+0.055 [+0.02, +0.11]** | k=0 probe 0.696 | **+0.055 [+0.02, +0.11]** | – |
| Laguna XS.2 / Pro | reduces_failing | 3 | S0 y_t=0 | 732,038 | 20,356 | 29 | 0.729 | k=0 probe 0.632 | **+0.097 [+0.05, +0.14]** | k=0 probe 0.632 | **+0.097 [+0.05, +0.14]** | 0.820 |
| Laguna XS.2 / Pro | reduces_failing | 3 | S1 pre-edit (y_t+1=0) | 720,731 | 9,389 | 29 | 0.693 | k=0 probe 0.591 | **+0.102 [+0.05, +0.15]** | k=0 probe 0.591 | **+0.102 [+0.05, +0.15]** | 0.815 |
| Laguna XS.2 / Pro | reduces_failing | 3 | S2 outcome y_T=1 | 53,346 | 14,165 | 21 | 0.720 | k=0 probe 0.684 | +0.036 [-0.00, +0.07] | k=0 probe 0.684 | +0.036 [-0.00, +0.07] | – |
| Laguna XS.2 / Pro | reduces_failing | 3 | S3 pre-edit & y_T=1 | 45,556 | 6,441 | 21 | 0.661 | position causal 0.631 | +0.030 [-0.04, +0.11] | position causal 0.631 | +0.030 [-0.04, +0.11] | – |
| Laguna XS.2 / Pro | reduces_failing | 3 | S4 y_T=0 | 677,938 | 6,191 | 16◦ | 0.723◦ | k=0 probe 0.678 | +0.045 [-0.03, +0.14] | k=0 probe 0.678 | +0.045 [-0.03, +0.14] | – |
| Laguna XS.2 / Pro | reduces_failing | 5 | S0 y_t=0 | 732,138 | 26,885 | 29 | 0.722 | k=0 probe 0.611 | **+0.111 [+0.07, +0.16]** | k=0 probe 0.611 | **+0.111 [+0.07, +0.16]** | 0.840 |
| Laguna XS.2 / Pro | reduces_failing | 5 | S1 pre-edit (y_t+1=0) | 720,633 | 17,159 | 29 | 0.687 | k=0 probe 0.581 | **+0.106 [+0.05, +0.16]** | k=0 probe 0.581 | **+0.106 [+0.05, +0.16]** | 0.825 |
| Laguna XS.2 / Pro | reduces_failing | 5 | S2 outcome y_T=1 | 53,346 | 19,590 | 21 | 0.704 | k=0 probe 0.669 | +0.035 [-0.01, +0.07] | k=0 probe 0.669 | +0.035 [-0.01, +0.07] | – |
| Laguna XS.2 / Pro | reduces_failing | 5 | S3 pre-edit & y_T=1 | 45,556 | 11,985 | 21 | 0.656 | position causal 0.658 | -0.003 [-0.08, +0.07] | position causal 0.658 | -0.003 [-0.08, +0.07] | – |
| Laguna XS.2 / Pro | reduces_failing | 5 | S4 y_T=0 | 678,105 | 7,295 | 16◦ | 0.728◦ | k=0 probe 0.678 | +0.050 [-0.03, +0.15] | k=0 probe 0.678 | +0.050 [-0.03, +0.15] | – |
| Laguna XS.2 / Pro | reduces_failing | 10 | S0 y_t=0 | 732,143 | 36,120 | 31 | 0.695 | position causal 0.572 | **+0.123 [+0.03, +0.25]** | position causal 0.572 | **+0.123 [+0.03, +0.25]** | 0.875 |
| Laguna XS.2 / Pro | reduces_failing | 10 | S1 pre-edit (y_t+1=0) | 720,613 | 27,372 | 31 | 0.668 | position causal 0.558 | **+0.110 [+0.01, +0.25]** | position causal 0.558 | **+0.110 [+0.01, +0.25]** | 0.869 |
| Laguna XS.2 / Pro | reduces_failing | 10 | S2 outcome y_T=1 | 53,346 | 28,381 | 24 | 0.653 | position causal 0.654 | -0.001 [-0.07, +0.08] | position causal 0.654 | -0.001 [-0.07, +0.08] | – |
| Laguna XS.2 / Pro | reduces_failing | 10 | S3 pre-edit & y_T=1 | 45,556 | 21,143 | 24 | 0.625 | position causal 0.641 | -0.015 [-0.08, +0.06] | position causal 0.641 | -0.015 [-0.08, +0.06] | – |
| Laguna XS.2 / Pro | reduces_failing | 10 | S4 y_T=0 | 678,188 | 7,739 | 18◦ | 0.701◦ | k=0 probe 0.620 | +0.081 [-0.01, +0.18] | k=0 probe 0.620 | +0.081 [-0.01, +0.18] | – |
| Laguna XS.2 / Pro | reduces_failing | 15 | S0 y_t=0 | 732,153 | 40,499 | 33 | 0.676 | position causal 0.542 | **+0.134 [+0.03, +0.28]** | position causal 0.542 | **+0.134 [+0.03, +0.28]** | 0.907 |
| Laguna XS.2 / Pro | reduces_failing | 15 | S1 pre-edit (y_t+1=0) | 720,623 | 31,941 | 33 | 0.652 | position causal 0.522 | **+0.130 [+0.02, +0.28]** | position causal 0.522 | **+0.130 [+0.02, +0.28]** | 0.902 |
| Laguna XS.2 / Pro | reduces_failing | 15 | S2 outcome y_T=1 | 53,338 | 34,116 | 27 | 0.588 | position causal 0.629 | -0.041 [-0.12, +0.06] | position causal 0.629 | -0.041 [-0.12, +0.06] | – |
| Laguna XS.2 / Pro | reduces_failing | 15 | S3 pre-edit & y_T=1 | 45,548 | 26,533 | 27 | 0.555 | position causal 0.611 | -0.057 [-0.13, +0.04] | position causal 0.611 | -0.057 [-0.13, +0.04] | – |
| Laguna XS.2 / Pro | reduces_failing | 15 | S4 y_T=0 | 678,276 | 6,383 | 18◦ | 0.667◦ | k=0 probe 0.582 | +0.085 [-0.04, +0.23] | k=0 probe 0.582 | +0.085 [-0.04, +0.23] | – |
| Laguna XS.2 / Pro | has_regressions | 2 | S0 y_t=0 | 769,086 | 8,813 | 14◦ | 0.793◦ | k=0 probe 0.665 | **+0.128 [+0.09, +0.17]** | k=0 probe 0.665 | **+0.128 [+0.09, +0.17]** | 0.866 |
| Laguna XS.2 / Pro | has_regressions | 2 | S1 pre-edit (y_t+1=0) | 762,758 | 2,842 | 14◦ | 0.739◦ | k=0 probe 0.619 | **+0.120 [+0.07, +0.19]** | k=0 probe 0.619 | **+0.120 [+0.07, +0.19]** | 0.838 |
| Laguna XS.2 / Pro | has_regressions | 2 | S2 outcome y_T=1 | 34,064 | 6,773 | 10◦ | 0.682◦ | position causal 0.673 | +0.009 [-0.06, +0.08] | position causal 0.673 | +0.009 [-0.06, +0.08] | – |
| Laguna XS.2 / Pro | has_regressions | 2 | S3 pre-edit & y_T=1 | 29,316 | 2,025 | 10◦ | 0.579◦ | position causal 0.659 | -0.080 [-0.16, +0.01] | position causal 0.659 | -0.080 [-0.16, +0.01] | – |
| Laguna XS.2 / Pro | has_regressions | 2 | S4 y_T=0 | 734,260 | 2,040 | 10◦ | 0.802◦ | k=0 probe 0.672 | **+0.130 [+0.05, +0.18]** | k=0 probe 0.672 | **+0.130 [+0.05, +0.18]** | – |
| Laguna XS.2 / Pro | has_regressions | 3 | S0 y_t=0 | 768,901 | 10,658 | 14◦ | 0.780◦ | k=0 probe 0.661 | **+0.119 [+0.09, +0.15]** | k=0 probe 0.661 | **+0.119 [+0.09, +0.15]** | 0.861 |
| Laguna XS.2 / Pro | has_regressions | 3 | S1 pre-edit (y_t+1=0) | 762,573 | 4,772 | 14◦ | 0.737◦ | k=0 probe 0.630 | **+0.107 [+0.07, +0.16]** | k=0 probe 0.630 | **+0.107 [+0.07, +0.16]** | 0.831 |
| Laguna XS.2 / Pro | has_regressions | 3 | S2 outcome y_T=1 | 34,065 | 8,068 | 10◦ | 0.666◦ | position causal 0.656 | +0.010 [-0.04, +0.08] | position causal 0.656 | +0.010 [-0.04, +0.08] | – |
| Laguna XS.2 / Pro | has_regressions | 3 | S3 pre-edit & y_T=1 | 29,317 | 3,320 | 10◦ | 0.586◦ | position causal 0.625 | -0.038 [-0.10, +0.08] | position causal 0.625 | -0.038 [-0.10, +0.08] | – |
| Laguna XS.2 / Pro | has_regressions | 3 | S4 y_T=0 | 734,082 | 2,590 | 10◦ | 0.779◦ | k=0 probe 0.672 | **+0.107 [+0.04, +0.16]** | k=0 probe 0.672 | **+0.107 [+0.04, +0.16]** | – |
| Laguna XS.2 / Pro | has_regressions | 5 | S0 y_t=0 | 769,000 | 14,209 | 14◦ | 0.763◦ | k=0 probe 0.646 | **+0.117 [+0.08, +0.15]** | k=0 probe 0.646 | **+0.117 [+0.08, +0.15]** | 0.866 |
| Laguna XS.2 / Pro | has_regressions | 5 | S1 pre-edit (y_t+1=0) | 762,474 | 8,397 | 14◦ | 0.728◦ | k=0 probe 0.617 | **+0.111 [+0.07, +0.16]** | k=0 probe 0.617 | **+0.111 [+0.07, +0.16]** | 0.847 |
| Laguna XS.2 / Pro | has_regressions | 5 | S2 outcome y_T=1 | 34,064 | 10,833 | 10◦ | 0.627◦ | position causal 0.624 | +0.003 [-0.05, +0.07] | position causal 0.624 | +0.003 [-0.05, +0.07] | – |
| Laguna XS.2 / Pro | has_regressions | 5 | S3 pre-edit & y_T=1 | 29,316 | 6,085 | 10◦ | 0.565◦ | position causal 0.601 | -0.036 [-0.09, +0.05] | position causal 0.601 | -0.036 [-0.09, +0.05] | – |
| Laguna XS.2 / Pro | has_regressions | 5 | S4 y_T=0 | 734,249 | 3,376 | 10◦ | 0.767◦ | k=0 probe 0.660 | **+0.107 [+0.04, +0.18]** | k=0 probe 0.660 | **+0.107 [+0.04, +0.18]** | – |
| Laguna XS.2 / Pro | has_regressions | 10 | S0 y_t=0 | 769,006 | 20,188 | 14◦ | 0.756◦ | k=0 probe 0.641 | **+0.114 [+0.07, +0.16]** | k=0 probe 0.641 | **+0.114 [+0.07, +0.16]** | 0.885 |
| Laguna XS.2 / Pro | has_regressions | 10 | S1 pre-edit (y_t+1=0) | 762,455 | 15,211 | 14◦ | 0.737◦ | k=0 probe 0.627 | **+0.111 [+0.06, +0.17]** | k=0 probe 0.627 | **+0.111 [+0.06, +0.17]** | 0.873 |
| Laguna XS.2 / Pro | has_regressions | 10 | S2 outcome y_T=1 | 34,065 | 16,034 | 10◦ | 0.606◦ | position causal 0.613 | -0.007 [-0.10, +0.08] | position causal 0.613 | -0.007 [-0.10, +0.08] | – |
| Laguna XS.2 / Pro | has_regressions | 10 | S3 pre-edit & y_T=1 | 29,317 | 11,698 | 10◦ | 0.576◦ | position causal 0.599 | -0.023 [-0.13, +0.08] | position causal 0.599 | -0.023 [-0.13, +0.08] | – |
| Laguna XS.2 / Pro | has_regressions | 10 | S4 y_T=0 | 734,332 | 4,154 | 10◦ | 0.743◦ | k=0 probe 0.642 | **+0.101 [+0.02, +0.19]** | k=0 probe 0.642 | **+0.101 [+0.02, +0.19]** | – |
| Laguna XS.2 / Pro | has_regressions | 15 | S0 y_t=0 | 769,022 | 25,079 | 15◦ | 0.744◦ | k=0 probe 0.626 | **+0.118 [+0.08, +0.17]** | k=0 probe 0.626 | **+0.118 [+0.08, +0.17]** | 0.896 |
| Laguna XS.2 / Pro | has_regressions | 15 | S1 pre-edit (y_t+1=0) | 762,471 | 19,894 | 15◦ | 0.729◦ | k=0 probe 0.609 | **+0.119 [+0.07, +0.18]** | k=0 probe 0.609 | **+0.119 [+0.07, +0.18]** | 0.886 |
| Laguna XS.2 / Pro | has_regressions | 15 | S2 outcome y_T=1 | 34,063 | 20,323 | 12◦ | 0.564◦ | position causal 0.623 | -0.059 [-0.16, +0.01] | position causal 0.623 | -0.059 [-0.16, +0.01] | – |
| Laguna XS.2 / Pro | has_regressions | 15 | S3 pre-edit & y_T=1 | 29,315 | 15,711 | 12◦ | 0.538◦ | position causal 0.605 | -0.066 [-0.18, +0.01] | position causal 0.605 | -0.066 [-0.18, +0.01] | – |
| Laguna XS.2 / Pro | has_regressions | 15 | S4 y_T=0 | 734,420 | 4,756 | 10◦ | 0.728◦ | k=0 probe 0.600 | **+0.128 [+0.07, +0.22]** | k=0 probe 0.600 | **+0.128 [+0.07, +0.22]** | – |
| Qwen3.6-35B-A3B / Pro | correct | 2 | S0 y_t=0 | 342,270 | 8,380 | 19◦ | 0.793◦ | k=0 probe 0.673 | **+0.121 [+0.08, +0.16]** | k=0 probe 0.673 | **+0.121 [+0.08, +0.16]** | 0.888 |
| Qwen3.6-35B-A3B / Pro | correct | 2 | S1 pre-edit (y_t+1=0) | 334,942 | 1,169 | 19◦ | 0.723◦ | k=0 probe 0.658 | **+0.064 [+0.03, +0.09]** | k=0 probe 0.658 | **+0.064 [+0.03, +0.09]** | 0.826 |
| Qwen3.6-35B-A3B / Pro | correct | 2 | S2 outcome y_T=1 | 50,911 | 7,583 | 17◦ | 0.783◦ | traj-shuffle 0.693 | **+0.090 [+0.03, +0.15]** | traj-shuffle 0.693 | **+0.090 [+0.03, +0.15]** | – |
| Qwen3.6-35B-A3B / Pro | correct | 2 | S3 pre-edit & y_T=1 | 44,125 | 914 | 17◦ | 0.687◦ | traj-shuffle 0.637 | +0.050 [-0.02, +0.12] | traj-shuffle 0.637 | +0.050 [-0.02, +0.12] | – |
| Qwen3.6-35B-A3B / Pro | correct | 2 | S4 y_T=0 | 291,359 | 797 | 3◦ | 0.816◦ | position causal 0.740 | +0.077 [-0.13, +0.29] | position causal 0.740 | +0.077 [-0.13, +0.29] | – |
| Qwen3.6-35B-A3B / Pro | correct | 3 | S0 y_t=0 | 342,315 | 9,277 | 21 | 0.789 | k=0 probe 0.664 | **+0.125 [+0.08, +0.16]** | k=0 probe 0.664 | **+0.125 [+0.08, +0.16]** | 0.888 |
| Qwen3.6-35B-A3B / Pro | correct | 3 | S1 pre-edit (y_t+1=0) | 334,929 | 2,072 | 21 | 0.708 | k=0 probe 0.627 | **+0.082 [+0.04, +0.12]** | k=0 probe 0.627 | **+0.082 [+0.04, +0.12]** | 0.851 |
| Qwen3.6-35B-A3B / Pro | correct | 3 | S2 outcome y_T=1 | 50,911 | 8,388 | 19◦ | 0.780◦ | traj-shuffle 0.694 | +0.086 [-0.00, +0.18] | traj-shuffle 0.694 | +0.086 [-0.00, +0.18] | – |
| Qwen3.6-35B-A3B / Pro | correct | 3 | S3 pre-edit & y_T=1 | 44,125 | 1,719 | 19◦ | 0.678◦ | traj-shuffle 0.644 | +0.034 [-0.03, +0.09] | traj-shuffle 0.644 | +0.034 [-0.03, +0.09] | – |
| Qwen3.6-35B-A3B / Pro | correct | 3 | S4 y_T=0 | 291,404 | 889 | 3◦ | 0.817◦ | position causal 0.750 | +0.067 [-0.14, +0.29] | position causal 0.750 | +0.067 [-0.14, +0.29] | – |
| Qwen3.6-35B-A3B / Pro | correct | 5 | S0 y_t=0 | 342,321 | 11,665 | 23 | 0.773 | k=0 probe 0.665 | **+0.108 [+0.06, +0.15]** | k=0 probe 0.665 | **+0.108 [+0.06, +0.15]** | 0.903 |
| Qwen3.6-35B-A3B / Pro | correct | 5 | S1 pre-edit (y_t+1=0) | 334,919 | 4,681 | 23 | 0.714 | k=0 probe 0.646 | **+0.069 [+0.02, +0.11]** | k=0 probe 0.646 | **+0.069 [+0.02, +0.11]** | 0.895 |
| Qwen3.6-35B-A3B / Pro | correct | 5 | S2 outcome y_T=1 | 50,911 | 10,808 | 21 | 0.762 | k=0 probe 0.636 | **+0.126 [+0.06, +0.17]** | k=0 probe 0.636 | **+0.126 [+0.06, +0.17]** | – |
| Qwen3.6-35B-A3B / Pro | correct | 5 | S3 pre-edit & y_T=1 | 44,125 | 4,265 | 21 | 0.694 | position rel_pos 0.635 | +0.059 [-0.05, +0.11] | k=0 probe 0.608 | **+0.086 [+0.03, +0.13]** | – |
| Qwen3.6-35B-A3B / Pro | correct | 5 | S4 y_T=0 | 291,410 | 857 | 3◦ | 0.818◦ | position causal 0.799 | +0.019 [-0.16, +0.37] | position causal 0.799 | +0.019 [-0.16, +0.37] | – |
| Qwen3.6-35B-A3B / Pro | correct | 10 | S0 y_t=0 | 342,315 | 18,853 | 25 | 0.731 | k=0 probe 0.653 | **+0.078 [+0.03, +0.12]** | k=0 probe 0.653 | **+0.078 [+0.03, +0.12]** | 0.920 |
| Qwen3.6-35B-A3B / Pro | correct | 10 | S1 pre-edit (y_t+1=0) | 334,913 | 11,798 | 25 | 0.692 | k=0 probe 0.641 | **+0.051 [+0.00, +0.10]** | k=0 probe 0.641 | **+0.051 [+0.00, +0.10]** | 0.915 |
| Qwen3.6-35B-A3B / Pro | correct | 10 | S2 outcome y_T=1 | 50,911 | 17,780 | 23 | 0.714 | position rel_pos 0.665 | +0.049 [-0.09, +0.14] | k=0 probe 0.645 | **+0.070 [+0.01, +0.13]** | – |
| Qwen3.6-35B-A3B / Pro | correct | 10 | S3 pre-edit & y_T=1 | 44,125 | 10,994 | 23 | 0.666 | position rel_pos 0.667 | -0.001 [-0.13, +0.10] | k=0 probe 0.628 | +0.038 [-0.02, +0.11] | – |
| Qwen3.6-35B-A3B / Pro | correct | 10 | S4 y_T=0 | 291,404 | 1,073 | 4◦ | 0.804◦ | position causal 0.709 | +0.095 [-0.03, +0.71] | position causal 0.709 | +0.095 [-0.03, +0.71] | – |
| Qwen3.6-35B-A3B / Pro | correct | 15 | S0 y_t=0 | 342,285 | 24,884 | 27 | 0.711 | k=0 probe 0.638 | **+0.073 [+0.02, +0.13]** | k=0 probe 0.638 | **+0.073 [+0.02, +0.13]** | 0.946 |
| Qwen3.6-35B-A3B / Pro | correct | 15 | S1 pre-edit (y_t+1=0) | 334,883 | 18,116 | 27 | 0.688 | k=0 probe 0.623 | **+0.065 [+0.02, +0.13]** | k=0 probe 0.623 | **+0.065 [+0.02, +0.13]** | 0.945 |
| Qwen3.6-35B-A3B / Pro | correct | 15 | S2 outcome y_T=1 | 50,911 | 24,286 | 25 | 0.672 | position rel_pos 0.705 | -0.033 [-0.15, +0.08] | k=0 probe 0.651 | +0.021 [-0.03, +0.07] | – |
| Qwen3.6-35B-A3B / Pro | correct | 15 | S3 pre-edit & y_T=1 | 44,125 | 17,641 | 25 | 0.645 | position rel_pos 0.700 | -0.055 [-0.17, +0.07] | k=0 probe 0.633 | +0.012 [-0.04, +0.08] | – |
| Qwen3.6-35B-A3B / Pro | correct | 15 | S4 y_T=0 | 291,374 | 598 | 5◦ | 0.800◦ | k=0 probe 0.597 | **+0.203 [+0.01, +0.29]** | k=0 probe 0.597 | **+0.203 [+0.01, +0.29]** | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 2 | S0 y_t=0 | 340,074 | 9,897 | 24 | 0.828 | k=0 probe 0.735 | **+0.093 [+0.08, +0.11]** | k=0 probe 0.735 | **+0.093 [+0.08, +0.11]** | 0.875 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 2 | S1 pre-edit (y_t+1=0) | 331,432 | 1,390 | 24 | 0.737 | k=0 probe 0.664 | **+0.072 [+0.05, +0.09]** | k=0 probe 0.664 | **+0.072 [+0.05, +0.09]** | 0.880 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 2 | S2 outcome y_T=1 | 55,901 | 8,824 | 20 | 0.780 | traj-shuffle 0.741 | +0.040 [-0.05, +0.13] | traj-shuffle 0.741 | +0.040 [-0.05, +0.13] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 2 | S3 pre-edit & y_T=1 | 48,195 | 1,253 | 20 | 0.685 | traj-shuffle 0.661 | +0.024 [-0.07, +0.11] | traj-shuffle 0.661 | +0.024 [-0.07, +0.11] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 2 | S4 y_T=0 | 284,173 | 1,073 | 6◦ | 0.786◦ | k=0 probe 0.673 | **+0.113 [+0.07, +0.13]** | k=0 probe 0.673 | **+0.113 [+0.07, +0.13]** | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 3 | S0 y_t=0 | 340,119 | 11,322 | 25 | 0.815 | k=0 probe 0.718 | **+0.097 [+0.08, +0.11]** | k=0 probe 0.718 | **+0.097 [+0.08, +0.11]** | 0.878 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 3 | S1 pre-edit (y_t+1=0) | 331,419 | 2,849 | 25 | 0.727 | k=0 probe 0.632 | **+0.095 [+0.07, +0.12]** | k=0 probe 0.632 | **+0.095 [+0.07, +0.12]** | 0.880 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 3 | S2 outcome y_T=1 | 55,901 | 10,134 | 21 | 0.765 | traj-shuffle 0.701 | +0.064 [-0.02, +0.16] | traj-shuffle 0.701 | +0.064 [-0.02, +0.16] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 3 | S3 pre-edit & y_T=1 | 48,195 | 2,563 | 21 | 0.673 | position rel_pos 0.632 | +0.041 [-0.06, +0.13] | traj-shuffle 0.622 | +0.051 [-0.02, +0.10] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 3 | S4 y_T=0 | 284,218 | 1,188 | 6◦ | 0.764◦ | k=0 probe 0.657 | **+0.107 [+0.06, +0.13]** | k=0 probe 0.657 | **+0.107 [+0.06, +0.13]** | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 5 | S0 y_t=0 | 340,125 | 14,938 | 27 | 0.797 | k=0 probe 0.702 | **+0.094 [+0.07, +0.12]** | k=0 probe 0.702 | **+0.094 [+0.07, +0.12]** | 0.887 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 5 | S1 pre-edit (y_t+1=0) | 331,409 | 6,669 | 27 | 0.740 | k=0 probe 0.646 | **+0.094 [+0.06, +0.12]** | k=0 probe 0.646 | **+0.094 [+0.06, +0.12]** | 0.889 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 5 | S2 outcome y_T=1 | 55,901 | 13,513 | 23 | 0.742 | k=0 probe 0.681 | **+0.061 [+0.03, +0.09]** | k=0 probe 0.681 | **+0.061 [+0.03, +0.09]** | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 5 | S3 pre-edit & y_T=1 | 48,195 | 6,049 | 23 | 0.685 | position rel_pos 0.631 | +0.054 [-0.02, +0.13] | k=0 probe 0.624 | **+0.061 [+0.01, +0.10]** | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 5 | S4 y_T=0 | 284,224 | 1,425 | 7◦ | 0.718◦ | traj-shuffle 0.608 | +0.110 [-0.10, +0.34] | traj-shuffle 0.608 | +0.110 [-0.10, +0.34] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 10 | S0 y_t=0 | 340,119 | 22,769 | 27 | 0.764 | k=0 probe 0.670 | **+0.094 [+0.05, +0.13]** | k=0 probe 0.670 | **+0.094 [+0.05, +0.13]** | 0.909 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 10 | S1 pre-edit (y_t+1=0) | 331,403 | 14,466 | 27 | 0.721 | k=0 probe 0.621 | **+0.100 [+0.04, +0.15]** | k=0 probe 0.621 | **+0.100 [+0.04, +0.15]** | 0.910 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 10 | S2 outcome y_T=1 | 55,901 | 21,129 | 23 | 0.680 | k=0 probe 0.659 | +0.021 [-0.04, +0.08] | k=0 probe 0.659 | +0.021 [-0.04, +0.08] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 10 | S3 pre-edit & y_T=1 | 48,195 | 13,447 | 23 | 0.635 | k=0 probe 0.608 | +0.027 [-0.06, +0.10] | k=0 probe 0.608 | +0.027 [-0.06, +0.10] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 10 | S4 y_T=0 | 284,218 | 1,640 | 7◦ | 0.688◦ | traj-shuffle 0.587 | +0.101 [-0.12, +0.34] | traj-shuffle 0.587 | +0.101 [-0.12, +0.34] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 15 | S0 y_t=0 | 340,089 | 28,729 | 29 | 0.732 | k=0 probe 0.653 | **+0.078 [+0.02, +0.14]** | k=0 probe 0.653 | **+0.078 [+0.02, +0.14]** | 0.930 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 15 | S1 pre-edit (y_t+1=0) | 331,373 | 20,861 | 29 | 0.710 | k=0 probe 0.614 | **+0.096 [+0.03, +0.17]** | k=0 probe 0.614 | **+0.096 [+0.03, +0.17]** | 0.928 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 15 | S2 outcome y_T=1 | 55,901 | 27,328 | 25 | 0.602 | k=0 probe 0.657 | -0.055 [-0.12, +0.02] | k=0 probe 0.657 | -0.055 [-0.12, +0.02] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 15 | S3 pre-edit & y_T=1 | 48,195 | 19,782 | 25 | 0.579 | k=0 probe 0.615 | -0.036 [-0.10, +0.05] | k=0 probe 0.615 | -0.036 [-0.10, +0.05] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 15 | S4 y_T=0 | 284,188 | 1,401 | 7◦ | 0.639◦ | traj-shuffle 0.520 | +0.119 [-0.02, +0.25] | traj-shuffle 0.520 | +0.119 [-0.02, +0.25] | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 2 | S0 y_t=0 | 358,127 | 1,975 | 7◦ | 0.853◦ | position causal 0.774 | +0.079 [-0.04, +0.32] | position causal 0.774 | +0.079 [-0.04, +0.32] | 0.790 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 2 | S1 pre-edit (y_t+1=0) | 356,451 | 299 | 7◦ | 0.792◦ | position causal 0.829 | -0.037 [-0.13, +0.22] | position causal 0.829 | -0.037 [-0.13, +0.22] | 0.629 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 2 | S2 outcome y_T=1 | 7,129 | 1,180 | 2◦ | 0.902◦ | k=0 probe 0.854 | **+0.047 [+0.00, +0.06]** | k=0 probe 0.854 | **+0.047 [+0.00, +0.06]** | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 2 | S3 pre-edit & y_T=1 | 6,031 | 82 | 2◦ | 0.753◦ | position rel_pos 0.824 | -0.071 [-0.61, +0.02] | position causal 0.758 | -0.005 [-0.49, +0.08] | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 2 | S4 y_T=0 | 350,998 | 795 | 5◦ | 0.864◦ | k=0 probe 0.798 | **+0.066 [+0.05, +0.10]** | k=0 probe 0.798 | **+0.066 [+0.05, +0.10]** | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 3 | S0 y_t=0 | 358,172 | 2,160 | 7◦ | 0.834◦ | position causal 0.773 | +0.062 [-0.06, +0.31] | position causal 0.773 | +0.062 [-0.06, +0.31] | 0.790 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 3 | S1 pre-edit (y_t+1=0) | 356,438 | 484 | 7◦ | 0.786◦ | position causal 0.767 | +0.019 [-0.10, +0.29] | position causal 0.767 | +0.019 [-0.10, +0.29] | 0.689 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 3 | S2 outcome y_T=1 | 7,129 | 1,289 | 2◦ | 0.877◦ | position causal 0.846 | +0.031 [-0.12, +0.08] | position causal 0.846 | +0.031 [-0.12, +0.08] | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 3 | S3 pre-edit & y_T=1 | 6,031 | 191 | 2◦ | 0.800◦ | position rel_pos 0.838 | -0.038 [-0.53, +0.03] | position causal 0.796 | +0.005 [-0.45, +0.07] | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 3 | S4 y_T=0 | 351,043 | 871 | 5◦ | 0.852◦ | k=0 probe 0.781 | **+0.072 [+0.04, +0.14]** | k=0 probe 0.781 | **+0.072 [+0.04, +0.14]** | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 5 | S0 y_t=0 | 358,178 | 2,591 | 8◦ | 0.822◦ | position causal 0.765 | +0.058 [-0.07, +0.30] | position causal 0.765 | +0.058 [-0.07, +0.30] | 0.779 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 5 | S1 pre-edit (y_t+1=0) | 356,428 | 1,014 | 8◦ | 0.761◦ | position causal 0.735 | +0.026 [-0.09, +0.25] | position causal 0.735 | +0.026 [-0.09, +0.25] | 0.685 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 5 | S2 outcome y_T=1 | 7,129 | 1,489 | 3◦ | 0.874◦ | position causal 0.844 | +0.030 [-0.26, +0.09] | position causal 0.844 | +0.030 [-0.26, +0.09] | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 5 | S3 pre-edit & y_T=1 | 6,031 | 391 | 3◦ | 0.811◦ | position rel_pos 0.866 | -0.055 [-0.54, +0.02] | position causal 0.789 | +0.022 [-0.30, +0.07] | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 5 | S4 y_T=0 | 351,049 | 1,102 | 5◦ | 0.806◦ | k=0 probe 0.737 | **+0.069 [+0.03, +0.20]** | k=0 probe 0.737 | **+0.069 [+0.03, +0.20]** | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 10 | S0 y_t=0 | 358,172 | 3,417 | 8◦ | 0.784◦ | position causal 0.743 | +0.040 [-0.09, +0.30] | position causal 0.743 | +0.040 [-0.09, +0.30] | 0.796 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 10 | S1 pre-edit (y_t+1=0) | 356,422 | 1,953 | 8◦ | 0.736◦ | position causal 0.677 | +0.059 [-0.05, +0.32] | position causal 0.677 | +0.059 [-0.05, +0.32] | 0.742 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 10 | S2 outcome y_T=1 | 7,129 | 2,070 | 3◦ | 0.799◦ | position rel_pos 0.834 | -0.036 [-0.43, +0.03] | position causal 0.832 | -0.033 [-0.17, +0.06] | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 10 | S3 pre-edit & y_T=1 | 6,031 | 972 | 3◦ | 0.711◦ | position rel_pos 0.841 | **-0.130 [-0.52, -0.06]** | position causal 0.767 | -0.056 [-0.37, +0.06] | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 10 | S4 y_T=0 | 351,043 | 1,347 | 5◦ | 0.786◦ | k=0 probe 0.726 | **+0.060 [+0.00, +0.20]** | k=0 probe 0.726 | **+0.060 [+0.00, +0.20]** | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 15 | S0 y_t=0 | 358,142 | 3,854 | 9◦ | 0.814◦ | position causal 0.709 | **+0.104 [+0.02, +0.32]** | position causal 0.709 | **+0.104 [+0.02, +0.32]** | 0.864 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 15 | S1 pre-edit (y_t+1=0) | 356,392 | 2,579 | 9◦ | 0.773◦ | position causal 0.656 | **+0.118 [+0.02, +0.33]** | position causal 0.656 | **+0.118 [+0.02, +0.33]** | 0.833 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 15 | S2 outcome y_T=1 | 7,129 | 2,849 | 4◦ | 0.687◦ | position causal 0.877 | -0.190 [-0.23, +0.02] | position causal 0.877 | -0.190 [-0.23, +0.02] | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 15 | S3 pre-edit & y_T=1 | 6,031 | 1,751 | 4◦ | 0.621◦ | position causal 0.841 | -0.221 [-0.36, +0.02] | position causal 0.841 | -0.221 [-0.36, +0.02] | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 15 | S4 y_T=0 | 351,013 | 1,005 | 5◦ | 0.778◦ | k=0 probe 0.680 | **+0.098 [+0.04, +0.23]** | k=0 probe 0.680 | **+0.098 [+0.04, +0.23]** | – |

#### E18, window max50: within y_t = 0, by edit turn and eventual outcome (AUC for y_(t+k); strongest control = highest point AUC among k=0 probe, traj-shuffle and both position baselines; 95% task-bootstrap CI, bold if it excludes 0; ◦ = fewer than 20 test tasks with a positive token)

| Model / benchmark | Probe | k | stratum | tokens | positives | tasks w/ positive | real | strongest control | real − strongest | strongest causal control | real − strongest causal | oracle y_T |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | correct | 2 | S0 y_t=0 | 168,064 | 2,905 | 15◦ | 0.767◦ | traj-shuffle 0.622 | **+0.145 [+0.07, +0.21]** | traj-shuffle 0.622 | **+0.145 [+0.07, +0.21]** | 0.876 |
| Laguna XS.2 / Verified | correct | 2 | S1 pre-edit (y_t+1=0) | 165,909 | 750 | 15◦ | 0.636◦ | k=0 probe 0.556 | **+0.080 [+0.01, +0.18]** | k=0 probe 0.556 | **+0.080 [+0.01, +0.18]** | 0.873 |
| Laguna XS.2 / Verified | correct | 2 | S3 pre-edit & y_T=1 | 38,952 | 735 | 15◦ | 0.721◦ | k=0 probe 0.652 | **+0.069 [+0.01, +0.17]** | k=0 probe 0.652 | **+0.069 [+0.01, +0.17]** | – |
| Laguna XS.2 / Verified | correct | 3 | S0 y_t=0 | 168,064 | 3,544 | 16◦ | 0.745◦ | traj-shuffle 0.557 | **+0.188 [+0.10, +0.28]** | traj-shuffle 0.557 | **+0.188 [+0.10, +0.28]** | 0.877 |
| Laguna XS.2 / Verified | correct | 3 | S1 pre-edit (y_t+1=0) | 165,909 | 1,389 | 16◦ | 0.622◦ | k=0 probe 0.540 | **+0.082 [+0.03, +0.17]** | k=0 probe 0.540 | **+0.082 [+0.03, +0.17]** | 0.875 |
| Laguna XS.2 / Verified | correct | 3 | S3 pre-edit & y_T=1 | 38,952 | 1,364 | 16◦ | 0.702◦ | k=0 probe 0.634 | **+0.068 [+0.01, +0.15]** | k=0 probe 0.634 | **+0.068 [+0.01, +0.15]** | – |
| Laguna XS.2 / Verified | correct | 5 | S0 y_t=0 | 168,064 | 5,309 | 19◦ | 0.716◦ | traj-shuffle 0.546 | **+0.170 [+0.09, +0.25]** | traj-shuffle 0.546 | **+0.170 [+0.09, +0.25]** | 0.882 |
| Laguna XS.2 / Verified | correct | 5 | S1 pre-edit (y_t+1=0) | 165,909 | 3,154 | 19◦ | 0.627◦ | k=0 probe 0.534 | **+0.093 [+0.02, +0.18]** | k=0 probe 0.534 | **+0.093 [+0.02, +0.18]** | 0.882 |
| Laguna XS.2 / Verified | correct | 5 | S3 pre-edit & y_T=1 | 38,952 | 3,114 | 19◦ | 0.709◦ | k=0 probe 0.632 | **+0.077 [+0.00, +0.17]** | k=0 probe 0.632 | **+0.077 [+0.00, +0.17]** | – |
| Laguna XS.2 / Verified | correct | 10 | S0 y_t=0 | 168,064 | 9,232 | 28 | 0.669 | traj-shuffle 0.497 | **+0.172 [+0.09, +0.26]** | traj-shuffle 0.497 | **+0.172 [+0.09, +0.26]** | 0.872 |
| Laguna XS.2 / Verified | correct | 10 | S1 pre-edit (y_t+1=0) | 165,909 | 7,209 | 28 | 0.616 | k=0 probe 0.497 | **+0.119 [+0.05, +0.19]** | k=0 probe 0.497 | **+0.119 [+0.05, +0.19]** | 0.867 |
| Laguna XS.2 / Verified | correct | 10 | S3 pre-edit & y_T=1 | 38,952 | 6,770 | 28 | 0.705 | k=0 probe 0.605 | **+0.100 [+0.01, +0.18]** | k=0 probe 0.605 | **+0.100 [+0.01, +0.18]** | – |
| Laguna XS.2 / Verified | correct | 15 | S0 y_t=0 | 168,064 | 13,585 | 34 | 0.654 | traj-shuffle 0.512 | **+0.142 [+0.07, +0.22]** | traj-shuffle 0.512 | **+0.142 [+0.07, +0.22]** | 0.878 |
| Laguna XS.2 / Verified | correct | 15 | S1 pre-edit (y_t+1=0) | 165,909 | 11,436 | 34 | 0.620 | k=0 probe 0.497 | **+0.123 [+0.07, +0.18]** | k=0 probe 0.497 | **+0.123 [+0.07, +0.18]** | 0.873 |
| Laguna XS.2 / Verified | correct | 15 | S3 pre-edit & y_T=1 | 38,952 | 10,664 | 34 | 0.712 | k=0 probe 0.611 | **+0.102 [+0.01, +0.18]** | k=0 probe 0.611 | **+0.102 [+0.01, +0.18]** | – |
| Laguna XS.2 / Verified | correct | 25 | S0 y_t=0 | 168,064 | 21,412 | 38 | 0.600 | traj-shuffle 0.572 | +0.028 [-0.05, +0.12] | traj-shuffle 0.572 | +0.028 [-0.05, +0.12] | 0.886 |
| Laguna XS.2 / Verified | correct | 25 | S1 pre-edit (y_t+1=0) | 165,909 | 19,400 | 38 | 0.577 | traj-shuffle 0.586 | -0.009 [-0.09, +0.08] | traj-shuffle 0.586 | -0.009 [-0.09, +0.08] | 0.883 |
| Laguna XS.2 / Verified | correct | 25 | S3 pre-edit & y_T=1 | 38,952 | 17,730 | 38 | 0.669 | traj-shuffle 0.606 | +0.063 [-0.02, +0.16] | traj-shuffle 0.606 | +0.063 [-0.02, +0.16] | – |
| Laguna XS.2 / Verified | correct | 50 | S0 y_t=0 | 166,737 | 36,008 | 39 | 0.504 | position causal 0.562 | -0.059 [-0.14, +0.03] | position causal 0.562 | -0.059 [-0.14, +0.03] | 0.961 |
| Laguna XS.2 / Verified | correct | 50 | S1 pre-edit (y_t+1=0) | 164,582 | 33,859 | 39 | 0.493 | position causal 0.560 | -0.067 [-0.14, +0.02] | position causal 0.560 | -0.067 [-0.14, +0.02] | 0.961 |
| Laguna XS.2 / Verified | correct | 50 | S3 pre-edit & y_T=1 | 38,952 | 32,798 | 39 | 0.770 | traj-shuffle 0.793 | -0.023 [-0.08, +0.05] | traj-shuffle 0.793 | -0.023 [-0.08, +0.05] | – |
| Laguna XS.2 / Verified | reduces_failing | 2 | S0 y_t=0 | 160,703 | 7,607 | 28 | 0.816 | traj-shuffle 0.650 | **+0.166 [+0.11, +0.23]** | traj-shuffle 0.650 | **+0.166 [+0.11, +0.23]** | 0.715 |
| Laguna XS.2 / Verified | reduces_failing | 2 | S1 pre-edit (y_t+1=0) | 155,361 | 2,265 | 28 | 0.712 | traj-shuffle 0.626 | +0.086 [-0.00, +0.18] | traj-shuffle 0.626 | +0.086 [-0.00, +0.18] | 0.705 |
| Laguna XS.2 / Verified | reduces_failing | 2 | S3 pre-edit & y_T=1 | 67,516 | 1,913 | 24 | 0.732 | position causal 0.653 | +0.079 [-0.10, +0.28] | position causal 0.653 | +0.079 [-0.10, +0.28] | – |
| Laguna XS.2 / Verified | reduces_failing | 3 | S0 y_t=0 | 160,703 | 9,948 | 30 | 0.786 | traj-shuffle 0.631 | **+0.155 [+0.09, +0.23]** | traj-shuffle 0.631 | **+0.155 [+0.09, +0.23]** | 0.724 |
| Laguna XS.2 / Verified | reduces_failing | 3 | S1 pre-edit (y_t+1=0) | 155,361 | 4,606 | 30 | 0.691 | position causal 0.629 | +0.062 [-0.08, +0.22] | position causal 0.629 | +0.062 [-0.08, +0.22] | 0.725 |
| Laguna XS.2 / Verified | reduces_failing | 3 | S3 pre-edit & y_T=1 | 67,516 | 4,046 | 25 | 0.719 | position causal 0.654 | +0.065 [-0.12, +0.27] | position causal 0.654 | +0.065 [-0.12, +0.27] | – |
| Laguna XS.2 / Verified | reduces_failing | 5 | S0 y_t=0 | 160,703 | 14,597 | 32 | 0.741 | traj-shuffle 0.630 | **+0.111 [+0.05, +0.19]** | traj-shuffle 0.630 | **+0.111 [+0.05, +0.19]** | 0.749 |
| Laguna XS.2 / Verified | reduces_failing | 5 | S1 pre-edit (y_t+1=0) | 155,361 | 9,355 | 32 | 0.667 | position causal 0.619 | +0.048 [-0.09, +0.20] | position causal 0.619 | +0.048 [-0.09, +0.20] | 0.754 |
| Laguna XS.2 / Verified | reduces_failing | 5 | S3 pre-edit & y_T=1 | 67,516 | 8,583 | 27 | 0.701 | k=0 probe 0.646 | **+0.056 [+0.01, +0.12]** | k=0 probe 0.646 | **+0.056 [+0.01, +0.12]** | – |
| Laguna XS.2 / Verified | reduces_failing | 10 | S0 y_t=0 | 160,703 | 25,183 | 40 | 0.679 | traj-shuffle 0.603 | **+0.076 [+0.02, +0.15]** | traj-shuffle 0.603 | **+0.076 [+0.02, +0.15]** | 0.782 |
| Laguna XS.2 / Verified | reduces_failing | 10 | S1 pre-edit (y_t+1=0) | 155,361 | 20,162 | 40 | 0.634 | traj-shuffle 0.574 | +0.060 [-0.01, +0.14] | traj-shuffle 0.574 | +0.060 [-0.01, +0.14] | 0.782 |
| Laguna XS.2 / Verified | reduces_failing | 10 | S3 pre-edit & y_T=1 | 67,516 | 18,771 | 34 | 0.691 | k=0 probe 0.635 | **+0.055 [+0.02, +0.11]** | k=0 probe 0.635 | **+0.055 [+0.02, +0.11]** | – |
| Laguna XS.2 / Verified | reduces_failing | 15 | S0 y_t=0 | 160,703 | 32,506 | 48 | 0.655 | k=0 probe 0.557 | **+0.098 [+0.05, +0.14]** | k=0 probe 0.557 | **+0.098 [+0.05, +0.14]** | 0.797 |
| Laguna XS.2 / Verified | reduces_failing | 15 | S1 pre-edit (y_t+1=0) | 155,361 | 27,394 | 48 | 0.621 | k=0 probe 0.547 | **+0.074 [+0.03, +0.12]** | k=0 probe 0.547 | **+0.074 [+0.03, +0.12]** | 0.798 |
| Laguna XS.2 / Verified | reduces_failing | 15 | S3 pre-edit & y_T=1 | 67,516 | 25,499 | 42 | 0.683 | k=0 probe 0.632 | **+0.052 [+0.00, +0.10]** | k=0 probe 0.632 | **+0.052 [+0.00, +0.10]** | – |
| Laguna XS.2 / Verified | reduces_failing | 25 | S0 y_t=0 | 160,703 | 44,518 | 53 | 0.589 | traj-shuffle 0.603 | -0.014 [-0.07, +0.05] | traj-shuffle 0.603 | -0.014 [-0.07, +0.05] | 0.830 |
| Laguna XS.2 / Verified | reduces_failing | 25 | S1 pre-edit (y_t+1=0) | 155,361 | 39,830 | 53 | 0.566 | traj-shuffle 0.599 | -0.033 [-0.09, +0.03] | traj-shuffle 0.599 | -0.033 [-0.09, +0.03] | 0.833 |
| Laguna XS.2 / Verified | reduces_failing | 25 | S3 pre-edit & y_T=1 | 67,516 | 37,185 | 47 | 0.654 | traj-shuffle 0.645 | +0.009 [-0.08, +0.08] | traj-shuffle 0.645 | +0.009 [-0.08, +0.08] | – |
| Laguna XS.2 / Verified | reduces_failing | 50 | S0 y_t=0 | 159,376 | 62,266 | 50 | 0.545 | position causal 0.519 | +0.026 [-0.05, +0.10] | position causal 0.519 | +0.026 [-0.05, +0.10] | 0.918 |
| Laguna XS.2 / Verified | reduces_failing | 50 | S1 pre-edit (y_t+1=0) | 154,034 | 57,458 | 50 | 0.543 | position causal 0.516 | +0.027 [-0.05, +0.11] | position causal 0.516 | +0.027 [-0.05, +0.11] | 0.918 |
| Laguna XS.2 / Verified | reduces_failing | 50 | S3 pre-edit & y_T=1 | 67,516 | 55,326 | 48 | 0.651 | k=0 probe 0.510 | +0.140 [-0.06, +0.25] | k=0 probe 0.510 | +0.140 [-0.06, +0.25] | – |
| Laguna XS.2 / Verified | has_regressions | 2 | S0 y_t=0 | 170,147 | 4,014 | 12◦ | 0.823◦ | k=0 probe 0.724 | **+0.099 [+0.06, +0.13]** | k=0 probe 0.724 | **+0.099 [+0.06, +0.13]** | 0.809 |
| Laguna XS.2 / Verified | has_regressions | 2 | S1 pre-edit (y_t+1=0) | 166,934 | 801 | 12◦ | 0.637◦ | traj-shuffle 0.616 | +0.020 [-0.09, +0.10] | traj-shuffle 0.616 | +0.020 [-0.09, +0.10] | 0.764 |
| Laguna XS.2 / Verified | has_regressions | 2 | S3 pre-edit & y_T=1 | 27,697 | 544 | 7◦ | 0.636◦ | position causal 0.672 | -0.036 [-0.18, +0.18] | position causal 0.672 | -0.036 [-0.18, +0.18] | – |
| Laguna XS.2 / Verified | has_regressions | 3 | S0 y_t=0 | 170,147 | 5,148 | 14◦ | 0.783◦ | k=0 probe 0.702 | **+0.081 [+0.05, +0.11]** | k=0 probe 0.702 | **+0.081 [+0.05, +0.11]** | 0.819 |
| Laguna XS.2 / Verified | has_regressions | 3 | S1 pre-edit (y_t+1=0) | 166,934 | 1,935 | 14◦ | 0.643◦ | k=0 probe 0.608 | **+0.035 [+0.01, +0.07]** | k=0 probe 0.608 | **+0.035 [+0.01, +0.07]** | 0.812 |
| Laguna XS.2 / Verified | has_regressions | 3 | S3 pre-edit & y_T=1 | 27,697 | 1,386 | 8◦ | 0.653◦ | position causal 0.656 | -0.003 [-0.15, +0.17] | position causal 0.656 | -0.003 [-0.15, +0.17] | – |
| Laguna XS.2 / Verified | has_regressions | 5 | S0 y_t=0 | 170,147 | 8,078 | 15◦ | 0.694◦ | k=0 probe 0.661 | +0.032 [-0.02, +0.09] | k=0 probe 0.661 | +0.032 [-0.02, +0.09] | 0.835 |
| Laguna XS.2 / Verified | has_regressions | 5 | S1 pre-edit (y_t+1=0) | 166,934 | 4,865 | 15◦ | 0.589◦ | k=0 probe 0.596 | -0.007 [-0.06, +0.06] | k=0 probe 0.596 | -0.007 [-0.06, +0.06] | 0.839 |
| Laguna XS.2 / Verified | has_regressions | 5 | S3 pre-edit & y_T=1 | 27,697 | 3,655 | 8◦ | 0.597◦ | position causal 0.663 | -0.067 [-0.22, +0.07] | position causal 0.663 | -0.067 [-0.22, +0.07] | – |
| Laguna XS.2 / Verified | has_regressions | 10 | S0 y_t=0 | 170,147 | 13,943 | 17◦ | 0.644◦ | k=0 probe 0.624 | +0.020 [-0.06, +0.09] | k=0 probe 0.624 | +0.020 [-0.06, +0.09] | 0.858 |
| Laguna XS.2 / Verified | has_regressions | 10 | S1 pre-edit (y_t+1=0) | 166,934 | 10,730 | 17◦ | 0.596◦ | k=0 probe 0.583 | +0.013 [-0.08, +0.10] | k=0 probe 0.583 | +0.013 [-0.08, +0.10] | 0.863 |
| Laguna XS.2 / Verified | has_regressions | 10 | S3 pre-edit & y_T=1 | 27,697 | 8,454 | 10◦ | 0.643◦ | position causal 0.670 | -0.027 [-0.11, +0.05] | position causal 0.670 | -0.027 [-0.11, +0.05] | – |
| Laguna XS.2 / Verified | has_regressions | 15 | S0 y_t=0 | 170,147 | 18,994 | 20 | 0.620 | k=0 probe 0.604 | +0.015 [-0.09, +0.10] | k=0 probe 0.604 | +0.015 [-0.09, +0.10] | 0.862 |
| Laguna XS.2 / Verified | has_regressions | 15 | S1 pre-edit (y_t+1=0) | 166,934 | 15,781 | 20 | 0.587 | k=0 probe 0.572 | +0.015 [-0.09, +0.11] | k=0 probe 0.572 | +0.015 [-0.09, +0.11] | 0.864 |
| Laguna XS.2 / Verified | has_regressions | 15 | S3 pre-edit & y_T=1 | 27,697 | 12,374 | 10◦ | 0.653◦ | position causal 0.690 | -0.037 [-0.13, +0.03] | position causal 0.690 | -0.037 [-0.13, +0.03] | – |
| Laguna XS.2 / Verified | has_regressions | 25 | S0 y_t=0 | 170,147 | 24,272 | 22 | 0.570 | traj-shuffle 0.548 | +0.022 [-0.08, +0.12] | traj-shuffle 0.548 | +0.022 [-0.08, +0.12] | 0.887 |
| Laguna XS.2 / Verified | has_regressions | 25 | S1 pre-edit (y_t+1=0) | 166,934 | 21,343 | 22 | 0.552 | traj-shuffle 0.533 | +0.019 [-0.09, +0.12] | traj-shuffle 0.533 | +0.019 [-0.09, +0.12] | 0.884 |
| Laguna XS.2 / Verified | has_regressions | 25 | S3 pre-edit & y_T=1 | 27,697 | 17,111 | 11◦ | 0.598◦ | position causal 0.627 | -0.029 [-0.11, +0.04] | position causal 0.627 | -0.029 [-0.11, +0.04] | – |
| Laguna XS.2 / Verified | has_regressions | 50 | S0 y_t=0 | 169,767 | 28,941 | 18◦ | 0.563◦ | k=0 probe 0.505 | +0.058 [-0.05, +0.15] | k=0 probe 0.505 | +0.058 [-0.05, +0.15] | 0.959 |
| Laguna XS.2 / Verified | has_regressions | 50 | S1 pre-edit (y_t+1=0) | 166,715 | 26,402 | 18◦ | 0.548◦ | k=0 probe 0.481 | +0.067 [-0.05, +0.16] | k=0 probe 0.481 | +0.067 [-0.05, +0.16] | 0.958 |
| Laguna XS.2 / Verified | has_regressions | 50 | S3 pre-edit & y_T=1 | 27,697 | 24,080 | 14◦ | 0.500◦ | position rel_pos 0.499 | +0.001 [-0.22, +0.22] | k=0 probe 0.460 | +0.040 [-0.06, +0.15] | – |
| Qwen3.6-35B-A3B / Verified | correct | 2 | S0 y_t=0 | 58,726 | 1,085 | 6◦ | 0.833◦ | position causal 0.744 | +0.089 [-0.04, +0.20] | position causal 0.744 | +0.089 [-0.04, +0.20] | 0.657 |
| Qwen3.6-35B-A3B / Verified | correct | 2 | S1 pre-edit (y_t+1=0) | 57,890 | 249 | 6◦ | 0.645◦ | position causal 0.668 | -0.022 [-0.14, +0.14] | position causal 0.668 | -0.022 [-0.14, +0.14] | 0.699 |
| Qwen3.6-35B-A3B / Verified | correct | 2 | S3 pre-edit & y_T=1 | 19,877 | 184 | 5◦ | 0.705◦ | position causal 0.793 | -0.088 [-0.19, +0.06] | position causal 0.793 | -0.088 [-0.19, +0.06] | – |
| Qwen3.6-35B-A3B / Verified | correct | 3 | S0 y_t=0 | 58,726 | 1,291 | 6◦ | 0.832◦ | position causal 0.730 | +0.102 [-0.03, +0.21] | position causal 0.730 | +0.102 [-0.03, +0.21] | 0.661 |
| Qwen3.6-35B-A3B / Verified | correct | 3 | S1 pre-edit (y_t+1=0) | 57,890 | 463 | 6◦ | 0.694◦ | position causal 0.651 | +0.044 [-0.09, +0.18] | position causal 0.651 | +0.044 [-0.09, +0.18] | 0.692 |
| Qwen3.6-35B-A3B / Verified | correct | 3 | S3 pre-edit & y_T=1 | 19,877 | 335 | 5◦ | 0.732◦ | position causal 0.785 | -0.053 [-0.19, +0.08] | position causal 0.785 | -0.053 [-0.19, +0.08] | – |
| Qwen3.6-35B-A3B / Verified | correct | 5 | S0 y_t=0 | 58,726 | 1,782 | 7◦ | 0.784◦ | position causal 0.694 | +0.091 [-0.02, +0.23] | position causal 0.694 | +0.091 [-0.02, +0.23] | 0.677 |
| Qwen3.6-35B-A3B / Verified | correct | 5 | S1 pre-edit (y_t+1=0) | 57,890 | 946 | 7◦ | 0.672◦ | position causal 0.605 | +0.067 [-0.04, +0.24] | position causal 0.605 | +0.067 [-0.04, +0.24] | 0.704 |
| Qwen3.6-35B-A3B / Verified | correct | 5 | S3 pre-edit & y_T=1 | 19,877 | 704 | 6◦ | 0.714◦ | position causal 0.700 | +0.013 [-0.08, +0.12] | position causal 0.700 | +0.013 [-0.08, +0.12] | – |
| Qwen3.6-35B-A3B / Verified | correct | 10 | S0 y_t=0 | 58,726 | 3,329 | 9◦ | 0.679◦ | position causal 0.651 | +0.028 [-0.12, +0.23] | position causal 0.651 | +0.028 [-0.12, +0.23] | 0.689 |
| Qwen3.6-35B-A3B / Verified | correct | 10 | S1 pre-edit (y_t+1=0) | 57,890 | 2,493 | 9◦ | 0.605◦ | position causal 0.599 | +0.006 [-0.14, +0.25] | position causal 0.599 | +0.006 [-0.14, +0.25] | 0.701 |
| Qwen3.6-35B-A3B / Verified | correct | 10 | S3 pre-edit & y_T=1 | 19,877 | 1,816 | 8◦ | 0.642◦ | position causal 0.737 | -0.096 [-0.18, +0.09] | position causal 0.737 | -0.096 [-0.18, +0.09] | – |
| Qwen3.6-35B-A3B / Verified | correct | 15 | S0 y_t=0 | 58,726 | 5,010 | 14◦ | 0.626◦ | position causal 0.566 | +0.060 [-0.11, +0.23] | position causal 0.566 | +0.060 [-0.11, +0.23] | 0.739 |
| Qwen3.6-35B-A3B / Verified | correct | 15 | S1 pre-edit (y_t+1=0) | 57,890 | 4,229 | 14◦ | 0.578◦ | position causal 0.524 | +0.054 [-0.11, +0.24] | position causal 0.524 | +0.054 [-0.11, +0.24] | 0.749 |
| Qwen3.6-35B-A3B / Verified | correct | 15 | S3 pre-edit & y_T=1 | 19,877 | 3,404 | 13◦ | 0.598◦ | position causal 0.674 | -0.075 [-0.19, +0.08] | position causal 0.674 | -0.075 [-0.19, +0.08] | – |
| Qwen3.6-35B-A3B / Verified | correct | 25 | S0 y_t=0 | 58,726 | 8,350 | 20 | 0.550 | k=0 probe 0.520 | +0.030 [-0.02, +0.10] | k=0 probe 0.520 | +0.030 [-0.02, +0.10] | 0.801 |
| Qwen3.6-35B-A3B / Verified | correct | 25 | S1 pre-edit (y_t+1=0) | 57,890 | 7,569 | 20 | 0.520 | traj-shuffle 0.505 | +0.016 [-0.08, +0.10] | traj-shuffle 0.505 | +0.016 [-0.08, +0.10] | 0.811 |
| Qwen3.6-35B-A3B / Verified | correct | 25 | S3 pre-edit & y_T=1 | 19,877 | 6,689 | 19◦ | 0.540◦ | position causal 0.601 | -0.062 [-0.15, +0.01] | position causal 0.601 | -0.062 [-0.15, +0.01] | – |
| Qwen3.6-35B-A3B / Verified | correct | 50 | S0 y_t=0 | 58,726 | 17,606 | 29 | 0.630 | position causal 0.517 | +0.113 [-0.02, +0.22] | position causal 0.517 | +0.113 [-0.02, +0.22] | 0.937 |
| Qwen3.6-35B-A3B / Verified | correct | 50 | S1 pre-edit (y_t+1=0) | 57,890 | 16,959 | 29 | 0.630 | position rel_pos 0.509 | +0.121 [-0.02, +0.24] | position causal 0.506 | +0.124 [-0.01, +0.24] | 0.939 |
| Qwen3.6-35B-A3B / Verified | correct | 50 | S3 pre-edit & y_T=1 | 19,877 | 16,362 | 29 | 0.521 | position rel_pos 0.589 | -0.068 [-0.21, +0.06] | k=0 probe 0.564 | -0.043 [-0.12, +0.09] | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 2 | S0 y_t=0 | 56,911 | 1,717 | 10◦ | 0.898◦ | k=0 probe 0.730 | **+0.168 [+0.08, +0.25]** | k=0 probe 0.730 | **+0.168 [+0.08, +0.25]** | 0.673 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 2 | S1 pre-edit (y_t+1=0) | 55,601 | 407 | 10◦ | 0.777◦ | position causal 0.619 | **+0.158 [+0.01, +0.28]** | position causal 0.619 | **+0.158 [+0.01, +0.28]** | 0.667 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 2 | S3 pre-edit & y_T=1 | 27,155 | 334 | 8◦ | 0.742◦ | position causal 0.615 | +0.127 [-0.03, +0.26] | position causal 0.615 | +0.127 [-0.03, +0.26] | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3 | S0 y_t=0 | 56,911 | 2,099 | 10◦ | 0.875◦ | k=0 probe 0.695 | **+0.181 [+0.09, +0.26]** | k=0 probe 0.695 | **+0.181 [+0.09, +0.26]** | 0.670 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3 | S1 pre-edit (y_t+1=0) | 55,601 | 796 | 10◦ | 0.771◦ | position causal 0.599 | **+0.172 [+0.00, +0.30]** | position causal 0.599 | **+0.172 [+0.00, +0.30]** | 0.658 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 3 | S3 pre-edit & y_T=1 | 27,155 | 636 | 8◦ | 0.729◦ | position causal 0.596 | +0.134 [-0.03, +0.27] | position causal 0.596 | +0.134 [-0.03, +0.27] | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 5 | S0 y_t=0 | 56,911 | 2,987 | 12◦ | 0.844◦ | k=0 probe 0.659 | **+0.185 [+0.09, +0.26]** | k=0 probe 0.659 | **+0.185 [+0.09, +0.26]** | 0.664 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 5 | S1 pre-edit (y_t+1=0) | 55,601 | 1,684 | 12◦ | 0.760◦ | position causal 0.610 | +0.151 [-0.00, +0.27] | position causal 0.610 | +0.151 [-0.00, +0.27] | 0.651 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 5 | S3 pre-edit & y_T=1 | 27,155 | 1,314 | 9◦ | 0.711◦ | position causal 0.580 | +0.131 [-0.04, +0.29] | position causal 0.580 | +0.131 [-0.04, +0.29] | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 10 | S0 y_t=0 | 56,911 | 5,432 | 14◦ | 0.771◦ | k=0 probe 0.607 | **+0.165 [+0.10, +0.23]** | k=0 probe 0.607 | **+0.165 [+0.10, +0.23]** | 0.677 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 10 | S1 pre-edit (y_t+1=0) | 55,601 | 4,129 | 14◦ | 0.714◦ | position causal 0.568 | **+0.146 [+0.02, +0.25]** | position causal 0.568 | **+0.146 [+0.02, +0.25]** | 0.673 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 10 | S3 pre-edit & y_T=1 | 27,155 | 3,338 | 11◦ | 0.684◦ | position causal 0.576 | +0.108 [-0.03, +0.20] | position causal 0.576 | +0.108 [-0.03, +0.20] | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 15 | S0 y_t=0 | 56,911 | 7,944 | 19◦ | 0.703◦ | position causal 0.575 | **+0.128 [+0.00, +0.24]** | position causal 0.575 | **+0.128 [+0.00, +0.24]** | 0.717 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 15 | S1 pre-edit (y_t+1=0) | 55,601 | 6,696 | 19◦ | 0.658◦ | position causal 0.552 | +0.106 [-0.02, +0.22] | position causal 0.552 | +0.106 [-0.02, +0.22] | 0.716 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 15 | S3 pre-edit & y_T=1 | 27,155 | 5,809 | 16◦ | 0.620◦ | position causal 0.577 | +0.043 [-0.08, +0.14] | position causal 0.577 | +0.043 [-0.08, +0.14] | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 25 | S0 y_t=0 | 56,911 | 12,561 | 20 | 0.635 | position causal 0.564 | +0.071 [-0.01, +0.15] | position causal 0.564 | +0.071 [-0.01, +0.15] | 0.767 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 25 | S1 pre-edit (y_t+1=0) | 55,601 | 11,313 | 20 | 0.608 | position causal 0.547 | +0.061 [-0.02, +0.14] | position causal 0.547 | +0.061 [-0.02, +0.14] | 0.769 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 25 | S3 pre-edit & y_T=1 | 27,155 | 10,371 | 19◦ | 0.569◦ | traj-shuffle 0.592 | -0.023 [-0.11, +0.07] | traj-shuffle 0.592 | -0.023 [-0.11, +0.07] | – |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 50 | S0 y_t=0 | 56,911 | 24,595 | 29 | 0.551 | position causal 0.550 | +0.001 [-0.05, +0.06] | position causal 0.550 | +0.001 [-0.05, +0.06] | 0.914 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 50 | S1 pre-edit (y_t+1=0) | 55,601 | 23,381 | 29 | 0.540 | position causal 0.546 | -0.006 [-0.06, +0.05] | position causal 0.546 | -0.006 [-0.06, +0.05] | 0.916 |
| Qwen3.6-35B-A3B / Verified | reduces_failing | 50 | S3 pre-edit & y_T=1 | 27,155 | 22,679 | 29 | 0.451 | position rel_pos 0.609 | **-0.158 [-0.23, -0.05]** | traj-shuffle 0.593 | **-0.142 [-0.22, -0.03]** | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 2 | S0 y_t=0 | 59,290 | 381 | 4◦ | 0.826◦ | k=0 probe 0.680 | **+0.146 [+0.00, +0.20]** | k=0 probe 0.680 | **+0.146 [+0.00, +0.20]** | 0.715 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 2 | S1 pre-edit (y_t+1=0) | 59,027 | 118 | 4◦ | 0.738◦ | k=0 probe 0.600 | +0.138 [-0.01, +0.18] | k=0 probe 0.600 | +0.138 [-0.01, +0.18] | 0.665 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 2 | S3 pre-edit & y_T=1 | 10,132 | 59 | 3◦ | 0.798◦ | k=0 probe 0.660 | **+0.138 [+0.00, +0.19]** | k=0 probe 0.660 | **+0.138 [+0.00, +0.19]** | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 3 | S0 y_t=0 | 59,290 | 609 | 4◦ | 0.810◦ | k=0 probe 0.661 | **+0.148 [+0.02, +0.20]** | k=0 probe 0.661 | **+0.148 [+0.02, +0.20]** | 0.686 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 3 | S1 pre-edit (y_t+1=0) | 59,027 | 346 | 4◦ | 0.759◦ | k=0 probe 0.620 | **+0.139 [+0.03, +0.19]** | k=0 probe 0.620 | **+0.139 [+0.03, +0.19]** | 0.646 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 3 | S3 pre-edit & y_T=1 | 10,132 | 160 | 3◦ | 0.834◦ | k=0 probe 0.671 | **+0.162 [+0.04, +0.24]** | k=0 probe 0.671 | **+0.162 [+0.04, +0.24]** | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 5 | S0 y_t=0 | 59,290 | 895 | 5◦ | 0.749◦ | k=0 probe 0.640 | **+0.109 [+0.04, +0.16]** | k=0 probe 0.640 | **+0.109 [+0.04, +0.16]** | 0.686 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 5 | S1 pre-edit (y_t+1=0) | 59,027 | 632 | 5◦ | 0.700◦ | k=0 probe 0.609 | **+0.092 [+0.03, +0.13]** | k=0 probe 0.609 | **+0.092 [+0.03, +0.13]** | 0.664 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 5 | S3 pre-edit & y_T=1 | 10,132 | 314 | 4◦ | 0.798◦ | k=0 probe 0.672 | **+0.126 [+0.06, +0.17]** | k=0 probe 0.672 | **+0.126 [+0.06, +0.17]** | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 10 | S0 y_t=0 | 59,290 | 1,639 | 6◦ | 0.692◦ | k=0 probe 0.588 | **+0.104 [+0.02, +0.18]** | k=0 probe 0.588 | **+0.104 [+0.02, +0.18]** | 0.691 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 10 | S1 pre-edit (y_t+1=0) | 59,027 | 1,376 | 6◦ | 0.648◦ | k=0 probe 0.563 | **+0.085 [+0.02, +0.16]** | k=0 probe 0.563 | **+0.085 [+0.02, +0.16]** | 0.681 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 10 | S3 pre-edit & y_T=1 | 10,132 | 724 | 6◦ | 0.714◦ | k=0 probe 0.612 | **+0.103 [+0.04, +0.15]** | k=0 probe 0.612 | **+0.103 [+0.04, +0.15]** | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 15 | S0 y_t=0 | 59,290 | 2,562 | 9◦ | 0.650◦ | k=0 probe 0.561 | **+0.089 [+0.02, +0.17]** | k=0 probe 0.561 | **+0.089 [+0.02, +0.17]** | 0.700 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 15 | S1 pre-edit (y_t+1=0) | 59,027 | 2,299 | 9◦ | 0.624◦ | k=0 probe 0.544 | **+0.081 [+0.02, +0.16]** | k=0 probe 0.544 | **+0.081 [+0.02, +0.16]** | 0.695 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 15 | S3 pre-edit & y_T=1 | 10,132 | 1,258 | 6◦ | 0.693◦ | k=0 probe 0.589 | **+0.104 [+0.03, +0.16]** | k=0 probe 0.589 | **+0.104 [+0.03, +0.16]** | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 25 | S0 y_t=0 | 59,290 | 5,137 | 11◦ | 0.584◦ | position rel_pos 0.567 | +0.018 [-0.15, +0.23] | traj-shuffle 0.547 | +0.037 [-0.03, +0.12] | 0.727 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 25 | S1 pre-edit (y_t+1=0) | 59,027 | 4,874 | 11◦ | 0.570◦ | position rel_pos 0.560 | +0.010 [-0.16, +0.23] | traj-shuffle 0.535 | +0.035 [-0.03, +0.12] | 0.726 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 25 | S3 pre-edit & y_T=1 | 10,132 | 2,854 | 8◦ | 0.591◦ | position rel_pos 0.633 | -0.042 [-0.24, +0.28] | position causal 0.632 | -0.041 [-0.19, +0.22] | – |
| Qwen3.6-35B-A3B / Verified | has_regressions | 50 | S0 y_t=0 | 59,290 | 10,044 | 15◦ | 0.637◦ | position causal 0.575 | +0.062 [-0.00, +0.15] | position causal 0.575 | +0.062 [-0.00, +0.15] | 0.911 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 50 | S1 pre-edit (y_t+1=0) | 59,027 | 9,866 | 15◦ | 0.636◦ | position causal 0.578 | +0.058 [-0.01, +0.15] | position causal 0.578 | +0.058 [-0.01, +0.15] | 0.910 |
| Qwen3.6-35B-A3B / Verified | has_regressions | 50 | S3 pre-edit & y_T=1 | 10,132 | 8,439 | 13◦ | 0.662◦ | position causal 0.794 | **-0.132 [-0.24, -0.07]** | position causal 0.794 | **-0.132 [-0.24, -0.07]** | – |
| Laguna XS.2 / Pro | correct | 2 | S0 y_t=0 | 125,585 | 638 | 7◦ | 0.803◦ | position rel_pos 0.797 | +0.007 [-0.09, +0.13] | k=0 probe 0.739 | **+0.064 [+0.00, +0.11]** | 0.920 |
| Laguna XS.2 / Pro | correct | 2 | S1 pre-edit (y_t+1=0) | 125,048 | 249 | 7◦ | 0.720◦ | position rel_pos 0.772 | -0.052 [-0.15, +0.21] | k=0 probe 0.714 | +0.006 [-0.02, +0.04] | 0.877 |
| Laguna XS.2 / Pro | correct | 2 | S3 pre-edit & y_T=1 | 13,267 | 214 | 6◦ | 0.698◦ | position rel_pos 0.831 | -0.134 [-0.21, +0.00] | k=0 probe 0.715 | -0.017 [-0.05, +0.03] | – |
| Laguna XS.2 / Pro | correct | 3 | S0 y_t=0 | 125,581 | 724 | 8◦ | 0.784◦ | position rel_pos 0.797 | -0.013 [-0.11, +0.10] | k=0 probe 0.731 | +0.053 [-0.01, +0.09] | 0.931 |
| Laguna XS.2 / Pro | correct | 3 | S1 pre-edit (y_t+1=0) | 125,044 | 335 | 8◦ | 0.706◦ | position rel_pos 0.778 | -0.072 [-0.15, +0.10] | k=0 probe 0.703 | +0.003 [-0.03, +0.04] | 0.912 |
| Laguna XS.2 / Pro | correct | 3 | S3 pre-edit & y_T=1 | 13,267 | 311 | 7◦ | 0.678◦ | position rel_pos 0.799 | -0.121 [-0.19, +0.07] | k=0 probe 0.692 | -0.014 [-0.06, +0.05] | – |
| Laguna XS.2 / Pro | correct | 5 | S0 y_t=0 | 125,565 | 1,623 | 9◦ | 0.784◦ | position rel_pos 0.813 | -0.029 [-0.10, +0.03] | k=0 probe 0.741 | +0.043 [-0.01, +0.09] | 0.936 |
| Laguna XS.2 / Pro | correct | 5 | S1 pre-edit (y_t+1=0) | 125,028 | 1,234 | 9◦ | 0.763◦ | position rel_pos 0.811 | -0.049 [-0.11, +0.02] | k=0 probe 0.736 | +0.027 [-0.03, +0.08] | 0.932 |
| Laguna XS.2 / Pro | correct | 5 | S3 pre-edit & y_T=1 | 13,267 | 1,188 | 8◦ | 0.753◦ | position rel_pos 0.850 | **-0.098 [-0.17, -0.03]** | k=0 probe 0.751 | +0.001 [-0.03, +0.04] | – |
| Laguna XS.2 / Pro | correct | 10 | S0 y_t=0 | 125,543 | 3,320 | 11◦ | 0.716◦ | position rel_pos 0.775 | -0.059 [-0.16, +0.04] | k=0 probe 0.712 | +0.003 [-0.06, +0.07] | 0.951 |
| Laguna XS.2 / Pro | correct | 10 | S1 pre-edit (y_t+1=0) | 125,006 | 2,931 | 11◦ | 0.701◦ | position rel_pos 0.768 | -0.068 [-0.17, +0.05] | k=0 probe 0.707 | -0.006 [-0.07, +0.06] | 0.951 |
| Laguna XS.2 / Pro | correct | 10 | S3 pre-edit & y_T=1 | 13,267 | 2,893 | 10◦ | 0.715◦ | position rel_pos 0.836 | **-0.121 [-0.24, -0.01]** | k=0 probe 0.747 | -0.032 [-0.10, +0.03] | – |
| Laguna XS.2 / Pro | correct | 15 | S0 y_t=0 | 125,509 | 5,292 | 10◦ | 0.680◦ | position rel_pos 0.702 | -0.023 [-0.12, +0.06] | k=0 probe 0.675 | +0.005 [-0.07, +0.07] | 0.958 |
| Laguna XS.2 / Pro | correct | 15 | S1 pre-edit (y_t+1=0) | 124,972 | 4,903 | 10◦ | 0.669◦ | position rel_pos 0.693 | -0.024 [-0.14, +0.07] | k=0 probe 0.668 | +0.001 [-0.07, +0.07] | 0.957 |
| Laguna XS.2 / Pro | correct | 15 | S3 pre-edit & y_T=1 | 13,267 | 4,830 | 10◦ | 0.707◦ | position rel_pos 0.789 | -0.082 [-0.18, +0.00] | k=0 probe 0.736 | -0.029 [-0.08, +0.02] | – |
| Laguna XS.2 / Pro | correct | 25 | S0 y_t=0 | 125,298 | 8,022 | 13◦ | 0.618◦ | position rel_pos 0.604 | +0.014 [-0.10, +0.11] | k=0 probe 0.590 | +0.028 [-0.06, +0.10] | 0.971 |
| Laguna XS.2 / Pro | correct | 25 | S1 pre-edit (y_t+1=0) | 124,761 | 7,584 | 13◦ | 0.607◦ | position rel_pos 0.593 | +0.014 [-0.11, +0.12] | k=0 probe 0.581 | +0.026 [-0.06, +0.10] | 0.971 |
| Laguna XS.2 / Pro | correct | 25 | S3 pre-edit & y_T=1 | 13,267 | 7,519 | 13◦ | 0.624◦ | position rel_pos 0.699 | -0.075 [-0.19, +0.04] | k=0 probe 0.642 | -0.018 [-0.10, +0.06] | – |
| Laguna XS.2 / Pro | correct | 50 | S0 y_t=0 | 124,588 | 12,656 | 19◦ | 0.603◦ | k=0 probe 0.537 | +0.066 [-0.01, +0.14] | k=0 probe 0.537 | +0.066 [-0.01, +0.14] | 0.993 |
| Laguna XS.2 / Pro | correct | 50 | S1 pre-edit (y_t+1=0) | 124,045 | 12,218 | 19◦ | 0.597◦ | k=0 probe 0.530 | +0.068 [-0.01, +0.15] | k=0 probe 0.530 | +0.068 [-0.01, +0.15] | 0.993 |
| Laguna XS.2 / Pro | correct | 50 | S3 pre-edit & y_T=1 | 13,267 | 12,171 | 19◦ | 0.651◦ | position rel_pos 0.767 | -0.116 [-0.29, +0.05] | k=0 probe 0.641 | +0.009 [-0.18, +0.10] | – |
| Laguna XS.2 / Pro | reduces_failing | 2 | S0 y_t=0 | 122,895 | 2,931 | 14◦ | 0.864◦ | k=0 probe 0.814 | **+0.050 [+0.02, +0.12]** | k=0 probe 0.814 | **+0.050 [+0.02, +0.12]** | 0.882 |
| Laguna XS.2 / Pro | reduces_failing | 2 | S1 pre-edit (y_t+1=0) | 121,162 | 1,357 | 14◦ | 0.839◦ | k=0 probe 0.803 | **+0.035 [+0.02, +0.07]** | k=0 probe 0.803 | **+0.035 [+0.02, +0.07]** | 0.861 |
| Laguna XS.2 / Pro | reduces_failing | 2 | S3 pre-edit & y_T=1 | 15,066 | 1,138 | 11◦ | 0.862◦ | position rel_pos 0.845 | +0.017 [-0.10, +0.10] | k=0 probe 0.844 | +0.018 [-0.00, +0.08] | – |
| Laguna XS.2 / Pro | reduces_failing | 3 | S0 y_t=0 | 122,891 | 3,722 | 16◦ | 0.870◦ | k=0 probe 0.808 | **+0.062 [+0.03, +0.14]** | k=0 probe 0.808 | **+0.062 [+0.03, +0.14]** | 0.878 |
| Laguna XS.2 / Pro | reduces_failing | 3 | S1 pre-edit (y_t+1=0) | 121,158 | 2,148 | 16◦ | 0.846◦ | k=0 probe 0.796 | **+0.050 [+0.03, +0.11]** | k=0 probe 0.796 | **+0.050 [+0.03, +0.11]** | 0.861 |
| Laguna XS.2 / Pro | reduces_failing | 3 | S3 pre-edit & y_T=1 | 15,066 | 1,792 | 11◦ | 0.887◦ | k=0 probe 0.856 | **+0.031 [+0.00, +0.13]** | k=0 probe 0.856 | **+0.031 [+0.00, +0.13]** | – |
| Laguna XS.2 / Pro | reduces_failing | 5 | S0 y_t=0 | 122,875 | 4,951 | 16◦ | 0.843◦ | k=0 probe 0.779 | **+0.064 [+0.03, +0.14]** | k=0 probe 0.779 | **+0.064 [+0.03, +0.14]** | 0.868 |
| Laguna XS.2 / Pro | reduces_failing | 5 | S1 pre-edit (y_t+1=0) | 121,142 | 3,366 | 16◦ | 0.815◦ | k=0 probe 0.756 | **+0.059 [+0.03, +0.12]** | k=0 probe 0.756 | **+0.059 [+0.03, +0.12]** | 0.850 |
| Laguna XS.2 / Pro | reduces_failing | 5 | S3 pre-edit & y_T=1 | 15,066 | 2,710 | 11◦ | 0.867◦ | k=0 probe 0.840 | **+0.027 [+0.00, +0.10]** | k=0 probe 0.840 | **+0.027 [+0.00, +0.10]** | – |
| Laguna XS.2 / Pro | reduces_failing | 10 | S0 y_t=0 | 122,853 | 7,704 | 21 | 0.774 | k=0 probe 0.722 | **+0.052 [+0.01, +0.13]** | k=0 probe 0.722 | **+0.052 [+0.01, +0.13]** | 0.850 |
| Laguna XS.2 / Pro | reduces_failing | 10 | S1 pre-edit (y_t+1=0) | 121,120 | 6,149 | 21 | 0.744 | k=0 probe 0.694 | **+0.050 [+0.01, +0.11]** | k=0 probe 0.694 | **+0.050 [+0.01, +0.11]** | 0.832 |
| Laguna XS.2 / Pro | reduces_failing | 10 | S3 pre-edit & y_T=1 | 15,066 | 4,650 | 13◦ | 0.768◦ | k=0 probe 0.778 | -0.011 [-0.05, +0.05] | k=0 probe 0.778 | -0.011 [-0.05, +0.05] | – |
| Laguna XS.2 / Pro | reduces_failing | 15 | S0 y_t=0 | 122,819 | 9,565 | 24 | 0.732 | k=0 probe 0.660 | **+0.071 [+0.02, +0.15]** | k=0 probe 0.660 | **+0.071 [+0.02, +0.15]** | 0.878 |
| Laguna XS.2 / Pro | reduces_failing | 15 | S1 pre-edit (y_t+1=0) | 121,086 | 8,138 | 24 | 0.704 | k=0 probe 0.630 | **+0.074 [+0.03, +0.14]** | k=0 probe 0.630 | **+0.074 [+0.03, +0.14]** | 0.863 |
| Laguna XS.2 / Pro | reduces_failing | 15 | S3 pre-edit & y_T=1 | 15,066 | 6,536 | 17◦ | 0.713◦ | position rel_pos 0.719 | -0.006 [-0.18, +0.09] | k=0 probe 0.712 | +0.001 [-0.06, +0.08] | – |
| Laguna XS.2 / Pro | reduces_failing | 25 | S0 y_t=0 | 122,608 | 13,231 | 28 | 0.683 | k=0 probe 0.578 | **+0.105 [+0.04, +0.18]** | k=0 probe 0.578 | **+0.105 [+0.04, +0.18]** | 0.884 |
| Laguna XS.2 / Pro | reduces_failing | 25 | S1 pre-edit (y_t+1=0) | 120,875 | 11,662 | 28 | 0.663 | k=0 probe 0.545 | **+0.118 [+0.06, +0.19]** | k=0 probe 0.545 | **+0.118 [+0.06, +0.19]** | 0.875 |
| Laguna XS.2 / Pro | reduces_failing | 25 | S3 pre-edit & y_T=1 | 15,066 | 9,367 | 21 | 0.536 | position rel_pos 0.581 | -0.045 [-0.22, +0.11] | k=0 probe 0.577 | -0.041 [-0.12, +0.05] | – |
| Laguna XS.2 / Pro | reduces_failing | 50 | S0 y_t=0 | 121,898 | 15,890 | 26 | 0.690 | k=0 probe 0.542 | **+0.147 [+0.07, +0.24]** | k=0 probe 0.542 | **+0.147 [+0.07, +0.24]** | 0.971 |
| Laguna XS.2 / Pro | reduces_failing | 50 | S1 pre-edit (y_t+1=0) | 120,159 | 14,386 | 26 | 0.679 | traj-shuffle 0.539 | **+0.140 [+0.06, +0.22]** | traj-shuffle 0.539 | **+0.140 [+0.06, +0.22]** | 0.969 |
| Laguna XS.2 / Pro | reduces_failing | 50 | S3 pre-edit & y_T=1 | 15,066 | 13,674 | 24 | 0.457 | k=0 probe 0.489 | -0.032 [-0.19, +0.09] | k=0 probe 0.489 | -0.032 [-0.19, +0.09] | – |
| Laguna XS.2 / Pro | has_regressions | 2 | S0 y_t=0 | 123,977 | 1,449 | 5◦ | 0.730◦ | k=0 probe 0.742 | -0.011 [-0.08, +0.07] | k=0 probe 0.742 | -0.011 [-0.08, +0.07] | 0.618 |
| Laguna XS.2 / Pro | has_regressions | 2 | S1 pre-edit (y_t+1=0) | 122,875 | 454 | 5◦ | 0.678◦ | k=0 probe 0.673 | +0.005 [-0.10, +0.08] | k=0 probe 0.673 | +0.005 [-0.10, +0.08] | 0.675 |
| Laguna XS.2 / Pro | has_regressions | 2 | S3 pre-edit & y_T=1 | 9,463 | 194 | 3◦ | 0.328◦ | k=0 probe 0.557 | -0.229 [-0.38, +0.05] | k=0 probe 0.557 | -0.229 [-0.38, +0.05] | – |
| Laguna XS.2 / Pro | has_regressions | 3 | S0 y_t=0 | 123,973 | 1,338 | 5◦ | 0.778◦ | position causal 0.712 | +0.067 [-0.11, +0.18] | position causal 0.712 | +0.067 [-0.11, +0.18] | 0.652 |
| Laguna XS.2 / Pro | has_regressions | 3 | S1 pre-edit (y_t+1=0) | 122,871 | 712 | 5◦ | 0.743◦ | k=0 probe 0.655 | **+0.088 [+0.02, +0.14]** | k=0 probe 0.655 | **+0.088 [+0.02, +0.14]** | 0.638 |
| Laguna XS.2 / Pro | has_regressions | 3 | S3 pre-edit & y_T=1 | 9,463 | 251 | 3◦ | 0.531◦ | k=0 probe 0.518 | +0.013 [-0.12, +0.31] | k=0 probe 0.518 | +0.013 [-0.12, +0.31] | – |
| Laguna XS.2 / Pro | has_regressions | 5 | S0 y_t=0 | 123,957 | 1,626 | 5◦ | 0.786◦ | k=0 probe 0.661 | **+0.125 [+0.02, +0.18]** | k=0 probe 0.661 | **+0.125 [+0.02, +0.18]** | 0.655 |
| Laguna XS.2 / Pro | has_regressions | 5 | S1 pre-edit (y_t+1=0) | 122,855 | 1,223 | 5◦ | 0.788◦ | k=0 probe 0.651 | **+0.137 [+0.04, +0.18]** | k=0 probe 0.651 | **+0.137 [+0.04, +0.18]** | 0.613 |
| Laguna XS.2 / Pro | has_regressions | 5 | S3 pre-edit & y_T=1 | 9,463 | 368 | 4◦ | 0.571◦ | k=0 probe 0.501 | +0.071 [-0.11, +0.28] | k=0 probe 0.501 | +0.071 [-0.11, +0.28] | – |
| Laguna XS.2 / Pro | has_regressions | 10 | S0 y_t=0 | 123,935 | 2,571 | 9◦ | 0.680◦ | position causal 0.727 | -0.047 [-0.13, +0.27] | position causal 0.727 | -0.047 [-0.13, +0.27] | 0.787 |
| Laguna XS.2 / Pro | has_regressions | 10 | S1 pre-edit (y_t+1=0) | 122,833 | 2,168 | 9◦ | 0.666◦ | position causal 0.710 | -0.044 [-0.15, +0.27] | position causal 0.710 | -0.044 [-0.15, +0.27] | 0.787 |
| Laguna XS.2 / Pro | has_regressions | 10 | S3 pre-edit & y_T=1 | 9,463 | 1,390 | 6◦ | 0.387◦ | position causal 0.678 | -0.291 [-0.42, +0.08] | position causal 0.678 | -0.291 [-0.42, +0.08] | – |
| Laguna XS.2 / Pro | has_regressions | 15 | S0 y_t=0 | 123,901 | 3,871 | 13◦ | 0.682◦ | position causal 0.752 | -0.070 [-0.15, +0.20] | position causal 0.752 | -0.070 [-0.15, +0.20] | 0.856 |
| Laguna XS.2 / Pro | has_regressions | 15 | S1 pre-edit (y_t+1=0) | 122,799 | 3,468 | 13◦ | 0.677◦ | position causal 0.745 | -0.068 [-0.15, +0.22] | position causal 0.745 | -0.068 [-0.15, +0.22] | 0.864 |
| Laguna XS.2 / Pro | has_regressions | 15 | S3 pre-edit & y_T=1 | 9,463 | 2,719 | 9◦ | 0.387◦ | position causal 0.702 | -0.315 [-0.42, +0.14] | position causal 0.702 | -0.315 [-0.42, +0.14] | – |
| Laguna XS.2 / Pro | has_regressions | 25 | S0 y_t=0 | 123,690 | 6,495 | 13◦ | 0.699◦ | position causal 0.780 | -0.081 [-0.17, +0.17] | position causal 0.780 | -0.081 [-0.17, +0.17] | 0.890 |
| Laguna XS.2 / Pro | has_regressions | 25 | S1 pre-edit (y_t+1=0) | 122,588 | 6,222 | 13◦ | 0.704◦ | position causal 0.777 | -0.073 [-0.17, +0.18] | position causal 0.777 | -0.073 [-0.17, +0.18] | 0.887 |
| Laguna XS.2 / Pro | has_regressions | 25 | S3 pre-edit & y_T=1 | 9,463 | 5,053 | 10◦ | 0.335◦ | position causal 0.641 | -0.306 [-0.47, +0.26] | position causal 0.641 | -0.306 [-0.47, +0.26] | – |
| Laguna XS.2 / Pro | has_regressions | 50 | S0 y_t=0 | 122,980 | 8,615 | 13◦ | 0.694◦ | position causal 0.753 | -0.060 [-0.17, +0.15] | position causal 0.753 | -0.060 [-0.17, +0.15] | 0.941 |
| Laguna XS.2 / Pro | has_regressions | 50 | S1 pre-edit (y_t+1=0) | 121,872 | 8,357 | 13◦ | 0.700◦ | position causal 0.749 | -0.049 [-0.16, +0.16] | position causal 0.749 | -0.049 [-0.16, +0.16] | 0.939 |
| Laguna XS.2 / Pro | has_regressions | 50 | S3 pre-edit & y_T=1 | 9,463 | 7,486 | 11◦ | 0.249◦ | position rel_pos 0.696 | **-0.447 [-0.53, -0.06]** | position causal 0.419 | -0.170 [-0.40, +0.24] | – |
| Qwen3.6-35B-A3B / Pro | correct | 2 | S0 y_t=0 | 64,121 | 837 | 5◦ | 0.765◦ | k=0 probe 0.758 | +0.006 [-0.01, +0.02] | k=0 probe 0.758 | +0.006 [-0.01, +0.02] | 0.827 |
| Qwen3.6-35B-A3B / Pro | correct | 2 | S1 pre-edit (y_t+1=0) | 63,459 | 175 | 5◦ | 0.538◦ | k=0 probe 0.532 | +0.006 [-0.01, +0.06] | k=0 probe 0.532 | +0.006 [-0.01, +0.06] | 0.847 |
| Qwen3.6-35B-A3B / Pro | correct | 2 | S3 pre-edit & y_T=1 | 9,739 | 148 | 4◦ | 0.531◦ | k=0 probe 0.526 | +0.005 [-0.01, +0.07] | k=0 probe 0.526 | +0.005 [-0.01, +0.07] | – |
| Qwen3.6-35B-A3B / Pro | correct | 3 | S0 y_t=0 | 64,121 | 968 | 5◦ | 0.724◦ | k=0 probe 0.717 | +0.007 [-0.01, +0.02] | k=0 probe 0.717 | +0.007 [-0.01, +0.02] | 0.834 |
| Qwen3.6-35B-A3B / Pro | correct | 3 | S1 pre-edit (y_t+1=0) | 63,459 | 306 | 5◦ | 0.506◦ | k=0 probe 0.498 | +0.008 [-0.01, +0.05] | k=0 probe 0.498 | +0.008 [-0.01, +0.05] | 0.856 |
| Qwen3.6-35B-A3B / Pro | correct | 3 | S3 pre-edit & y_T=1 | 9,739 | 264 | 4◦ | 0.511◦ | position rel_pos 0.506 | +0.005 [-0.27, +0.23] | k=0 probe 0.504 | +0.007 [-0.01, +0.06] | – |
| Qwen3.6-35B-A3B / Pro | correct | 5 | S0 y_t=0 | 64,108 | 1,224 | 5◦ | 0.657◦ | k=0 probe 0.660 | -0.003 [-0.02, +0.02] | k=0 probe 0.660 | -0.003 [-0.02, +0.02] | 0.847 |
| Qwen3.6-35B-A3B / Pro | correct | 5 | S1 pre-edit (y_t+1=0) | 63,446 | 567 | 5◦ | 0.469◦ | k=0 probe 0.477 | -0.009 [-0.02, +0.04] | k=0 probe 0.477 | -0.009 [-0.02, +0.04] | 0.869 |
| Qwen3.6-35B-A3B / Pro | correct | 5 | S3 pre-edit & y_T=1 | 9,739 | 502 | 4◦ | 0.477◦ | position rel_pos 0.500 | -0.024 [-0.27, +0.30] | k=0 probe 0.488 | -0.011 [-0.03, +0.04] | – |
| Qwen3.6-35B-A3B / Pro | correct | 10 | S0 y_t=0 | 64,100 | 1,746 | 6◦ | 0.671◦ | k=0 probe 0.597 | +0.074 [-0.02, +0.15] | k=0 probe 0.597 | +0.074 [-0.02, +0.15] | 0.867 |
| Qwen3.6-35B-A3B / Pro | correct | 10 | S1 pre-edit (y_t+1=0) | 63,438 | 1,084 | 6◦ | 0.554◦ | k=0 probe 0.462 | +0.092 [-0.02, +0.25] | k=0 probe 0.462 | +0.092 [-0.02, +0.25] | 0.891 |
| Qwen3.6-35B-A3B / Pro | correct | 10 | S3 pre-edit & y_T=1 | 9,739 | 1,000 | 5◦ | 0.529◦ | k=0 probe 0.477 | +0.051 [-0.07, +0.25] | k=0 probe 0.477 | +0.051 [-0.07, +0.25] | – |
| Qwen3.6-35B-A3B / Pro | correct | 15 | S0 y_t=0 | 64,101 | 2,180 | 9◦ | 0.664◦ | k=0 probe 0.567 | +0.097 [-0.00, +0.20] | k=0 probe 0.567 | +0.097 [-0.00, +0.20] | 0.876 |
| Qwen3.6-35B-A3B / Pro | correct | 15 | S1 pre-edit (y_t+1=0) | 63,439 | 1,523 | 9◦ | 0.586◦ | traj-shuffle 0.461 | +0.125 [-0.10, +0.36] | traj-shuffle 0.461 | +0.125 [-0.10, +0.36] | 0.895 |
| Qwen3.6-35B-A3B / Pro | correct | 15 | S3 pre-edit & y_T=1 | 9,739 | 1,407 | 9◦ | 0.544◦ | traj-shuffle 0.522 | +0.022 [-0.16, +0.23] | traj-shuffle 0.522 | +0.022 [-0.16, +0.23] | – |
| Qwen3.6-35B-A3B / Pro | correct | 25 | S0 y_t=0 | 64,099 | 3,370 | 14◦ | 0.656◦ | k=0 probe 0.498 | **+0.158 [+0.03, +0.25]** | k=0 probe 0.498 | **+0.158 [+0.03, +0.25]** | 0.925 |
| Qwen3.6-35B-A3B / Pro | correct | 25 | S1 pre-edit (y_t+1=0) | 63,437 | 2,836 | 14◦ | 0.629◦ | traj-shuffle 0.462 | **+0.167 [+0.03, +0.29]** | traj-shuffle 0.462 | **+0.167 [+0.03, +0.29]** | 0.923 |
| Qwen3.6-35B-A3B / Pro | correct | 25 | S3 pre-edit & y_T=1 | 9,739 | 2,729 | 14◦ | 0.562◦ | position rel_pos 0.559 | +0.003 [-0.16, +0.17] | traj-shuffle 0.493 | +0.069 [-0.07, +0.21] | – |
| Qwen3.6-35B-A3B / Pro | correct | 50 | S0 y_t=0 | 64,079 | 8,216 | 22 | 0.656 | k=0 probe 0.492 | **+0.164 [+0.08, +0.24]** | k=0 probe 0.492 | **+0.164 [+0.08, +0.24]** | 0.974 |
| Qwen3.6-35B-A3B / Pro | correct | 50 | S1 pre-edit (y_t+1=0) | 63,417 | 7,682 | 22 | 0.645 | k=0 probe 0.473 | **+0.171 [+0.08, +0.25]** | k=0 probe 0.473 | **+0.171 [+0.08, +0.25]** | 0.973 |
| Qwen3.6-35B-A3B / Pro | correct | 50 | S3 pre-edit & y_T=1 | 9,739 | 7,573 | 21 | 0.595 | position rel_pos 0.630 | -0.034 [-0.20, +0.15] | k=0 probe 0.508 | +0.087 [-0.02, +0.18] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 2 | S0 y_t=0 | 65,951 | 1,033 | 6◦ | 0.909◦ | k=0 probe 0.827 | **+0.082 [+0.06, +0.18]** | k=0 probe 0.827 | **+0.082 [+0.06, +0.18]** | 0.667 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 2 | S1 pre-edit (y_t+1=0) | 65,080 | 162 | 6◦ | 0.884◦ | k=0 probe 0.760 | +0.124 [-0.08, +0.17] | k=0 probe 0.760 | +0.124 [-0.08, +0.17] | 0.541 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 2 | S3 pre-edit & y_T=1 | 7,884 | 33 | 4◦ | 0.653◦ | k=0 probe 0.667 | -0.014 [-0.25, +0.05] | k=0 probe 0.667 | -0.014 [-0.25, +0.05] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 3 | S0 y_t=0 | 65,951 | 1,116 | 6◦ | 0.894◦ | k=0 probe 0.821 | **+0.073 [+0.05, +0.15]** | k=0 probe 0.821 | **+0.073 [+0.05, +0.15]** | 0.666 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 3 | S1 pre-edit (y_t+1=0) | 65,080 | 245 | 6◦ | 0.858◦ | k=0 probe 0.752 | +0.106 [-0.03, +0.15] | k=0 probe 0.752 | +0.106 [-0.03, +0.15] | 0.578 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 3 | S3 pre-edit & y_T=1 | 7,884 | 68 | 4◦ | 0.629◦ | k=0 probe 0.643 | -0.014 [-0.21, +0.05] | k=0 probe 0.643 | -0.014 [-0.21, +0.05] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 5 | S0 y_t=0 | 65,938 | 1,317 | 6◦ | 0.887◦ | k=0 probe 0.790 | **+0.097 [+0.05, +0.16]** | k=0 probe 0.790 | **+0.097 [+0.05, +0.16]** | 0.655 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 5 | S1 pre-edit (y_t+1=0) | 65,067 | 446 | 6◦ | 0.853◦ | k=0 probe 0.693 | **+0.160 [+0.02, +0.21]** | k=0 probe 0.693 | **+0.160 [+0.02, +0.21]** | 0.585 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 5 | S3 pre-edit & y_T=1 | 7,884 | 129 | 4◦ | 0.643◦ | k=0 probe 0.604 | +0.039 [-0.10, +0.11] | k=0 probe 0.604 | +0.039 [-0.10, +0.11] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 10 | S0 y_t=0 | 65,930 | 1,928 | 7◦ | 0.877◦ | k=0 probe 0.753 | **+0.123 [+0.07, +0.19]** | k=0 probe 0.753 | **+0.123 [+0.07, +0.19]** | 0.690 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 10 | S1 pre-edit (y_t+1=0) | 65,059 | 1,057 | 7◦ | 0.824◦ | k=0 probe 0.681 | **+0.143 [+0.05, +0.22]** | k=0 probe 0.681 | **+0.143 [+0.05, +0.22]** | 0.688 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 10 | S3 pre-edit & y_T=1 | 7,884 | 519 | 5◦ | 0.738◦ | k=0 probe 0.694 | +0.045 [-0.12, +0.21] | k=0 probe 0.694 | +0.045 [-0.12, +0.21] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 15 | S0 y_t=0 | 65,931 | 2,412 | 9◦ | 0.860◦ | k=0 probe 0.721 | **+0.139 [+0.07, +0.22]** | k=0 probe 0.721 | **+0.139 [+0.07, +0.22]** | 0.704 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 15 | S1 pre-edit (y_t+1=0) | 65,060 | 1,541 | 9◦ | 0.818◦ | k=0 probe 0.654 | **+0.164 [+0.06, +0.26]** | k=0 probe 0.654 | **+0.164 [+0.06, +0.26]** | 0.709 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 15 | S3 pre-edit & y_T=1 | 7,884 | 816 | 8◦ | 0.702◦ | k=0 probe 0.653 | +0.048 [-0.06, +0.23] | k=0 probe 0.653 | +0.048 [-0.06, +0.23] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 25 | S0 y_t=0 | 65,929 | 3,601 | 11◦ | 0.815◦ | k=0 probe 0.673 | **+0.142 [+0.06, +0.24]** | k=0 probe 0.673 | **+0.142 [+0.06, +0.24]** | 0.741 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 25 | S1 pre-edit (y_t+1=0) | 65,058 | 2,730 | 11◦ | 0.780◦ | k=0 probe 0.619 | **+0.161 [+0.05, +0.28]** | k=0 probe 0.619 | **+0.161 [+0.05, +0.28]** | 0.754 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 25 | S3 pre-edit & y_T=1 | 7,884 | 1,659 | 11◦ | 0.621◦ | position causal 0.698 | -0.076 [-0.20, +0.21] | position causal 0.698 | -0.076 [-0.20, +0.21] | – |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 50 | S0 y_t=0 | 65,909 | 7,029 | 18◦ | 0.706◦ | position rel_pos 0.565 | **+0.141 [+0.00, +0.31]** | k=0 probe 0.553 | +0.152 [-0.03, +0.34] | 0.919 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 50 | S1 pre-edit (y_t+1=0) | 65,038 | 6,592 | 18◦ | 0.701◦ | position rel_pos 0.574 | +0.127 [-0.01, +0.30] | k=0 probe 0.544 | +0.157 [-0.03, +0.36] | 0.914 |
| Qwen3.6-35B-A3B / Pro | reduces_failing | 50 | S3 pre-edit & y_T=1 | 7,884 | 5,709 | 16◦ | 0.469◦ | k=0 probe 0.649 | -0.180 [-0.33, +0.00] | k=0 probe 0.649 | -0.180 [-0.33, +0.00] | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 2 | S0 y_t=0 | 66,736 | 113 | 1◦ | 0.916◦ | position causal 0.939 | -0.022 [-0.07, +0.03] | position causal 0.939 | -0.022 [-0.07, +0.03] | 0.493 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 2 | S1 pre-edit (y_t+1=0) | 66,667 | 44 | 1◦ | 0.919◦ | position causal 0.909 | +0.010 [-0.06, +0.08] | position causal 0.909 | +0.010 [-0.06, +0.08] | 0.493 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 2 | S3 pre-edit & y_T=1 | 890 | 0 | 0◦ | –◦ | – | – | – | – | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 3 | S0 y_t=0 | 66,736 | 119 | 1◦ | 0.921◦ | position causal 0.943 | -0.022 [-0.07, +0.03] | position causal 0.943 | -0.022 [-0.07, +0.03] | 0.493 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 3 | S1 pre-edit (y_t+1=0) | 66,667 | 50 | 1◦ | 0.919◦ | position causal 0.913 | +0.006 [-0.06, +0.07] | position causal 0.913 | +0.006 [-0.06, +0.07] | 0.493 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 3 | S3 pre-edit & y_T=1 | 890 | 0 | 0◦ | –◦ | – | – | – | – | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 5 | S0 y_t=0 | 66,723 | 243 | 1◦ | 0.947◦ | k=0 probe 0.912 | **+0.035 [+0.02, +0.05]** | k=0 probe 0.912 | **+0.035 [+0.02, +0.05]** | 0.493 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 5 | S1 pre-edit (y_t+1=0) | 66,654 | 174 | 1◦ | 0.954◦ | k=0 probe 0.918 | **+0.036 [+0.02, +0.05]** | k=0 probe 0.918 | **+0.036 [+0.02, +0.05]** | 0.493 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 5 | S3 pre-edit & y_T=1 | 890 | 0 | 0◦ | –◦ | – | – | – | – | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 10 | S0 y_t=0 | 66,715 | 56 | 2◦ | 0.826◦ | traj-shuffle 0.810 | +0.016 [-0.08, +0.10] | traj-shuffle 0.810 | +0.016 [-0.08, +0.10] | 0.547 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 10 | S1 pre-edit (y_t+1=0) | 66,646 | 56 | 2◦ | 0.827◦ | traj-shuffle 0.810 | +0.017 [-0.08, +0.10] | traj-shuffle 0.810 | +0.017 [-0.08, +0.10] | 0.547 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 10 | S3 pre-edit & y_T=1 | 890 | 6 | 1◦ | 0.718◦ | position rel_pos 0.891 | **-0.173 [-0.18, -0.15]** | k=0 probe 0.614 | **+0.104 [+0.08, +0.11]** | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 15 | S0 y_t=0 | 66,716 | 89 | 2◦ | 0.711◦ | k=0 probe 0.579 | **+0.131 [+0.10, +0.16]** | k=0 probe 0.579 | **+0.131 [+0.10, +0.16]** | 0.651 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 15 | S1 pre-edit (y_t+1=0) | 66,647 | 89 | 2◦ | 0.711◦ | k=0 probe 0.580 | **+0.132 [+0.10, +0.16]** | k=0 probe 0.580 | **+0.132 [+0.10, +0.16]** | 0.651 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 15 | S3 pre-edit & y_T=1 | 890 | 28 | 1◦ | 0.561◦ | position rel_pos 0.869 | **-0.308 [-0.31, -0.23]** | traj-shuffle 0.571 | -0.010 [-0.02, +0.08] | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 25 | S0 y_t=0 | 66,714 | 320 | 4◦ | 0.754◦ | k=0 probe 0.654 | **+0.100 [+0.01, +0.19]** | k=0 probe 0.654 | **+0.100 [+0.01, +0.19]** | 0.631 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 25 | S1 pre-edit (y_t+1=0) | 66,645 | 320 | 4◦ | 0.754◦ | k=0 probe 0.654 | **+0.100 [+0.01, +0.19]** | k=0 probe 0.654 | **+0.100 [+0.01, +0.19]** | 0.631 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 25 | S3 pre-edit & y_T=1 | 890 | 88 | 1◦ | 0.409◦ | position rel_pos 0.693 | **-0.283 [-0.29, -0.20]** | traj-shuffle 0.602 | **-0.193 [-0.20, -0.07]** | – |
| Qwen3.6-35B-A3B / Pro | has_regressions | 50 | S0 y_t=0 | 66,694 | 759 | 4◦ | 0.841◦ | position rel_pos 0.664 | **+0.177 [+0.09, +0.47]** | k=0 probe 0.565 | **+0.275 [+0.18, +0.53]** | 0.850 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 50 | S1 pre-edit (y_t+1=0) | 66,625 | 759 | 4◦ | 0.841◦ | position rel_pos 0.664 | **+0.177 [+0.09, +0.47]** | k=0 probe 0.566 | **+0.275 [+0.18, +0.53]** | 0.850 |
| Qwen3.6-35B-A3B / Pro | has_regressions | 50 | S3 pre-edit & y_T=1 | 890 | 535 | 2◦ | 0.574◦ | position rel_pos 0.693 | **-0.119 [-0.17, -0.06]** | k=0 probe 0.601 | -0.027 [-0.12, +0.07] | – |

#### E18 (b): last-turn label y_T of `correct` vs the final-patch evaluation (all trajectories of each run)

| Run | trajectories | y_T undefined (no labelled edit) | y_T=1, resolved | y_T=0, resolved | y_T=1, not resolved | y_T=0, not resolved | agreement (y_T defined) | last-edit label agreement | last edit issued in the last turn |
|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | 4,991 | 316 | 2,806 | 63 | 43 | 1,763 | 0.977 | 0.980 | 92 |
| Qwen3.6-35B-A3B / Verified | 4,998 | 476 | 2,902 | 99 | 24 | 1,497 | 0.973 | 0.974 | 50 |
| Laguna XS.2 / Pro | 6,921 | 873 | 789 | 22 | 13 | 5,224 | 0.994 | 0.996 | 375 |
| Qwen3.6-35B-A3B / Pro | 5,804 | 913 | 842 | 42 | 17 | 3,990 | 0.988 | 0.988 | 61 |


#### E11: k=0 probes on Pro, pooled vs language-/repo-controlled (validation-best layer as in E17; pooled AUC over all test tokens; 95% task-bootstrap CI; bold if a difference's CI excludes 0)

| Model | Probe | layer | pooled AUC | same-language pairs | same-repo pairs | language-only baseline | repo-only baseline | real − language-only | real − repo-only |
|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Pro | compiles | L10 | 0.786 [0.725, 0.843] | 0.646 [0.584, 0.700] | 0.649 [0.569, 0.714] | 0.778 [0.688, 0.856] | 0.833 [0.757, 0.897] | +0.008 [-0.04, +0.06] | -0.046 [-0.09, +0.00] |
| Laguna XS.2 / Pro | correct | L20 | 0.794 [0.746, 0.839] | 0.785 [0.734, 0.830] | 0.781 [0.710, 0.833] | 0.562 [0.415, 0.697] | 0.573 [0.442, 0.695] | **+0.232 [+0.10, +0.37]** | **+0.221 [+0.11, +0.33]** |
| Laguna XS.2 / Pro | has_regressions | L20 | 0.740 [0.677, 0.808] | 0.650 [0.569, 0.736] | 0.644 [0.539, 0.719] | 0.823 [0.752, 0.881] | 0.793 [0.706, 0.889] | **-0.084 [-0.16, -0.00]** | -0.053 [-0.16, +0.03] |
| Laguna XS.2 / Pro | reduces_failing | L20 | 0.780 [0.733, 0.825] | 0.736 [0.693, 0.778] | 0.704 [0.665, 0.756] | 0.663 [0.533, 0.778] | 0.744 [0.656, 0.838] | **+0.117 [+0.03, +0.22]** | +0.036 [-0.03, +0.11] |
| Qwen3.6-35B-A3B / Pro | compiles | L39 | 0.720 [0.651, 0.788] | 0.578 [0.512, 0.635] | 0.568 [0.486, 0.632] | 0.756 [0.656, 0.854] | 0.819 [0.737, 0.891] | -0.037 [-0.10, +0.02] | **-0.100 [-0.15, -0.05]** |
| Qwen3.6-35B-A3B / Pro | correct | L30 | 0.822 [0.787, 0.861] | 0.811 [0.773, 0.855] | 0.815 [0.777, 0.857] | 0.506 [0.368, 0.667] | 0.538 [0.394, 0.708] | **+0.316 [+0.17, +0.46]** | **+0.284 [+0.12, +0.43]** |
| Qwen3.6-35B-A3B / Pro | has_regressions | L20 | 0.818 [0.774, 0.859] | 0.711 [0.648, 0.776] | 0.696 [0.621, 0.766] | 0.831 [0.777, 0.884] | 0.791 [0.702, 0.864] | -0.013 [-0.06, +0.03] | +0.027 [-0.04, +0.11] |
| Qwen3.6-35B-A3B / Pro | reduces_failing | L20 | 0.830 [0.780, 0.866] | 0.783 [0.733, 0.821] | 0.783 [0.730, 0.820] | 0.716 [0.593, 0.814] | 0.744 [0.640, 0.834] | **+0.114 [+0.02, +0.21]** | +0.086 [-0.00, +0.17] |

#### E11: compiles on Pro by language (◦ = fewer than 20 test tasks with both classes)

| Model | language | AUC | tokens | positives | positive rate | test tasks | tasks w/ both classes | Verified-trained probe AUC |
|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Pro | go | 0.641 [0.567, 0.708]◦ | 579,874 | 168,885 | 0.29 | 45 | 17 | 0.453 [0.399, 0.516] |
| Laguna XS.2 / Pro | js | 0.543 [0.412, 0.672]◦ | 165,454 | 141,234 | 0.85 | 21 | 10 | 0.451 [0.352, 0.585] |
| Laguna XS.2 / Pro | python | 0.674 [0.564, 0.766]◦ | 445,819 | 373,792 | 0.84 | 33 | 19 | 0.470 [0.380, 0.607] |
| Laguna XS.2 / Pro | ts | 0.602 [0.505, 0.693]◦ | 61,489 | 15,486 | 0.25 | 7 | 5 | 0.582 [0.418, 0.673] |
| Laguna XS.2 / Pro | **all (Verified-trained probe, L10)** | pooled 0.516 [0.476, 0.553]; same-language pairs 0.459 [0.412, 0.514] | | | | | | |
| Qwen3.6-35B-A3B / Pro | go | 0.564 [0.493, 0.645]◦ | 167,689 | 53,384 | 0.32 | 37 | 12 | 0.467 [0.408, 0.527] |
| Qwen3.6-35B-A3B / Pro | js | 0.489 [0.410, 0.580]◦ | 108,383 | 84,612 | 0.78 | 20 | 6 | 0.523 [0.418, 0.631] |
| Qwen3.6-35B-A3B / Pro | python | 0.686 [0.560, 0.733]◦ | 163,255 | 145,986 | 0.89 | 30 | 13 | 0.476 [0.429, 0.616] |
| Qwen3.6-35B-A3B / Pro | ts | 0.534 [0.463, 0.730]◦ | 28,177 | 9,346 | 0.33 | 7 | 4 | 0.507 [0.200, 0.600] |
| Qwen3.6-35B-A3B / Pro | **all (Verified-trained probe, L39)** | pooled 0.505 [0.464, 0.545]; same-language pairs 0.480 [0.444, 0.525] | | | | | | |

#### E11: compiles on Pro by repository (◦ = fewer than 20 test tasks with both classes)

| Model | repository | AUC | tokens | positive rate | test tasks | tasks w/ both classes |
|---|---|---|---|---|---|---|
| Laguna XS.2 / Pro | flipt-io/flipt | 0.688 [0.578, 0.772]◦ | 304,372 | 0.39 | 21 | 8 |
| Laguna XS.2 / Pro | ansible/ansible | 0.573 [0.529, 0.644]◦ | 240,229 | 0.90 | 16 | 10 |
| Laguna XS.2 / Pro | gravitational/teleport | 0.560 [0.387, 0.726]◦ | 146,377 | 0.08 | 12 | 2 |
| Laguna XS.2 / Pro | protonmail/webclients | 0.490 [0.353, 0.721]◦ | 69,632 | 0.87 | 11 | 4 |
| Laguna XS.2 / Pro | qutebrowser/qutebrowser | 0.614 [0.529, 0.712]◦ | 116,955 | 0.97 | 10 | 6 |
| Laguna XS.2 / Pro | internetarchive/openlibrary | 0.633 [0.491, 0.747]◦ | 88,635 | 0.51 | 7 | 3 |
| Laguna XS.2 / Pro | tutao/tutanota | 0.602 [0.505, 0.693]◦ | 61,489 | 0.25 | 7 | 5 |
| Laguna XS.2 / Pro | future-architect/vuls | 0.664 [0.394, 0.869]◦ | 63,831 | 0.30 | 6 | 4 |
| Laguna XS.2 / Pro | navidrome/navidrome | 0.596 [0.362, 0.730]◦ | 65,294 | 0.29 | 6 | 3 |
| Laguna XS.2 / Pro | NodeBB/NodeBB | 0.600 [0.541, 0.645]◦ | 43,490 | 0.99 | 5 | 4 |
| Laguna XS.2 / Pro | element-hq/element-web | 0.465 [0.285, 0.648]◦ | 52,332 | 0.72 | 5 | 2 |
| Qwen3.6-35B-A3B / Pro | flipt-io/flipt | 0.668 [0.601, 0.738]◦ | 73,946 | 0.43 | 14 | 7 |
| Qwen3.6-35B-A3B / Pro | ansible/ansible | 0.576 [0.526, 0.628]◦ | 70,351 | 0.96 | 13 | 7 |
| Qwen3.6-35B-A3B / Pro | gravitational/teleport | 0.504 [0.423, 0.639]◦ | 42,890 | 0.28 | 12 | 2 |
| Qwen3.6-35B-A3B / Pro | future-architect/vuls | 0.501 [0.435, 0.605]◦ | 40,422 | 0.24 | 9 | 3 |
| Qwen3.6-35B-A3B / Pro | internetarchive/openlibrary | 0.627 [0.517, 0.733]◦ | 56,045 | 0.75 | 9 | 2 |
| Qwen3.6-35B-A3B / Pro | protonmail/webclients | 0.420 [0.315, 0.618]◦ | 60,528 | 0.72 | 9 | 2 |
| Qwen3.6-35B-A3B / Pro | qutebrowser/qutebrowser | 0.604 [0.521, 0.716]◦ | 36,859 | 0.99 | 8 | 4 |
| Qwen3.6-35B-A3B / Pro | element-hq/element-web | 0.504 [0.348, 0.577]◦ | 32,794 | 0.80 | 7 | 2 |
| Qwen3.6-35B-A3B / Pro | tutao/tutanota | 0.534 [0.463, 0.730]◦ | 28,177 | 0.33 | 7 | 4 |
| Qwen3.6-35B-A3B / Pro | NodeBB/NodeBB | 0.584 [0.392, 0.658]◦ | 15,061 | 1.00 | 4 | 2 |
| Qwen3.6-35B-A3B / Pro | navidrome/navidrome | –◦ | 10,431 | 0.00 | 2 | 0 |

#### E11: correct on Pro by language (◦ = fewer than 20 test tasks with both classes)

| Model | language | AUC | tokens | positives | positive rate | test tasks | tasks w/ both classes |
|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Pro | go | 0.792 [0.720, 0.850]◦ | 579,874 | 26,093 | 0.04 | 45 | 10 |
| Laguna XS.2 / Pro | js | 0.787 [0.715, 0.932]◦ | 165,454 | 3,192 | 0.02 | 21 | 4 |
| Laguna XS.2 / Pro | python | 0.779 [0.689, 0.853]◦ | 445,819 | 33,211 | 0.07 | 33 | 10 |
| Laguna XS.2 / Pro | ts | 0.737 [0.601, 0.844]◦ | 61,489 | 3,434 | 0.06 | 7 | 4 |
| Qwen3.6-35B-A3B / Pro | go | 0.841 [0.786, 0.906]◦ | 167,689 | 11,152 | 0.07 | 37 | 8 |
| Qwen3.6-35B-A3B / Pro | js | 0.849 [0.777, 0.930]◦ | 108,383 | 3,213 | 0.03 | 20 | 4 |
| Qwen3.6-35B-A3B / Pro | python | 0.772 [0.725, 0.832]◦ | 163,255 | 11,548 | 0.07 | 30 | 13 |
| Qwen3.6-35B-A3B / Pro | ts | 0.882 [0.840, 0.934]◦ | 28,177 | 1,223 | 0.04 | 7 | 2 |

#### E11: has_regressions on Pro by language (◦ = fewer than 20 test tasks with both classes)

| Model | language | AUC | tokens | positives | positive rate | test tasks | tasks w/ both classes |
|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Pro | go | 0.821 [0.786, 0.851]◦ | 579,874 | 89 | 0.00 | 45 | 1 |
| Laguna XS.2 / Pro | js | 0.740 [0.649, 0.827]◦ | 165,454 | 4,746 | 0.03 | 21 | 2 |
| Laguna XS.2 / Pro | python | 0.647 [0.560, 0.736]◦ | 445,819 | 65,629 | 0.15 | 33 | 12 |
| Laguna XS.2 / Pro | ts | –◦ | 61,489 | 0 | 0.00 | 7 | 0 |
| Qwen3.6-35B-A3B / Pro | go | –◦ | 167,689 | 0 | 0.00 | 37 | 0 |
| Qwen3.6-35B-A3B / Pro | js | –◦ | 108,383 | 0 | 0.00 | 20 | 0 |
| Qwen3.6-35B-A3B / Pro | python | 0.711 [0.648, 0.776]◦ | 163,255 | 7,805 | 0.05 | 30 | 9 |
| Qwen3.6-35B-A3B / Pro | ts | –◦ | 28,177 | 0 | 0.00 | 7 | 0 |

#### E11: reduces_failing on Pro by language (◦ = fewer than 20 test tasks with both classes)

| Model | language | AUC | tokens | positives | positive rate | test tasks | tasks w/ both classes |
|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Pro | go | 0.724 [0.533, 0.836]◦ | 579,874 | 39,677 | 0.07 | 45 | 6 |
| Laguna XS.2 / Pro | js | 0.855 [0.718, 0.940]◦ | 165,454 | 12,362 | 0.07 | 21 | 6 |
| Laguna XS.2 / Pro | python | 0.738 [0.693, 0.780] | 445,819 | 102,457 | 0.23 | 33 | 22 |
| Laguna XS.2 / Pro | ts | 0.568 [0.505, 0.772]◦ | 61,489 | 5,217 | 0.08 | 7 | 1 |
| Qwen3.6-35B-A3B / Pro | go | 0.776 [0.703, 0.848]◦ | 167,689 | 6,132 | 0.04 | 37 | 5 |
| Qwen3.6-35B-A3B / Pro | js | 0.855 [0.779, 0.911]◦ | 108,383 | 5,117 | 0.05 | 20 | 6 |
| Qwen3.6-35B-A3B / Pro | python | 0.775 [0.711, 0.822] | 163,255 | 34,977 | 0.21 | 30 | 20 |
| Qwen3.6-35B-A3B / Pro | ts | 0.839 [0.812, 0.879]◦ | 28,177 | 1,514 | 0.05 | 7 | 1 |

#### E11b: k=0 probes on Verified, pooled vs repository-controlled, and the submitted runs' repository-split AUC (task split: validation-best layer as in E17, pooled AUC over all test tokens, 95% task-bootstrap CI; legacy: stored pooled test AUC of the submitted runs on their test repositories astropy, django, sphinx)

| Model | Probe | layer | pooled AUC | same-repo pairs | repo-only baseline | real − repo-only | pooled − same-repo | legacy split, test-best layer | legacy split, val-best layer |
|---|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | compiles | L10 | 0.691 [0.623, 0.783] | 0.709 [0.622, 0.779] | 0.692 [0.534, 0.833] | -0.001 [-0.14, +0.16] | -0.018 [-0.09, +0.07] | 0.571 (L30) | 0.566 (L20) |
| Laguna XS.2 / Verified | correct | L20 | 0.788 [0.740, 0.838] | 0.752 [0.688, 0.824] | 0.574 [0.479, 0.665] | **+0.215 [+0.12, +0.32]** | +0.036 [-0.01, +0.07] | 0.727 (L20) | 0.727 (L20) |
| Laguna XS.2 / Verified | has_regressions | L20 | 0.695 [0.629, 0.758] | 0.694 [0.630, 0.764] | 0.507 [0.330, 0.682] | **+0.188 [+0.05, +0.33]** | +0.002 [-0.06, +0.05] | 0.678 (L20) | 0.678 (L20) |
| Laguna XS.2 / Verified | reduces_failing | L20 | 0.773 [0.721, 0.826] | 0.765 [0.706, 0.823] | 0.450 [0.368, 0.539] | **+0.323 [+0.22, +0.43]** | +0.008 [-0.04, +0.05] | 0.708 (L20) | 0.708 (L20) |
| Qwen3.6-35B-A3B / Verified | compiles | L39 | 0.736 [0.666, 0.809] | 0.733 [0.653, 0.808] | 0.785 [0.595, 0.913] | -0.050 [-0.19, +0.14] | +0.003 [-0.06, +0.09] | 0.587 (L30) | 0.553 (L20) |
| Qwen3.6-35B-A3B / Verified | correct | L30 | 0.838 [0.795, 0.879] | 0.833 [0.787, 0.883] | 0.565 [0.482, 0.643] | **+0.273 [+0.19, +0.37]** | +0.005 [-0.03, +0.04] | 0.828 (L30) | 0.820 (L20) |
| Qwen3.6-35B-A3B / Verified | has_regressions | L30 | 0.770 [0.726, 0.809] | 0.759 [0.678, 0.806] | 0.521 [0.358, 0.767] | **+0.249 [+0.01, +0.41]** | +0.010 [-0.02, +0.08] | 0.721 (L30) | 0.594 (L10) |
| Qwen3.6-35B-A3B / Verified | reduces_failing | L30 | 0.832 [0.790, 0.870] | 0.830 [0.781, 0.879] | 0.448 [0.380, 0.531] | **+0.384 [+0.31, +0.46]** | +0.002 [-0.03, +0.03] | 0.808 (L30) | 0.807 (L20) |

#### E11b: compiles on Verified by repository (◦ = fewer than 20 test tasks with both classes)

| Model | repository | AUC | tokens | positives | positive rate | train positive rate | test tasks | tasks w/ both classes |
|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | django/django | 0.720 [0.650, 0.797]◦ | 545,474 | 494,087 | 0.91 | 0.96 | 34 | 8 |
| Laguna XS.2 / Verified | sympy/sympy | –◦ | 130,747 | 130,747 | 1.00 | 0.99 | 9 | 0 |
| Laguna XS.2 / Verified | scikit-learn/scikit-learn | 0.565 [0.512, 0.701]◦ | 93,789 | 91,504 | 0.98 | 0.98 | 6 | 3 |
| Laguna XS.2 / Verified | sphinx-doc/sphinx | 0.453 [0.402, 0.559]◦ | 94,724 | 94,249 | 0.99 | 0.90 | 6 | 1 |
| Laguna XS.2 / Verified | astropy/astropy | 0.586 [0.473, 0.725]◦ | 87,976 | 59,963 | 0.68 | 0.96 | 5 | 3 |
| Laguna XS.2 / Verified | pydata/xarray | 0.611 [0.494, 0.767]◦ | 62,974 | 55,703 | 0.88 | 0.95 | 5 | 2 |
| Laguna XS.2 / Verified | matplotlib/matplotlib | 0.439 [0.368, 0.636]◦ | 41,319 | 41,287 | 1.00 | 0.97 | 4 | 1 |
| Laguna XS.2 / Verified | pylint-dev/pylint | 0.724 [0.713, 0.807]◦ | 23,473 | 12,333 | 0.53 | 0.41 | 2 | 2 |
| Laguna XS.2 / Verified | pytest-dev/pytest | 0.674 [0.639, 0.756]◦ | 23,097 | 11,164 | 0.48 | 0.74 | 2 | 1 |
| Laguna XS.2 / Verified | mwaskom/seaborn | 0.634 [0.634, 0.634]◦ | 17,034 | 16,099 | 0.95 | 1.00 | 1 | 1 |
| Laguna XS.2 / Verified | psf/requests | 0.477 [0.477, 0.477]◦ | 10,284 | 10,269 | 1.00 | 0.87 | 1 | 1 |
| Qwen3.6-35B-A3B / Verified | django/django | 0.749 [0.659, 0.840]◦ | 273,151 | 258,427 | 0.95 | 0.97 | 34 | 8 |
| Qwen3.6-35B-A3B / Verified | sympy/sympy | 0.815 [0.777, 0.841]◦ | 79,054 | 79,029 | 1.00 | 1.00 | 9 | 1 |
| Qwen3.6-35B-A3B / Verified | scikit-learn/scikit-learn | 0.696 [0.607, 0.747]◦ | 32,755 | 32,237 | 0.98 | 0.99 | 6 | 3 |
| Qwen3.6-35B-A3B / Verified | sphinx-doc/sphinx | 0.617 [0.585, 0.651]◦ | 39,568 | 39,346 | 0.99 | 0.94 | 6 | 1 |
| Qwen3.6-35B-A3B / Verified | astropy/astropy | 0.637 [0.607, 0.768]◦ | 49,438 | 38,068 | 0.77 | 0.93 | 5 | 3 |
| Qwen3.6-35B-A3B / Verified | pydata/xarray | 0.611 [0.589, 0.833]◦ | 32,909 | 28,802 | 0.88 | 0.95 | 5 | 4 |
| Qwen3.6-35B-A3B / Verified | matplotlib/matplotlib | 0.686 [0.596, 0.768]◦ | 20,532 | 20,263 | 0.99 | 0.98 | 4 | 3 |
| Qwen3.6-35B-A3B / Verified | pylint-dev/pylint | 0.757 [0.755, 0.779]◦ | 14,364 | 8,802 | 0.61 | 0.64 | 2 | 2 |
| Qwen3.6-35B-A3B / Verified | pytest-dev/pytest | 0.695 [0.678, 0.708]◦ | 20,903 | 9,292 | 0.44 | 0.81 | 2 | 1 |
| Qwen3.6-35B-A3B / Verified | mwaskom/seaborn | –◦ | 1,306 | 1,306 | 1.00 | 1.00 | 1 | 0 |
| Qwen3.6-35B-A3B / Verified | psf/requests | –◦ | 6,707 | 6,707 | 1.00 | 0.74 | 1 | 0 |

#### E11b: correct on Verified by repository (◦ = fewer than 20 test tasks with both classes)

| Model | repository | AUC | tokens | positives | positive rate | train positive rate | test tasks | tasks w/ both classes |
|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | django/django | 0.742 [0.669, 0.822] | 545,474 | 234,398 | 0.43 | 0.40 | 34 | 29 |
| Laguna XS.2 / Verified | sympy/sympy | 0.852 [0.665, 0.959]◦ | 130,747 | 44,017 | 0.34 | 0.28 | 9 | 7 |
| Laguna XS.2 / Verified | scikit-learn/scikit-learn | 0.781 [0.644, 0.959]◦ | 93,789 | 33,394 | 0.36 | 0.36 | 6 | 6 |
| Laguna XS.2 / Verified | sphinx-doc/sphinx | 0.780 [0.618, 0.866]◦ | 94,724 | 8,895 | 0.09 | 0.34 | 6 | 3 |
| Laguna XS.2 / Verified | astropy/astropy | 0.867 [0.792, 0.971]◦ | 87,976 | 31,144 | 0.35 | 0.20 | 5 | 4 |
| Laguna XS.2 / Verified | pydata/xarray | 0.845 [0.786, 0.959]◦ | 62,974 | 24,636 | 0.39 | 0.36 | 5 | 3 |
| Laguna XS.2 / Verified | matplotlib/matplotlib | 0.768 [0.695, 0.990]◦ | 41,319 | 16,937 | 0.41 | 0.23 | 4 | 3 |
| Laguna XS.2 / Verified | pylint-dev/pylint | –◦ | 23,473 | 0 | 0.00 | 0.06 | 2 | 0 |
| Laguna XS.2 / Verified | pytest-dev/pytest | 0.796 [0.783, 0.933]◦ | 23,097 | 3,598 | 0.16 | 0.42 | 2 | 1 |
| Laguna XS.2 / Verified | mwaskom/seaborn | 0.638 [0.638, 0.638]◦ | 17,034 | 98 | 0.01 | 0.05 | 1 | 1 |
| Laguna XS.2 / Verified | psf/requests | 0.831 [0.831, 0.831]◦ | 10,284 | 2,879 | 0.28 | 0.13 | 1 | 1 |
| Qwen3.6-35B-A3B / Verified | django/django | 0.829 [0.773, 0.884] | 273,151 | 91,301 | 0.33 | 0.29 | 34 | 26 |
| Qwen3.6-35B-A3B / Verified | sympy/sympy | 0.837 [0.707, 0.961]◦ | 79,054 | 25,545 | 0.32 | 0.23 | 9 | 7 |
| Qwen3.6-35B-A3B / Verified | scikit-learn/scikit-learn | 0.907 [0.789, 0.992]◦ | 32,755 | 7,773 | 0.24 | 0.34 | 6 | 6 |
| Qwen3.6-35B-A3B / Verified | sphinx-doc/sphinx | 0.902 [0.784, 0.963]◦ | 39,568 | 7,456 | 0.19 | 0.24 | 6 | 3 |
| Qwen3.6-35B-A3B / Verified | astropy/astropy | 0.913 [0.865, 0.991]◦ | 49,438 | 8,817 | 0.18 | 0.15 | 5 | 4 |
| Qwen3.6-35B-A3B / Verified | pydata/xarray | 0.900 [0.844, 0.990]◦ | 32,909 | 7,451 | 0.23 | 0.28 | 5 | 3 |
| Qwen3.6-35B-A3B / Verified | matplotlib/matplotlib | 0.781 [0.711, 0.976]◦ | 20,532 | 6,199 | 0.30 | 0.18 | 4 | 3 |
| Qwen3.6-35B-A3B / Verified | pylint-dev/pylint | –◦ | 14,364 | 0 | 0.00 | 0.03 | 2 | 0 |
| Qwen3.6-35B-A3B / Verified | pytest-dev/pytest | 0.917 [0.910, 0.993]◦ | 20,903 | 873 | 0.04 | 0.27 | 2 | 1 |
| Qwen3.6-35B-A3B / Verified | mwaskom/seaborn | –◦ | 1,306 | 0 | 0.00 | 0.00 | 1 | 0 |
| Qwen3.6-35B-A3B / Verified | psf/requests | 0.819 [0.819, 0.819]◦ | 6,707 | 619 | 0.09 | 0.11 | 1 | 1 |

#### E11b: has_regressions on Verified by repository (◦ = fewer than 20 test tasks with both classes)

| Model | repository | AUC | tokens | positives | positive rate | train positive rate | test tasks | tasks w/ both classes |
|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | django/django | 0.692 [0.618, 0.775]◦ | 545,474 | 64,561 | 0.12 | 0.05 | 34 | 13 |
| Laguna XS.2 / Verified | sympy/sympy | –◦ | 130,747 | 0 | 0.00 | 0.09 | 9 | 0 |
| Laguna XS.2 / Verified | scikit-learn/scikit-learn | 0.714 [0.606, 0.848]◦ | 93,789 | 25,219 | 0.27 | 0.15 | 6 | 2 |
| Laguna XS.2 / Verified | sphinx-doc/sphinx | 0.628 [0.438, 0.817]◦ | 94,724 | 5,756 | 0.06 | 0.06 | 6 | 2 |
| Laguna XS.2 / Verified | astropy/astropy | 0.621 [0.550, 0.759]◦ | 87,976 | 3,111 | 0.04 | 0.22 | 5 | 1 |
| Laguna XS.2 / Verified | pydata/xarray | 0.744 [0.498, 0.764]◦ | 62,974 | 2,701 | 0.04 | 0.18 | 5 | 5 |
| Laguna XS.2 / Verified | matplotlib/matplotlib | 0.794 [0.307, 0.866]◦ | 41,319 | 15,373 | 0.37 | 0.18 | 4 | 3 |
| Laguna XS.2 / Verified | pylint-dev/pylint | –◦ | 23,473 | 0 | 0.00 | 0.00 | 2 | 0 |
| Laguna XS.2 / Verified | pytest-dev/pytest | 0.697 [0.658, 0.884]◦ | 23,097 | 925 | 0.04 | 0.11 | 2 | 2 |
| Laguna XS.2 / Verified | mwaskom/seaborn | 0.717 [0.717, 0.717]◦ | 17,034 | 4,181 | 0.25 | 0.00 | 1 | 1 |
| Laguna XS.2 / Verified | psf/requests | 0.622 [0.622, 0.622]◦ | 10,284 | 3,858 | 0.38 | 0.39 | 1 | 1 |
| Qwen3.6-35B-A3B / Verified | django/django | 0.757 [0.640, 0.808]◦ | 273,151 | 21,551 | 0.08 | 0.03 | 34 | 11 |
| Qwen3.6-35B-A3B / Verified | sympy/sympy | 0.811 [0.783, 0.845]◦ | 79,054 | 228 | 0.00 | 0.06 | 9 | 1 |
| Qwen3.6-35B-A3B / Verified | scikit-learn/scikit-learn | 0.890 [0.822, 0.909]◦ | 32,755 | 5,399 | 0.16 | 0.06 | 6 | 2 |
| Qwen3.6-35B-A3B / Verified | sphinx-doc/sphinx | 0.777 [0.694, 0.803]◦ | 39,568 | 689 | 0.02 | 0.05 | 6 | 1 |
| Qwen3.6-35B-A3B / Verified | astropy/astropy | 0.715 [0.617, 0.869]◦ | 49,438 | 3,057 | 0.06 | 0.14 | 5 | 3 |
| Qwen3.6-35B-A3B / Verified | pydata/xarray | 0.817 [0.741, 0.852]◦ | 32,909 | 1,276 | 0.04 | 0.12 | 5 | 1 |
| Qwen3.6-35B-A3B / Verified | matplotlib/matplotlib | 0.709 [0.495, 0.731]◦ | 20,532 | 1,879 | 0.09 | 0.13 | 4 | 2 |
| Qwen3.6-35B-A3B / Verified | pylint-dev/pylint | –◦ | 14,364 | 0 | 0.00 | 0.00 | 2 | 0 |
| Qwen3.6-35B-A3B / Verified | pytest-dev/pytest | –◦ | 20,903 | 0 | 0.00 | 0.07 | 2 | 0 |
| Qwen3.6-35B-A3B / Verified | mwaskom/seaborn | 0.664 [0.664, 0.664]◦ | 1,306 | 107 | 0.08 | 0.06 | 1 | 1 |
| Qwen3.6-35B-A3B / Verified | psf/requests | 0.697 [0.697, 0.697]◦ | 6,707 | 3,858 | 0.58 | 0.36 | 1 | 1 |

#### E11b: reduces_failing on Verified by repository (◦ = fewer than 20 test tasks with both classes)

| Model | repository | AUC | tokens | positives | positive rate | train positive rate | test tasks | tasks w/ both classes |
|---|---|---|---|---|---|---|---|---|
| Laguna XS.2 / Verified | django/django | 0.758 [0.691, 0.822] | 545,474 | 235,376 | 0.43 | 0.39 | 34 | 25 |
| Laguna XS.2 / Verified | sympy/sympy | 0.831 [0.657, 0.954]◦ | 130,747 | 41,886 | 0.32 | 0.39 | 9 | 8 |
| Laguna XS.2 / Verified | scikit-learn/scikit-learn | 0.750 [0.562, 0.961]◦ | 93,789 | 34,540 | 0.37 | 0.39 | 6 | 6 |
| Laguna XS.2 / Verified | sphinx-doc/sphinx | 0.796 [0.730, 0.936]◦ | 94,724 | 30,658 | 0.32 | 0.44 | 6 | 5 |
| Laguna XS.2 / Verified | astropy/astropy | 0.864 [0.791, 0.963]◦ | 87,976 | 59,300 | 0.67 | 0.26 | 5 | 5 |
| Laguna XS.2 / Verified | pydata/xarray | 0.876 [0.765, 0.967]◦ | 62,974 | 24,915 | 0.40 | 0.42 | 5 | 4 |
| Laguna XS.2 / Verified | matplotlib/matplotlib | 0.714 [0.660, 0.990]◦ | 41,319 | 16,969 | 0.41 | 0.28 | 4 | 3 |
| Laguna XS.2 / Verified | pylint-dev/pylint | 0.585 [0.566, 0.617]◦ | 23,473 | 369 | 0.02 | 0.27 | 2 | 1 |
| Laguna XS.2 / Verified | pytest-dev/pytest | 0.766 [0.751, 0.932]◦ | 23,097 | 3,732 | 0.16 | 0.49 | 2 | 1 |
| Laguna XS.2 / Verified | mwaskom/seaborn | 0.670 [0.670, 0.670]◦ | 17,034 | 3,494 | 0.21 | 0.24 | 1 | 1 |
| Laguna XS.2 / Verified | psf/requests | 0.848 [0.848, 0.848]◦ | 10,284 | 2,894 | 0.28 | 0.29 | 1 | 1 |
| Qwen3.6-35B-A3B / Verified | django/django | 0.822 [0.762, 0.876] | 273,151 | 98,764 | 0.36 | 0.29 | 34 | 25 |
| Qwen3.6-35B-A3B / Verified | sympy/sympy | 0.891 [0.804, 0.972]◦ | 79,054 | 19,488 | 0.25 | 0.33 | 9 | 9 |
| Qwen3.6-35B-A3B / Verified | scikit-learn/scikit-learn | 0.896 [0.794, 0.988]◦ | 32,755 | 8,291 | 0.25 | 0.35 | 6 | 6 |
| Qwen3.6-35B-A3B / Verified | sphinx-doc/sphinx | 0.922 [0.852, 0.963]◦ | 39,568 | 10,884 | 0.28 | 0.31 | 6 | 5 |
| Qwen3.6-35B-A3B / Verified | astropy/astropy | 0.892 [0.820, 0.992]◦ | 49,438 | 20,187 | 0.41 | 0.22 | 5 | 5 |
| Qwen3.6-35B-A3B / Verified | pydata/xarray | 0.869 [0.805, 0.987]◦ | 32,909 | 7,675 | 0.23 | 0.31 | 5 | 4 |
| Qwen3.6-35B-A3B / Verified | matplotlib/matplotlib | 0.729 [0.372, 0.979]◦ | 20,532 | 6,468 | 0.32 | 0.21 | 4 | 4 |
| Qwen3.6-35B-A3B / Verified | pylint-dev/pylint | –◦ | 14,364 | 0 | 0.00 | 0.11 | 2 | 0 |
| Qwen3.6-35B-A3B / Verified | pytest-dev/pytest | 0.881 [0.564, 0.993]◦ | 20,903 | 888 | 0.04 | 0.36 | 2 | 2 |
| Qwen3.6-35B-A3B / Verified | mwaskom/seaborn | 0.763 [0.763, 0.763]◦ | 1,306 | 138 | 0.11 | 0.26 | 1 | 1 |
| Qwen3.6-35B-A3B / Verified | psf/requests | 0.836 [0.836, 0.836]◦ | 6,707 | 619 | 0.09 | 0.14 | 1 | 1 |
