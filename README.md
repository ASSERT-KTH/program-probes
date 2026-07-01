# program-probes

Measures whether a language model's internal hidden states linearly predict properties of its own agentic output before those properties are realised.

The experiment runs a coding agent (mini-SWE-agent) on SWE-bench, records hidden states at every assistant turn, and trains linear probes to predict per-edit properties such as *does the code currently compile?* and *are all tests currently passing?*

## Clone

This repo vendors `SWE-bench_Pro-os` as a git submodule, so clone recursively:

```bash
git clone --recurse-submodules https://github.com/ASSERT-KTH/program-probes
# already cloned without --recurse-submodules?
git submodule update --init --recursive
```

## Data

The labeled agent trajectories used in the paper are released on the Hugging Face Hub:

- **[ASSERT-KTH/latent-programming-horizons-trajs](https://huggingface.co/datasets/ASSERT-KTH/latent-programming-horizons-trajs)**

These are the outputs of steps 1–2 of the [pipeline](#pipeline) (generation + labeling). Downloading them lets you skip the GPU generation step and start from hidden-state extraction (step 3) onward. Point `--traj-dir` / `--label-dir` at the downloaded trajectories in the commands below.

## Pipeline

```
slurm/swebench_run_thin.sh       run agent on SWE-bench instances (GPU)
                                 → generations/swebench/<run_id>/

slurm/swebench_label.sh          replay each edit in a Modal sandbox, compute
                                 per-edit labels (compiles, test pass/fail)
                                 → generations/swebench/<run_id>/labels/

slurm/extract_swebench_thin.sh   tokenise trajectories, extract hidden states
                                 at all probe layers (GPU, array job)
                                 → outputs/swebench/<run_id>/*.pt

run_attach_labels_swebench.py    attach per-step probe labels to activation files
                                 (CPU — re-run freely if label logic changes)
                                 → outputs/swebench/<run_id>/*_labels.pt

slurm/build_cache.sh             aggregate per-trajectory .pt files into
                                 per-(probe, layer) tensors (CPU)
                                 → cache/swebench/<run_id>/<probe>/layer_N.pt

slurm/probe_sweep_coordinator.sh  per-(probe, layer): create W&B sweep, launch N
                                  parallel workers, auto-submit final training
                                  → results/swebench/<run_id>/<probe>/results.pt

slurm/figures.sh                 AUC heatmaps, layer plots, lookahead horizon,
                                 barplots, LaTeX tables
                                 → paper/figures/  +  paper/*.tex
```

## Setup

### Berzelius

```bash
source scripts/berzelius_env.sh --sync
```

This loads `buildenv-gcccuda/12.4.1-gcc13.3.0` and syncs the virtualenv with Python 3.12.

```bash
# Version/module check without installing:
source scripts/berzelius_env.sh --check-only

# Use a different CUDA/GCC module:
export BERZELIUS_MODULES="buildenv-gcccuda/12.1.1-gcc12.3.0"
source scripts/berzelius_env.sh --sync
```

### Other environments

```bash
uv sync --frozen
```

### W&B

The sweep coordinator and probe training use W&B for experiment tracking. Set your API key:

```bash
export WANDB_API_KEY="..."
```

### Modal (labeling)

The trajectory labeler replays edit steps in remote Modal sandboxes. Authenticate once:

```bash
uv run modal token new
```

For non-interactive jobs:

```bash
export MODAL_TOKEN_ID="..."
export MODAL_TOKEN_SECRET="..."
```

### SSL certificates (Berzelius)

Add to `~/.bashrc` to fix TLS errors on RHEL 8:

```bash
export SSL_CERT_FILE=/etc/pki/tls/cert.pem
```

## Running the experiment (Berzelius SLURM)

The commands below reproduce the SWE-bench Verified experiment for Laguna-XS2. To run on a different model, swap `laguna_xs2` for your model name and point to the corresponding configs. For SWE-bench Pro, change `swebench` to `swebench_pro` in all paths.

### 1. Generate trajectories

```bash
sbatch --array=0-3 slurm/swebench_run_thin.sh \
  --run-config configs/runs/laguna_xs2_swebench_full.yaml
```

Output: `generations/swebench/laguna_xs2_full/`

### 2. Label trajectories

```bash
sbatch slurm/swebench_label.sh \
  --config configs/labeling/swebench_labeler.yaml
```

Each edit step is replayed in a Modal sandbox: git reset to clean HEAD, apply the cumulative diff, infer `compiles` from the pytest log, run the SWE-bench eval script.
Output: `generations/swebench/laguna_xs2_full/labels/`

### 3. Extract hidden states

```bash
sbatch --array=0-7 slurm/extract_swebench_thin.sh \
  --model-config configs/models/laguna_xs2.yaml \
  --generation-config configs/generation_laguna_xs2.yaml \
  --traj-dir generations/swebench/laguna_xs2_full \
  --output-dir outputs/swebench/laguna_xs2_full
```

Extracts hidden states at all layers listed in `model_config.probe_layers` (currently `[0, 10, 20, 30, 39]`). Uses chunked KV-cache inference (`--chunk-size 8192`) to handle very long trajectories without OOM.
Output: `outputs/swebench/laguna_xs2_full/<instance_id>.pt`

### 4. Attach labels

```bash
sbatch slurm/attach_labels_swebench.sh \
  --input-dir outputs/swebench/laguna_xs2_full \
  --traj-dir generations/swebench/laguna_xs2_full \
  --label-dir generations/swebench/laguna_xs2_full/labels \
  --probe currently_compiles currently_correct currently_has_regressions currently_reduces_failing \
  --generation-config configs/generation_laguna_xs2.yaml
```

Writes `<instance_id>_labels.pt` alongside each activation file. CPU-only — safe to re-run if label logic changes.

### 5. Build cache

```bash
sbatch slurm/build_cache.sh \
  --run-id laguna_xs2_full \
  --probe currently_compiles currently_correct currently_has_regressions currently_reduces_failing \
  --output-dir outputs/swebench \
  --cache-dir cache/swebench
```

Aggregates all per-trajectory files into `cache/swebench/laguna_xs2_full/<probe>/layer_N.pt` — one tensor per (probe, layer) covering the full train/val/test split.

### 6. Probe sweep + final training

One coordinator job per (probe, layer). Each coordinator creates a W&B sweep, launches `--n-agents` parallel workers that each run `--count / --n-agents` trials, then automatically submits final training once all workers complete.

```bash
for probe in currently_compiles currently_correct currently_has_regressions currently_reduces_failing; do
  for layer in 0 10 20 30 39; do
    sbatch slurm/probe_sweep_coordinator.sh \
      --model-config configs/models/laguna_xs2.yaml \
      --layer $layer \
      --probe $probe \
      --probe-arch linear \
      --run-id laguna_xs2_full_pooled \
      --cache-dir cache/swebench \
      --cache-run-id laguna_xs2_full \
      --results-dir results/swebench \
      --n-bins 1 \
      --n-agents 4 \
      --count 20
  done
done
```

This fans out to 20 coordinator jobs (4 probes × 5 layers), each spawning 4 parallel sweep workers and 1 dependent final training job. Results accumulate into `results/swebench/laguna_xs2_full_pooled/<probe>/results.pt` via merge-on-save, so layers can complete in any order.

Each W&B sweep and its runs are grouped under `<run_id>/<probe>/layer_<N>` and tagged with `layer_<N>` for easy filtering.

### 7. Paper figures

```bash
sbatch slurm/figures.sh \
  --results-dir results/swebench \
  --probes currently_compiles currently_correct currently_reduces_failing currently_has_regressions \
  --model-run-ids laguna_xs2_full_pooled qwen36_35b_a3b_full_pooled \
  --shuffled-run-ids laguna_xs2_full_pooled_shuffled qwen36_35b_a3b_full_pooled_shuffled
```

Generates all figures under `paper/figures/` and LaTeX tables under `paper/`. Also writes `paper/figures/manifest.json` for the dashboard gallery.

## Tests

```bash
uv run pytest
```

All tests run without GPU, network access, or real model downloads. Complete in under 30 seconds.

## Probes

A probe asks: *does the model's hidden state at a given point in generation linearly encode a specific property of its eventual output?*

### Implemented probes

| Probe | Type | Label |
|---|---|---|
| `currently_compiles` | dynamic | At each edit, do all changed `.py` files compile? |
| `currently_correct` | dynamic | At each edit, do all evaluation tests pass? |
| `currently_reduces_failing` | dynamic | Did the number of failing tests decrease vs the previous edit? |
| `currently_has_regressions` | dynamic | Did any previously-passing test start failing? |
| `will_resolve` | static | Does the final patch resolve the issue? |

Dynamic probes require an `edit_history` with per-edit `test_results`. The carry-forward step in `run_attach_labels_swebench.py` expands one label per edit into one label per stride step; tokens before the first edit are excluded from training.

### SWE-bench label format

Each `_labels.json` produced by the labeler contains:

```json
{
  "instance_id": "astropy__astropy-12907",
  "edits": [
    {"cmd_idx": -1, "compiles": true, "test_results": {"resolved": false, ...}},
    {"cmd_idx": 9,  "compiles": true, "test_results": {"resolved": true,  ...}}
  ]
}
```

`cmd_idx = -1` is the clean-checkout baseline used as the comparison point for delta probes.

### Adding a probe

1. Create `src/probes/myprobe.py` subclassing `ProbeAdapter`.
2. Set `name`, `is_dynamic`, and implement `compute_label(ctx)`.
3. Register the name in `src/extract._load_probe`.
4. Pass `--probe myprobe` to any entrypoint.

## Adding a model

1. Create `src/models/mymodel.py` subclassing `ModelAdapter`.
2. Implement `load`, `get_layer_modules`, `get_hidden_dim`, `tokenize`, `generate`.
3. Add `configs/models/mymodel.yaml` with `model_id`, `probe_layers`, and `adapter`.
4. Register the adapter name in `src/extract._load_model_adapter`.

The `probe_layers` list in the model config controls which transformer layers are extracted and probed. The sweep targets `probe_layers[len(probe_layers)//2]` (middle layer) when no `--layer` override is given.
