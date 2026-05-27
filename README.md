# program-probes

Measures whether a language model's internal hidden states linearly predict properties of its own output before those properties are realised.

Pipeline: inference + activation hooks → per-sample `.pt` files → per-(layer, probe) cache tensors → linear probe training with W&B sweep → accuracy heatmaps → static dashboard.

## Pipeline overview

There are two pipeline variants depending on whether the generations come from a standard completion model or an agentic loop.

### Standard pipeline (completion model)

```
run_generate.py          generate token sequences → generations/<run_id>_shard*.json
        │
        ▼
src/extract.py           load shards, extract hidden states, compute probe labels
run_extract.py           → outputs/<run_id>/<sample_id>_gen<n>.pt
        │
        ▼
run_build_cache.py       concatenate .pt files into per-(layer, probe) tensors
                         → cache/<run_id>/<probe>_layer<n>.pt
        │
        ▼
run_probe.py sweep       W&B hyperparameter sweep → pick best lr/wd/batch-size/loss
run_probe.py final       train linear probe on best hparams → W&B run with metrics
                          (or pass --from-sweep to auto-load best HPs from a sweep)
        │
        ▼
run_figures.py           accuracy heatmaps per layer × relative-position bin
run_export_dashboard.py  → dashboard/index.html  (open in browser)
```

Probe labels for this path are **static** (`will_be_correct`) or computed by the task adapter at label time. No per-edit replay is needed.

### Agentic pipeline (SWE-bench / mini-SWE-agent)

```
mini-SWE-agent           run agent on SWE-bench instances inside Modal sandboxes
run_swebench_*.py        → generations/swebench/<run_id>/<instance_id>.json
        │                  (trajectory JSON: messages, bash history, diffs, outcome)
        ▼
run_labeler.py           replay each edit step in a fresh Modal sandbox:
swebench_labeler.py        git checkout HEAD, apply cumulative diff, infer compiles
                           from the pytest eval log, run SWE-bench eval script
                           → test_results {passed/failed/resolved}
                         → generations/swebench/<run_id>/<instance_id>_labels.json
        │
        │  trajectories where any edit has test_results=None (patch apply failure
        │  or unparseable eval output) are discarded with a logged warning
        ▼
run_extract_swebench.py  tokenise full conversation, build extraction mask over
                         assistant turns, extract hidden states only (GPU, expensive)
                         → outputs/swebench/<run_id>/<instance_id>.pt  (activations only)
        │
        ▼
run_attach_labels_swebench.py  load activations + trajectory + _labels.json,
                               compute probe labels via carry-forward (CPU, fast)
                               Labels take effect from turn N+1 after the command at
                               turn N (off-by-one: the edit at step N is not visible
                               to the model until the next assistant turn).
                               → outputs/swebench/<run_id>_labeled/<instance_id>.pt
        │
        ▼
run_build_cache.py  →  run_probe.py  →  run_figures.py   (same as standard path)
```

Dynamic probes (`currently_correct`, `currently_compiles`, `currently_reduces_failing`, `currently_has_regressions`) only make sense in the agentic path because they require `EditEvent` history with per-edit `test_results`. The carry-forward step in `run_attach_labels_swebench.py` expands one label per edit into one label per stride step; steps before the first edit are masked (`None` → `-1` → excluded from training).

## Setup

### Berzelius

Use the repo bootstrap script so the CUDA/GCC build module is loaded before
`uv sync` installs GPU packages such as vLLM:

```bash
source scripts/berzelius_env.sh --sync
```

For Berzelius users, by default this loads `buildenv-gcccuda/12.4.1-gcc13.3.0` and syncs the repo
with Python 3.12. To use a different Berzelius module:

```bash
export BERZELIUS_MODULES="buildenv-gcccuda/12.1.1-gcc12.3.0"
source scripts/berzelius_env.sh --sync
```

For a version/module check without installing packages:

```bash
source scripts/berzelius_env.sh --check-only
```

### Other Environments

```bash
uv sync --frozen
```

### Modal authentication

Modal sandboxes run commands in remote containers, not in the local Python
virtual environment. Authenticate once with the Modal CLI:

```bash
uv run modal token new
```

Verify the active token with:

```bash
uv run modal token info
```

For non-interactive jobs, provide credentials through environment variables:

```bash
export MODAL_TOKEN_ID="..."
export MODAL_TOKEN_SECRET="..."
```

If you already have token values, store them for the current Modal profile with:

```bash
uv run modal token set \
  --token-id "$MODAL_TOKEN_ID" \
  --token-secret "$MODAL_TOKEN_SECRET"
```

Do not commit Modal tokens to repo YAML or scripts. The Modal SDK reads these
credentials automatically when `ModalSandboxEnvironment` creates a sandbox.

### Agent Trajectory Smoke Tests

To start vLLM, run mini-SWE-agent locally, and save a formatted trajectory:

```bash
uv run python tests/run_mini_swe_with_vllm.py \
  --model-config configs/models/qwen3_8b.yaml \
  --generation-config configs/generation.yaml \
  --vllm-config configs/agents/vllm_launch.yaml \
  --mini-swe-config configs/agents/mini_swe_local.yaml \
  --trajectory-output outputs/agent_trajectories/local_smoke.json
```

To execute mini-SWE-agent bash commands in a Modal sandbox:

```bash
uv run python tests/run_mini_swe_with_vllm_modal.py \
  --model-config configs/models/qwen3_8b.yaml \
  --generation-config configs/generation.yaml \
  --vllm-config configs/agents/vllm_launch.yaml \
  --mini-swe-config configs/agents/mini_swe_local.yaml \
  --trajectory-output outputs/agent_trajectories/modal_smoke.json
```

To run mini-SWE-agent on a single SWE-bench Verified instance in Modal:

```bash
uv run python tests/run_swebench_single_instance.py \
  --model-config configs/models/qwen3_8b.yaml \
  --generation-config configs/generation.yaml \
  --vllm-config configs/agents/vllm_launch.yaml \
  --agent-config configs/agents/mini_swe_swebench.yaml \
  --trajectory-output outputs/agent_trajectories/swebench_single.json
```

Pass `--instance-id <id>` to target a specific SWE-bench Verified instance
(default: `astropy__astropy-12907`). The trajectory JSON contains mini-SWE
messages, bash command history with per-command git diffs, the final patch,
and the pass/fail outcome from the SWE-bench eval script.

## Running locally

### 1. Extract activations

```bash
uv run python run_extract.py \
  --model-config configs/models/qwen3_8b.yaml \
  --task-config configs/tasks/humaneval.yaml \
  --hardware-config configs/hardware/single_a100.yaml \
  --generation-config configs/generation.yaml \
  --probe will_be_correct \
  --run-id my_run
```

### 2. Build cache

```bash
uv run python run_build_cache.py \
  --run-id my_run \
  --probe will_be_correct
```

### 3. Probe training

Probe training has three modes: **manual**, **sweep-only**, and **sweep→final** (one-shot).

#### Loss function

The `--loss` flag selects the objective:

| `--loss` | Behavior |
|---|---|
| `cross_entropy` (default) | Standard `CrossEntropyLoss` — no class weighting |
| `weighted_cross_entropy` | `CrossEntropyLoss(weight=[1.0, λ · n_neg/n_pos])` — adaptive per-bin class balancing |

When using `weighted_cross_entropy`, the `--pos-weight` λ parameter (default 1.0) multiplies the inverse class ratio:
- λ = 1.0 → exactly balanced classes per bin
- λ > 1.0 → penalize false negatives more (up-weight positives)
- λ < 1.0 → penalize false positives more (down-weight positives)

#### Mode A: Manual (specify all HPs)

```bash
uv run python run_probe.py \
  --run-id my_run --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml \
  final \
  --lr 1e-3 --weight-decay 1e-4 --batch-size 512 --patience 10 \
  --loss weighted_cross_entropy --pos-weight 1.0
```

#### Mode B: Sweep only

```bash
uv run python run_probe.py \
  --run-id my_run --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml \
  sweep --count 50
```

The sweep explores `lr`, `weight_decay`, `batch_size`, `patience`, `loss` (cross_entropy vs weighted_cross_entropy), and `pos_weight` (log-uniform in [0.1, 10.0]) via Bayesian optimisation. Copy the best HPs from the W&B dashboard.

#### Mode C: Sweep → final (fully automatic)

```bash
uv run python run_probe.py \
  --run-id my_run --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml \
  sweep --count 50 --then-final
```

Runs the sweep, waits for completion, fetches the best config from W&B via `fetch_best_sweep_config()`, and trains on all probe layers — no manual copy-paste.

#### Mode D: Final from an existing sweep

```bash
uv run python run_probe.py \
  --run-id my_run --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml \
  final --from-sweep abc123xyz
```

Queries `sweep.best_run()` via the W&B API to auto-populate `lr`, `weight_decay`, `batch_size`, `patience`, `loss`, and `pos_weight`.

### 4. Figures

```bash
uv run python run_figures.py \
  --run-id my_run \
  --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml
```

### 5. Export dashboard

```bash
uv run python run_export_dashboard.py \
  --run-id my_run \
  --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml \
  --task-config configs/tasks/humaneval.yaml
```

Then open `dashboard/index.html` directly in a browser.

## SLURM (Berzelius)

```bash
sbatch slurm/extract.sh \
  --model-config configs/models/qwen3_8b.yaml \
  --task-config configs/tasks/humaneval.yaml \
  --hardware-config configs/hardware/single_a100.yaml \
  --generation-config configs/generation.yaml \
  --probe will_be_correct \
  --run-id my_run

# 8-GPU run:
sbatch --gpus=8 slurm/extract.sh \
  --hardware-config configs/hardware/eight_a100.yaml ...

sbatch slurm/build_cache.sh --run-id my_run --probe will_be_correct

# W&B sweep as job array (30 agents):
sbatch --array=0-29 slurm/probe_sweep.sh \
  --run-id my_run --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml

# Final with manual HPs:
sbatch slurm/probe_final.sh \
  --run-id my_run --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml \
  --lr 1e-3 --weight-decay 1e-4 --batch-size 512 --patience 10 \
  --loss weighted_cross_entropy --pos-weight 1.0

# Final auto-loaded from sweep:
sbatch slurm/probe_final.sh \
  --run-id my_run --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml \
  --from-sweep abc123xyz

sbatch slurm/figures.sh --run-id my_run --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml

sbatch slurm/export_dashboard.sh \
  --run-id my_run --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml \
  --task-config configs/tasks/humaneval.yaml
```

## Tests

```bash
uv run pytest
```

All tests run without GPU, network access, or real model downloads. Complete in under 30 seconds.

## SWE-bench labeling

Agent trajectories generated by mini-SWE-agent inside Modal sandboxes can be labeled
by replaying each edit step to compute per-step probe labels (compiles, test results,
resolution status).

The labeler replays each edit step of an agent trajectory in a fresh Modal sandbox:
reset to clean HEAD, apply the cumulative diff, infer `compiles` from the pytest eval
log, and run the SWE-bench eval script. Labels are written alongside trajectories as
`_labels.json` files.

```bash
# Full run (CPU-only — no GPU needed):
sbatch slurm/swebench_label.sh --config configs/labeling/swebench_labeler.yaml

# Quick test on specific instances:
uv run python run_labeler.py --config configs/labeling/swebench_labeler_test.yaml
```

Config format (`SwebenchLabelerConfig` in `src/configs.py`):

```yaml
trajectory_dir: generations/swebench/qwen3_8b_test
output_dir: generations/swebench/qwen3_8b_test/labels
modal_app_name: program-probes-labeler
sandbox_timeout: 3600     # sandbox lifetime (shared across edits of one instance)
eval_timeout: 600         # per eval-script run
resume: true              # skip trajectories that already have _labels.json
# instances:              # optional — limit to specific instance IDs
#   - astropy__astropy-12907
```

Each `_labels.json` entry records per-edit `compiles` and `test_results`
(`{passed, failed, error, resolved}`). These feed directly into the dynamic probes
(such as `currently_correct` and `currently_compiles`).

## Probes

A probe asks: *does the model's hidden state at a given point in generation linearly encode a specific property of its eventual output?*

### Static vs dynamic

| | Static probe | Dynamic probe |
|---|---|---|
| **Label** | One bool for the entire generation | One bool per edit step |
| **Example** | Will the final code be correct? | At each edit, does the code compile? |
| **Return type** | `bool \| None` | `list[bool \| None]` |
| **Interface** | `is_dynamic = False` | `is_dynamic = True` |

### Edit history and carry-forward

Probes consume `TrajectoryContext`, which holds an `edit_history: list[EditEvent]`.
Each `EditEvent` records the state after one code change (`step_idx`, `compiles`,
`test_results`). Probes return one label per edit — they are stride-agnostic.

The extraction pipeline later expands per-edit labels to per-stride-step labels via
carry-forward: each stride step inherits the label from the most recent edit at or
before that step. Steps before the first edit are masked (`None`) during training.

### Implemented probes

```
will_be_correct           (static)   Did the final edit resolve the issue?
                                     Uses edit_history[-1].test_results["resolved"]
                                     when available; falls back to task_adapter.check_correct().

currently_correct         (dynamic)  At each edit, do all evaluation tests pass?
                                     Uses test_results["resolved"] (SWE-bench gold
                                     standard) if present, otherwise checks that
                                     len(test_results["failed"]) == 0.

currently_compiles        (dynamic)  At each edit, does every changed .py file compile
                                     without errors?  Reads the compiles field directly.

currently_reduces_failing (dynamic)  At each edit, did the number of failing tests decrease
                                     compared to the previous edit?  The clean-checkout
                                     baseline (cmd_idx = -1) provides the initial count.

currently_has_regressions (dynamic)  At each edit, did any test that previously passed
                                     now fail?  First edit compares against the clean
                                     checkout baseline.
```

### SWE-bench label format

Labels are produced by `src/labeling/swebench_labeler.py`. Each `_labels.json` file contains:

```json
{
  "instance_id": "astropy__astropy-12907",
  "edits": [
    {"cmd_idx": -1, "compiles": true, "test_results": {..., "resolved": false}},
    {"cmd_idx": 9,  "compiles": true, "test_results": {..., "resolved": true}}
  ]
}
```

The baseline entry (`cmd_idx = -1`) captures the clean checkout before any edits.
It initialises the predecessor state for delta probes (`currently_reduces_failing`,
`currently_has_regressions`) so the first real edit has a meaningful comparison point.

### Adding a probe

1. Create `src/probes/myprobe.py` subclassing `ProbeAdapter`.
2. Set `name`, `is_dynamic`, and implement `compute_label(ctx)`.
3. Register the probe name in `src/extract._load_probe`.
4. Pass `--probe myprobe` to any entrypoint.

## HumanEval split note

HumanEval has 164 problems. The 70/15/15 group-level split gives approximately 115/25/24 problems. For a larger dataset, switch to MBPP (374 problems) by changing one line in `configs/tasks/humaneval.yaml`:

```yaml
dataset: mbpp
adapter: mbpp
```

## Extension guide

### Add a model adapter

1. Create `src/models/mymodel.py` subclassing `ModelAdapter`.
2. Implement `load`, `get_layer_modules`, `get_hidden_dim`, `tokenize`, `generate`.
3. Add a YAML in `configs/models/` and register the adapter name in `src/extract._load_model_adapter`.

### Add a task adapter

1. Create `src/tasks/mytask.py` subclassing `TaskAdapter`.
2. Implement `load_dataset`, `format_prompt`, `check_correct`, `group_id`.
3. Add a YAML in `configs/tasks/` and register in `src/extract._load_task_adapter`.

### Add an agent scaffold

1. Subclass `AgentAdapter` from `src/agents/base.py` and implement `run_episode`.
2. Wire it into `run_extract.py` via a new `--agent-config` flag.
3. Dynamic probes become available automatically once `edit_history` is populated.

### Add a probe

1. Create `src/probes/myprobe.py` subclassing `ProbeAdapter`.
2. Set `name`, `is_dynamic`, and implement `compute_label`.
3. Register the probe name in `src/extract._load_probe`.
4. Pass `--probe myprobe` to any entrypoint.

## Practical problems

### SSL certificate errors on Berzelius

`uv` ships its own Python 3.12 binary linked against an OpenSSL that looks for `/etc/ssl/cert.pem`.
That path does not exist on RHEL 8 (Berzelius uses `/etc/pki/tls/cert.pem` instead), so Python's ssl
module finds no CA bundle and any outbound TLS connection — including Modal's gRPC channel — fails with:

```
ssl.SSLCertVerificationError: certificate verify failed: unable to get local issuer certificate
```

Fix: add the following to `~/.bashrc` (or `~/.bash_profile`) on Berzelius:

```bash
export SSL_CERT_FILE=/etc/pki/tls/cert.pem
```
