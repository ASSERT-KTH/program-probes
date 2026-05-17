# program-probes

Measures whether a language model's internal hidden states linearly predict properties of its own output before those properties are realised.

Pipeline: inference + activation hooks → per-sample `.pt` files → per-(layer, probe) cache tensors → linear probe training with W&B sweep → accuracy heatmaps → static dashboard.

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

### 3. Probe sweep (W&B)

```bash
uv run python run_probe.py sweep \
  --run-id my_run \
  --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml
```

### 4. Probe final run

```bash
uv run python run_probe.py final \
  --run-id my_run \
  --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --batch-size 512 \
  --patience 10
```

### 5. Figures

```bash
uv run python run_figures.py \
  --run-id my_run \
  --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml
```

### 6. Export dashboard

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

sbatch slurm/probe_final.sh \
  --run-id my_run --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml \
  --lr 1e-3 --weight-decay 1e-4 --batch-size 512 --patience 10

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
