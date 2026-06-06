#!/bin/bash
# Compute per-turn mean NLL of tool output tokens for SWE-bench trajectories.
# Runs one forward pass per trajectory (no sampling); same GPU requirements as extraction.
#
# Laguna XS2 (single shard):
#   sbatch slurm/build_tool_nll.sh \
#     --model-config configs/models/laguna_xs2.yaml \
#     --generation-config configs/generation_laguna_xs2.yaml \
#     --traj-dir generations/swebench/laguna_xs2_full \
#     --output cache/swebench/laguna_xs2_full/tool_nll_index.pt
#
# Laguna XS2 (array job, 8 shards):
#   sbatch --array=0-7 slurm/build_tool_nll.sh \
#     --model-config configs/models/laguna_xs2.yaml \
#     --generation-config configs/generation_laguna_xs2.yaml \
#     --traj-dir generations/swebench/laguna_xs2_full \
#     --output cache/swebench/laguna_xs2_full/tool_nll_index.pt
#
# After all shards complete, merge with:
#   uv run python -c "
#   import torch; from pathlib import Path
#   shards = sorted(Path('cache/swebench/laguna_xs2_full').glob('tool_nll_index.pt.shard*.pt'))
#   merged = {}
#   for p in shards:
#       merged.update(torch.load(p, weights_only=False))
#   torch.save(merged, 'cache/swebench/laguna_xs2_full/tool_nll_index.pt')
#   print(f'Merged {len(shards)} shards → {len(merged)} samples')
#   "
#
#SBATCH -J pp-tool-nll
#SBATCH -p berzelius
#SBATCH --gpus=4
#SBATCH -t 06:00:00
#SBATCH -o logs/build_tool_nll_%j.out
#SBATCH -e logs/build_tool_nll_%j.err

set -euo pipefail
mkdir -p logs

module load buildenv-gcccuda/12.4.1-gcc13.3.0
unset CPATH
export LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"

RANK=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python build_tool_nll_index.py \
  --shard-rank "$RANK" \
  --num-shards "$NUM_SHARDS" \
  "$@"
