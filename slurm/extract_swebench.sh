#!/bin/bash
# Extract hidden states from SWE-bench trajectories for probe training.
#
# Single shard:
#   sbatch slurm/extract_swebench.sh \
#     --model-config configs/models/qwen36_27b.yaml \
#     --generation-config configs/generation.yaml \
#     --traj-dir generations/swebench/qwen36_27b_test \
#     --probe will_resolve \
#     --output-dir outputs/swebench/qwen36_27b_test
#
# Array job (4 shards):
#   sbatch --array=0-3 slurm/extract_swebench.sh \
#     --model-config configs/models/qwen36_27b.yaml \
#     --generation-config configs/generation.yaml \
#     --traj-dir generations/swebench/qwen36_27b_test \
#     --probe will_resolve \
#     --output-dir outputs/swebench/qwen36_27b_test
#
#SBATCH -J pp-extract-swebench
#SBATCH -p berzelius
#SBATCH --gpus=2
#SBATCH -C fat
#SBATCH -t 12:00:00
#SBATCH -o logs/extract_swebench_%A_%a.out
#SBATCH -e logs/extract_swebench_%A_%a.err

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

uv run python run_extract_swebench.py \
  --shard-rank "$RANK" \
  --num-shards "$NUM_SHARDS" \
  --extraction-batch-size 1 \
  "$@"
