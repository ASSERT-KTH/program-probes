#!/bin/bash
# Extract hidden states from SWE-bench trajectories — thin A100 nodes (4 GPUs, no fat constraint).
#
# Array job (8 shards):
#   sbatch --array=0-7 slurm/extract_swebench_thin.sh \
#     --model-config configs/models/laguna_xs2.yaml \
#     --generation-config configs/generation_laguna_xs2.yaml \
#     --traj-dir generations/swebench_pro/laguna_xs2_full \
#     --output-dir outputs/swebench_pro/laguna_xs2_full
#
#SBATCH -J pp-extract-swebench
#SBATCH -p berzelius
#SBATCH --gpus=4
#SBATCH -t 12:00:00
#SBATCH -o logs/extract_swebench_%A_%a.out
#SBATCH -e logs/extract_swebench_%A_%a.err

set -euo pipefail
mkdir -p logs

module load buildenv-gcccuda/12.4.1-gcc13.3.0
unset CPATH
export LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/software/sse/manual/GCC/13.3.0/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"

RANK=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python run_extract_swebench.py \
  --shard-rank "$RANK" \
  --num-shards "$NUM_SHARDS" \
  --extraction-batch-size 1 \
  --chunk-size 8192 \
  "$@"
