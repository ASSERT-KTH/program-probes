#!/bin/bash
#SBATCH -J pp-generate
#SBATCH -p berzelius
#SBATCH --gpus=4
#SBATCH -t 12:00:00
#SBATCH -o logs/generate_%A_%a.out
#SBATCH -e logs/generate_%A_%a.err

set -euo pipefail
mkdir -p logs

# Load GCC 13.3.0 for vLLM CUDA runtime compatibility
module load buildenv-gcccuda/12.4.1-gcc13.3.0
unset CPATH
export LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"

RANK=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}

uv run python run_generate.py \
  --shard-rank "$RANK" \
  --num-shards "$NUM_SHARDS" \
  "$@"
