#!/bin/bash
#SBATCH -J pp-extract
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 06:00:00
#SBATCH -o logs/extract_%A_%a.out
#SBATCH -e logs/extract_%A_%a.err

set -euo pipefail
mkdir -p logs

RANK=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}

uv run python run_extract.py \
  --shard-rank "$RANK" \
  --num-shards "$NUM_SHARDS" \
  "$@"
