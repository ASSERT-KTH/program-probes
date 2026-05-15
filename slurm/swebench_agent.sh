#!/bin/bash
#SBATCH -J pp-swebench
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -o logs/swebench_%A_%a.out
#SBATCH -e logs/swebench_%A_%a.err

set -euo pipefail
mkdir -p logs

module load buildenv-gcccuda/12.4.1-gcc13.3.0
unset CPATH
export LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"

RANK=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}

uv run python run_swebench_agent.py \
  --model-config configs/models/qwen3_8b.yaml \
  --generation-config configs/generation.yaml \
  --vllm-config configs/agents/vllm_launch.yaml \
  --agent-config configs/agents/mini_swe_swebench.yaml \
  --task-config configs/tasks/swe_bench_verified.yaml \
  --output-dir generations/swebench \
  --shard-rank "$RANK" \
  --num-shards "$NUM_SHARDS" \
  --resume \
  "$@"
