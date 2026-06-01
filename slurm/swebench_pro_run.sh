#!/bin/bash
# SWE-bench Pro agent run — SLURM launcher.
#
# Single shard:
#   sbatch slurm/swebench_pro_run.sh --run-config configs/runs/qwen36_27b_swebench_pro_test.yaml
#
# Array job (4 shards):
#   sbatch --array=0-3 slurm/swebench_pro_run.sh \
#     --run-config configs/runs/qwen36_27b_swebench_pro_full.yaml
#
# Prerequisite: initialise the SWE-bench_Pro-os submodule (one-time):
#   git submodule update --init SWE-bench_Pro-os
#
#SBATCH -J pp-swebench-pro
#SBATCH -p berzelius
#SBATCH --gpus=8
#SBATCH -C fat
#SBATCH -t 12:00:00
#SBATCH -o logs/swebench_pro_%A_%a.out
#SBATCH -e logs/swebench_pro_%A_%a.err

set -euo pipefail
mkdir -p logs

module load buildenv-gcccuda/12.4.1-gcc13.3.0
unset CPATH
export LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"
export SSL_CERT_FILE=/etc/pki/tls/cert.pem

RANK=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}

uv run python run_swebench_pro_agent.py \
  --shard-rank "$RANK" \
  --num-shards "$NUM_SHARDS" \
  --resume \
  "$@"
