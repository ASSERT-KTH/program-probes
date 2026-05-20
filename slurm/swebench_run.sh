#!/bin/bash
# SWE-bench Verified agent run — SLURM launcher.
#
# Single shard:
#   sbatch swebench_run.sh --run-config configs/runs/qwen3_8b_swebench_test.yaml
#
# Array job (8 shards):
#   sbatch --array=0-7 swebench_run.sh --run-config configs/runs/qwen3_8b_swebench_test.yaml
#
#SBATCH -J pp-swebench
#SBATCH -p berzelius
#SBATCH --gpus=8
#SBATCH -C fat
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
export SSL_CERT_FILE=/etc/pki/tls/cert.pem

RANK=${SLURM_ARRAY_TASK_ID:-0}
# Allow overriding total shard count independently of array size
# e.g. when resubmitting a subset: sbatch --array=1,3 -v NUM_SHARDS=4 ...
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}

uv run python run_swebench_agent.py \
  --shard-rank "$RANK" \
  --num-shards "$NUM_SHARDS" \
  --resume \
  "$@"
