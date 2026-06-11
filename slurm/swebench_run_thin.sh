#!/bin/bash
# SWE-bench Verified agent run — thin A100 nodes (no -C fat constraint).
#
# Single shard:
#   sbatch slurm/swebench_run_thin.sh \
#     --run-config configs/runs/qwen35_2b_swebench_test.yaml
#
# Array job (4 shards):
#   sbatch --array=0-3 slurm/swebench_run_thin.sh \
#     --run-config configs/runs/qwen35_2b_swebench_full.yaml
#
#SBATCH -J pp-swebench
#SBATCH -p berzelius
#SBATCH --gpus=8
#SBATCH -t 20:00:00
#SBATCH -o logs/swebench_%A_%a.out
#SBATCH -e logs/swebench_%A_%a.err

set -euo pipefail
mkdir -p logs

module load buildenv-gcccuda/12.4.1-gcc13.3.0
unset CPATH
export LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/software/sse/manual/GCC/13.3.0/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"
export SSL_CERT_FILE=/etc/pki/tls/cert.pem
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

RANK=${SLURM_ARRAY_TASK_ID:-0}
# NUM_SHARDS must be set explicitly when resubmitting a subset of array tasks
# (SLURM_ARRAY_TASK_COUNT reflects the subset size, not the total).
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}

uv run python run_swebench_agent.py \
  --shard-rank "$RANK" \
  --num-shards "$NUM_SHARDS" \
  --resume \
  "$@"
