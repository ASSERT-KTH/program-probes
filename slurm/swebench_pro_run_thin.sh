#!/bin/bash
# SWE-bench Pro agent run — thin A100 nodes (4 GPUs, no fat constraint).
#
# Single shard:
#   sbatch --gpus=4 slurm/swebench_pro_run_thin.sh \
#     --run-config configs/runs/laguna_xs2_swebench_pro_test.yaml
#
# Array job (5 shards):
#   sbatch --array=0-4 slurm/swebench_pro_run_thin.sh \
#     --run-config configs/runs/laguna_xs2_swebench_pro_full.yaml
#
#SBATCH -J pp-swebench-pro
#SBATCH -p berzelius
#SBATCH --gpus=4
#SBATCH -t 32:00:00
#SBATCH -o logs/swebench_pro_%A_%a.out
#SBATCH -e logs/swebench_pro_%A_%a.err

set -euo pipefail
mkdir -p logs

module load buildenv-gcccuda/12.4.1-gcc13.3.0
unset CPATH
export LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/software/sse/manual/GCC/13.3.0/lib64:${LD_LIBRARY_PATH:-}"
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
