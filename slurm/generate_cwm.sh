#!/bin/bash
#SBATCH -J pp-generate-cwm
#SBATCH -p berzelius
#SBATCH --gpus=2
#SBATCH -C fat
#SBATCH -t 08:00:00
#SBATCH -o logs/generate_cwm_%j.out
#SBATCH -e logs/generate_cwm_%j.err

set -euo pipefail
mkdir -p logs

module load buildenv-gcccuda/12.4.1-gcc13.3.0
unset CPATH
export LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"

uv run python run_generate.py \
  --model-config configs/models/cwm.yaml \
  --generation-config configs/generation_cwm.yaml \
  "$@"
