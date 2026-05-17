#!/bin/bash
#SBATCH -J install-vllm
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -t 01:30:00
#SBATCH -o logs/install_vllm12_%j.out
#SBATCH -e logs/install_vllm12_%j.err

set -euo pipefail
mkdir -p logs

echo "=== Loading GCC 13.3.0 ==="
module load buildenv-gcccuda/12.4.1-gcc13.3.0

# The module sets CPATH to CUDA 12.4.1 headers which conflict with CUDA 13.0 nvcc.
# Unset CPATH so cmake/nvcc use their own built-in CUDA 13.0 include paths.
unset CPATH
# Keep LIBRARY_PATH but put CUDA 13.0 first to override CUDA 12.4.1 module libs.
export LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"

echo "=== Environment ==="
gcc --version | head -1
nvcc --version | head -1
.venv/bin/python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)"

echo "=== Installing vllm ==="
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=16

uv run python -m ensurepip
.venv/bin/python -m pip install vllm

echo "=== Done ==="
.venv/bin/python -c "import vllm; print('vllm:', vllm.__version__)"
