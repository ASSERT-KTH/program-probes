#!/usr/bin/env bash
set -euo pipefail

# Shared environment bootstrap for Berzelius jobs.
# Usage:
#   source scripts/berzelius_env.sh --check-only
#   source scripts/berzelius_env.sh --sync
#
# By default this loads the CUDA 12.4 + GCC 13 build environment needed when
# uv has to build vLLM from source on Berzelius. Override with:
#   export BERZELIUS_MODULES="buildenv-gcccuda/12.1.1-gcc12.3.0"
# or disable module loading with:
#   export BERZELIUS_LOAD_MODULES=0

MODE="${1:---check-only}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_BERZELIUS_MODULES="buildenv-gcccuda/12.4.1-gcc13.3.0"

if [[ "${BERZELIUS_PURGE_MODULES:-0}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module purge
fi

if [[ "${BERZELIUS_LOAD_MODULES:-1}" == "1" ]]; then
  BERZELIUS_MODULES="${BERZELIUS_MODULES:-${DEFAULT_BERZELIUS_MODULES}}"
  if ! command -v module >/dev/null 2>&1; then
    echo "BERZELIUS_MODULES is set but the module command is unavailable." >&2
    return 1 2>/dev/null || exit 1
  fi
  # shellcheck disable=SC2206
  _modules=(${BERZELIUS_MODULES})
  for _module in "${_modules[@]}"; do
    module load "${_module}"
  done
fi

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it once with: python -m pip install --user uv" >&2
  return 1 2>/dev/null || exit 1
fi

case "${MODE}" in
  --sync)
    echo "CUDA_HOME=${CUDA_HOME:-unset}"
    mkdir -p logs
    if command -v nvcc >/dev/null 2>&1; then
      nvcc --version | sed -n '1,4p'
    else
      echo "WARNING: nvcc is not on PATH; vLLM source builds may fail." >&2
    fi
    uv sync --frozen --python 3.12
    ;;
  --check-only)
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use --check-only or --sync." >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

uv run --python 3.12 python - <<'PY'
import importlib.metadata as md
import sys

packages = [
    "torch",
    "transformers",
    "accelerate",
    "pydantic",
    "vllm",
    "mini-swe-agent",
    "litellm",
    "modal",
]

print(f"python={sys.version.split()[0]}")
for package in packages:
    try:
        version = md.version(package)
    except md.PackageNotFoundError:
        version = "not-installed"
    print(f"{package}={version}")

try:
    import torch
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_count={torch.cuda.device_count()}")
except Exception as exc:
    print(f"cuda_check_error={exc!r}")
PY
