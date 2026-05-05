#!/bin/bash
#SBATCH -J pp-build-cache
#SBATCH -p berzelius-cpu
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o logs/build_cache_%j.out
#SBATCH -e logs/build_cache_%j.err

set -euo pipefail
mkdir -p logs

uv run python run_build_cache.py "$@"
