#!/bin/bash
# Brief E9 (scripts/rebuttal_text_baseline.py), eval + CPU TF-IDF logistic regressions.
# Usage: sbatch -A berzelius-2026-259 --exclude=node226 slurm/text_baseline.sh horizon --run laguna_verified \
#          --probe currently_correct --output paper_rebuttal_candidate/text_baseline/json/<name>.json
#SBATCH -J pp-e9
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=128G
#SBATCH -t 05:00:00
#SBATCH -o logs/e9_%j.out
#SBATCH -e logs/e9_%j.err
set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
uv run python scripts/rebuttal_text_baseline.py "$@"
