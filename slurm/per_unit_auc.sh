#!/bin/bash
# Brief E12 (scripts/rebuttal_per_unit_auc.py), eval only.
# Usage: sbatch -A berzelius-2026-259 --exclude=node226 slurm/per_unit_auc.sh --run laguna_verified \
#          --probe currently_correct --output paper_rebuttal_candidate/per_unit_auc/json/<name>.json
#SBATCH -J pp-e12
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o logs/e12_%j.out
#SBATCH -e logs/e12_%j.err
set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
uv run python scripts/rebuttal_per_unit_auc.py "$@"
