#!/bin/bash
# Brief PO (scripts/rebuttal_prompt_only.py), eval + small CPU logistic regressions.
# Usage: sbatch -A berzelius-2026-259 --exclude=node226 slurm/prompt_only.sh horizon --run laguna_verified \
#          --probe currently_correct --output paper_rebuttal_candidate/prompt_only/json/<name>.json
#SBATCH -J pp-po
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=96G
#SBATCH -t 05:00:00
#SBATCH -o logs/po_%j.out
#SBATCH -e logs/po_%j.err
set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
uv run python scripts/rebuttal_prompt_only.py "$@"
