#!/bin/bash
# Brief T1K1 (scripts/rebuttal_table1_k1.py), eval + small CPU logistic regressions, k=1 only.
# Usage: sbatch -A berzelius-2026-259 --exclude=node226 slurm/table1_k1.sh --run laguna_verified \
#          --probe currently_correct --output paper_rebuttal_candidate/table1_k1/json/<name>.json
#SBATCH -J pp-t1k1
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o logs/t1k1_%j.out
#SBATCH -e logs/t1k1_%j.err
set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
uv run python scripts/rebuttal_table1_k1.py "$@"
