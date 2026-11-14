#!/bin/bash
# Brief E12H (scripts/rebuttal_per_traj_horizon.py), eval only.
# Usage: sbatch -A berzelius-2026-259 --exclude=node226 slurm/per_traj_horizon.sh --run laguna_verified \
#          --probe currently_correct --output paper_rebuttal_candidate/per_traj_horizon/json/pth_<run>_<probe>.json
#SBATCH -J pp-e12h
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o logs/e12h_%j.out
#SBATCH -e logs/e12h_%j.err
set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
uv run python scripts/rebuttal_per_traj_horizon.py "$@"
