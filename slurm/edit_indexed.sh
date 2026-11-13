#!/bin/bash
# Brief EI (scripts/rebuttal_edit_indexed.py): edit-indexed horizon, new linear probes on CPU.
# Usage: sbatch -A berzelius-2026-259 --exclude=node226 slurm/edit_indexed.sh --run laguna_verified \
#          --probe currently_correct --output paper_rebuttal_candidate/edit_indexed/json/<name>.json
#SBATCH -J pp-ei
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -o logs/ei_%j.out
#SBATCH -e logs/ei_%j.err
set -euo pipefail
mkdir -p logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
uv run python scripts/rebuttal_edit_indexed.py "$@"
