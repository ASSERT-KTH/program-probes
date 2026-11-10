#!/bin/bash
# Brief E14 (scripts/rebuttal_population_counts.py): population counts after filtering (no probes).
# Usage: sbatch -A <account> slurm/population_counts.sh --run laguna_verified --output paper_rebuttal_candidate/counts/json/counts_laguna_verified.json
#SBATCH -J pp-counts
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/counts_%j.out
#SBATCH -e logs/counts_%j.err
set -euo pipefail
mkdir -p logs
uv run python scripts/rebuttal_population_counts.py "$@"
