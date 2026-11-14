#!/bin/bash
# Rebuttal E18 (scripts/rebuttal_edit_outcome_strata.py).
# Usage: sbatch -A <account> slurm/edit_outcome_strata.sh strata --bench ... --output reports/rerun/phase/e18_<name>.json
#SBATCH -J pp-e18
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -o logs/e18_%j.out
#SBATCH -e logs/e18_%j.err
set -euo pipefail
mkdir -p logs
uv run python scripts/rebuttal_edit_outcome_strata.py "$@"
