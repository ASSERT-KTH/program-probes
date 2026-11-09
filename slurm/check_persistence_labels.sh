#!/bin/bash
# Label/persistence sanity checks for the lookahead caches.
# Usage: sbatch slurm/check_persistence_labels.sh --only laguna_xs2_pro_full --output reports/rerun/persistence_label_check_laguna_xs2_pro_full.json
#SBATCH -J pp-pchk
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/pchk_%j.out
#SBATCH -e logs/pchk_%j.err
set -euo pipefail
mkdir -p logs
uv run python scripts/rebuttal_check_persistence_labels.py "$@"
