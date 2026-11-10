#!/bin/bash
# Brief T-POOL (scripts/rebuttal_transfer_pooled.py): pooled transfer AUC, eval-only.
# Usage: sbatch -A <account> slurm/transfer_pooled.sh --model laguna --direction v2p --probe currently_correct --output <json>
#SBATCH -J pp-tpool
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o logs/tpool_%j.out
#SBATCH -e logs/tpool_%j.err
set -euo pipefail
mkdir -p logs
uv run python scripts/rebuttal_transfer_pooled.py "$@"
