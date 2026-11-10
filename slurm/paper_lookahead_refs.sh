#!/bin/bash
# Reference curves for the paper's lookahead figures (scripts/rebuttal_paper_lookahead_refs.py).
# Usage: sbatch -A <account> slurm/paper_lookahead_refs.sh --bench swebench --pooled laguna_xs2_full_pooled \
#          --cache-prefix laguna_xs2_full --probe currently_correct --max-k 15 --output reports/rerun/lookahead_refs/<name>.json
#SBATCH -J pp-larefs
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/larefs_%j.out
#SBATCH -e logs/larefs_%j.err
set -euo pipefail
mkdir -p logs
uv run python scripts/rebuttal_paper_lookahead_refs.py "$@"
