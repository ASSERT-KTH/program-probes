#!/bin/bash
# Rebuttal E2/E16/E17 (scripts/rebuttal_phase_confound.py).
# Usage: sbatch -A <account> slurm/phase_confound.sh lookahead --bench swebench_pro --pooled qwen36_35b_a3b_pro_full_pooled \
#          --cache-prefix qwen36_35b_a3b_pro_full --probe currently_correct --max-k 15 --output reports/rerun/phase/<name>.json
#SBATCH -J pp-phase
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -o logs/phase_%j.out
#SBATCH -e logs/phase_%j.err
set -euo pipefail
mkdir -p logs
uv run python scripts/rebuttal_phase_confound.py "$@"
