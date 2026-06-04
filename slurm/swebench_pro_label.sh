#!/bin/bash
# SLURM launcher for SWE-bench Pro trajectory labeling.
#
# Usage:
#   sbatch slurm/swebench_pro_label.sh --config configs/labeling/laguna_xs2_swebench_pro_test_labeler.yaml
#
#SBATCH -J pp-pro-label
#SBATCH -p berzelius-cpu
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH -o logs/pro_labeler_%A_%a.out
#SBATCH -e logs/pro_labeler_%A_%a.err

set -euo pipefail
mkdir -p logs

uv run python run_labeler_pro.py "$@"
