#!/bin/bash
# SLURM launcher for SWE-bench trajectory labeling.
#
# Single run:
#   sbatch slurm/swebench_label.sh --config configs/labeling/swebench_labeler.yaml
#
# Array job (N shards):
#   sbatch --array=0-$((N-1)) slurm/swebench_label.sh --config configs/labeling/swebench_labeler.yaml
#
#SBATCH -J pp-label
#SBATCH -p berzelius-cpu
#SBATCH -t 12:00:00
#SBATCH -o logs/labeler_%A_%a.out
#SBATCH -e logs/labeler_%A_%a.err

set -euo pipefail
mkdir -p logs

uv run python run_labeler.py "$@"
