#!/bin/bash
# Attach probe labels to activation .pt files (CPU-only step).
#
# Example:
#   sbatch slurm/attach_labels_swebench.sh \
#     --input-dir outputs/swebench/qwen36_27b_test \
#     --traj-dir generations/swebench/qwen36_27b_test \
#     --label-dir generations/swebench/qwen36_27b_test/labels \
#     --output-dir outputs/swebench/qwen36_27b_test_labeled \
#     --probe will_resolve currently_correct_swe \
#     --generation-config configs/generation.yaml
#
#SBATCH -J pp-attach-labels-swebench
#SBATCH -p berzelius-cpu
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/attach_labels_swebench_%j.out
#SBATCH -e logs/attach_labels_swebench_%j.err

set -euo pipefail
mkdir -p logs

uv run python run_attach_labels_swebench.py "$@"
