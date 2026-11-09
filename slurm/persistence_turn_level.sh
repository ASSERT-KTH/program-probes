#!/bin/bash
# Turn-level persistence baseline at k=1 (see scripts/rebuttal_persistence_turn_level.py).
# Usage: sbatch slurm/persistence_turn_level.sh --bench swebench_pro --prefix laguna_xs2_pro_full --output reports/rerun/persistence_turn_level_laguna_xs2_pro_full.json
#SBATCH -J pp-pturn
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/pturn_%j.out
#SBATCH -e logs/pturn_%j.err
set -euo pipefail
mkdir -p logs
uv run python scripts/rebuttal_persistence_turn_level.py "$@"
