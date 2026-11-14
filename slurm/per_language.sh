#!/bin/bash
# Rebuttal E11 (scripts/rebuttal_per_language.py).
#SBATCH -J pp-lang
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o logs/lang_%j.out
#SBATCH -e logs/lang_%j.err
set -euo pipefail
mkdir -p logs
uv run python scripts/rebuttal_per_language.py "$@"
