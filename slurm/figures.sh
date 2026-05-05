#!/bin/bash
#SBATCH -J pp-figures
#SBATCH -p berzelius-cpu
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o logs/figures_%j.out
#SBATCH -e logs/figures_%j.err

set -euo pipefail
mkdir -p logs

uv run python run_figures.py "$@"
