#!/bin/bash
#SBATCH -J pp-export-dashboard
#SBATCH -p berzelius-cpu
#SBATCH -n 8
#SBATCH --mem=100G
#SBATCH -t 02:00:00
#SBATCH -o logs/export_dashboard_%j.out
#SBATCH -e logs/export_dashboard_%j.err

set -euo pipefail
mkdir -p logs

uv run python run_export_dashboard.py "$@"
