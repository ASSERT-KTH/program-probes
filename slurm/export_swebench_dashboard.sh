#!/bin/bash
#SBATCH -J pp-export-swebench-dashboard
#SBATCH -p berzelius-cpu
#SBATCH -n 8
#SBATCH --mem=100G
#SBATCH -t 02:00:00
#SBATCH -o logs/export_swebench_dashboard_%j.out
#SBATCH -e logs/export_swebench_dashboard_%j.err

set -euo pipefail
mkdir -p logs

uv run python run_export_swebench_dashboard.py "$@"
