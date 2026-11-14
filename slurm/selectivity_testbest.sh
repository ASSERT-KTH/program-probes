#!/bin/bash
# Request SEL-CI (scripts/rebuttal_selectivity_testbest.py): k=0 selectivity CIs at best test layers.
# Usage: sbatch -A <account> slurm/selectivity_testbest.sh --bench swebench --pooled <pooled> --cache-prefix <cache> --probe <probe> --output <json>
#SBATCH -J pp-selci
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o logs/selci_%j.out
#SBATCH -e logs/selci_%j.err
set -euo pipefail
mkdir -p logs
uv run python scripts/rebuttal_selectivity_testbest.py "$@"
