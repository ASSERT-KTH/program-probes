#!/bin/bash
#SBATCH -J pp-probe-final
#SBATCH -p berzelius-cpu
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o logs/probe_final_%j.out
#SBATCH -e logs/probe_final_%j.err

set -euo pipefail
mkdir -p logs

uv run python run_probe.py "$@"
