#!/bin/bash
#SBATCH -J pp-probe-sweep
#SBATCH -p berzelius-cpu
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o logs/probe_sweep_%j.out
#SBATCH -e logs/probe_sweep_%j.err

set -euo pipefail
mkdir -p logs

uv run python run_probe.py sweep "$@"
