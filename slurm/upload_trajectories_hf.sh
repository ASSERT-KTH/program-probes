#!/bin/bash
#SBATCH -J pp-upload-trajs
#SBATCH -p berzelius-cpu
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/upload_trajectories_hf_%j.out
#SBATCH -e logs/upload_trajectories_hf_%j.err

set -euo pipefail
mkdir -p logs

uv run python scripts/upload_trajectories_hf.py "$@"
