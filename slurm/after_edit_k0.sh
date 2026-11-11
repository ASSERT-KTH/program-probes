#!/bin/bash
#SBATCH -J ae0
#SBATCH -p berzelius-cpu
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/ae0_%j.out
# Brief AE0: k=0 probe AUC right after an edit (eval only). Usage: sbatch -A <acct> slurm/after_edit_k0.sh <run>
set -euo pipefail
cd /proj/assert-berzelius/users/x_andaf/program-probes
RUN=$1
uv run python scripts/rebuttal_after_edit_k0.py --run "$RUN" \
    --output paper_rebuttal_candidate/after_edit_k0/json/ae0_${RUN}.json --edits-dump tmp/ae0_edits_${RUN}.json
