#!/bin/bash
# Coordinator for a single (probe, layer) sweep.
# Creates one W&B sweep, submits N parallel worker jobs, then submits final
# training as a dependent job that runs after all workers complete.
#
# Usage — one call per (probe, layer):
#   sbatch slurm/probe_sweep_coordinator.sh \
#     --model-config configs/models/laguna_xs2.yaml \
#     --layer 0 \
#     --probe currently_correct \
#     --probe-arch linear \
#     --run-id laguna_xs2_full_pooled \
#     --cache-dir cache/swebench \
#     --cache-run-id laguna_xs2_full \
#     --results-dir results/swebench \
#     --n-bins 1 \
#     --n-agents 4 \
#     --count 5
#
#SBATCH -J pp-probe-coord
#SBATCH -p berzelius-cpu
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o logs/probe_sweep_coordinator_%j.out
#SBATCH -e logs/probe_sweep_coordinator_%j.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# Parse --n-agents and --count; pass everything else through to run_probe.py
# ---------------------------------------------------------------------------
N_AGENTS=4
COUNT=20
PROBE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-agents) N_AGENTS="$2"; shift 2 ;;
        --count)    COUNT="$2";    shift 2 ;;
        *)          PROBE_ARGS+=("$1"); shift ;;
    esac
done

COUNT_PER_AGENT=$(( COUNT / N_AGENTS ))

# ---------------------------------------------------------------------------
# 1. Create the sweep — layer name baked in via --layer in PROBE_ARGS
# ---------------------------------------------------------------------------
echo "[coordinator] Creating sweep (n_agents=${N_AGENTS}, count=${COUNT}, count_per_agent=${COUNT_PER_AGENT})"
SWEEP_ID=$(uv run python run_probe.py "${PROBE_ARGS[@]}" create-sweep)
echo "[coordinator] Sweep ID: ${SWEEP_ID}"

# ---------------------------------------------------------------------------
# 2. Submit N parallel worker jobs
# ---------------------------------------------------------------------------
WORKER_IDS=()
for i in $(seq 1 "${N_AGENTS}"); do
    JID=$(sbatch --parsable \
        slurm/probe_sweep.sh \
        "${PROBE_ARGS[@]}" sweep --sweep-id "${SWEEP_ID}" --count "${COUNT_PER_AGENT}")
    WORKER_IDS+=("${JID}")
    echo "[coordinator] Worker ${i} → job ${JID}"
done

# ---------------------------------------------------------------------------
# 3. Submit final training, dependent on all workers completing successfully
# ---------------------------------------------------------------------------
DEP="afterok:$(IFS=:; echo "${WORKER_IDS[*]}")"
FINAL_JID=$(sbatch --parsable --dependency="${DEP}" \
    slurm/probe_final.sh \
    "${PROBE_ARGS[@]}" final --from-sweep "${SWEEP_ID}")
echo "[coordinator] Final training → job ${FINAL_JID} (depends on ${WORKER_IDS[*]})"
