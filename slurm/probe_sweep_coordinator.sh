#!/bin/bash
#SBATCH -J pp-probe-coord
#SBATCH -p berzelius-cpu
#SBATCH -n 8
#SBATCH --mem=150G
#SBATCH -t 04:00:00
#SBATCH -o logs/probe_sweep_coordinator_%j.out
#SBATCH -e logs/probe_sweep_coordinator_%j.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# Parse --n-agents and --count; collect everything else as probe_args
# ---------------------------------------------------------------------------
N_AGENTS=2
COUNT=10
PROBE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-agents)  N_AGENTS="$2"; shift 2 ;;
        --count)     COUNT="$2";    shift 2 ;;
        *)           PROBE_ARGS+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Create the sweep and capture only the ID (printed to stdout by create_sweep)
# ---------------------------------------------------------------------------
echo "[coordinator] Creating sweep (n_agents=${N_AGENTS}, count_per_agent=${COUNT})"
SWEEP_ID=$(uv run python run_probe.py "${PROBE_ARGS[@]}" create-sweep)
echo "[coordinator] Sweep ID: ${SWEEP_ID}"

# ---------------------------------------------------------------------------
# 2. Submit N-1 additional agent jobs
# ---------------------------------------------------------------------------
AGENT_SLURM_TIME="${SLURM_TIMELIMIT:-04:00:00}"

for i in $(seq 2 "${N_AGENTS}"); do
    JOB_ID=$(sbatch --parsable \
        --time="${AGENT_SLURM_TIME}" \
        slurm/probe_sweep.sh \
        "${PROBE_ARGS[@]}" sweep --sweep-id "${SWEEP_ID}" --count "${COUNT}")
    echo "[coordinator] Submitted agent ${i} as SLURM job ${JOB_ID}"
done

# ---------------------------------------------------------------------------
# 3. Run as agent 1
# ---------------------------------------------------------------------------
echo "[coordinator] Running as agent 1"
uv run python run_probe.py "${PROBE_ARGS[@]}" sweep --sweep-id "${SWEEP_ID}" --count "${COUNT}"
