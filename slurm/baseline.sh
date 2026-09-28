#!/bin/bash
# Computes the naive-persistence baseline for every k in an existing lookahead
# sweep (all shift{k}_max{K} caches share one job instead of one job per k).
#
# Usage:
#   sbatch slurm/baseline.sh \
#     --model-config configs/models/laguna_xs2.yaml \
#     --cache-dir cache/swebench \
#     --results-dir results/swebench \
#     --cache-base laguna_xs2_full \
#     --results-base laguna_xs2_full_pooled \
#     --max-k 50 \
#     --probes currently_compiles currently_correct currently_has_regressions currently_reduces_failing
#
#SBATCH -J pp-baseline
#SBATCH -p berzelius-cpu
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/baseline_%j.out
#SBATCH -e logs/baseline_%j.err

set -euo pipefail
mkdir -p logs

MODEL_CONFIG=""
CACHE_DIR=""
RESULTS_DIR=""
CACHE_BASE=""
RESULTS_BASE=""
MAX_K=""
RUN_SUFFIX=""
PROBES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-config) MODEL_CONFIG="$2"; shift 2 ;;
        --cache-dir)    CACHE_DIR="$2";     shift 2 ;;
        --results-dir)  RESULTS_DIR="$2";   shift 2 ;;
        --cache-base)   CACHE_BASE="$2";    shift 2 ;;
        --results-base) RESULTS_BASE="$2";  shift 2 ;;
        --max-k)        MAX_K="$2";         shift 2 ;;
        --run-suffix)   RUN_SUFFIX="$2";    shift 2 ;;
        --probes)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                PROBES+=("$1"); shift
            done
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

SFX=""
if [[ -n "$RUN_SUFFIX" ]]; then
    SFX="_${RUN_SUFFIX}"
fi

# k=0 has no y_original in the cache (label_shift=0 stores no shift), so the
# persistence baseline is undefined there — start from k=1.
for k in $(seq 1 "$MAX_K"); do
    for probe in "${PROBES[@]}"; do
        out_path="${RESULTS_DIR}/${RESULTS_BASE}_shift${k}_max${MAX_K}${SFX}/${probe}/persistence_baseline.pt"
        if [[ -f "$out_path" ]]; then
            echo "[skip] already computed: $out_path"
            continue
        fi
        uv run python run_probe.py \
            --run-id "${RESULTS_BASE}_shift${k}_max${MAX_K}${SFX}" \
            --probe "$probe" \
            --model-config "$MODEL_CONFIG" \
            --cache-dir "$CACHE_DIR" \
            --cache-run-id "${CACHE_BASE}_shift${k}_max${MAX_K}" \
            --results-dir "$RESULTS_DIR" \
            --layer 0 \
            baseline-persistence
    done
done
