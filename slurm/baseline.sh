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
# With --run-suffix after_edit, the baseline is restricted to the same
# after-edit targets as the after-edit probes, using the edit index at
# cache/<cache-base>/edit_step_index.pt (override with --edit-index-run-id).
# --n-eval-bins must match the probe runs being compared against (default 10).
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
EDIT_INDEX_RUN_ID=""
N_EVAL_BINS=10
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
        --edit-index-run-id) EDIT_INDEX_RUN_ID="$2"; shift 2 ;;
        --n-eval-bins)  N_EVAL_BINS="$2";   shift 2 ;;
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

EXTRA_ARGS=(--n-bins 1 --n-eval-bins "$N_EVAL_BINS")  # --n-bins is unused by the baseline
if [[ "$RUN_SUFFIX" == *after_edit* ]]; then
    EXTRA_ARGS+=(--after-edit-only --edit-index-run-id "${EDIT_INDEX_RUN_ID:-$CACHE_BASE}")
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
            "${EXTRA_ARGS[@]}" \
            baseline-persistence
    done
done
