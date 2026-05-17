#!/bin/bash
#SBATCH -J pp-gen-test-cwm
#SBATCH -p berzelius
#SBATCH --gpus=2
#SBATCH -C fat
#SBATCH -t 00:30:00
#SBATCH -o logs/test_generate_cwm_%j.out
#SBATCH -e logs/test_generate_cwm_%j.err

set -euo pipefail
mkdir -p logs

module load buildenv-gcccuda/12.4.1-gcc13.3.0
unset CPATH
export LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"

echo "=== HumanEval+ (1 sample) ==="
uv run python run_generate.py \
  --model-config configs/models/cwm.yaml \
  --task-config configs/tasks/humaneval.yaml \
  --generation-config configs/generation_cwm_test.yaml \
  --run-id test_cwm_humaneval \
  --max-samples 1

echo "=== Done ==="
cat generations/test_cwm_humaneval_shard000.json | python -c "
import json, sys
data = json.load(sys.stdin)
for entry in data:
    print('--- HumanEval sample', entry['sample_id'])
    print('raw_text:', entry['raw_text'][:500])
"
