#!/bin/bash
#SBATCH -J pp-gen-test-qwen35
#SBATCH -p berzelius
#SBATCH --gpus=1
#SBATCH -C thin
#SBATCH -t 00:30:00
#SBATCH -o logs/test_generate_qwen35_%j.out
#SBATCH -e logs/test_generate_qwen35_%j.err

set -euo pipefail
mkdir -p logs

module load buildenv-gcccuda/12.4.1-gcc13.3.0
unset CPATH
export LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"

echo "=== HumanEval+ (1 sample) ==="
uv run python run_generate.py \
  --model-config configs/models/qwen35_2b.yaml \
  --task-config configs/tasks/humaneval.yaml \
  --generation-config configs/generation_qwen35_test.yaml \
  --run-id test_qwen35_humaneval \
  --max-samples 1

echo "=== HumanEval-Fix (1 sample) ==="
uv run python run_generate.py \
  --model-config configs/models/qwen35_2b.yaml \
  --task-config configs/tasks/humaneval_fix.yaml \
  --generation-config configs/generation_qwen35_test.yaml \
  --run-id test_qwen35_humaneval_fix \
  --max-samples 1

echo "=== Done ==="
cat generations/test_qwen35_humaneval_shard000.json | python -c "
import json, sys
data = json.load(sys.stdin)
for entry in data:
    print('--- HumanEval sample', entry['sample_id'])
    print('raw_text:', entry['raw_text'][:300])
"
cat generations/test_qwen35_humaneval_fix_shard000.json | python -c "
import json, sys
data = json.load(sys.stdin)
for entry in data:
    print('--- HumanEval-Fix sample', entry['sample_id'])
    print('raw_text:', entry['raw_text'][:300])
"
