#!/bin/bash
#SBATCH -J pp-eval-qwen35
#SBATCH -p berzelius
#SBATCH --gpus=4
#SBATCH -t 04:00:00
#SBATCH -o logs/eval_passrate_qwen35_%j.out
#SBATCH -e logs/eval_passrate_qwen35_%j.err

set -euo pipefail
mkdir -p logs

module load buildenv-gcccuda/12.4.1-gcc13.3.0
unset CPATH
export LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"

RUN_ID="eval_qwen35_passrate"

echo "=== Generating HumanEval+ (n=1, all samples) ==="
uv run python run_generate.py \
  --model-config configs/models/qwen35_2b.yaml \
  --task-config configs/tasks/humaneval.yaml \
  --generation-config configs/generation_qwen35_test.yaml \
  --run-id "${RUN_ID}_humaneval"

echo "=== Generating HumanEval-Fix (n=1, all samples) ==="
uv run python run_generate.py \
  --model-config configs/models/qwen35_2b.yaml \
  --task-config configs/tasks/humaneval_fix.yaml \
  --generation-config configs/generation_qwen35_test.yaml \
  --run-id "${RUN_ID}_humaneval_fix"

echo "=== Computing pass rates ==="
uv run python - <<'EOF'
import json
from pathlib import Path
from src.tasks.humaneval import HumanEvalAdapter
from src.tasks.humaneval_fix import HumanEvalFixAdapter
from src.configs import TaskConfig

tasks = [
    (
        "HumanEval+",
        HumanEvalAdapter(),
        TaskConfig(dataset="humaneval", adapter="humaneval"),
        "generations/eval_qwen35_passrate_humaneval_shard000.json",
    ),
    (
        "HumanEval-Fix",
        HumanEvalFixAdapter(),
        TaskConfig(dataset="bigcode/humanevalpack", adapter="humaneval_fix"),
        "generations/eval_qwen35_passrate_humaneval_fix_shard000.json",
    ),
]

for label, adapter, task_config, gen_file in tasks:
    with open(gen_file) as f:
        entries = json.load(f)
    correct = sum(adapter.check_correct(e["raw_text"], e["task_sample"]) for e in entries)
    total = len(entries)
    print(f"{label}: {correct}/{total} = {correct/total:.1%}")
EOF
