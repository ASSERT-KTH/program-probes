#!/bin/bash
#SBATCH -J pp-finish-qwen3
#SBATCH -p berzelius-cpu
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o logs/finish_qwen3_%j.out
#SBATCH -e logs/finish_qwen3_%j.err

set -euo pipefail
mkdir -p logs
cd /proj/assert-berzelius/users/x_andaf/program-probes

# Find best hyperparameters from local W&B runs for sweep 67j274ft
BEST=$(uv run python3 -c "
import yaml, json, pathlib
best_loss = float('inf')
best_cfg = None
for cfg_path in pathlib.Path('wandb').glob('run-*/files/config.yaml'):
    cfg_text = cfg_path.read_text()
    if '67j274ft' not in cfg_text:
        continue
    summary_path = cfg_path.parent / 'wandb-summary.json'
    if not summary_path.exists():
        continue
    cfg = yaml.safe_load(cfg_text)
    summary = json.loads(summary_path.read_text())
    bin_losses = [v for k, v in summary.items() if k.endswith('/val_loss')]
    if not bin_losses:
        continue
    val_loss = sum(bin_losses) / len(bin_losses)
    if val_loss < best_loss:
        best_loss = val_loss
        best_cfg = cfg
if best_cfg:
    print(best_cfg['lr']['value'], best_cfg['weight_decay']['value'], int(best_cfg['batch_size']['value']), int(best_cfg['patience']['value']))
else:
    raise RuntimeError('No finished sweep runs found for 67j274ft')
")

LR=$(echo $BEST | awk '{print $1}')
WD=$(echo $BEST | awk '{print $2}')
BS=$(echo $BEST | awk '{print $3}')
PAT=$(echo $BEST | awk '{print $4}')
echo "Best params: lr=$LR wd=$WD bs=$BS patience=$PAT"

FINAL_JOB=$(sbatch --parsable \
  slurm/probe_final.sh \
  --run-id cruxeval_fix_qwen3_8b \
  --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml \
  final \
  --lr $LR --weight-decay $WD --batch-size $BS --patience $PAT)
echo "Final probe: $FINAL_JOB"

FIGURES_JOB=$(sbatch --parsable \
  --dependency=afterok:${FINAL_JOB} \
  slurm/figures.sh \
  --run-id cruxeval_fix_qwen3_8b \
  --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml)
echo "Figures: $FIGURES_JOB"

DASH_JOB=$(sbatch --parsable \
  --dependency=afterok:${FINAL_JOB} \
  slurm/export_dashboard.sh \
  --run-id cruxeval_fix_qwen3_8b \
  --probe will_be_correct \
  --model-config configs/models/qwen3_8b.yaml \
  --task-config configs/tasks/cruxeval_fix.yaml)
echo "Dashboard: $DASH_JOB"
