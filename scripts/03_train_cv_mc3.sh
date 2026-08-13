#!/usr/bin/env bash
# =====================================================================
#  MC3-18 (RGB-only) training with source-wise CROSS-VALIDATION
#  StratifiedGroupKFold (10 folds, leakage-free). Runs on GPU (CUDA).
#
#  Reproduces exactly the hyperparameters of results/models/foldN/config.json.
#  10 folds follows Kohavi (1995): best bias/variance trade-off for
#  cross-validation-based accuracy estimation.
#
#  Usage:
#     bash scripts/03_train_cv_mc3.sh <CLIPS_FOLDER>
#  Example:
#     bash scripts/03_train_cv_mc3.sh data/analysis_data/clips
#
#  At the end, for the honest number (CV mean +- std):
#     python scripts/04_cv_summary.py
# =====================================================================
set -euo pipefail

DATA="${1:-data/analysis_data/clips}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$DATA" ]; then
  echo "ERROR: clips folder not found: $DATA"
  echo "Place the labeled clips (filename contains the class) in that folder. See data/README.md"
  exit 1
fi

# 1 GPU (avoids BatchNorm collapse with DataParallel at small batch size)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for FOLD in 0 1 2 3 4 5 6 7 8 9; do
  echo "==================== FOLD ${FOLD} ===================="
  python "$ROOT/src/main.py" \
    --data_dir "$DATA" \
    --output_dir "$ROOT/results/models/fold${FOLD}" \
    --model mc3_18 --binary_tachi \
    --num_frames 64 --image_size 112 --batch_size 4 \
    --epochs_head 5 --epochs_finetune 40 \
    --lr_head 1e-4 --lr_backbone 1e-5 --weight_decay 0.01 \
    --seed 42 --balance_strategy class_weights --pos_weight_mode full \
    --patience 40 --grad_accum_steps 4 --scheduler cosine --warmup_epochs 2 \
    --freeze_early true --val_smooth_window 3 --label_smoothing 0.1 --dropout 0.4 \
    --spatial_mode full_frame \
    --input_modality rgb \
    --train_split 0.75 --val_split 0.15 --test_split 0.10 \
    --group_split --group_fold "${FOLD}" \
    --num_workers 4
done

echo "==================== CV SUMMARY ===================="
python "$ROOT/scripts/04_cv_summary.py"
