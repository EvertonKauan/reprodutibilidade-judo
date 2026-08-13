# =====================================================================
#  MC3-18 (RGB-only) training with source-wise CROSS-VALIDATION
#  StratifiedGroupKFold (10 folds, leakage-free). Runs on GPU (CUDA).
#
#  Reproduces exactly the hyperparameters of results/models/mc3_18_rgb/foldN/config.json.
#  10 folds follows Kohavi (1995): best bias/variance trade-off for
#  cross-validation-based accuracy estimation.
#
#  Usage:
#     powershell -ExecutionPolicy Bypass -File scripts\03_train_cv_mc3.ps1 <CLIPS_FOLDER>
#  Example:
#     powershell -ExecutionPolicy Bypass -File scripts\03_train_cv_mc3.ps1 data\analysis_data\clips
#
#  At the end, for the honest number (CV mean +- std):
#     python scripts\04_cv_summary.py
# =====================================================================

param(
    [string]$Data = "data\analysis_data\clips"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $Data)) {
    Write-Error "Clips folder not found: $Data. Place the labeled clips (filename contains the class) in that folder. See data\README.md"
    exit 1
}

if (-not $env:CUDA_VISIBLE_DEVICES) {
    $env:CUDA_VISIBLE_DEVICES = "0"
}

foreach ($Fold in 0..9) {
    Write-Host "==================== FOLD $Fold ===================="
    python "$Root\src\main.py" `
        --data_dir "$Data" `
        --output_dir "$Root\results\models\mc3_18_rgb\fold$Fold" `
        --model mc3_18 --binary_tachi `
        --num_frames 64 --image_size 112 --batch_size 4 `
        --epochs_head 5 --epochs_finetune 40 `
        --lr_head 1e-4 --lr_backbone 1e-5 --weight_decay 0.01 `
        --seed 42 --balance_strategy class_weights --pos_weight_mode full `
        --patience 40 --grad_accum_steps 4 --scheduler cosine --warmup_epochs 2 `
        --freeze_early true --val_smooth_window 3 --label_smoothing 0.1 --dropout 0.4 `
        --spatial_mode full_frame `
        --input_modality rgb `
        --train_split 0.75 --val_split 0.15 --test_split 0.10 `
        --group_split --group_fold $Fold `
        --num_workers 4
}

Write-Host "==================== CV SUMMARY ===================="
python "$Root\scripts\04_cv_summary.py"
