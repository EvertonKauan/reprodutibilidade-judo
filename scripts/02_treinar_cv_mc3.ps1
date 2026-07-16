# =====================================================================
#  Treino do MC3 v2 (40 epocas, two-stream) com VALIDACAO CRUZADA por fonte
#  StratifiedGroupKFold (7 folds, sem vazamento). Roda em GPU (CUDA).  [Windows]
#
#  Uso:
#     powershell -ExecutionPolicy Bypass -File scripts\02_treinar_cv_mc3.ps1 <PASTA_DOS_CLIPES>
#  Ex.:
#     powershell -ExecutionPolicy Bypass -File scripts\02_treinar_cv_mc3.ps1 data\input_data\clips
# =====================================================================
param([string]$Data = "data\analysis_data\clips")

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent

if (-not (Test-Path $Data)) {
    Write-Host "ERRO: pasta de clipes nao encontrada: $Data" -ForegroundColor Red
    Write-Host "Coloque os clipes rotulados nessa pasta. Veja data\README.md"
    exit 1
}

if (-not $env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES = "0" }

foreach ($Fold in 0..6) {
    Write-Host "==================== FOLD $Fold ====================" -ForegroundColor Cyan
    python (Join-Path $Root "src\main.py") `
        --data_dir $Data `
        --output_dir (Join-Path $Root "results\cv_mc3_v2_40ep\fold$Fold") `
        --model mc3_18 --binary_tachi `
        --num_frames 64 --image_size 112 --batch_size 4 `
        --epochs_head 5 --epochs_finetune 40 `
        --lr_head 1e-4 --lr_backbone 1e-5 --weight_decay 0.01 `
        --seed 42 --balance_strategy class_weights --pos_weight_mode full `
        --patience 40 --grad_accum_steps 4 --scheduler cosine --warmup_epochs 2 `
        --freeze_early true --val_smooth_window 3 --label_smoothing 0.1 --dropout 0.4 `
        --spatial_mode full_frame `
        --input_modality two_stream --flow_bound 20.0 `
        --group_split --group_fold $Fold `
        --num_workers 4
    if ($LASTEXITCODE -ne 0) { throw "Falha no fold $Fold" }
}

Write-Host "==================== RESUMO DA CV ====================" -ForegroundColor Cyan
python (Join-Path $Root "scripts\04_resumo_cv.py")
