#!/usr/bin/env bash
# =====================================================================
#  Treino do MC3 v2 (40 epocas, two-stream) com VALIDACAO CRUZADA por fonte
#  StratifiedGroupKFold (7 folds, sem vazamento). Roda em GPU (CUDA).
#
#  Reproduz exatamente os hiperparametros de results/cv_mc3_v2_40ep/foldN/config.json.
#
#  Uso:
#     bash scripts/02_treinar_cv_mc3.sh <PASTA_DOS_CLIPES>
#  Ex.:
#     bash scripts/02_treinar_cv_mc3.sh data/analysis_data/clips
#
#  Ao final, para o numero honesto (media +- desvio da CV):
#     python scripts/04_resumo_cv.py
# =====================================================================
set -euo pipefail

DATA="${1:-data/analysis_data/clips}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$DATA" ]; then
  echo "ERRO: pasta de clipes nao encontrada: $DATA"
  echo "Coloque os clipes rotulados (nome contem a classe) nessa pasta. Veja data/README.md"
  exit 1
fi

# 1 GPU (evita o colapso de BatchNorm com DataParallel em batch pequeno)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for FOLD in 0 1 2 3 4 5 6; do
  echo "==================== FOLD ${FOLD} ===================="
  python "$ROOT/src/main.py" \
    --data_dir "$DATA" \
    --output_dir "$ROOT/results/cv_mc3_v2_40ep/fold${FOLD}" \
    --model mc3_18 --binary_tachi \
    --num_frames 64 --image_size 112 --batch_size 4 \
    --epochs_head 5 --epochs_finetune 40 \
    --lr_head 1e-4 --lr_backbone 1e-5 --weight_decay 0.01 \
    --seed 42 --balance_strategy class_weights --pos_weight_mode full \
    --patience 40 --grad_accum_steps 4 --scheduler cosine --warmup_epochs 2 \
    --freeze_early true --val_smooth_window 3 --label_smoothing 0.1 --dropout 0.4 \
    --spatial_mode full_frame \
    --input_modality two_stream --flow_bound 20.0 \
    --group_split --group_fold "${FOLD}" \
    --num_workers 4
done

echo "==================== RESUMO DA CV ===================="
python "$ROOT/scripts/04_resumo_cv.py"
