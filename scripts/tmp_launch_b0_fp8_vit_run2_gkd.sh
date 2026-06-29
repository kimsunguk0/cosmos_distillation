#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN_ID="run2_20k_fp8vit_gkd_from_step006250_val512_b8"
RUN_ROOT="b0_fp8_vit_step006250_20260618"
OUT_DIR="outputs/checkpoints/${RUN_ROOT}/${RUN_ID}"
SUMMARY_JSON="outputs/reports/${RUN_ROOT}/${RUN_ID}_summary.json"
LOG_PATH="logs/${RUN_ROOT}/${RUN_ID}.train.log"
INIT_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
CORPUS="data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl"
CONFIG="configs/train/stage_b0_fp8_old_recipe_20k_gkd.yaml"

mkdir -p "logs/${RUN_ROOT}" "outputs/reports/${RUN_ROOT}" "${OUT_DIR}" /tmp/triton-cache
exec > >(tee -a "${LOG_PATH}") 2>&1

export PATH="/home/pm97/workspace/sukim/distillation/cosmos_distillation/.venv/bin:${PATH}"
export TRITON_CACHE_DIR="/tmp/triton-cache"
export COSMOS_DATA_ROOT="/home/pm97/workspace/dataset/distill_dataset"
export COSMOS_STUDENT_MODEL="/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b"
export ALPAMAYO_MODEL_PATH="/home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B"
export ALPAMAYO_SRC="/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/src"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export COSMOS_DATALOADER_NUM_WORKERS="4"

echo "START ${RUN_ID} $(date -Is)"
echo "config=${CONFIG}"
echo "init=${INIT_CKPT}"
echo "output=${OUT_DIR}"
echo "summary=${SUMMARY_JSON}"

.venv/bin/python -u scripts/09_train_distill.py \
  --corpus-jsonl "${CORPUS}" \
  --stage-config "${CONFIG}" \
  --student-model /home/pm97/workspace/sukim/base_weights/cosmos-reason-2b \
  --batch-size 8 \
  --eval-every-epochs 0.5 \
  --save-every-epochs 0.5 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --output-dir "${OUT_DIR}" \
  --summary-json "${SUMMARY_JSON}" \
  --pin-memory \
  --persistent-workers \
  --skip-asset-check \
  --init-checkpoint-dir "${INIT_CKPT}" \
  --max-steps 2500 \
  --epochs 1.0 \
  --max-train-samples 20000 \
  --max-val-samples 512 \
  --multi-gpu off \
  --log-every-steps 10 \
  --qat-quantization fp8_pcpt_vit \
  --qat-calib-samples 128

echo "END ${RUN_ID} $(date -Is)"
