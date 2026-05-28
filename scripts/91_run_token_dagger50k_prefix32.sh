#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN_ID="no_nav_token_dagger50k_prefix32_b16_$(date +%Y%m%d_%H%M%S)"
CORPUS="data/corpus/no_nav_token_dagger50k_prefix32_b16.jsonl"
REPORT="outputs/reports/no_nav_distill/token_dagger50k_prefix32_b16.json"
OUTDIR="outputs/checkpoints/no_nav_token_dagger/${RUN_ID}"
SUMMARY="outputs/reports/no_nav_distill/${RUN_ID}_summary.json"
BASE_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_200k/no_nav_bp3_camera_labeled_official_200k_from_20kbest_20260514_092509/best_decode"
CONFIG="configs/train/stage_bp3_no_nav_token_dagger_prefix32.yaml"

echo "[token-dagger] run_id=${RUN_ID}"
echo "[token-dagger] build corpus -> ${CORPUS}"
.venv/bin/python -u scripts/90_build_token_dagger_corpus.py \
  --corpus-jsonl data/corpus/no_nav_teacher_pair_300chunks_semantic_balanced_50k.jsonl \
  --student-checkpoint-dir "${BASE_CKPT}" \
  --teacher-model-path /home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B \
  --alpamayo-src /home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/src \
  --split train \
  --max-samples 50000 \
  --prefix-tokens 32 \
  --batch-size 16 \
  --log-every 160 \
  --output-jsonl "${CORPUS}" \
  --report-json "${REPORT}"

echo "[token-dagger] corpus line count"
wc -l "${CORPUS}"

echo "[token-dagger] train -> ${OUTDIR}"
.venv/bin/python -u scripts/09_train_distill.py \
  --corpus-jsonl "${CORPUS}" \
  --stage-config "${CONFIG}" \
  --init-checkpoint-dir "${BASE_CKPT}" \
  --batch-size 16 \
  --max-train-samples 50000 \
  --skip-asset-check \
  --output-dir "${OUTDIR}" \
  --summary-json "${SUMMARY}" \
  --log-every-steps 10

echo "[token-dagger] done run_id=${RUN_ID}"
