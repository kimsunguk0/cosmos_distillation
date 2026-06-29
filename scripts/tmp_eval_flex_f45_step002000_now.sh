#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

CORPUS="data/corpus/flex_heldout256_stage2val_seed42.jsonl"
B0_TRAJONLY="outputs/reports/b0_step006250_flexheldout256_decode_trajonly_summary.json"
RUN_NAME="flex_f45_nods_free_run_target256_from_f42_s8000_lr2e7_20260607"
STEP="${STEP:-step_002000}"
CKPT="outputs/checkpoints/${RUN_NAME}/${STEP}"
LOG_PATH="outputs/logs/${RUN_NAME}_${STEP}_eval_now.log"
DECODE_SUMMARY="outputs/reports/${RUN_NAME}_${STEP}_decode_trajonly_summary.json"
PARITY_SUMMARY="outputs/reports/${RUN_NAME}_${STEP}_b0_trajonly_parity_summary.json"

mkdir -p outputs/logs outputs/reports
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"f45_step_eval_now_start\",\"checkpoint\":\"${CKPT}\"}"

.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  --corpus-jsonl "${CORPUS}" \
  --split val \
  --num-samples 256 \
  --max-new-tokens 160 \
  --prompt-mode joint \
  --target-mode traj_only \
  --image-prompt-style camera_labeled \
  --prompt-text-style official_alpamayo \
  --fuse-history-tokens \
  --geometry-reference teacher \
  --batch-size 1 \
  --samples-per-row 1 \
  --skip-overlays \
  --disable-failure-tags \
  --checkpoint-dir "${CKPT}" \
  --output-dir "outputs/reports/${RUN_NAME}_${STEP}_decode_trajonly" \
  --summary-json "${DECODE_SUMMARY}"

.venv/bin/python -u scripts/109_compare_decode_to_free_run_targets.py \
  --decode-summary "${DECODE_SUMMARY}" \
  --target-summary "${B0_TRAJONLY}" \
  --corpus-jsonl "${CORPUS}" \
  --split val \
  --num-samples 256 \
  --summary-json "${PARITY_SUMMARY}"

echo "{\"event\":\"f45_step_eval_now_done\",\"checkpoint\":\"${CKPT}\",\"summary\":\"${PARITY_SUMMARY}\"}"
