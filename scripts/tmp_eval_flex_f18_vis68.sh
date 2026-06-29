#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN_NAME="flex_f18_lowce_lora4_from_f8_ablation16_s1200_lr2e6_ce05_20260606"
CKPT_DIR="outputs/checkpoints/${RUN_NAME}/final"
LOG_PATH="outputs/logs/${RUN_NAME}_vis68_eval.log"
mkdir -p outputs/logs
exec > >(tee -a "${LOG_PATH}") 2>&1

COMMON_ARGS=(
  --corpus-jsonl data/corpus/vis_4per_category_val.jsonl
  --checkpoint-dir "${CKPT_DIR}"
  --split val
  --num-samples 68
  --max-new-tokens 192
  --prompt-mode joint
  --target-mode joint
  --image-prompt-style camera_labeled
  --prompt-text-style official_alpamayo
  --fuse-history-tokens
  --geometry-reference teacher
  --batch-size 1
  --samples-per-row 1
  --skip-overlays
  --disable-failure-tags
)

.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  "${COMMON_ARGS[@]}" \
  --output-dir "outputs/reports/${RUN_NAME}_vis68_decode_normal" \
  --summary-json "outputs/reports/${RUN_NAME}_vis68_decode_normal_summary.json"

.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  "${COMMON_ARGS[@]}" \
  --image-ablation camera_shuffle \
  --output-dir "outputs/reports/${RUN_NAME}_vis68_decode_camera_shuffle" \
  --summary-json "outputs/reports/${RUN_NAME}_vis68_decode_camera_shuffle_summary.json"

.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  "${COMMON_ARGS[@]}" \
  --image-ablation black \
  --output-dir "outputs/reports/${RUN_NAME}_vis68_decode_black" \
  --summary-json "outputs/reports/${RUN_NAME}_vis68_decode_black_summary.json"

echo "{\"event\":\"vis68_eval_done\",\"run_name\":\"${RUN_NAME}\"}"
