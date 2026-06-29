#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN_NAME="flex_f17_pairdelta_lora4_from_f8_ablation16_s1200_lr2e6_20260606_step600"
CKPT_DIR="outputs/checkpoints/flex_f17_pairdelta_lora4_from_f8_ablation16_s1200_lr2e6_20260606/step_000600"
LOG_PATH="outputs/logs/${RUN_NAME}_eval.log"
mkdir -p outputs/logs
exec > >(tee -a "${LOG_PATH}") 2>&1

.venv/bin/python -u scripts/104_eval_flex_teacher_parity.py \
  --corpus-jsonl data/corpus/vis_4per_category_val.jsonl \
  --teacher-checkpoint-dir outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250 \
  --student-checkpoint-dir "${CKPT_DIR}" \
  --split val \
  --num-samples 16 \
  --batch-size 1 \
  --summary-json "outputs/reports/${RUN_NAME}_eval_vis16_summary.json"

COMMON_ARGS=(
  --corpus-jsonl data/corpus/vis_4per_category_val.jsonl
  --checkpoint-dir "${CKPT_DIR}"
  --split val
  --num-samples 16
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
  --output-dir "outputs/reports/${RUN_NAME}_vis16_decode_normal" \
  --summary-json "outputs/reports/${RUN_NAME}_vis16_decode_normal_summary.json"

.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  "${COMMON_ARGS[@]}" \
  --image-ablation camera_shuffle \
  --output-dir "outputs/reports/${RUN_NAME}_vis16_decode_camera_shuffle" \
  --summary-json "outputs/reports/${RUN_NAME}_vis16_decode_camera_shuffle_summary.json"

.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  "${COMMON_ARGS[@]}" \
  --image-ablation black \
  --output-dir "outputs/reports/${RUN_NAME}_vis16_decode_black" \
  --summary-json "outputs/reports/${RUN_NAME}_vis16_decode_black_summary.json"

echo "{\"event\":\"step600_eval_done\",\"run_name\":\"${RUN_NAME}\"}"
