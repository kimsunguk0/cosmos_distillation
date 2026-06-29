#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN_NAME="flex_f19_pairdelta_from_f18_ablation16_s600_lr1e6_ce02_20260606"
CKPT_DIR="outputs/checkpoints/${RUN_NAME}/final"
LOG_PATH="outputs/logs/${RUN_NAME}_posteval.log"
mkdir -p outputs/logs
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"posteval_wait_start\",\"run_name\":\"${RUN_NAME}\",\"ckpt_dir\":\"${CKPT_DIR}\"}"

while [[ ! -f "${CKPT_DIR}/train_config.json" ]]; do
  sleep 30
done

echo "{\"event\":\"posteval_checkpoint_ready\",\"ckpt_dir\":\"${CKPT_DIR}\"}"

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

echo "{\"event\":\"posteval_done\",\"run_name\":\"${RUN_NAME}\"}"
