#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

MAIN_SESSION="flex_f45_nods256"
CORPUS="data/corpus/flex_heldout256_stage2val_seed42.jsonl"
B0_TRAJONLY="outputs/reports/b0_step006250_flexheldout256_decode_trajonly_summary.json"
RUN_NAME="flex_f45_nods_free_run_target256_from_f42_s8000_lr2e7_20260607"
CKPT_ROOT="outputs/checkpoints/${RUN_NAME}"
LOG_PATH="outputs/logs/${RUN_NAME}_checkpoint_eval_wait.log"

mkdir -p outputs/logs outputs/reports
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"f45_checkpoint_eval_wait_start\",\"main_session\":\"${MAIN_SESSION}\"}"

while tmux has-session -t "${MAIN_SESSION}" 2>/dev/null; do
  echo "{\"event\":\"f45_checkpoint_eval_waiting\",\"main_session\":\"${MAIN_SESSION}\"}"
  sleep 300
done

echo "{\"event\":\"f45_checkpoint_eval_begin\",\"run_name\":\"${RUN_NAME}\"}"

COMMON_DECODE_ARGS=(
  --corpus-jsonl "${CORPUS}"
  --split val
  --num-samples 256
  --max-new-tokens 160
  --prompt-mode joint
  --target-mode traj_only
  --image-prompt-style camera_labeled
  --prompt-text-style official_alpamayo
  --fuse-history-tokens
  --geometry-reference teacher
  --batch-size 1
  --samples-per-row 1
  --skip-overlays
  --disable-failure-tags
)

for STEP in step_002000 step_004000 step_006000 step_008000 final; do
  CKPT="${CKPT_ROOT}/${STEP}"
  if [[ ! -d "${CKPT}" ]]; then
    echo "{\"event\":\"f45_checkpoint_eval_skip_missing\",\"checkpoint\":\"${CKPT}\"}"
    continue
  fi
  DECODE_SUMMARY="outputs/reports/${RUN_NAME}_${STEP}_decode_trajonly_summary.json"
  PARITY_SUMMARY="outputs/reports/${RUN_NAME}_${STEP}_b0_trajonly_parity_summary.json"
  if [[ ! -f "${DECODE_SUMMARY}" ]]; then
    echo "{\"event\":\"f45_checkpoint_decode_start\",\"checkpoint\":\"${CKPT}\",\"summary\":\"${DECODE_SUMMARY}\"}"
    .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
      "${COMMON_DECODE_ARGS[@]}" \
      --checkpoint-dir "${CKPT}" \
      --output-dir "outputs/reports/${RUN_NAME}_${STEP}_decode_trajonly" \
      --summary-json "${DECODE_SUMMARY}"
  else
    echo "{\"event\":\"f45_checkpoint_decode_reuse\",\"checkpoint\":\"${CKPT}\",\"summary\":\"${DECODE_SUMMARY}\"}"
  fi
  echo "{\"event\":\"f45_checkpoint_compare_start\",\"checkpoint\":\"${CKPT}\",\"summary\":\"${PARITY_SUMMARY}\"}"
  .venv/bin/python -u scripts/109_compare_decode_to_free_run_targets.py \
    --decode-summary "${DECODE_SUMMARY}" \
    --target-summary "${B0_TRAJONLY}" \
    --corpus-jsonl "${CORPUS}" \
    --split val \
    --num-samples 256 \
    --summary-json "${PARITY_SUMMARY}"
done

echo "{\"event\":\"f45_checkpoint_eval_done\",\"run_name\":\"${RUN_NAME}\"}"
