#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

CORPUS="data/corpus/flex_heldout256_stage2val_seed42.jsonl"
B0_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
F29_CKPT="outputs/checkpoints/flex_f29_normal_anchor_from_f28_overfit16_s3000_lr2e6_20260607/final"
F30B_CKPT="outputs/checkpoints/flex_f30b_studentgreedy_safe_from_f29_overfit16_s2000_lr5e7_20260607/final"
B0_NORMAL="outputs/reports/b0_step006250_flexheldout256_decode_normal_summary.json"
RUN_NAME="flex_f31_trajonly_oraclecot_overfit16_20260607"
LOG_PATH="outputs/logs/${RUN_NAME}.log"

mkdir -p outputs/logs
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"f31_start\",\"run_name\":\"${RUN_NAME}\"}"

COMMON_DECODE_ARGS=(
  --corpus-jsonl "${CORPUS}"
  --split val
  --num-samples 16
  --max-new-tokens 160
  --prompt-mode joint
  --target-mode traj_only
  --oracle-cot-prefix
  --image-prompt-style camera_labeled
  --prompt-text-style official_alpamayo
  --fuse-history-tokens
  --geometry-reference teacher
  --batch-size 1
  --samples-per-row 1
  --skip-overlays
  --disable-failure-tags
)

for label in b0 f29 f30b; do
  case "${label}" in
    b0) ckpt="${B0_CKPT}" ;;
    f29) ckpt="${F29_CKPT}" ;;
    f30b) ckpt="${F30B_CKPT}" ;;
  esac
  echo "{\"event\":\"f31_decode_start\",\"label\":\"${label}\",\"ckpt\":\"${ckpt}\"}"
  .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
    "${COMMON_DECODE_ARGS[@]}" \
    --checkpoint-dir "${ckpt}" \
    --output-dir "outputs/reports/${RUN_NAME}_${label}_decode_trajonly_oraclecot" \
    --summary-json "outputs/reports/${RUN_NAME}_${label}_decode_trajonly_oraclecot_summary.json"

  .venv/bin/python -u scripts/109_compare_decode_to_free_run_targets.py \
    --decode-summary "outputs/reports/${RUN_NAME}_${label}_decode_trajonly_oraclecot_summary.json" \
    --target-summary "${B0_NORMAL}" \
    --corpus-jsonl "${CORPUS}" \
    --split val \
    --num-samples 16 \
    --summary-json "outputs/reports/${RUN_NAME}_${label}_b0_joint_target_parity_summary.json"
done

echo "{\"event\":\"f31_done\",\"run_name\":\"${RUN_NAME}\"}"
