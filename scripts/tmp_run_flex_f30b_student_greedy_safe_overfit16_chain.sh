#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

CORPUS="data/corpus/flex_heldout256_stage2val_seed42.jsonl"
B0_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
F29_CKPT="outputs/checkpoints/flex_f29_normal_anchor_from_f28_overfit16_s3000_lr2e6_20260607/final"
B0_NORMAL="outputs/reports/b0_step006250_flexheldout256_decode_normal_summary.json"
RUN_NAME="flex_f30b_studentgreedy_safe_from_f29_overfit16_s2000_lr5e7_20260607"
LOG_PATH="outputs/logs/${RUN_NAME}_chain.log"

mkdir -p outputs/logs
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"f30b_chain_start\",\"run_name\":\"${RUN_NAME}\",\"corpus\":\"${CORPUS}\",\"b0_normal\":\"${B0_NORMAL}\"}"

.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl "${CORPUS}" \
  --teacher-checkpoint-dir "${B0_CKPT}" \
  --student-checkpoint-dir "${F29_CKPT}" \
  --output-dir "outputs/checkpoints/${RUN_NAME}" \
  --split val \
  --max-train-samples 16 \
  --image-ablations normal \
  --paired-ablation none \
  --max-steps 2000 \
  --batch-size 1 \
  --learning-rate 5e-7 \
  --weight-decay 0.0 \
  --grad-clip-norm 5.0 \
  --log-every 50 \
  --save-every 1000 \
  --seed 42 \
  --traj-kl-weight 0.0 \
  --text-kl-weight 0.02 \
  --format-kl-weight 0.02 \
  --boundary-cos-weight 0.0 \
  --boundary-norm-weight 0.0 \
  --boundary-mse-weight 0.0 \
  --pairwise-boundary-delta-cos-weight 0.0 \
  --pairwise-boundary-delta-norm-weight 0.0 \
  --pairwise-traj-logprob-delta-weight 0.0 \
  --pairwise-free-run-margin-weight 0.0 \
  --free-run-token-targets normal="${B0_NORMAL}" \
  --free-run-token-ce-weight 1.0 \
  --free-run-token-ce-modes normal \
  --free-run-end-token-ce-weight 0.10 \
  --prefix-token-ce-weight 0.10 \
  --free-run-token-context-source student_greedy \
  --student-greedy-context-refresh-steps 250 \
  --student-greedy-invalid-context target \
  --cache-teacher-targets \
  --cache-collated-batches \
  --train-flex \
  --unfreeze-lora-last-n-layers 4 \
  --unfreeze-multimodal-projector \
  --summary-json "outputs/reports/${RUN_NAME}_train_summary.json"

F30B_CKPT="outputs/checkpoints/${RUN_NAME}/final"

COMMON_DECODE_ARGS=(
  --corpus-jsonl "${CORPUS}"
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

echo "{\"event\":\"f30b_decode_start\",\"mode\":\"compressed\"}"
.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  "${COMMON_DECODE_ARGS[@]}" \
  --checkpoint-dir "${F30B_CKPT}" \
  --output-dir "outputs/reports/${RUN_NAME}_overfit16_decode_normal_compressed" \
  --summary-json "outputs/reports/${RUN_NAME}_overfit16_decode_normal_compressed_summary.json"

.venv/bin/python -u scripts/109_compare_decode_to_free_run_targets.py \
  --decode-summary "outputs/reports/${RUN_NAME}_overfit16_decode_normal_compressed_summary.json" \
  --target-summary "${B0_NORMAL}" \
  --corpus-jsonl "${CORPUS}" \
  --split val \
  --num-samples 16 \
  --summary-json "outputs/reports/${RUN_NAME}_overfit16_b0_parity_compressed_summary.json"

echo "{\"event\":\"f30b_chain_done\",\"run_name\":\"${RUN_NAME}\"}"
