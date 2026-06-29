#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

CORPUS="data/corpus/flex_heldout256_stage2val_seed42.jsonl"
B0_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
F4B_CKPT="outputs/checkpoints/flex_f4b_paircontrast_cache_vis68_from_f3_s3000_lr5e6_gc5_20260606/final"
B0_PREFIX="outputs/reports/b0_step006250_flexheldout256"
RUN_NAME="flex_f25_f4b_margin_heldout256_s3000_lr1e6_m005_20260606"
LOG_PATH="outputs/logs/${RUN_NAME}_chain.log"

mkdir -p outputs/logs
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"f25_chain_start\",\"run_name\":\"${RUN_NAME}\",\"corpus\":\"${CORPUS}\"}"

COMMON_DECODE_ARGS=(
  --corpus-jsonl "${CORPUS}"
  --split val
  --num-samples 256
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

if [[ ! -f "${B0_PREFIX}_decode_normal_summary.json" ]]; then
  echo "{\"event\":\"f25_b0_decode_start\",\"mode\":\"normal\"}"
  .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
    "${COMMON_DECODE_ARGS[@]}" \
    --checkpoint-dir "${B0_CKPT}" \
    --output-dir "${B0_PREFIX}_decode_normal" \
    --summary-json "${B0_PREFIX}_decode_normal_summary.json"
  echo "{\"event\":\"f25_b0_decode_done\",\"mode\":\"normal\"}"
fi

if [[ ! -f "${B0_PREFIX}_decode_camera_shuffle_summary.json" ]]; then
  echo "{\"event\":\"f25_b0_decode_start\",\"mode\":\"camera_shuffle\"}"
  .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
    "${COMMON_DECODE_ARGS[@]}" \
    --checkpoint-dir "${B0_CKPT}" \
    --image-ablation camera_shuffle \
    --output-dir "${B0_PREFIX}_decode_camera_shuffle" \
    --summary-json "${B0_PREFIX}_decode_camera_shuffle_summary.json"
  echo "{\"event\":\"f25_b0_decode_done\",\"mode\":\"camera_shuffle\"}"
fi

echo "{\"event\":\"f25_train_start\"}"
.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl "${CORPUS}" \
  --teacher-checkpoint-dir "${B0_CKPT}" \
  --student-checkpoint-dir "${F4B_CKPT}" \
  --output-dir "outputs/checkpoints/${RUN_NAME}" \
  --split val \
  --max-train-samples 256 \
  --paired-ablation camera_shuffle \
  --max-steps 3000 \
  --batch-size 2 \
  --learning-rate 1e-6 \
  --grad-clip-norm 5.0 \
  --log-every 100 \
  --save-every 1500 \
  --seed 42 \
  --traj-kl-weight 1.0 \
  --text-kl-weight 0.20 \
  --format-kl-weight 0.05 \
  --boundary-cos-weight 0.05 \
  --boundary-norm-weight 0.10 \
  --pairwise-boundary-delta-cos-weight 0.05 \
  --pairwise-boundary-delta-norm-weight 0.05 \
  --pairwise-traj-logprob-delta-weight 0.10 \
  --pairwise-free-run-margin-weight 0.05 \
  --pairwise-free-run-margin 0.10 \
  --free-run-token-targets normal="${B0_PREFIX}_decode_normal_summary.json",camera_shuffle="${B0_PREFIX}_decode_camera_shuffle_summary.json" \
  --free-run-token-ce-weight 0.0 \
  --no-free-run-token-force-context \
  --cache-teacher-targets \
  --cache-collated-batches \
  --train-flex \
  --unfreeze-lora-last-n-layers 4 \
  --unfreeze-multimodal-projector \
  --summary-json "outputs/reports/${RUN_NAME}_train_summary.json"
echo "{\"event\":\"f25_train_done\"}"

F25_CKPT="outputs/checkpoints/${RUN_NAME}/final"

.venv/bin/python -u scripts/104_eval_flex_teacher_parity.py \
  --corpus-jsonl "${CORPUS}" \
  --teacher-checkpoint-dir "${B0_CKPT}" \
  --student-checkpoint-dir "${F25_CKPT}" \
  --split val \
  --num-samples 256 \
  --batch-size 1 \
  --summary-json "outputs/reports/${RUN_NAME}_eval_heldout256_summary.json"

.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  "${COMMON_DECODE_ARGS[@]}" \
  --checkpoint-dir "${F25_CKPT}" \
  --output-dir "outputs/reports/${RUN_NAME}_heldout256_decode_normal" \
  --summary-json "outputs/reports/${RUN_NAME}_heldout256_decode_normal_summary.json"

.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  "${COMMON_DECODE_ARGS[@]}" \
  --checkpoint-dir "${F25_CKPT}" \
  --image-ablation camera_shuffle \
  --output-dir "outputs/reports/${RUN_NAME}_heldout256_decode_camera_shuffle" \
  --summary-json "outputs/reports/${RUN_NAME}_heldout256_decode_camera_shuffle_summary.json"

.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  "${COMMON_DECODE_ARGS[@]}" \
  --checkpoint-dir "${F25_CKPT}" \
  --image-ablation black \
  --output-dir "outputs/reports/${RUN_NAME}_heldout256_decode_black" \
  --summary-json "outputs/reports/${RUN_NAME}_heldout256_decode_black_summary.json"

echo "{\"event\":\"f25_chain_done\",\"run_name\":\"${RUN_NAME}\"}"
