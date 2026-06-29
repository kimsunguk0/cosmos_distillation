#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

CORPUS="data/corpus/flex_heldout256_stage2val_seed42.jsonl"
B0_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
PERIMAGE_F0="outputs/checkpoints/flex_f8_factorized_per_image_k896_camtime_untrained_from_b0_20260606"
B0_TRAJONLY="outputs/reports/b0_step006250_flexheldout16_decode_trajonly_summary.json"
RUN_NAME="flex_f35_trajstate_clean_f0_trajonly_overfit16_s3000_lr2e6_20260607"
LOG_PATH="outputs/logs/${RUN_NAME}_chain.log"

mkdir -p outputs/logs
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"f35_chain_start\",\"run_name\":\"${RUN_NAME}\",\"corpus\":\"${CORPUS}\"}"

COMMON_DECODE_ARGS=(
  --corpus-jsonl "${CORPUS}"
  --split val
  --num-samples 16
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

if [[ ! -f "${B0_TRAJONLY}" ]]; then
  echo "{\"event\":\"f35_b0_trajonly_decode_start\"}"
  .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
    "${COMMON_DECODE_ARGS[@]}" \
    --checkpoint-dir "${B0_CKPT}" \
    --output-dir "outputs/reports/b0_step006250_flexheldout16_decode_trajonly" \
    --summary-json "${B0_TRAJONLY}"
fi

.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl "${CORPUS}" \
  --teacher-checkpoint-dir "${B0_CKPT}" \
  --student-checkpoint-dir "${PERIMAGE_F0}" \
  --output-dir "outputs/checkpoints/${RUN_NAME}" \
  --split val \
  --max-train-samples 16 \
  --prompt-mode-override joint \
  --target-mode-override traj_only \
  --image-ablations normal \
  --paired-ablation none \
  --max-steps 3000 \
  --batch-size 1 \
  --learning-rate 2e-6 \
  --weight-decay 0.0 \
  --grad-clip-norm 5.0 \
  --log-every 50 \
  --save-every 1000 \
  --seed 42 \
  --traj-kl-weight 0.0 \
  --text-kl-weight 0.0 \
  --format-kl-weight 0.0 \
  --boundary-cos-weight 0.0 \
  --boundary-norm-weight 0.0 \
  --boundary-mse-weight 0.0 \
  --pairwise-boundary-delta-cos-weight 0.0 \
  --pairwise-boundary-delta-norm-weight 0.0 \
  --pairwise-traj-logprob-delta-weight 0.0 \
  --pairwise-free-run-margin-weight 0.0 \
  --free-run-token-targets normal="${B0_TRAJONLY}" \
  --free-run-token-ce-weight 1.0 \
  --free-run-token-ce-modes normal \
  --free-run-end-token-ce-weight 0.05 \
  --prefix-token-ce-weight 0.0 \
  --traj-state-cos-weight 0.5 \
  --traj-state-norm-weight 0.05 \
  --traj-state-mse-weight 0.0 \
  --free-run-token-force-context \
  --free-run-token-context-source target \
  --cache-teacher-targets \
  --cache-collated-batches \
  --train-flex \
  --unfreeze-lora-last-n-layers 4 \
  --unfreeze-multimodal-projector \
  --summary-json "outputs/reports/${RUN_NAME}_train_summary.json"

F35_CKPT="outputs/checkpoints/${RUN_NAME}/final"

echo "{\"event\":\"f35_decode_start\",\"mode\":\"trajonly_compressed\"}"
.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  "${COMMON_DECODE_ARGS[@]}" \
  --checkpoint-dir "${F35_CKPT}" \
  --output-dir "outputs/reports/${RUN_NAME}_overfit16_decode_trajonly" \
  --summary-json "outputs/reports/${RUN_NAME}_overfit16_decode_trajonly_summary.json"

.venv/bin/python -u scripts/109_compare_decode_to_free_run_targets.py \
  --decode-summary "outputs/reports/${RUN_NAME}_overfit16_decode_trajonly_summary.json" \
  --target-summary "${B0_TRAJONLY}" \
  --corpus-jsonl "${CORPUS}" \
  --split val \
  --num-samples 16 \
  --summary-json "outputs/reports/${RUN_NAME}_overfit16_b0_trajonly_parity_summary.json"

echo "{\"event\":\"f35_chain_done\",\"run_name\":\"${RUN_NAME}\"}"
