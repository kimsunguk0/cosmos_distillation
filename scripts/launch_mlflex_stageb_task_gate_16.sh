#!/usr/bin/env bash
set -euo pipefail

CORPUS="${CORPUS:-data/corpus/flex_heldout256_stage2val_seed42.jsonl}"
B0_CKPT="${B0_CKPT:-outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250}"
MLFLEX_INIT="${MLFLEX_INIT:-outputs/checkpoints/mlflex_stagea_prealign16_s500_20260608/final}"
RUN_NAME="${RUN_NAME:-mlflex_stageb_task_gate16_s500_20260608}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-16}"
MAX_STEPS="${MAX_STEPS:-500}"

OUT_DIR="outputs/checkpoints/${RUN_NAME}"
SUMMARY_JSON="outputs/reports/${RUN_NAME}_summary.json"

.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl "${CORPUS}" \
  --teacher-checkpoint-dir "${B0_CKPT}" \
  --student-checkpoint-dir "${MLFLEX_INIT}" \
  --output-dir "${OUT_DIR}" \
  --split val \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --max-steps "${MAX_STEPS}" \
  --batch-size 1 \
  --learning-rate 1e-6 \
  --flex-lr 5e-5 \
  --lora-lr 1e-6 \
  --traj-kl-weight 1.0 \
  --text-kl-weight 0.05 \
  --format-kl-weight 0.05 \
  --boundary-cos-weight 0.02 \
  --boundary-norm-weight 0.02 \
  --boundary-mse-weight 0.0 \
  --traj-state-cos-weight 0.0 \
  --traj-state-norm-weight 0.0 \
  --traj-state-mse-weight 0.0 \
  --cache-teacher-targets \
  --preserve-flex-positions \
  --flex-selection-strategy uniform \
  --flex-scene-deepstack \
  --image-feature-tokens-per-image 32 \
  --image-feature-mse-weight 0.2 \
  --image-feature-cos-weight 0.02 \
  --image-feature-norm-weight 0.01 \
  --deepstack-feature-tokens-per-image 32 \
  --deepstack-feature-mse-weight 0.2 \
  --deepstack-feature-cos-weight 0.02 \
  --deepstack-feature-norm-weight 0.01 \
  --train-flex \
  --unfreeze-all-lora \
  --save-every 100 \
  --summary-json "${SUMMARY_JSON}"
