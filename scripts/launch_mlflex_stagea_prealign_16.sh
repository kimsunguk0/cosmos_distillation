#!/usr/bin/env bash
set -euo pipefail

CORPUS="${CORPUS:-data/corpus/flex_heldout256_stage2val_seed42.jsonl}"
B0_CKPT="${B0_CKPT:-outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250}"
MLFLEX_CKPT="${MLFLEX_CKPT:-outputs/checkpoints/mlflex_f0_k512_camtime_from_b0_20260608_smoke}"
RUN_NAME="${RUN_NAME:-mlflex_stagea_prealign16_s500_20260608}"
OUT_DIR="${OUT_DIR:-outputs/checkpoints/${RUN_NAME}}"
SUMMARY_JSON="${SUMMARY_JSON:-outputs/reports/${RUN_NAME}_summary.json}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-16}"
MAX_STEPS="${MAX_STEPS:-500}"

if [[ ! -d "${B0_CKPT}" ]]; then
  echo "{\"event\":\"missing_b0_checkpoint\",\"path\":\"${B0_CKPT}\"}" >&2
  exit 2
fi

if [[ ! -d "${MLFLEX_CKPT}" ]]; then
  echo "{\"event\":\"missing_mlflex_checkpoint\",\"path\":\"${MLFLEX_CKPT}\"}" >&2
  exit 2
fi

mkdir -p "$(dirname "${SUMMARY_JSON}")"

.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl "${CORPUS}" \
  --teacher-checkpoint-dir "${B0_CKPT}" \
  --student-checkpoint-dir "${MLFLEX_CKPT}" \
  --output-dir "${OUT_DIR}" \
  --split val \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --max-steps "${MAX_STEPS}" \
  --batch-size 1 \
  --learning-rate 1e-4 \
  --flex-lr 1e-4 \
  --traj-kl-weight 0.0 \
  --text-kl-weight 0.0 \
  --format-kl-weight 0.0 \
  --boundary-cos-weight 0.0 \
  --boundary-norm-weight 0.0 \
  --boundary-mse-weight 0.0 \
  --traj-state-cos-weight 0.0 \
  --traj-state-norm-weight 0.0 \
  --traj-state-mse-weight 0.0 \
  --cache-teacher-targets \
  --preserve-flex-positions \
  --flex-selection-strategy uniform \
  --flex-scene-deepstack \
  --image-feature-tokens-per-image 32 \
  --image-feature-mse-weight 1.0 \
  --image-feature-cos-weight 0.1 \
  --image-feature-norm-weight 0.05 \
  --deepstack-feature-tokens-per-image 32 \
  --deepstack-feature-mse-weight 1.0 \
  --deepstack-feature-cos-weight 0.1 \
  --deepstack-feature-norm-weight 0.05 \
  --train-flex \
  --save-every 100 \
  --summary-json "${SUMMARY_JSON}"
