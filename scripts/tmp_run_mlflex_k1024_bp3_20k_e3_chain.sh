#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN_TAG="${RUN_TAG:-20260608}"
B0_CKPT="${B0_CKPT:-outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250}"
F0_DIR="${F0_DIR:-outputs/checkpoints/mlflex_f0_k1024_camtime_from_b0_${RUN_TAG}}"
PREALIGN_RUN="${PREALIGN_RUN:-mlflex_k1024_stagea_prealign16_s500_${RUN_TAG}}"
PREALIGN_DIR="${PREALIGN_DIR:-outputs/checkpoints/${PREALIGN_RUN}}"
MAIN_RUN="${MAIN_RUN:-mlflex_k1024_bp3_20k_e3_b16_${RUN_TAG}}"
MAIN_DIR="${MAIN_DIR:-outputs/checkpoints/${MAIN_RUN}}"
CORPUS_20K="${CORPUS_20K:-data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl}"
PREFLIGHT_CORPUS="${PREFLIGHT_CORPUS:-data/corpus/flex_heldout256_stage2val_seed42.jsonl}"
CONFIG="${CONFIG:-configs/train/stage_mlflex_k1024_bp3_hidden_gc_20k_e3.yaml}"
LOG_PATH="${LOG_PATH:-outputs/logs/${MAIN_RUN}_chain.log}"

mkdir -p outputs/logs outputs/reports outputs/checkpoints
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"mlflex_k1024_chain_start\",\"time\":\"$(date -Is)\",\"run\":\"${MAIN_RUN}\",\"f0_dir\":\"${F0_DIR}\",\"prealign_dir\":\"${PREALIGN_DIR}\",\"main_dir\":\"${MAIN_DIR}\"}"

if [[ ! -d "${F0_DIR}" ]]; then
  echo "{\"event\":\"create_f0_start\",\"time\":\"$(date -Is)\",\"output_dir\":\"${F0_DIR}\"}"
  .venv/bin/python -u scripts/103_make_flex_untrained_checkpoint.py \
    --base-checkpoint-dir "${B0_CKPT}" \
    --output-dir "${F0_DIR}" \
    --architecture multi_level \
    --tokens-per-image 64 \
    --expected-images-per-sample 16 \
    --input-hidden-size 2048 \
    --hidden-size 1024 \
    --num-layers 1 \
    --num-heads 8 \
    --mlp-ratio 4.0 \
    --dropout 0.0 \
    --use-camera-time-embeddings \
    --use-local-slot-embeddings \
    --max-camera-types 16 \
    --num-deepstack-levels 3 \
    --compression-mode per_image \
    --selection-strategy uniform
  echo "{\"event\":\"create_f0_done\",\"time\":\"$(date -Is)\",\"output_dir\":\"${F0_DIR}\"}"
else
  echo "{\"event\":\"create_f0_skip_exists\",\"time\":\"$(date -Is)\",\"output_dir\":\"${F0_DIR}\"}"
fi

.venv/bin/python -u scripts/111_mlflex_forward_smoke.py \
  --checkpoint-dir "${F0_DIR}" \
  --corpus-jsonl "${PREFLIGHT_CORPUS}" \
  --split val \
  --sample-index 0 \
  --summary-json "outputs/reports/mlflex_f0_k1024_forward_smoke_summary.json"

if [[ ! -d "${PREALIGN_DIR}/final" ]]; then
  echo "{\"event\":\"prealign_start\",\"time\":\"$(date -Is)\",\"output_dir\":\"${PREALIGN_DIR}\"}"
  .venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
    --corpus-jsonl "${PREFLIGHT_CORPUS}" \
    --teacher-checkpoint-dir "${B0_CKPT}" \
    --student-checkpoint-dir "${F0_DIR}" \
    --output-dir "${PREALIGN_DIR}" \
    --split val \
    --max-train-samples 16 \
    --max-steps 500 \
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
    --image-feature-tokens-per-image 64 \
    --image-feature-mse-weight 1.0 \
    --image-feature-cos-weight 0.1 \
    --image-feature-norm-weight 0.05 \
    --deepstack-feature-tokens-per-image 64 \
    --deepstack-feature-mse-weight 1.0 \
    --deepstack-feature-cos-weight 0.1 \
    --deepstack-feature-norm-weight 0.05 \
    --train-flex \
    --save-every 100 \
    --summary-json "outputs/reports/${PREALIGN_RUN}_summary.json"
  echo "{\"event\":\"prealign_done\",\"time\":\"$(date -Is)\",\"output_dir\":\"${PREALIGN_DIR}\"}"
else
  echo "{\"event\":\"prealign_skip_exists\",\"time\":\"$(date -Is)\",\"output_dir\":\"${PREALIGN_DIR}\"}"
fi

.venv/bin/python -u scripts/111_mlflex_forward_smoke.py \
  --checkpoint-dir "${PREALIGN_DIR}/final" \
  --corpus-jsonl "${PREFLIGHT_CORPUS}" \
  --split val \
  --sample-index 0 \
  --summary-json "outputs/reports/${PREALIGN_RUN}_forward_smoke_summary.json"

if [[ ! -d "${MAIN_DIR}/final" ]]; then
  echo "{\"event\":\"main_train_start\",\"time\":\"$(date -Is)\",\"output_dir\":\"${MAIN_DIR}\",\"config\":\"${CONFIG}\"}"
  .venv/bin/python -u scripts/09_train_distill.py \
    --corpus-jsonl "${CORPUS_20K}" \
    --stage-config "${CONFIG}" \
    --student-model /home/pm97/workspace/sukim/base_weights/cosmos-reason-2b \
    --init-checkpoint-dir "${PREALIGN_DIR}/final" \
    --max-train-samples 20000 \
    --max-val-samples 512 \
    --batch-size 16 \
    --epochs 3.0 \
    --eval-every-epochs 0.5 \
    --save-every-epochs 1.0 \
    --log-every-steps 1 \
    --output-dir "${MAIN_DIR}" \
    --summary-json "outputs/reports/${MAIN_RUN}_summary.json"
  echo "{\"event\":\"main_train_done\",\"time\":\"$(date -Is)\",\"output_dir\":\"${MAIN_DIR}\"}"
else
  echo "{\"event\":\"main_train_skip_exists\",\"time\":\"$(date -Is)\",\"output_dir\":\"${MAIN_DIR}\"}"
fi

echo "{\"event\":\"mlflex_k1024_chain_done\",\"time\":\"$(date -Is)\",\"run\":\"${MAIN_RUN}\",\"summary\":\"outputs/reports/${MAIN_RUN}_summary.json\"}"
