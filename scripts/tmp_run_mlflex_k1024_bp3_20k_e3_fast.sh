#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN_TAG="${RUN_TAG:-20260609_fast}"
PREALIGN_DIR="${PREALIGN_DIR:-outputs/checkpoints/mlflex_k1024_stagea_prealign16_s500_20260608}"
MAIN_RUN="${MAIN_RUN:-mlflex_k1024_bp3_20k_e3_b16_fast_${RUN_TAG}}"
MAIN_DIR="${MAIN_DIR:-outputs/checkpoints/${MAIN_RUN}}"
CORPUS_20K="${CORPUS_20K:-data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl}"
CONFIG="${CONFIG:-configs/train/stage_mlflex_k1024_bp3_hidden_gc_20k_e3.yaml}"
SUMMARY_JSON="${SUMMARY_JSON:-outputs/reports/${MAIN_RUN}_summary.json}"
LOG_PATH="${LOG_PATH:-outputs/logs/${MAIN_RUN}.log}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-100}"

mkdir -p outputs/logs outputs/reports outputs/checkpoints
exec > >(tee -a "${LOG_PATH}") 2>&1

if [[ "${FORCE_RESTART:-0}" == "1" ]]; then
  rm -rf "${MAIN_DIR}" "${SUMMARY_JSON}"
fi

if [[ ! -d "${PREALIGN_DIR}/final" ]]; then
  echo "{\"event\":\"missing_prealign\",\"time\":\"$(date -Is)\",\"prealign_dir\":\"${PREALIGN_DIR}\"}"
  exit 1
fi

echo "{\"event\":\"mlflex_k1024_fast_train_start\",\"time\":\"$(date -Is)\",\"run\":\"${MAIN_RUN}\",\"main_dir\":\"${MAIN_DIR}\",\"config\":\"${CONFIG}\",\"num_workers\":${NUM_WORKERS},\"prefetch_factor\":${PREFETCH_FACTOR},\"log_every_steps\":${LOG_EVERY_STEPS}}"

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
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --pin-memory \
  --persistent-workers \
  --log-every-steps "${LOG_EVERY_STEPS}" \
  --output-dir "${MAIN_DIR}" \
  --summary-json "${SUMMARY_JSON}"

echo "{\"event\":\"mlflex_k1024_fast_train_done\",\"time\":\"$(date -Is)\",\"run\":\"${MAIN_RUN}\",\"summary\":\"${SUMMARY_JSON}\"}"
