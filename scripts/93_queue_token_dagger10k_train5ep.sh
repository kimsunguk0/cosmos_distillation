#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

SOURCE_RUN_ID="${SOURCE_RUN_ID:-no_nav_token_dagger_fixedbase10k_prefix32_b16_20260521_012242}"
RUN_ID="${RUN_ID:-no_nav_token_dagger10k_prefix32_5ep_b16_$(date +%Y%m%d_%H%M%S)}"

CORPUS_DIR="data/corpus/no_nav_token_dagger_fixedbase10k_prefix32_b16"
CORPUS="${CORPUS_DIR}/${SOURCE_RUN_ID}.jsonl"
BASE_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_200k/no_nav_bp3_camera_labeled_official_200k_from_20kbest_20260514_092509/best_decode"
CONFIG="configs/train/stage_bp3_no_nav_token_dagger_prefix32.yaml"
OUTDIR="outputs/checkpoints/no_nav_token_dagger10k/${RUN_ID}"
SUMMARY="outputs/reports/no_nav_distill/${RUN_ID}_summary.json"
LOG="outputs/logs/${RUN_ID}.log"

TARGET_LINES="${TARGET_LINES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
POLL_SEC="${POLL_SEC:-60}"

mkdir -p "$(dirname "${LOG}")" "$(dirname "${SUMMARY}")" "${OUTDIR}"

echo "[queue-dagger10k] run_id=${RUN_ID}"
echo "[queue-dagger10k] waiting corpus=${CORPUS} target_lines=${TARGET_LINES}"

while true; do
  if [[ -s "${CORPUS}" ]]; then
    lines="$(wc -l < "${CORPUS}" | tr -d ' ')"
  else
    lines=0
  fi
  echo "[queue-dagger10k] $(date -Is) corpus_lines=${lines}/${TARGET_LINES}"
  if [[ "${lines}" -ge "${TARGET_LINES}" ]]; then
    break
  fi
  sleep "${POLL_SEC}"
done

echo "[queue-dagger10k] corpus ready; starting 5ep train -> ${OUTDIR}"
COSMOS_DATALOADER_NUM_WORKERS="${NUM_WORKERS}" \
COSMOS_DATALOADER_PREFETCH_FACTOR="${PREFETCH_FACTOR}" \
COSMOS_DATALOADER_PIN_MEMORY=1 \
COSMOS_DATALOADER_PERSISTENT_WORKERS=1 \
.venv/bin/python -u scripts/09_train_distill.py \
  --corpus-jsonl "${CORPUS}" \
  --stage-config "${CONFIG}" \
  --init-checkpoint-dir "${BASE_CKPT}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --max-train-samples "${TARGET_LINES}" \
  --eval-every-epochs 1 \
  --save-every-epochs 1 \
  --early-stop-patience 0 \
  --skip-asset-check \
  --output-dir "${OUTDIR}" \
  --summary-json "${SUMMARY}" \
  --log-every-steps 20 \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --pin-memory \
  --persistent-workers

echo "[queue-dagger10k] done run_id=${RUN_ID} summary=${SUMMARY}"
