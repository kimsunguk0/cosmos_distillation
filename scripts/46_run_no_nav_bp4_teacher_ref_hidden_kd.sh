#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-no_nav_bp4_teacher_ref_hidden_kd_b${BATCH_SIZE:-8}_$(date +%Y%m%d_%H%M%S)}"
INIT_CHECKPOINT_DIR="${INIT_CHECKPOINT_DIR:-$ROOT_DIR/outputs/checkpoints/no_nav_bp3_h200fast_b4/no_nav_bp3_h200fast_b4_from_step2288_20260504_053208/final}"
MAX_STEPS="${MAX_STEPS:-1500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-25}"
STAGE_CONFIG="${STAGE_CONFIG:-$ROOT_DIR/configs/train/stage_bp4_no_nav_teacher_ref_hidden_kd.yaml}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/checkpoints/no_nav_bp4_teacher_ref_hidden_kd/$RUN_ID}"
SUMMARY_JSON="${SUMMARY_JSON:-$ROOT_DIR/outputs/reports/no_nav_distill/${RUN_ID}_summary.json}"
TRAIN_LOG="${TRAIN_LOG:-$ROOT_DIR/logs/no_nav_distill/${RUN_ID}.train.log}"
EVAL_LOG="${EVAL_LOG:-$ROOT_DIR/logs/no_nav_distill/${RUN_ID}.eval.log}"
REPORT_PREFIX="${REPORT_PREFIX:-$ROOT_DIR/outputs/reports/no_nav_distill/${RUN_ID}}"

mkdir -p "$ROOT_DIR/logs/no_nav_distill" "$ROOT_DIR/outputs/reports/no_nav_distill" "$(dirname "$OUTPUT_DIR")"

COSMOS_DATA_ROOT=/home/pm97/workspace/dataset/distill_dataset \
COSMOS_DATALOADER_NUM_WORKERS="${COSMOS_DATALOADER_NUM_WORKERS:-8}" \
COSMOS_DATALOADER_PREFETCH_FACTOR="${COSMOS_DATALOADER_PREFETCH_FACTOR:-4}" \
COSMOS_DATALOADER_PIN_MEMORY="${COSMOS_DATALOADER_PIN_MEMORY:-1}" \
COSMOS_DATALOADER_PERSISTENT_WORKERS="${COSMOS_DATALOADER_PERSISTENT_WORKERS:-1}" \
COSMOS_SKIP_ASSET_CHECK=1 \
STAGE_CONFIG="$STAGE_CONFIG" \
BATCH_SIZE="$BATCH_SIZE" \
MAX_STEPS="$MAX_STEPS" \
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-999}" \
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-0.05}" \
INIT_CHECKPOINT_DIR="$INIT_CHECKPOINT_DIR" \
OUTPUT_DIR="$OUTPUT_DIR" \
SUMMARY_JSON="$SUMMARY_JSON" \
bash "$ROOT_DIR/scripts/43_train_no_nav_backbone_pilot.sh" --multi-gpu off --log-every-steps "$LOG_EVERY_STEPS" \
  > "$TRAIN_LOG" 2>&1

if [[ "${DATA_ONLY_DRY_RUN:-0}" == "1" || "${COSMOS_SKIP_FINAL_SAVE:-0}" == "1" ]]; then
  echo "skip eval because DATA_ONLY_DRY_RUN=${DATA_ONLY_DRY_RUN:-0} COSMOS_SKIP_FINAL_SAVE=${COSMOS_SKIP_FINAL_SAVE:-0}" > "$EVAL_LOG"
  exit 0
fi

CHECKPOINT_DIR="$OUTPUT_DIR/final" \
COSMOS_DATA_ROOT=/home/pm97/workspace/dataset/distill_dataset \
CORPUS_JSONL="$ROOT_DIR/data/corpus/no_nav_teacher_pair_300chunks.jsonl" \
REPORT_PREFIX="$REPORT_PREFIX" \
VAL_EVAL_SAMPLES="${VAL_EVAL_SAMPLES:-4760}" \
TRAIN_EVAL_SAMPLES="${TRAIN_EVAL_SAMPLES:-128}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}" \
bash "$ROOT_DIR/scripts/39_eval_h200_checkpoint.sh" > "$EVAL_LOG" 2>&1

echo "done $RUN_ID" >> "$EVAL_LOG"
