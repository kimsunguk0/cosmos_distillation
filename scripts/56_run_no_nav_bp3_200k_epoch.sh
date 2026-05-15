#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-no_nav_bp3_200k_epoch_gc_b16_from_bp3final_$(date +%Y%m%d_%H%M%S)}"
INIT_CHECKPOINT_DIR="${INIT_CHECKPOINT_DIR:-$ROOT_DIR/outputs/checkpoints/no_nav_bp3_h200fast_b4/no_nav_bp3_h200fast_b4_from_step2288_20260504_053208/final}"
BP3_STAGE_CONFIG="${BP3_STAGE_CONFIG:-$ROOT_DIR/configs/train/stage_bp3_no_nav_traj_topk_kd_gc_decode_eval.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/checkpoints/no_nav_bp3_200k_epoch/$RUN_ID}"
SUMMARY_JSON="${SUMMARY_JSON:-$ROOT_DIR/outputs/reports/no_nav_distill/${RUN_ID}_summary.json}"
TRAIN_LOG="${TRAIN_LOG:-$ROOT_DIR/logs/no_nav_distill/${RUN_ID}.train.log}"
EVAL_LOG="${EVAL_LOG:-$ROOT_DIR/logs/no_nav_distill/${RUN_ID}.eval.log}"
REPORT_PREFIX="${REPORT_PREFIX:-$ROOT_DIR/outputs/reports/no_nav_distill/${RUN_ID}}"

mkdir -p "$ROOT_DIR/logs/no_nav_distill" "$ROOT_DIR/outputs/reports/no_nav_distill" "$(dirname "$OUTPUT_DIR")"

echo "[run] id=$RUN_ID"
echo "[run] init=$INIT_CHECKPOINT_DIR"
echo "[run] output=$OUTPUT_DIR"
echo "[run] train_log=$TRAIN_LOG"
echo "[run] summary=$SUMMARY_JSON"

COSMOS_DATA_ROOT=/home/pm97/workspace/dataset/distill_dataset \
COSMOS_DATALOADER_NUM_WORKERS="${COSMOS_DATALOADER_NUM_WORKERS:-8}" \
COSMOS_DATALOADER_PREFETCH_FACTOR="${COSMOS_DATALOADER_PREFETCH_FACTOR:-4}" \
COSMOS_DATALOADER_PIN_MEMORY="${COSMOS_DATALOADER_PIN_MEMORY:-1}" \
COSMOS_DATALOADER_PERSISTENT_WORKERS="${COSMOS_DATALOADER_PERSISTENT_WORKERS:-1}" \
COSMOS_SKIP_ASSET_CHECK=1 \
STAGE_CONFIG="$BP3_STAGE_CONFIG" \
CORPUS_JSONL="$ROOT_DIR/data/corpus/no_nav_teacher_pair_300chunks.jsonl" \
INIT_CHECKPOINT_DIR="$INIT_CHECKPOINT_DIR" \
BATCH_SIZE="${BATCH_SIZE:-16}" \
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-200000}" \
EPOCHS="${EPOCHS:-1.0}" \
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-0.25}" \
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-0.25}" \
OUTPUT_DIR="$OUTPUT_DIR" \
SUMMARY_JSON="$SUMMARY_JSON" \
bash "$ROOT_DIR/scripts/43_train_no_nav_backbone_pilot.sh" --multi-gpu off --log-every-steps "${LOG_EVERY_STEPS:-50}" \
  > "$TRAIN_LOG" 2>&1

if [[ "${EVAL_CHECKPOINT_SUITE:-1}" == "1" ]]; then
  RUN_ID="$RUN_ID" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  CORPUS_JSONL="$ROOT_DIR/data/corpus/no_nav_teacher_pair_300chunks.jsonl" \
  VAL_EVAL_SAMPLES="${VAL_EVAL_SAMPLES:-204}" \
  TRAIN_EVAL_SAMPLES="${TRAIN_EVAL_SAMPLES:-64}" \
  PREFILL_QC_SAMPLES="${PREFILL_QC_SAMPLES:-128}" \
  bash "$ROOT_DIR/scripts/57_eval_no_nav_bp3_checkpoint_suite.sh" > "$EVAL_LOG" 2>&1
else
  CHECKPOINT_DIR="$OUTPUT_DIR/final" \
  COSMOS_DATA_ROOT=/home/pm97/workspace/dataset/distill_dataset \
  CORPUS_JSONL="$ROOT_DIR/data/corpus/no_nav_teacher_pair_300chunks.jsonl" \
  REPORT_PREFIX="$REPORT_PREFIX" \
  VAL_EVAL_SAMPLES="${VAL_EVAL_SAMPLES:-204}" \
  TRAIN_EVAL_SAMPLES="${TRAIN_EVAL_SAMPLES:-64}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}" \
  GEOMETRY_REFERENCE="${GEOMETRY_REFERENCE:-teacher}" \
  bash "$ROOT_DIR/scripts/39_eval_h200_checkpoint.sh" > "$EVAL_LOG" 2>&1
fi

echo "[run] done id=$RUN_ID"
