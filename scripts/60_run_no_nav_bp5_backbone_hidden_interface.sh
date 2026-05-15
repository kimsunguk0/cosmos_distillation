#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# BP5/B-plan: keep CoT/text/traj token supervision, but make teacher planning
# interface hidden alignment a first-class objective.
PROFILE="${PROFILE:-sanity}"
BASE_RUN_ID="${BASE_RUN_ID:-no_nav_bp3_200k_epoch_gc_b16_eval64_from_bp3final_20260508_072958}"
BASE_RUN_DIR="${BASE_RUN_DIR:-$ROOT_DIR/outputs/checkpoints/no_nav_bp3_200k_epoch/$BASE_RUN_ID}"
INIT_CHECKPOINT_NAME="${INIT_CHECKPOINT_NAME:-step_012500}"
INIT_CHECKPOINT_DIR="${INIT_CHECKPOINT_DIR:-$BASE_RUN_DIR/$INIT_CHECKPOINT_NAME}"

case "$PROFILE" in
  sanity)
    DEFAULT_BATCH_SIZE=24
    DEFAULT_MAX_STEPS=600
    DEFAULT_MAX_TRAIN_SAMPLES=4096
    DEFAULT_MAX_VAL_SAMPLES=128
    DEFAULT_EVAL_EVERY_EPOCHS=0.25
    DEFAULT_SAVE_EVERY_EPOCHS=0.50
    ;;
  pilot)
    DEFAULT_BATCH_SIZE=24
    DEFAULT_MAX_STEPS=3000
    DEFAULT_MAX_TRAIN_SAMPLES=50000
    DEFAULT_MAX_VAL_SAMPLES=1024
    DEFAULT_EVAL_EVERY_EPOCHS=0.10
    DEFAULT_SAVE_EVERY_EPOCHS=0.25
    ;;
  full)
    DEFAULT_BATCH_SIZE=24
    DEFAULT_MAX_STEPS=12500
    DEFAULT_MAX_TRAIN_SAMPLES=
    DEFAULT_MAX_VAL_SAMPLES=2048
    DEFAULT_EVAL_EVERY_EPOCHS=0.05
    DEFAULT_SAVE_EVERY_EPOCHS=0.25
    ;;
  *)
    echo "unknown PROFILE=$PROFILE; expected sanity|pilot|full" >&2
    exit 2
    ;;
esac

RUN_ID="${RUN_ID:-no_nav_bp5_hidden_interface_${PROFILE}_from_${INIT_CHECKPOINT_NAME}_$(date +%Y%m%d_%H%M%S)}"
CORPUS_JSONL="${CORPUS_JSONL:-$ROOT_DIR/data/corpus/no_nav_teacher_pair_300chunks.jsonl}"
STAGE_CONFIG="${STAGE_CONFIG:-$ROOT_DIR/configs/train/stage_bp5_no_nav_vlm_interface_hidden_kd.yaml}"
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
MAX_STEPS="${MAX_STEPS:-$DEFAULT_MAX_STEPS}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-$DEFAULT_MAX_TRAIN_SAMPLES}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-$DEFAULT_MAX_VAL_SAMPLES}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-$DEFAULT_EVAL_EVERY_EPOCHS}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-$DEFAULT_SAVE_EVERY_EPOCHS}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/checkpoints/no_nav_bp5_hidden_interface/$RUN_ID}"
SUMMARY_JSON="${SUMMARY_JSON:-$ROOT_DIR/outputs/reports/no_nav_distill/${RUN_ID}_summary.json}"
TRAIN_LOG="${TRAIN_LOG:-$ROOT_DIR/logs/no_nav_distill/${RUN_ID}.train.log}"
EVAL_LOG="${EVAL_LOG:-$ROOT_DIR/logs/no_nav_distill/${RUN_ID}.eval.log}"
REPORT_PREFIX="${REPORT_PREFIX:-$ROOT_DIR/outputs/reports/no_nav_distill/${RUN_ID}}"
RUN_FINAL_DECODE_EVAL="${RUN_FINAL_DECODE_EVAL:-0}"

mkdir -p "$ROOT_DIR/logs/no_nav_distill" "$ROOT_DIR/outputs/reports/no_nav_distill" "$(dirname "$OUTPUT_DIR")"

if [[ ! -d "$INIT_CHECKPOINT_DIR" ]]; then
  echo "missing INIT_CHECKPOINT_DIR=$INIT_CHECKPOINT_DIR" >&2
  exit 1
fi

echo "[bp5] run_id=$RUN_ID profile=$PROFILE init=$INIT_CHECKPOINT_DIR" | tee "$TRAIN_LOG"
echo "[bp5] stage_config=$STAGE_CONFIG corpus=$CORPUS_JSONL batch=$BATCH_SIZE max_steps=$MAX_STEPS max_train=${MAX_TRAIN_SAMPLES:-all}" | tee -a "$TRAIN_LOG"

COSMOS_DATA_ROOT=/home/pm97/workspace/dataset/distill_dataset \
COSMOS_DATALOADER_NUM_WORKERS="${COSMOS_DATALOADER_NUM_WORKERS:-8}" \
COSMOS_DATALOADER_PREFETCH_FACTOR="${COSMOS_DATALOADER_PREFETCH_FACTOR:-4}" \
COSMOS_DATALOADER_PIN_MEMORY="${COSMOS_DATALOADER_PIN_MEMORY:-1}" \
COSMOS_DATALOADER_PERSISTENT_WORKERS="${COSMOS_DATALOADER_PERSISTENT_WORKERS:-1}" \
COSMOS_SKIP_ASSET_CHECK=1 \
CORPUS_JSONL="$CORPUS_JSONL" \
STAGE_CONFIG="$STAGE_CONFIG" \
BATCH_SIZE="$BATCH_SIZE" \
MAX_STEPS="$MAX_STEPS" \
MAX_TRAIN_SAMPLES="$MAX_TRAIN_SAMPLES" \
MAX_VAL_SAMPLES="$MAX_VAL_SAMPLES" \
EVAL_EVERY_EPOCHS="$EVAL_EVERY_EPOCHS" \
SAVE_EVERY_EPOCHS="$SAVE_EVERY_EPOCHS" \
INIT_CHECKPOINT_DIR="$INIT_CHECKPOINT_DIR" \
OUTPUT_DIR="$OUTPUT_DIR" \
SUMMARY_JSON="$SUMMARY_JSON" \
bash "$ROOT_DIR/scripts/43_train_no_nav_backbone_pilot.sh" --multi-gpu off --log-every-steps "${LOG_EVERY_STEPS:-20}" \
  >> "$TRAIN_LOG" 2>&1

if [[ "$RUN_FINAL_DECODE_EVAL" != "1" ]]; then
  echo "skip final decode eval because RUN_FINAL_DECODE_EVAL=$RUN_FINAL_DECODE_EVAL; use hidden QC + TF val first" > "$EVAL_LOG"
  echo "done $RUN_ID" >> "$EVAL_LOG"
  exit 0
fi

CHECKPOINT_DIR="$OUTPUT_DIR/final" \
COSMOS_DATA_ROOT=/home/pm97/workspace/dataset/distill_dataset \
CORPUS_JSONL="$CORPUS_JSONL" \
REPORT_PREFIX="$REPORT_PREFIX" \
VAL_EVAL_SAMPLES="${VAL_EVAL_SAMPLES:-204}" \
TRAIN_EVAL_SAMPLES="${TRAIN_EVAL_SAMPLES:-64}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}" \
bash "$ROOT_DIR/scripts/39_eval_h200_checkpoint.sh" > "$EVAL_LOG" 2>&1

echo "done $RUN_ID" >> "$EVAL_LOG"
