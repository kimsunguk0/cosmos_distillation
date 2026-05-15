#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/31_local_env.sh"

STAGE_CONFIG="${STAGE_CONFIG:?STAGE_CONFIG is required}"
STAGE_BASENAME="$(basename "${STAGE_CONFIG%.yaml}")"
CORPUS_JSONL="${CORPUS_JSONL:-$ROOT_DIR/data/corpus/human_coc_teacher_full.jsonl}"
STUDENT_MODEL="${STUDENT_MODEL:-$COSMOS_STUDENT_MODEL}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/checkpoints/$STAGE_BASENAME}"
SUMMARY_JSON="${SUMMARY_JSON:-$ROOT_DIR/outputs/reports/${STAGE_BASENAME}_train_summary.json}"
REPORT_PREFIX="${REPORT_PREFIX:-$ROOT_DIR/outputs/reports/$STAGE_BASENAME}"
BATCH_SIZE="${BATCH_SIZE:-}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-999}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-999}"
SKIP_EVAL_IF_NO_CHECKPOINT="${SKIP_EVAL_IF_NO_CHECKPOINT:-1}"

TRAIN_ARGS=(
  --corpus-jsonl "$CORPUS_JSONL"
  --stage-config "$STAGE_CONFIG"
  --student-model "$STUDENT_MODEL"
  --eval-every-epochs "$EVAL_EVERY_EPOCHS"
  --save-every-epochs "$SAVE_EVERY_EPOCHS"
  --num-workers "${COSMOS_DATALOADER_NUM_WORKERS:-0}"
  --prefetch-factor "${COSMOS_DATALOADER_PREFETCH_FACTOR:-2}"
  --output-dir "$OUTPUT_DIR"
  --summary-json "$SUMMARY_JSON"
)

if [[ "${COSMOS_DATALOADER_PIN_MEMORY:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--pin-memory)
else
  TRAIN_ARGS+=(--no-pin-memory)
fi
if [[ "${COSMOS_DATALOADER_PERSISTENT_WORKERS:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--persistent-workers)
else
  TRAIN_ARGS+=(--no-persistent-workers)
fi

if [[ -n "$BATCH_SIZE" ]]; then
  TRAIN_ARGS+=(--batch-size "$BATCH_SIZE")
fi

if [[ -n "${INIT_CHECKPOINT_DIR:-}" ]]; then
  TRAIN_ARGS+=(--init-checkpoint-dir "$INIT_CHECKPOINT_DIR")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  TRAIN_ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "${EPOCHS:-}" ]]; then
  TRAIN_ARGS+=(--epochs "$EPOCHS")
fi
if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  TRAIN_ARGS+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi
if [[ -n "${LEARNING_RATE:-}" ]]; then
  TRAIN_ARGS+=(--learning-rate "$LEARNING_RATE")
fi

"$VENV_PYTHON" "$ROOT_DIR/scripts/09_train_distill.py" "${TRAIN_ARGS[@]}" "$@"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/final}"

if [[ ! -e "$CHECKPOINT_DIR" ]]; then
  if [[ "$SKIP_EVAL_IF_NO_CHECKPOINT" == "1" ]]; then
    echo "[skip-eval] checkpoint missing at $CHECKPOINT_DIR"
    exit 0
  fi
  echo "[error] checkpoint missing at $CHECKPOINT_DIR" >&2
  exit 1
fi

CHECKPOINT_DIR="$CHECKPOINT_DIR" \
CORPUS_JSONL="$CORPUS_JSONL" \
STUDENT_MODEL="$STUDENT_MODEL" \
REPORT_PREFIX="$REPORT_PREFIX" \
VAL_EVAL_SAMPLES="${VAL_EVAL_SAMPLES:-204}" \
TRAIN_EVAL_SAMPLES="${TRAIN_EVAL_SAMPLES:-64}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}" \
"$ROOT_DIR/scripts/39_eval_h200_checkpoint.sh"
