#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/31_local_env.sh"

CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}"
CORPUS_JSONL="${CORPUS_JSONL:-$ROOT_DIR/data/corpus/human_coc_teacher_full.jsonl}"
STUDENT_MODEL="${STUDENT_MODEL:-$COSMOS_STUDENT_MODEL}"
REPORT_PREFIX="${REPORT_PREFIX:-$ROOT_DIR/outputs/reports/$(basename "$CHECKPOINT_DIR")}"
VAL_EVAL_SAMPLES="${VAL_EVAL_SAMPLES:-64}"
TRAIN_EVAL_SAMPLES="${TRAIN_EVAL_SAMPLES:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DECODE_FAILURE_TAGS="${DECODE_FAILURE_TAGS:-0}"
GEOMETRY_REFERENCE="${GEOMETRY_REFERENCE:-teacher}"

VAL_OUTPUT_DIR="${VAL_OUTPUT_DIR:-${REPORT_PREFIX}_val${VAL_EVAL_SAMPLES}_decode}"
VAL_SUMMARY_JSON="${VAL_SUMMARY_JSON:-${REPORT_PREFIX}_val${VAL_EVAL_SAMPLES}_decode_summary.json}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${REPORT_PREFIX}_train64_decode}"
TRAIN_SUMMARY_JSON="${TRAIN_SUMMARY_JSON:-${REPORT_PREFIX}_train64_decode_summary.json}"

FAILURE_TAG_ARGS=()
if [[ "$DECODE_FAILURE_TAGS" == "0" || "$DECODE_FAILURE_TAGS" == "false" ]]; then
  FAILURE_TAG_ARGS+=(--disable-failure-tags)
fi

"$VENV_PYTHON" "$ROOT_DIR/scripts/25_decode_checkpoint_overlays.py" \
  --corpus-jsonl "$CORPUS_JSONL" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --student-model "$STUDENT_MODEL" \
  --split val \
  --num-samples "$VAL_EVAL_SAMPLES" \
  --prompt-mode joint \
  --target-mode joint \
  --geometry-reference "$GEOMETRY_REFERENCE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --skip-overlays \
  "${FAILURE_TAG_ARGS[@]}" \
  --output-dir "$VAL_OUTPUT_DIR" \
  --summary-json "$VAL_SUMMARY_JSON"

"$VENV_PYTHON" "$ROOT_DIR/scripts/25_decode_checkpoint_overlays.py" \
  --corpus-jsonl "$CORPUS_JSONL" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --student-model "$STUDENT_MODEL" \
  --split train \
  --num-samples "$TRAIN_EVAL_SAMPLES" \
  --prompt-mode joint \
  --target-mode joint \
  --geometry-reference "$GEOMETRY_REFERENCE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --skip-overlays \
  "${FAILURE_TAG_ARGS[@]}" \
  --output-dir "$TRAIN_OUTPUT_DIR" \
  --summary-json "$TRAIN_SUMMARY_JSON"
