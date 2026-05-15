#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/31_local_env.sh"

CORPUS_JSONL="${CORPUS_JSONL:-$ROOT_DIR/data/corpus/human_coc_teacher_full.jsonl}"
STAGE_CONFIG="${STAGE_CONFIG:-$ROOT_DIR/configs/train/stage_h200_clean_human900_stage0.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/checkpoints/h200_stage0_overfit4}"
SUMMARY_JSON="${SUMMARY_JSON:-$ROOT_DIR/outputs/reports/h200_stage0_overfit4_train_summary.json}"
DECODE_OUTPUT_DIR="${DECODE_OUTPUT_DIR:-$ROOT_DIR/outputs/reports/h200_stage0_overfit4_decode}"
DECODE_SUMMARY_JSON="${DECODE_SUMMARY_JSON:-$ROOT_DIR/outputs/reports/h200_stage0_overfit4_decode_summary.json}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_STEPS="${MAX_STEPS:-200}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-4}"
EVAL_SPLIT="${EVAL_SPLIT:-train}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4}"

"$VENV_PYTHON" "$ROOT_DIR/scripts/09_train_distill.py" \
  --corpus-jsonl "$CORPUS_JSONL" \
  --stage-config "$STAGE_CONFIG" \
  --student-model "$COSMOS_STUDENT_MODEL" \
  --batch-size "$BATCH_SIZE" \
  --max-train-samples "$TRAIN_SAMPLES" \
  --max-steps "$MAX_STEPS" \
  --eval-every-epochs 999 \
  --save-every-epochs 999 \
  --log-every-steps 10 \
  --output-dir "$OUTPUT_DIR" \
  --summary-json "$SUMMARY_JSON" \
  "$@"

"$VENV_PYTHON" "$ROOT_DIR/scripts/25_decode_checkpoint_overlays.py" \
  --corpus-jsonl "$CORPUS_JSONL" \
  --checkpoint-dir "$OUTPUT_DIR/final" \
  --student-model "$COSMOS_STUDENT_MODEL" \
  --split "$EVAL_SPLIT" \
  --num-samples "$EVAL_SAMPLES" \
  --prompt-mode joint \
  --target-mode joint \
  --max-new-tokens 256 \
  --skip-overlays \
  --output-dir "$DECODE_OUTPUT_DIR" \
  --summary-json "$DECODE_SUMMARY_JSON"
