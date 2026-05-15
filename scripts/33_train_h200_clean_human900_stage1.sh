#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/31_local_env.sh"

CORPUS_JSONL="${CORPUS_JSONL:-$ROOT_DIR/data/corpus/human_coc_teacher_full.jsonl}"
STAGE_CONFIG="${STAGE_CONFIG:-$ROOT_DIR/configs/train/stage_h200_clean_human900_stage1.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/checkpoints/h200_clean_human900_stage1}"
SUMMARY_JSON="${SUMMARY_JSON:-$ROOT_DIR/outputs/reports/h200_clean_human900_stage1_train_summary.json}"

"$VENV_PYTHON" "$ROOT_DIR/scripts/09_train_distill.py" \
  --corpus-jsonl "$CORPUS_JSONL" \
  --stage-config "$STAGE_CONFIG" \
  --student-model "$COSMOS_STUDENT_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --summary-json "$SUMMARY_JSON" \
  "$@"
