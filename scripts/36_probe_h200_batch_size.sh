#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/31_local_env.sh"

CORPUS_JSONL="${CORPUS_JSONL:-$ROOT_DIR/data/corpus/human_coc_teacher_full.jsonl}"
STAGE_CONFIG="${STAGE_CONFIG:-$ROOT_DIR/configs/train/stage_h200_clean_human900_stage0.yaml}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-64}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$ROOT_DIR/outputs/checkpoints/h200_stage0_batch_probe}"
SUMMARY_BASE_DIR="${SUMMARY_BASE_DIR:-$ROOT_DIR/outputs/reports/h200_stage0_batch_probe}"
PROBE_MAX_STEPS="${PROBE_MAX_STEPS:-1}"

if [[ -n "${BATCH_CANDIDATES:-}" ]]; then
  read -r -a CANDIDATES <<<"$BATCH_CANDIDATES"
else
  CANDIDATES=(32 24 16 12 8 6 4 2 1)
fi

mkdir -p "$OUTPUT_BASE_DIR" "$SUMMARY_BASE_DIR"
BEST_BATCH_SIZE=""

for BATCH_SIZE in "${CANDIDATES[@]}"; do
  if (( BATCH_SIZE > MAX_TRAIN_SAMPLES )); then
    continue
  fi

  OUTPUT_DIR="$OUTPUT_BASE_DIR/bs_${BATCH_SIZE}"
  SUMMARY_JSON="$SUMMARY_BASE_DIR/bs_${BATCH_SIZE}.json"
  LOG_PATH="$SUMMARY_BASE_DIR/bs_${BATCH_SIZE}.log"
  rm -rf "$OUTPUT_DIR"

  set +e
  "$VENV_PYTHON" "$ROOT_DIR/scripts/09_train_distill.py" \
    --corpus-jsonl "$CORPUS_JSONL" \
    --stage-config "$STAGE_CONFIG" \
    --student-model "$COSMOS_STUDENT_MODEL" \
    --batch-size "$BATCH_SIZE" \
    --max-train-samples "$MAX_TRAIN_SAMPLES" \
    --max-steps "$PROBE_MAX_STEPS" \
    --eval-every-epochs 999 \
    --save-every-epochs 999 \
    --log-every-steps 1 \
    --output-dir "$OUTPUT_DIR" \
    --summary-json "$SUMMARY_JSON" \
    >"$LOG_PATH" 2>&1
  STATUS=$?
  set -e

  if [[ "$STATUS" -eq 0 ]]; then
    BEST_BATCH_SIZE="$BATCH_SIZE"
    break
  fi

  if grep -Eqi "out of memory|cuda out of memory|CUDA error: out of memory" "$LOG_PATH"; then
    continue
  fi

  cat "$LOG_PATH" >&2
  exit "$STATUS"
done

if [[ -z "$BEST_BATCH_SIZE" ]]; then
  echo "No candidate batch size completed successfully." >&2
  exit 1
fi

echo "$BEST_BATCH_SIZE"
