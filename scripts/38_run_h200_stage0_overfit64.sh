#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/31_local_env.sh"

CORPUS_JSONL="${CORPUS_JSONL:-$ROOT_DIR/data/corpus/human_coc_teacher_full.jsonl}"
STAGE_CONFIG="${STAGE_CONFIG:-$ROOT_DIR/configs/train/stage_h200_clean_human900_stage0.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/checkpoints/h200_stage0_overfit64}"
SUMMARY_JSON="${SUMMARY_JSON:-$ROOT_DIR/outputs/reports/h200_stage0_overfit64_train_summary.json}"
DECODE_OUTPUT_DIR="${DECODE_OUTPUT_DIR:-$ROOT_DIR/outputs/reports/h200_stage0_overfit64_decode}"
DECODE_SUMMARY_JSON="${DECODE_SUMMARY_JSON:-$ROOT_DIR/outputs/reports/h200_stage0_overfit64_decode_summary.json}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-64}"
MAX_STEPS="${MAX_STEPS:-1200}"
EVAL_SPLIT="${EVAL_SPLIT:-train}"
EVAL_SAMPLES="${EVAL_SAMPLES:-64}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-10}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
INIT_CHECKPOINT_DIR="${INIT_CHECKPOINT_DIR:-}"
AUTO_RESUME="${AUTO_RESUME:-1}"

if [[ -z "${BATCH_SIZE:-}" ]]; then
  BATCH_SIZE="$(
    PROBE_MAX_STEPS="${PROBE_MAX_STEPS:-2}" \
    MAX_TRAIN_SAMPLES="$TRAIN_SAMPLES" \
    CORPUS_JSONL="$CORPUS_JSONL" \
    STAGE_CONFIG="$STAGE_CONFIG" \
    "$ROOT_DIR/scripts/36_probe_h200_batch_size.sh" | tail -n 1
  )"
else
  BATCH_SIZE="${BATCH_SIZE}"
fi

if [[ -z "$INIT_CHECKPOINT_DIR" && "$AUTO_RESUME" == "1" && -d "$OUTPUT_DIR" ]]; then
  LATEST_STEP_CHECKPOINT="$(
    find "$OUTPUT_DIR" -maxdepth 1 -mindepth 1 -type d -name 'step_*' | sort | tail -n 1
  )"
  if [[ -n "$LATEST_STEP_CHECKPOINT" ]]; then
    INIT_CHECKPOINT_DIR="$LATEST_STEP_CHECKPOINT"
  fi
fi

echo "[stage0-overfit64] using batch_size=$BATCH_SIZE"
if [[ -n "$INIT_CHECKPOINT_DIR" ]]; then
  echo "[stage0-overfit64] init_checkpoint_dir=$INIT_CHECKPOINT_DIR"
fi

TRAIN_ARGS=(
  --corpus-jsonl "$CORPUS_JSONL"
  --stage-config "$STAGE_CONFIG"
  --student-model "$COSMOS_STUDENT_MODEL"
  --batch-size "$BATCH_SIZE"
  --max-train-samples "$TRAIN_SAMPLES"
  --max-steps "$MAX_STEPS"
  --eval-every-epochs 999
  --save-every-epochs "$SAVE_EVERY_EPOCHS"
  --log-every-steps "$LOG_EVERY_STEPS"
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

if [[ -n "$INIT_CHECKPOINT_DIR" ]]; then
  TRAIN_ARGS+=(--init-checkpoint-dir "$INIT_CHECKPOINT_DIR")
fi

"$VENV_PYTHON" "$ROOT_DIR/scripts/09_train_distill.py" \
  "${TRAIN_ARGS[@]}" \
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
