#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export COSMOS_DATA_ROOT="${COSMOS_DATA_ROOT:-/home/pm97/workspace/dataset/distill_dataset}"
source "$ROOT_DIR/scripts/31_local_env.sh"

DISTILL_DATASET_ROOT="${DISTILL_DATASET_ROOT:-/home/pm97/workspace/dataset/distill_dataset}"
CORPUS_JSONL="${CORPUS_JSONL:-$ROOT_DIR/data/corpus/no_nav_teacher_pair_300chunks.jsonl}"
CORPUS_SUMMARY_JSON="${CORPUS_SUMMARY_JSON:-$ROOT_DIR/outputs/reports/no_nav_teacher_pair_300chunks_summary.json}"
STAGE_CONFIG="${STAGE_CONFIG:-$ROOT_DIR/configs/train/stage_bp1_no_nav_teacher_pair_ce.yaml}"
STAGE_BASENAME="$(basename "${STAGE_CONFIG%.yaml}")"
STUDENT_MODEL="${STUDENT_MODEL:-$COSMOS_STUDENT_MODEL}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/checkpoints/$STAGE_BASENAME}"
SUMMARY_JSON="${SUMMARY_JSON:-$ROOT_DIR/outputs/reports/${STAGE_BASENAME}_train_summary.json}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-999}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-999}"
BATCH_SIZE="${BATCH_SIZE:-1}"

MANIFEST_0="${MANIFEST_0:-$DISTILL_DATASET_ROOT/teacher_cache/no_nav/text/manifest/no_nav_teacher_infer_manifest.parquet}"
MANIFEST_1="${MANIFEST_1:-$DISTILL_DATASET_ROOT/reports/no_nav/next50_after_250/manifest/no_nav_teacher_infer_manifest.parquet}"

if [[ "${REBUILD_CORPUS:-0}" == "1" || ! -s "$CORPUS_JSONL" ]]; then
  BUILD_OUTPUT_JSONL="${CORPUS_BUILD_TMP_JSONL:-/tmp/$(basename "$CORPUS_JSONL")}"
  BUILD_SUMMARY_JSON="${CORPUS_BUILD_TMP_SUMMARY:-/tmp/$(basename "$CORPUS_SUMMARY_JSON")}"
  BUILD_ARGS=(
    --manifest-parquet "$MANIFEST_0"
    --manifest-parquet "$MANIFEST_1"
    --output-jsonl "$BUILD_OUTPUT_JSONL"
    --summary-json "$BUILD_SUMMARY_JSON"
    --reported-output-jsonl "$CORPUS_JSONL"
    --split-policy "${SPLIT_POLICY:-hash_clip}"
    --val-fraction "${VAL_FRACTION:-0.02}"
  )
  if [[ -n "${MAX_CORPUS_RECORDS:-}" ]]; then
    BUILD_ARGS+=(--max-records "$MAX_CORPUS_RECORDS")
  fi
  if [[ "${SKIP_IMAGE_STAT:-0}" == "1" ]]; then
    BUILD_ARGS+=(--skip-image-stat)
  fi
  "$VENV_PYTHON" "$ROOT_DIR/scripts/42_build_no_nav_teacher_pair_corpus.py" "${BUILD_ARGS[@]}"
  if [[ "$BUILD_OUTPUT_JSONL" != "$CORPUS_JSONL" ]]; then
    mkdir -p "$(dirname "$CORPUS_JSONL")"
    cp "$BUILD_OUTPUT_JSONL" "$CORPUS_JSONL"
  fi
  if [[ "$BUILD_SUMMARY_JSON" != "$CORPUS_SUMMARY_JSON" ]]; then
    mkdir -p "$(dirname "$CORPUS_SUMMARY_JSON")"
    cp "$BUILD_SUMMARY_JSON" "$CORPUS_SUMMARY_JSON"
  fi
fi

TRAIN_ARGS=(
  --corpus-jsonl "$CORPUS_JSONL"
  --stage-config "$STAGE_CONFIG"
  --student-model "$STUDENT_MODEL"
  --batch-size "$BATCH_SIZE"
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
if [[ "${DATA_ONLY_DRY_RUN:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--data-only-dry-run)
fi
if [[ "${COSMOS_SKIP_ASSET_CHECK:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--skip-asset-check)
else
  TRAIN_ARGS+=(--no-skip-asset-check)
fi
if [[ "${COSMOS_SKIP_FINAL_SAVE:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--skip-final-save)
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
if [[ -n "${MAX_VAL_SAMPLES:-}" ]]; then
  TRAIN_ARGS+=(--max-val-samples "$MAX_VAL_SAMPLES")
fi
if [[ -n "${LEARNING_RATE:-}" ]]; then
  TRAIN_ARGS+=(--learning-rate "$LEARNING_RATE")
fi

"$VENV_PYTHON" "$ROOT_DIR/scripts/09_train_distill.py" "${TRAIN_ARGS[@]}" "$@"
