#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
STAGE_CONFIG="${STAGE_CONFIG:-$ROOT_DIR/configs/train/stage_flex_2b_k512_camtime_smoke.yaml}"
CORPUS_JSONL="${CORPUS_JSONL:-$ROOT_DIR/data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl}"
INIT_CHECKPOINT_DIR="${INIT_CHECKPOINT_DIR:-$ROOT_DIR/outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/checkpoints/flex_2b_k512_camtime_smoke_${RUN_TAG}}"
SUMMARY_JSON="${SUMMARY_JSON:-$ROOT_DIR/outputs/reports/flex_2b_k512_camtime_smoke_${RUN_TAG}_train_summary.json}"
REPORT_PREFIX="${REPORT_PREFIX:-$ROOT_DIR/outputs/reports/flex_2b_k512_camtime_smoke_${RUN_TAG}}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2048}"
MAX_STEPS="${MAX_STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-999}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"
SKIP_EVAL_IF_NO_CHECKPOINT="${SKIP_EVAL_IF_NO_CHECKPOINT:-1}"

mkdir -p "$(dirname "$SUMMARY_JSON")" "$OUTPUT_DIR"

echo "===== FLEX_2B_K512_CAMTIME_SMOKE START $(date -Iseconds) ====="
echo "stage_config=$STAGE_CONFIG"
echo "corpus_jsonl=$CORPUS_JSONL"
echo "init_checkpoint_dir=$INIT_CHECKPOINT_DIR"
echo "output_dir=$OUTPUT_DIR"
echo "max_train_samples=$MAX_TRAIN_SAMPLES"
echo "max_steps=$MAX_STEPS"
echo "batch_size=$BATCH_SIZE"

STAGE_CONFIG="$STAGE_CONFIG" \
CORPUS_JSONL="$CORPUS_JSONL" \
INIT_CHECKPOINT_DIR="$INIT_CHECKPOINT_DIR" \
OUTPUT_DIR="$OUTPUT_DIR" \
SUMMARY_JSON="$SUMMARY_JSON" \
REPORT_PREFIX="$REPORT_PREFIX" \
MAX_TRAIN_SAMPLES="$MAX_TRAIN_SAMPLES" \
MAX_STEPS="$MAX_STEPS" \
BATCH_SIZE="$BATCH_SIZE" \
EVAL_EVERY_EPOCHS="$EVAL_EVERY_EPOCHS" \
SAVE_EVERY_EPOCHS="$SAVE_EVERY_EPOCHS" \
SKIP_EVAL_IF_NO_CHECKPOINT="$SKIP_EVAL_IF_NO_CHECKPOINT" \
bash "$ROOT_DIR/scripts/40_train_eval_h200_stage.sh"
