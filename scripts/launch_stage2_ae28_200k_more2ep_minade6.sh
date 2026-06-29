#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_STAGE2_DIR="${BASE_STAGE2_DIR:-outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603}"
RESUME_CKPT="${RESUME_CKPT:-${BASE_STAGE2_DIR}/final.pt}"
SPLIT_CACHE="${SPLIT_CACHE:-outputs/action_expert/stage2_heldout200k_val10k_seed42_20260603/split_cache_200k_10k_seed42.json}"
OUT_DIR="${OUT_DIR:-outputs/action_expert/stage2_200k_more2ep_b8_nt16_minade6_${RUN_TAG}}"
START_STEP="${START_STEP:-25000}"
END_STEP="${END_STEP:-75000}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-1.0}"
EVAL_SELECTION_METHOD="${EVAL_SELECTION_METHOD:-single}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$BATCH_SIZE}"
EVAL_PATH_BATCH_SIZE="${EVAL_PATH_BATCH_SIZE:-6}"
EVAL_VECTORIZE_PATHS="${EVAL_VECTORIZE_PATHS:-1}"

EVAL_VECTORIZE_ARGS=()
if [[ "$EVAL_VECTORIZE_PATHS" == "1" ]]; then
  EVAL_VECTORIZE_ARGS+=(--eval-vectorize-paths)
fi

mkdir -p "$OUT_DIR" "$(dirname "$SPLIT_CACHE")"
LOG_FILE="${OUT_DIR}/launch.log"

{
  echo "===== STAGE2_200K_AE28_MORE2EP_MINADE6 START $(date -Is) ====="
  echo "out_dir=${OUT_DIR}"
  echo "resume_ckpt=${RESUME_CKPT}"
  echo "split_cache=${SPLIT_CACHE}"
  echo "start_step=${START_STEP}"
  echo "end_step=${END_STEP}"
  echo "eval_temperature=${EVAL_TEMPERATURE}"
  echo "eval_selection_method=${EVAL_SELECTION_METHOD}"
  echo "batch_size=${BATCH_SIZE}"
  echo "eval_batch_size=${EVAL_BATCH_SIZE}"
  echo "eval_path_batch_size=${EVAL_PATH_BATCH_SIZE}"
  echo "eval_vectorize_paths=${EVAL_VECTORIZE_PATHS}"

  .venv/bin/python -u scripts/84_train_student_ae28_official.py \
    --student-checkpoint-dir outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250 \
    --num-samples 200000 \
    --val-samples 10000 \
    --split-cache-json "$SPLIT_CACHE" \
    --split-scan-all \
    --eval-samples 1024 \
    --eval-train-samples 512 \
    --batch-size "$BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --eval-every 2500 \
    --log-every 100 \
    --train-ade-every 0 \
    --prefix-mode student_free \
    --ae-init-mode student_backbone_init \
    --target-source teacher \
    --expert-lr 1e-4 \
    --proj-lr 1e-4 \
    --num-time-samples 16 \
    --grad-clip-norm 5.0 \
    --no-norm-bias-decay \
    --eval-temperature "$EVAL_TEMPERATURE" \
    --eval-num-paths 6 \
    --eval-selection-method "$EVAL_SELECTION_METHOD" \
    --eval-seed-mode fixed \
    "${EVAL_VECTORIZE_ARGS[@]}" \
    --eval-path-batch-size "$EVAL_PATH_BATCH_SIZE" \
    --eval-log-rows 0 \
    --cleanup-every 0 \
    --eval-cleanup-every 0 \
    --attn-implementation flash_attention_2 \
    --save-every 2500 \
    --skip-initial-eval \
    --seed 42 \
    --steps "$END_STEP" \
    --start-step "$START_STEP" \
    --resume-ae-checkpoint "$RESUME_CKPT" \
    --lr-warmup-steps 0 \
    --min-lr 1e-6 \
    --allow-train-cache-mutation \
    --fused-adamw \
    --output-dir "$OUT_DIR"

  echo "===== STAGE2_200K_AE28_MORE2EP_MINADE6 END $(date -Is) ====="
} 2>&1 | tee -a "$LOG_FILE"
