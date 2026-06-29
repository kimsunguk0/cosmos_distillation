#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
STAGE2_DIR="${STAGE2_DIR:-outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603}"
CKPT="${CKPT:-${STAGE2_DIR}/best.pt}"
SPLIT_CACHE="${SPLIT_CACHE:-outputs/action_expert/stage2_heldout200k_val10k_seed42_20260603/split_cache_200k_10k_seed42.json}"
OUT_DIR="${OUT_DIR:-outputs/action_expert/stage2_200k_best_minade6_eval_${RUN_TAG}}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1024}"
EVAL_TRAIN_SAMPLES="${EVAL_TRAIN_SAMPLES:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_PATH_BATCH_SIZE="${EVAL_PATH_BATCH_SIZE:-6}"
SEED="${SEED:-42}"

mkdir -p "$OUT_DIR"
SWEEP_JSON="${OUT_DIR}/eval_sweep_minade6.json"
LOG_PATH="${OUT_DIR}/run.log"

cat > "$SWEEP_JSON" <<'JSON'
[
  {
    "label": "temp1p0_single_n6",
    "eval_temperature": 1.0,
    "eval_num_paths": 6,
    "eval_selection_method": "single"
  },
  {
    "label": "temp0p85_single_n6",
    "eval_temperature": 0.85,
    "eval_num_paths": 6,
    "eval_selection_method": "single"
  }
]
JSON

{
  echo "{\"event\":\"stage2_best_minade6_eval_launch\",\"time\":\"$(date -Is)\",\"ckpt\":\"${CKPT}\",\"out_dir\":\"${OUT_DIR}\",\"eval_samples\":${EVAL_SAMPLES},\"eval_train_samples\":${EVAL_TRAIN_SAMPLES},\"eval_num_paths\":6,\"seed\":${SEED}}"

  .venv/bin/python -u scripts/84_train_student_ae28_official.py \
    --student-checkpoint-dir outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250 \
    --num-samples 200000 \
    --val-samples 10000 \
    --split-cache-json "$SPLIT_CACHE" \
    --split-scan-all \
    --eval-samples "$EVAL_SAMPLES" \
    --eval-train-samples "$EVAL_TRAIN_SAMPLES" \
    --batch-size 8 \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --prefix-mode student_free \
    --ae-init-mode student_backbone_init \
    --target-source teacher \
    --eval-num-paths 6 \
    --eval-selection-method single \
    --eval-seed-mode fixed \
    --eval-vectorize-paths \
    --eval-path-batch-size "$EVAL_PATH_BATCH_SIZE" \
    --eval-log-rows 0 \
    --cleanup-every 0 \
    --eval-cleanup-every 0 \
    --attn-implementation flash_attention_2 \
    --seed "$SEED" \
    --resume-ae-checkpoint "$CKPT" \
    --eval-only \
    --eval-sweep-json "@${SWEEP_JSON}" \
    --output-dir "$OUT_DIR"

  echo "{\"event\":\"stage2_best_minade6_eval_done\",\"time\":\"$(date -Is)\",\"out_dir\":\"${OUT_DIR}\"}"
} >> "$LOG_PATH" 2>&1
