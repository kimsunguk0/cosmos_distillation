#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CKPT="${CKPT:-outputs/action_expert/q3_cosine_cooldown_from_q2best_s26000_2k_20260603_0505/best.pt}"
SPLIT_CACHE="${SPLIT_CACHE:-outputs/action_expert/stage1_heldout20k_val2k_s10000_seed42_full444k_20260531/split_cache_20k_2k_seed42.json}"
OUT_DIR="${OUT_DIR:-outputs/action_expert/q3_minade6_temp_sweep_seed42_evalbase1042_${RUN_TAG}}"
EVAL_SAMPLES="${EVAL_SAMPLES:-512}"
EVAL_TRAIN_SAMPLES="${EVAL_TRAIN_SAMPLES:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
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
  echo "{\"event\":\"q3_minade6_temp_sweep_launch\",\"time\":\"$(date -Is)\",\"ckpt\":\"${CKPT}\",\"out_dir\":\"${OUT_DIR}\",\"eval_samples\":${EVAL_SAMPLES},\"eval_num_paths\":6,\"seed\":${SEED}}"

  .venv/bin/python -u scripts/84_train_student_ae28_official.py \
    --student-checkpoint-dir outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250 \
    --num-samples 20000 \
    --val-samples 2000 \
    --split-cache-json "$SPLIT_CACHE" \
    --eval-samples "$EVAL_SAMPLES" \
    --eval-train-samples "$EVAL_TRAIN_SAMPLES" \
    --batch-size 2 \
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

  echo "{\"event\":\"q3_minade6_temp_sweep_done\",\"time\":\"$(date -Is)\",\"out_dir\":\"${OUT_DIR}\"}"
} >> "$LOG_PATH" 2>&1
