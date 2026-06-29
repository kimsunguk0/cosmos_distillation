#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/outputs/checkpoints/b0_fp8_gkd_20260618}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT_DIR/outputs/reports/b0_fp8_gkd_20260618}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/logs/b0_fp8_gkd_20260618}"
CORPUS_20K="${CORPUS_20K:-$ROOT_DIR/data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl}"

mkdir -p "$RUN_ROOT" "$REPORT_ROOT" "$LOG_ROOT"

wait_for_gpu() {
  local min_free_mb="${WAIT_FREE_MB:-120000}"
  local sleep_sec="${WAIT_GPU_SLEEP_SEC:-60}"
  while true; do
    local free_mb
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ -n "$free_mb" && "$free_mb" -ge "$min_free_mb" ]]; then
      echo "$(date -Is) gpu_ready free_mb=${free_mb} threshold_mb=${min_free_mb}"
      return 0
    fi
    echo "$(date -Is) gpu_wait free_mb=${free_mb:-unknown} threshold_mb=${min_free_mb}"
    sleep "$sleep_sec"
  done
}

run_stage() {
  local name="$1"
  local stage_config="$2"
  local batch_size="$3"
  local max_train_samples="$4"
  local max_steps="$5"
  local eval_every_epochs="$6"
  local save_every_epochs="$7"
  local qat_calib_samples="$8"

  local output_dir="$RUN_ROOT/$name"
  local summary_json="$REPORT_ROOT/${name}_summary.json"
  local log_path="$LOG_ROOT/${name}.train.log"

  echo "$(date -Is) start ${name}"
  echo "  config=${stage_config}"
  echo "  output=${output_dir}"
  echo "  log=${log_path}"

  wait_for_gpu

  COSMOS_DATA_ROOT=/home/pm97/workspace/dataset/distill_dataset \
  COSMOS_DATALOADER_NUM_WORKERS="${COSMOS_DATALOADER_NUM_WORKERS:-4}" \
  COSMOS_DATALOADER_PREFETCH_FACTOR="${COSMOS_DATALOADER_PREFETCH_FACTOR:-2}" \
  COSMOS_DATALOADER_PIN_MEMORY="${COSMOS_DATALOADER_PIN_MEMORY:-1}" \
  COSMOS_DATALOADER_PERSISTENT_WORKERS="${COSMOS_DATALOADER_PERSISTENT_WORKERS:-1}" \
  COSMOS_SKIP_ASSET_CHECK=1 \
  STAGE_CONFIG="$ROOT_DIR/$stage_config" \
  CORPUS_JSONL="$CORPUS_20K" \
  BATCH_SIZE="$batch_size" \
  MAX_TRAIN_SAMPLES="$max_train_samples" \
  MAX_STEPS="$max_steps" \
  EPOCHS=1.0 \
  EVAL_EVERY_EPOCHS="$eval_every_epochs" \
  SAVE_EVERY_EPOCHS="$save_every_epochs" \
  OUTPUT_DIR="$output_dir" \
  SUMMARY_JSON="$summary_json" \
  STUDENT_MODEL=/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b \
  bash "$ROOT_DIR/scripts/43_train_no_nav_backbone_pilot.sh" \
    --multi-gpu off \
    --log-every-steps "${LOG_EVERY_STEPS:-25}" \
    --qat-quantization fp8_pcpt \
    --qat-calib-samples "$qat_calib_samples" \
    > "$log_path" 2>&1

  echo "$(date -Is) done ${name}"
}

run_stage \
  "run0_smoke16_fp8_old_recipe" \
  "configs/train/stage_b0_fp8_old_recipe_smoke16.yaml" \
  1 \
  16 \
  "${RUN0_MAX_STEPS:-32}" \
  0 \
  1.0 \
  "${RUN0_QAT_CALIB_SAMPLES:-8}"

run_stage \
  "run1_20k_fp8_old_recipe_offpolicy" \
  "configs/train/stage_b0_fp8_old_recipe_20k_offpolicy.yaml" \
  "${RUN_BATCH_SIZE:-16}" \
  20000 \
  "${RUN1_MAX_STEPS:-1250}" \
  "${RUN_EVAL_EVERY_EPOCHS:-0.5}" \
  "${RUN_SAVE_EVERY_EPOCHS:-0.5}" \
  "${RUN_QAT_CALIB_SAMPLES:-128}"

run_stage \
  "run2_20k_fp8_old_recipe_late_onpolicy" \
  "configs/train/stage_b0_fp8_old_recipe_20k_late_onpolicy.yaml" \
  "${RUN_BATCH_SIZE:-16}" \
  20000 \
  "${RUN2_MAX_STEPS:-1250}" \
  "${RUN_EVAL_EVERY_EPOCHS:-0.5}" \
  "${RUN_SAVE_EVERY_EPOCHS:-0.5}" \
  "${RUN_QAT_CALIB_SAMPLES:-128}"

echo "$(date -Is) b0_fp8_gkd_runs_done"
