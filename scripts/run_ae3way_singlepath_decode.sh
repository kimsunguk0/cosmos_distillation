#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/pm97/workspace/sukim/distillation/cosmos_distillation"
cd "${ROOT}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE="outputs/action_expert/ae3way_singlepath_20260722"
mkdir -p "${BASE}"
DRIVER_LOG="${BASE}/driver.log"
SWEEP_JSON='[{"label":"np1_t0p1","eval_temperature":0.1,"eval_num_paths":1,"eval_selection_method":"single"},{"label":"np1_t0p5","eval_temperature":0.5,"eval_num_paths":1,"eval_selection_method":"single"},{"label":"np1_t0p85","eval_temperature":0.85,"eval_num_paths":1,"eval_selection_method":"single"},{"label":"np1_t1p0","eval_temperature":1.0,"eval_num_paths":1,"eval_selection_method":"single"},{"label":"np6_t0p85_oracle","eval_temperature":0.85,"eval_num_paths":6,"eval_selection_method":"oracle_best"}]'

say() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "${DRIVER_LOG}"
}

run_one() {
  local label="$1"
  local AE_BEST="$2"
  local BACKBONE="$3"
  local OUT="${BASE}/${label}"
  local log="${BASE}/${label}.log"

  if [[ -s "${OUT}/summary.json" ]]; then
    say "skip ${label}: ${OUT}/summary.json already exists"
    return 0
  fi

  say "start ${label}: backbone=${BACKBONE} ae=${AE_BEST} out=${OUT}"
  .venv/bin/python -u scripts/84_train_student_ae28_official.py \
    --student-checkpoint-dir "${BACKBONE}" \
    --resume-ae-checkpoint "${AE_BEST}" \
    --eval-only \
    --num-samples 200000 --val-samples 10000 \
    --split-cache-json outputs/action_expert/stage2_heldout200k_val10k_seed42_20260603/split_cache_200k_10k_seed42.json --split-scan-all \
    --eval-batch-size 8 \
    --prefix-mode teacher_forced --ae-init-mode teacher_compressed --target-source teacher \
    --eval-samples 1024 --eval-seed-mode fixed --eval-vectorize-paths \
    --eval-num-paths 1 --eval-selection-method single --eval-temperature 1.0 \
    --eval-sweep-json "${SWEEP_JSON}" \
    --teacher-load-device cpu --device cuda:0 --attn-implementation flash_attention_2 --seed 42 \
    --output-dir "${OUT}" >> "${log}" 2>&1
  local status=$?

  if [[ "${status}" -eq 0 ]]; then
    say "finish ${label}: success"
  else
    say "finish ${label}: failed exit=${status}"
  fi
  return "${status}"
}

say "driver start: base=${BASE}"

run_one ce_fullft \
  outputs/action_expert/ae3way_20260720_ce_fullft_teacherforced/best.pt \
  outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/fullft_lr3e5_200k_e1/best_decode
status=$?
if [[ "${status}" -ne 0 ]]; then
  say "driver abort: ce_fullft failed exit=${status}"
  exit "${status}"
fi

run_one r1kd_fullft \
  outputs/action_expert/ae3way_20260720_r1kd_fullft_teacherforced/best.pt \
  outputs/checkpoints/stepb_r1kd_200k/r1kd_200k_20260717/fullft_lr3e5_r1kd_200k_e1/best_decode
status=$?
if [[ "${status}" -ne 0 ]]; then
  say "driver abort: r1kd_fullft failed exit=${status}"
  exit "${status}"
fi

run_one lora_ce \
  outputs/action_expert/ae3way_20260720_lora_ce_teacherforced/best.pt \
  outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/lprime_lora_lr2e4_200k_e1/best_decode
status=$?
if [[ "${status}" -ne 0 ]]; then
  say "driver abort: lora_ce failed exit=${status}"
  exit "${status}"
fi

say "driver finish: all runs complete or skipped"
touch "${BASE}/_COMPLETE"
