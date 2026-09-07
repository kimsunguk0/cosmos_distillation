#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-.venv/bin/python}"
CORPUS_JSONL="${CORPUS_JSONL:-data/corpus/val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/reports/kv_probe_444k_ce_20260722}"
DRIVER_LOG="${OUT_DIR}/driver.log"

CKPT_444K_CE="outputs/checkpoints/stepb_ceonly_444k/ceonly_444k_20260718/fullft_lr3e5_ceonly_444k_e1/best_decode"
CKPT_200K_CE="outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/fullft_lr3e5_200k_e1/best_decode"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUT_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${DRIVER_LOG}"
}

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    log "missing required path: ${path}"
    exit 1
  fi
}

summary_has_r2() {
  local summary_json="$1"
  [[ -s "${summary_json}" ]] && grep -q '"hidden_action_r2"' "${summary_json}" && grep -q '"r2"' "${summary_json}"
}

run_ce_r2_probe() {
  local section_name="$1"
  local checkpoint_dir="$2"
  local summary_json="$3"
  local run_log="$4"

  log "${section_name}: requested checkpoint=${checkpoint_dir}"
  if summary_has_r2 "${summary_json}"; then
    log "${section_name}: skip existing R2 summary ${summary_json}"
    return 0
  fi

  log "${section_name}: start hidden/KV representation compare with hidden_action_r2.CE.r2"
  if "${PYTHON}" -u scripts/112_extract_hidden_kv_repr_compare.py \
    --corpus-jsonl "${CORPUS_JSONL}" \
    --checkpoint-dir "CE=${checkpoint_dir}" \
    --ce-baseline-dir CE \
    --split val \
    --num-samples 512 \
    --batch-size 2 \
    --device cuda \
    --image-prompt-style camera_labeled \
    --prompt-text-style official_alpamayo \
    --metric-token-cap 0 \
    --seed 42 \
    --r2-ridge-alpha 1.0 \
    --r2-val-fraction 0.25 \
    --compute-hidden-action-r2 \
    --output-json "${summary_json}" >>"${run_log}" 2>&1; then
    log "${section_name}: done ${summary_json}"
  else
    log "${section_name}: FAIL; see ${run_log}"
    exit 1
  fi
}

log "driver start: output_dir=${OUT_DIR}"
require_path "${PYTHON}"
require_path "${CORPUS_JSONL}"
require_path "${CKPT_444K_CE}"
require_path "${CKPT_200K_CE}"

# 444K CE-only backbone: new target.
run_ce_r2_probe \
  "444K CE" \
  "${CKPT_444K_CE}" \
  "${OUT_DIR}/444k_ce_hidden_kv_repr_compare_val512.json" \
  "${OUT_DIR}/444k_ce.log"

# 200K CE-only backbone: re-verification anchor, expected hidden_action_r2.CE.r2 ~= 0.684.
run_ce_r2_probe \
  "200K CE anchor" \
  "${CKPT_200K_CE}" \
  "${OUT_DIR}/200k_ce_hidden_kv_repr_compare_val512.json" \
  "${OUT_DIR}/200k_ce.log"

log "driver done"
