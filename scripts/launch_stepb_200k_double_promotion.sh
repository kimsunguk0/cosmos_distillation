#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/pm97/workspace/sukim/distillation/cosmos_distillation}"
cd "${ROOT}"

RUN="${RUN:-0}"
RUN_ID="${RUN_ID:-double200k_$(date -u +%Y%m%dT%H%M%SZ)}"

STUDENT="${STUDENT:-outputs/checkpoints/stepa_q2_vqa_fullft_repaired_v1_bs8_e1/step_003488}"
CORPUS_200K="${CORPUS_200K:-data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl}"

OUT_ROOT="${OUT_ROOT:-outputs/checkpoints/stepb_200k_double_promotion/${RUN_ID}}"
REPORT_ROOT="${REPORT_ROOT:-outputs/reports/stepb_200k_double_promotion/${RUN_ID}}"

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LOG_EVERY="${LOG_EVERY:-25}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-0.05}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-0.05}"
MAX_KEEP_CHECKPOINTS="${MAX_KEEP_CHECKPOINTS:-4}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-999}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

mkdir -p "${OUT_ROOT}" "${REPORT_ROOT}"
STATUS_JSONL="${REPORT_ROOT}/queue_status.jsonl"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

status_line() {
  local name="$1"
  local state="$2"
  local exit_code="$3"
  local log_path="$4"
  printf '{"time_utc":"%s","run_id":"%s","name":"%s","state":"%s","exit_code":%s,"log":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${RUN_ID}" "${name}" "${state}" "${exit_code}" "${log_path}" \
    >> "${STATUS_JSONL}"
}

run_train() {
  local name="$1"
  local config="$2"
  local disable_lora="$3"
  local log_path="${REPORT_ROOT}/${name}.log"
  local cmd=(
    .venv/bin/python -u scripts/09_train_distill.py
    --corpus-jsonl "${CORPUS_200K}"
    --student-model "${STUDENT}"
    --stage-config "${config}"
    --output-dir "${OUT_ROOT}/${name}"
    --summary-json "${REPORT_ROOT}/${name}_summary.json"
    --batch-size "${BATCH_SIZE}"
    --epochs 1.0
    --num-workers "${NUM_WORKERS}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    --pin-memory
    --persistent-workers
    --skip-asset-check
    --eval-every-epochs "${EVAL_EVERY_EPOCHS}"
    --save-every-epochs "${SAVE_EVERY_EPOCHS}"
    --max-keep-checkpoints "${MAX_KEEP_CHECKPOINTS}"
    --grad-clip-norm "${GRAD_CLIP_NORM}"
    --early-stop-stage stage_a
    --early-stop-patience "${EARLY_STOP_PATIENCE}"
    --log-every-steps "${LOG_EVERY}"
    --skip-final-save
  )
  if [[ "${disable_lora}" == "1" ]]; then
    cmd+=(--disable-lora)
  fi

  printf '\n# %s\n' "${name}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  status_line "${name}" "planned" 0 "${log_path}"

  if [[ "${RUN}" != "1" ]]; then
    return 0
  fi

  status_line "${name}" "started" 0 "${log_path}"
  "${cmd[@]}" > "${log_path}" 2>&1
  local exit_code=$?
  if [[ "${exit_code}" == "0" ]]; then
    status_line "${name}" "completed" 0 "${log_path}"
  else
    status_line "${name}" "failed" "${exit_code}" "${log_path}"
    printf '[queue] %s failed with exit_code=%s; continuing\n' "${name}" "${exit_code}" >&2
  fi
  return 0
}

run_train \
  fullft_lr3e5_200k_e1 \
  configs/train/stepb_fullft_lr3e5_200k_e1.yaml \
  1

run_train \
  lprime_lora_lr2e4_200k_e1 \
  configs/train/stepb_lprime_lora_lr2e4_200k_e1.yaml \
  0

printf '\n[queue] run_id=%s run=%s status=%s\n' "${RUN_ID}" "${RUN}" "${STATUS_JSONL}"
