#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/pm97/workspace/sukim/distillation/cosmos_distillation}"
cd "${ROOT}"

RUN="${RUN:-0}"
RUN_ID="${RUN_ID:-weekend_$(date -u +%Y%m%dT%H%M%SZ)}"

STUDENT="${STUDENT:-outputs/checkpoints/stepa_q2_vqa_fullft_repaired_v1_bs8_e1/step_003488}"
STUDENT_8B="${STUDENT_8B:-/home/pm97/workspace/sukim/base_weights/Cosmos-Reason2-8B}"
CORPUS_20K="${CORPUS_20K:-data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl}"
CORPUS_200K="${CORPUS_200K:-data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl}"

OUT_ROOT="${OUT_ROOT:-outputs/checkpoints/stepb_weekend_queue/${RUN_ID}}"
REPORT_ROOT="${REPORT_ROOT:-outputs/reports/stepb_weekend_queue/${RUN_ID}}"

BATCH_SIZE="${BATCH_SIZE:-8}"
BATCH_SIZE_8B="${BATCH_SIZE_8B:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LOG_EVERY="${LOG_EVERY:-25}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-0.2}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-0.3}"
MAX_KEEP_CHECKPOINTS="${MAX_KEEP_CHECKPOINTS:-4}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

RUN_R1="${RUN_R1:-1}"
RUN_SOUP_1E4="${RUN_SOUP_1E4:-1}"
RUN_LPRIME_200K="${RUN_LPRIME_200K:-1}"
RUN_8B_SMOKE="${RUN_8B_SMOKE:-0}"

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
  local corpus="$3"
  local student="$4"
  local disable_lora="$5"
  local epochs="$6"
  local batch_size="$7"
  local max_steps="${8:-}"
  local log_path="${REPORT_ROOT}/${name}.log"

  local cmd=(
    .venv/bin/python -u scripts/09_train_distill.py
    --corpus-jsonl "${corpus}"
    --student-model "${student}"
    --stage-config "${config}"
    --output-dir "${OUT_ROOT}/${name}"
    --summary-json "${REPORT_ROOT}/${name}_summary.json"
    --batch-size "${batch_size}"
    --epochs "${epochs}"
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
  )
  if [[ "${disable_lora}" == "1" ]]; then
    cmd+=(--disable-lora)
  fi
  if [[ -n "${max_steps}" ]]; then
    cmd+=(--max-steps "${max_steps}")
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

if [[ "${RUN_R1}" == "1" ]]; then
  run_train \
    r1_fullft_lr3e5_tailkl_tau1_20k \
    configs/train/stepb_ladder_r1_tailkl.yaml \
    "${CORPUS_20K}" "${STUDENT}" 1 3.0 "${BATCH_SIZE}"
fi

if [[ "${RUN_SOUP_1E4}" == "1" ]]; then
  run_train \
    r0_c_bp3_lora_lr1e4_20k \
    configs/train/stepb_ladder_r0_c_bp3_lora_lr1e4.yaml \
    "${CORPUS_20K}" "${STUDENT}" 0 3.0 "${BATCH_SIZE}"
fi

if [[ "${RUN_LPRIME_200K}" == "1" ]]; then
  run_train \
    lprime_lora_lr1e4_200k_e1 \
    configs/train/stepb_lprime_lora_lr1e4_200k_e1.yaml \
    "${CORPUS_200K}" "${STUDENT}" 0 1.0 "${BATCH_SIZE}"
fi

if [[ "${RUN_8B_SMOKE}" == "1" ]]; then
  run_train \
    r0_8b_lprime_lora_lr1e4_smoke100 \
    configs/train/stepb_ladder_r0_lprime_lora_lr1e4.yaml \
    "${CORPUS_20K}" "${STUDENT_8B}" 0 1.0 "${BATCH_SIZE_8B}" 100
fi

printf '\n[queue] run_id=%s run=%s status=%s\n' "${RUN_ID}" "${RUN}" "${STATUS_JSONL}"
