#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/pm97/workspace/sukim/distillation/cosmos_distillation}"
cd "${ROOT}"

CORPUS="${CORPUS:-data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl}"
STUDENT="${STUDENT:-outputs/checkpoints/stepa_q2_vqa_fullft_repaired_v1_bs8_e1/step_003488}"
OUT_ROOT="${OUT_ROOT:-outputs/checkpoints/stepb_ladder_20k}"
REPORT_ROOT="${REPORT_ROOT:-outputs/reports/stepb_ladder_20k}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-3}"
LOG_EVERY="${LOG_EVERY:-25}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
RUN="${RUN:-0}"
INCLUDE_R1="${INCLUDE_R1:-0}"
R1_DISABLE_LORA="${R1_DISABLE_LORA:-1}"
R1_LR="${R1_LR:-3.0e-5}"
RUN_LOW_LR_FULLFT="${RUN_LOW_LR_FULLFT:-0}"

mkdir -p "${OUT_ROOT}" "${REPORT_ROOT}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

common_args=(
  .venv/bin/python -u scripts/09_train_distill.py
  --corpus-jsonl "${CORPUS}"
  --student-model "${STUDENT}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --pin-memory
  --persistent-workers
  --skip-asset-check
  --eval-every-epochs 1.0
  --save-every-epochs 0.3
  --skip-final-save
  --grad-clip-norm 1.0
  --early-stop-stage stage_a
  --early-stop-patience 999
  --log-every-steps "${LOG_EVERY}"
)

run_one() {
  local name="$1"
  local config="$2"
  local disable_lora="$3"
  local lr_override="${4:-}"
  local cmd=("${common_args[@]}"
    --stage-config "${config}"
    --output-dir "${OUT_ROOT}/${name}"
    --summary-json "${REPORT_ROOT}/${name}_summary.json"
  )
  if [[ "${disable_lora}" == "1" ]]; then
    cmd+=(--disable-lora)
  fi
  if [[ -n "${lr_override}" ]]; then
    cmd+=(--learning-rate "${lr_override}")
  fi
  printf '\n# %s\n' "${name}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "${RUN}" == "1" ]]; then
    "${cmd[@]}"
  fi
}

run_one r0_c_bp3_lora_lr2e5 configs/train/stepb_ladder_r0_c_bp3_lora_lr2e5.yaml 0
run_one r0_l_lora_lr2e5 configs/train/stepb_ladder_r0_l_lora_lr2e5.yaml 0
run_one r0_lprime_lora_lr1e4 configs/train/stepb_ladder_r0_lprime_lora_lr1e4.yaml 0
run_one r0_f_fullft_lr1e5 configs/train/stepb_ladder_r0_f_fullft_lr1e5.yaml 1

if [[ "${RUN_LOW_LR_FULLFT}" == "1" ]]; then
  run_one r0_f_fullft_lr5e6 configs/train/stepb_ladder_r0_f_fullft_lr5e6.yaml 1
fi

if [[ "${INCLUDE_R1}" == "1" ]]; then
  run_one r1_tailkl configs/train/stepb_ladder_r1_tailkl.yaml "${R1_DISABLE_LORA}" "${R1_LR}"
fi
