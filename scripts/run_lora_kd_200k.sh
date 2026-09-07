#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/pm97/workspace/sukim/distillation/cosmos_distillation"
cd "${ROOT}"

PY=".venv/bin/python"
NAME="lprime_lora_lr2e4_200k_e1_r1kd"
RUN_ID="lora_kd_200k_20260723"

CORPUS_200K="data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl"
STUDENT="outputs/checkpoints/stepa_q2_vqa_fullft_repaired_v1_bs8_e1/step_003488"
CONFIG="configs/train/stepb_lprime_lora_lr2e4_200k_e1_r1kd.yaml"

OUT_ROOT="outputs/checkpoints/stepb_lora_kd_200k/${RUN_ID}"
REPORT_ROOT="outputs/reports/stepb_lora_kd_200k/${RUN_ID}"
OUT_DIR="${OUT_ROOT}/${NAME}"
SUMMARY_JSON="${REPORT_ROOT}/${NAME}_summary.json"
DRIVER_LOG="${REPORT_ROOT}/driver.log"
TRAIN_LOG="${REPORT_ROOT}/${NAME}.log"
PID_FILE="${REPORT_ROOT}/${NAME}.pid"

FINAL_STEP_CHECKPOINT="${OUT_DIR}/step_025000/checkpoint_manifest.json"
FINAL_DIR_CHECKPOINT="${OUT_DIR}/final/checkpoint_manifest.json"

mkdir -p "${OUT_DIR}" "${REPORT_ROOT}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

say() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >> "${DRIVER_LOG}"
}

if [[ -f "${FINAL_STEP_CHECKPOINT}" ]]; then
  say "SKIP ${NAME}: final step checkpoint exists at ${FINAL_STEP_CHECKPOINT}"
  exit 0
fi

if [[ -f "${FINAL_DIR_CHECKPOINT}" ]]; then
  say "SKIP ${NAME}: final checkpoint exists at ${FINAL_DIR_CHECKPOINT}"
  exit 0
fi

if [[ -f "${PID_FILE}" ]]; then
  pid="$(<"${PID_FILE}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && [[ -d "/proc/${pid}" ]]; then
    say "SKIP ${NAME}: process already running with pid ${pid}"
    exit 0
  fi
fi

cmd=(
  "${PY}" -u scripts/09_train_distill.py
  --corpus-jsonl "${CORPUS_200K}"
  --student-model "${STUDENT}"
  --stage-config "${CONFIG}"
  --output-dir "${OUT_DIR}"
  --summary-json "${SUMMARY_JSON}"
  --batch-size 8
  --epochs 1.0
  --num-workers 8
  --prefetch-factor 2
  --pin-memory
  --persistent-workers
  --skip-asset-check
  --eval-every-epochs 0.05
  --save-every-epochs 0.05
  --max-keep-checkpoints 4
  --grad-clip-norm 1.0
  --early-stop-stage stage_a
  --early-stop-patience 999
  --log-every-steps 25
  --skip-final-save
)

printf -v cmd_q '%q ' "${cmd[@]}"
say "START ${NAME}"
say "command=${cmd_q}"
say "logs train=${TRAIN_LOG} summary=${SUMMARY_JSON}"

nohup "${cmd[@]}" >> "${TRAIN_LOG}" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${PID_FILE}"
say "launched pid=${pid}"

disown "${pid}" 2>/dev/null || true
