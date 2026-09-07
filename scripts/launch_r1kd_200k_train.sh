#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/pm97/workspace/sukim/distillation/cosmos_distillation"
cd "${ROOT}"
PY=".venv/bin/python"

RUN_ID="r1kd_200k_20260717"
CORPUS_200K="data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl"
STEPA_INIT="outputs/checkpoints/stepa_q2_vqa_fullft_repaired_v1_bs8_e1/step_003488"
CONFIG="configs/train/stepb_fullft_lr3e5_200k_e1_r1kd.yaml"

OUT_ROOT="outputs/checkpoints/stepb_r1kd_200k/${RUN_ID}"
REPORT_ROOT="outputs/reports/stepb_r1kd_200k/${RUN_ID}"
NAME="fullft_lr3e5_r1kd_200k_e1"
LOG="${REPORT_ROOT}/${NAME}.log"
mkdir -p "${OUT_ROOT}" "${REPORT_ROOT}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

{
  printf 'launch_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'config=%s\n' "${CONFIG}"
  printf 'corpus=%s\n' "${CORPUS_200K}"
  printf 'init=%s\n' "${STEPA_INIT}"
  printf 'out=%s\n' "${OUT_ROOT}/${NAME}"
} > "${LOG}"

"${PY}" -u scripts/09_train_distill.py \
  --corpus-jsonl "${CORPUS_200K}" \
  --student-model "${STEPA_INIT}" \
  --stage-config "${CONFIG}" \
  --output-dir "${OUT_ROOT}/${NAME}" \
  --summary-json "${REPORT_ROOT}/${NAME}_summary.json" \
  --batch-size 8 \
  --epochs 1.0 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --pin-memory \
  --persistent-workers \
  --skip-asset-check \
  --eval-every-epochs 0.2 \
  --save-every-epochs 0.3 \
  --max-keep-checkpoints 4 \
  --grad-clip-norm 1.0 \
  --early-stop-stage stage_a \
  --early-stop-patience 4 \
  --log-every-steps 25 \
  --disable-lora \
  >> "${LOG}" 2>&1

printf 'exit_code=%s\nfinished_utc=%s\n' "$?" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG}"
