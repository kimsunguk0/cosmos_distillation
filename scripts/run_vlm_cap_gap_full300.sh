#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/pm97/workspace/sukim/distillation/cosmos_distillation"
PY="/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/.venv/bin/python"
OUT="${OUT:-outputs/eval/vlm_cap_gap_human_ood_20260701_full300}"
MANIFEST="${MANIFEST:-${OUT}/manifest.jsonl}"
SAMPLE_N="${SAMPLE_N:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
LOG_DIR="${OUT}/logs"

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${MANIFEST}" ]]; then
  "${PY}" scripts/vlm_cap_gap_eval.py build-manifest \
    --output-dir "${OUT}" \
    --limit-clips 300 \
    --overwrite \
    | tee "${LOG_DIR}/manifest.log"
fi

for MODEL in 2b 8b 32b; do
  TS="$(date -u +%Y%m%dT%H%M%SZ)"
  LOG="${LOG_DIR}/${MODEL}_${TS}.log"
  echo "[${TS}] starting ${MODEL} sample_n=${SAMPLE_N}" | tee -a "${LOG}"
  "${PY}" scripts/vlm_cap_gap_eval.py run-model \
    --output-dir "${OUT}" \
    --manifest "${MANIFEST}" \
    --model-key "${MODEL}" \
    --limit-clips 300 \
    --tasks T1_pos,T1_neg,T2,T3,T4 \
    --sample-n "${SAMPLE_N}" \
    --sample-temperature 0.7 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --log-every 25 \
    2>&1 | tee -a "${LOG}"

  "${PY}" scripts/vlm_cap_gap_eval.py score \
    --output-dir "${OUT}" \
    --models 2b,8b,32b \
    2>&1 | tee -a "${LOG}"
done

"${PY}" scripts/vlm_cap_gap_eval.py score \
  --output-dir "${OUT}" \
  --models 2b,8b,32b \
  2>&1 | tee -a "${LOG_DIR}/final_score.log"

"${PY}" scripts/export_vlm_open_judge_pack.py \
  --output-dir "${OUT}" \
  --out-name open_judge_codex55_xhigh_full300 \
  --limit-per-task 50 \
  2>&1 | tee -a "${LOG_DIR}/judge_pack.log"

echo "DONE ${OUT}"
