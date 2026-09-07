#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/pm97/workspace/sukim/distillation/cosmos_distillation}"
PY="${PY:-/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/.venv/bin/python}"
OUT="${OUT:-outputs/eval/vlm_cap_gap_human_ood_20260701_full300}"
JUDGE_NAME="${JUDGE_NAME:-open_judge_codex55_xhigh_full300}"
JUDGE_LIMIT_PER_TASK="${JUDGE_LIMIT_PER_TASK:-50}"
EXPECTED_ROWS="${EXPECTED_ROWS:-9000}"
EXPECTED_CLIPS="${EXPECTED_CLIPS:-300}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"

cd "${ROOT}"
mkdir -p "${OUT}/logs"
mkdir -p "${OUT}/locks"

row_count() {
  local model="$1"
  local path="${OUT}/runs/${model}/predictions.jsonl"
  if [[ -f "${path}" ]]; then
    wc -l < "${path}"
  else
    echo 0
  fi
}

if [[ "${ALLOW_PARTIAL}" != "1" ]]; then
  r2="$(row_count 2b)"
  r8="$(row_count 8b)"
  r32="$(row_count 32b)"
  if [[ "${r2}" -lt "${EXPECTED_ROWS}" || "${r8}" -lt "${EXPECTED_ROWS}" || "${r32}" -lt "${EXPECTED_ROWS}" ]]; then
    echo "Postprocess refused partial rows: 2b=${r2}/${EXPECTED_ROWS} 8b=${r8}/${EXPECTED_ROWS} 32b=${r32}/${EXPECTED_ROWS}"
    echo "Set ALLOW_PARTIAL=1 only for intentional debug scoring."
    exit 1
  fi
fi

LOCK_DIR="${OUT}/locks/postprocess.lock"
if mkdir "${LOCK_DIR}" 2>/dev/null; then
  trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT
else
  echo "Postprocess already running or lock exists: ${LOCK_DIR}"
  exit 0
fi

"${PY}" scripts/vlm_cap_gap_eval.py score \
  --output-dir "${OUT}" \
  --models 2b,8b,32b

"${PY}" scripts/export_vlm_open_judge_pack.py \
  --output-dir "${OUT}" \
  --out-name "${JUDGE_NAME}" \
  --limit-per-task "${JUDGE_LIMIT_PER_TASK}"

"${PY}" scripts/export_vlm_cap_gap_qualitative_dump.py \
  --output-dir "${OUT}" \
  --limit 12

JUDGE_DIR="${OUT}/${JUDGE_NAME}"
"${PY}" scripts/check_vlm_codex_judge_ready.py \
  --output-dir "${OUT}" \
  --judge-name "${JUDGE_NAME}" \
  --expected-pairs "$((JUDGE_LIMIT_PER_TASK * 2))" \
  --out "${JUDGE_DIR}/judge_ready.json"

"${PY}" scripts/audit_vlm_cap_gap_completion.py \
  --output-dir "${OUT}" \
  --expected-clips "${EXPECTED_CLIPS}" \
  --expected-rows-per-model "${EXPECTED_ROWS}" \
  --judge-name "${JUDGE_NAME}" \
  --expected-judge-pairs "$((JUDGE_LIMIT_PER_TASK * 2))" \
  > "${OUT}/completion_audit_pre_judge.json"
echo "Pre-judge completion audit wrote ${OUT}/completion_audit_pre_judge.json"

if [[ -f "${JUDGE_DIR}/judgments_blind.jsonl" ]]; then
  "${PY}" scripts/validate_vlm_open_judge.py \
    --judge-dir "${JUDGE_DIR}" \
    --out "${JUDGE_DIR}/judgments_validation.json"
  "${PY}" scripts/score_vlm_open_judge.py \
    --judge-dir "${JUDGE_DIR}" \
    --append-report "${OUT}/report.md"
  "${PY}" scripts/summarize_vlm_cap_gap_results.py \
    --output-dir "${OUT}" \
    --judge-name "${JUDGE_NAME}" \
    --out "${OUT}/decision_brief.md"
  "${PY}" scripts/audit_vlm_cap_gap_completion.py \
    --output-dir "${OUT}" \
    --expected-clips "${EXPECTED_CLIPS}" \
    --expected-rows-per-model "${EXPECTED_ROWS}" \
    --judge-name "${JUDGE_NAME}" \
    --expected-judge-pairs "$((JUDGE_LIMIT_PER_TASK * 2))" \
    --require-judge \
    > "${OUT}/completion_audit_final.json"
  echo "Final completion audit wrote ${OUT}/completion_audit_final.json"
  "${PY}" scripts/summarize_vlm_cap_gap_results.py \
    --output-dir "${OUT}" \
    --judge-name "${JUDGE_NAME}" \
    --out "${OUT}/decision_brief.md"
  echo "Decision brief wrote ${OUT}/decision_brief.md"
else
  echo "Judge pack ready at ${JUDGE_DIR}"
  echo "Waiting for codex-5.5 xhigh to write ${JUDGE_DIR}/judgments_blind.jsonl"
fi
