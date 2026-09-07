#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/pm97/workspace/sukim/distillation/cosmos_distillation}"
PY="${PY:-/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/.venv/bin/python}"
OUT="${OUT:-outputs/eval/vlm_cap_gap_human_ood_20260701_full300}"
EXPECTED_ROWS="${EXPECTED_ROWS:-9000}"
POLL_SECONDS="${POLL_SECONDS:-600}"
LOG_DIR="${OUT}/logs"
SCHEDULER_SESSION="${SCHEDULER_SESSION:-vlm_gap_full300_scheduler}"

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

row_count() {
  local model="$1"
  local path="${OUT}/runs/${model}/predictions.jsonl"
  if [[ -f "${path}" ]]; then
    wc -l < "${path}"
  else
    echo 0
  fi
}

utc_now() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

while true; do
  r2="$(row_count 2b)"
  r8="$(row_count 8b)"
  r32="$(row_count 32b)"
  echo "[$(utc_now)] postprocess_watch rows: 2b=${r2} 8b=${r8} 32b=${r32}"
  if [[ "${r2}" -ge "${EXPECTED_ROWS}" && "${r8}" -ge "${EXPECTED_ROWS}" && "${r32}" -ge "${EXPECTED_ROWS}" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

echo "[$(utc_now)] rows complete; running postprocess"
while tmux has-session -t "${SCHEDULER_SESSION}" 2>/dev/null; do
  echo "[$(utc_now)] ${SCHEDULER_SESSION} still alive; waiting before watcher postprocess"
  sleep 300
done

JUDGE_DIR="${OUT}/${JUDGE_NAME:-open_judge_codex55_xhigh_full300}"
if [[ -f "${OUT}/completion_audit_pre_judge.json" && ! -f "${JUDGE_DIR}/judgments_blind.jsonl" ]]; then
  echo "[$(utc_now)] pre-judge postprocess already complete and no blind judgments yet; watcher will not rerun"
  exit 0
fi

bash scripts/run_vlm_cap_gap_postprocess.sh 2>&1 | tee -a "${LOG_DIR}/postprocess_watch.log"
echo "[$(utc_now)] postprocess complete"
