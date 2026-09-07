#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/pm97/workspace/sukim/distillation/cosmos_distillation}"
PY="${PY:-/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/.venv/bin/python}"
OUT="${OUT:-outputs/eval/vlm_cap_gap_human_ood_20260701_full300}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-14400}"
SCHEDULER_SESSION="${SCHEDULER_SESSION:-vlm_gap_full300_scheduler}"
SCHEDULER_SCRIPT="${SCHEDULER_SCRIPT:-scripts/run_vlm_cap_gap_full300_scheduler.sh}"
LOG_DIR="${OUT}/logs"
LOCK_DIR="${OUT}/locks/monitor_4h.lock"

cd "${ROOT}"
mkdir -p "${LOG_DIR}"
mkdir -p "${OUT}/locks"

if mkdir "${LOCK_DIR}" 2>/dev/null; then
  trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT
else
  echo "monitor_lock_exists=${LOCK_DIR} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 0
fi

utc_now() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

utc_from_epoch() {
  python3 - "$1" <<'PY'
from __future__ import annotations
import datetime as dt
import sys

epoch = int(sys.argv[1])
print(dt.datetime.fromtimestamp(epoch, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
PY
}

sleep_until_epoch() {
  local target_epoch="$1"
  while true; do
    local now_epoch remaining
    now_epoch="$(date -u +%s)"
    remaining=$((target_epoch - now_epoch))
    if [[ "${remaining}" -le 0 ]]; then
      return
    fi
    sleep "${remaining}"
  done
}

status_json() {
  "${PY}" scripts/vlm_cap_gap_status.py --output-dir "${OUT}"
}

expected_total_rows() {
  status_json | python3 -c 'import json,sys; print(json.load(sys.stdin).get("total_expected_rows", 0))'
}

current_total_rows() {
  status_json | python3 -c 'import json,sys; print(json.load(sys.stdin).get("total_rows", 0))'
}

ensure_scheduler() {
  local total expected
  total="$(current_total_rows || echo 0)"
  expected="$(expected_total_rows || echo 0)"
  if [[ "${expected}" -gt 0 && "${total}" -ge "${expected}" ]]; then
    echo "scheduler_check: all model rows complete (${total}/${expected}); no restart needed"
    return
  fi
  if tmux has-session -t "${SCHEDULER_SESSION}" 2>/dev/null; then
    echo "scheduler_check: ${SCHEDULER_SESSION} alive"
    return
  fi
  echo "scheduler_check: ${SCHEDULER_SESSION} missing; restarting"
  tmux new-session -d -s "${SCHEDULER_SESSION}" \
    "cd ${ROOT} && bash ${SCHEDULER_SCRIPT} 2>&1 | tee -a ${LOG_DIR}/scheduler.log"
}

snapshot() {
  local label="$1"
  echo "===== ${label} $(utc_now) ====="
  ensure_scheduler
  echo "--- status ---"
  status_json
  echo "--- tmux ---"
  tmux list-sessions 2>/dev/null | rg 'vlm_gap_full300|$' || true
  echo "--- gpu ---"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  echo "--- scheduler tail ---"
  tail -n 30 "${LOG_DIR}/scheduler.log" 2>/dev/null || true
  echo "===== end $(utc_now) ====="
}

START_EPOCH="$(date -u +%s)"
CHECK_INDEX=0

snapshot "initial"
while true; do
  CHECK_INDEX=$((CHECK_INDEX + 1))
  next_epoch=$((START_EPOCH + CHECK_INDEX * INTERVAL_SECONDS))
  next_check="$(utc_from_epoch "${next_epoch}")"
  echo "next_check_utc=${next_check} interval_seconds=${INTERVAL_SECONDS} schedule_start_utc=$(utc_from_epoch "${START_EPOCH}") check_index=${CHECK_INDEX}"
  sleep_until_epoch "${next_epoch}"
  snapshot "periodic"
done
