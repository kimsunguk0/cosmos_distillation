#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

WAIT_LOG="outputs/logs/flex_f50_wait_f49_20260607.log"
F49_SUMMARY="outputs/reports/flex_f49_nods_alllora_target32_from_f42_s8000_lr2e7_20260607_final_b0_trajonly_parity_summary.json"
RUN_SCRIPT="scripts/tmp_run_flex_f50_residual_alllora_target32_from_f42_chain.sh"
ADE_PASS_THRESHOLD="${ADE_PASS_THRESHOLD:-0.8}"
CHECK_EVERY_SEC="${CHECK_EVERY_SEC:-120}"

mkdir -p outputs/logs

event() {
  local payload="$1"
  printf '%s\n' "${payload}" | tee -a "${WAIT_LOG}"
}

event "{\"event\":\"f50_wait_start\",\"f49_summary\":\"${F49_SUMMARY}\",\"ade_pass_threshold\":${ADE_PASS_THRESHOLD},\"check_every_sec\":${CHECK_EVERY_SEC}}"

while [[ ! -s "${F49_SUMMARY}" ]]; do
  f49_tmux_alive=0
  if tmux has-session -t flex_f49_wait_alllora32 2>/dev/null; then
    f49_tmux_alive=1
  fi
  event "{\"event\":\"f50_wait_poll\",\"f49_summary_exists\":0,\"f49_wait_tmux_alive\":${f49_tmux_alive}}"
  if [[ "${f49_tmux_alive}" == "0" ]]; then
    event "{\"event\":\"f50_wait_error_f49_ended_without_summary\",\"f49_summary\":\"${F49_SUMMARY}\"}"
    exit 1
  fi
  sleep "${CHECK_EVERY_SEC}"
done

ade="$(
  .venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/reports/flex_f49_nods_alllora_target32_from_f42_s8000_lr2e7_20260607_final_b0_trajonly_parity_summary.json")
o = json.loads(p.read_text())
print(float(o.get("avg_target_ade_m", 999999.0)))
PY
)"

should_skip="$(
  .venv/bin/python - <<PY
ade = float("${ade}")
threshold = float("${ADE_PASS_THRESHOLD}")
print(1 if ade < threshold else 0)
PY
)"

if [[ "${should_skip}" == "1" ]]; then
  event "{\"event\":\"f50_skip_f49_passed\",\"f49_ade_m\":${ade},\"threshold\":${ADE_PASS_THRESHOLD}}"
  exit 0
fi

event "{\"event\":\"f50_start_f49_failed\",\"f49_ade_m\":${ade},\"threshold\":${ADE_PASS_THRESHOLD},\"run_script\":\"${RUN_SCRIPT}\"}"
exec bash "${RUN_SCRIPT}"
