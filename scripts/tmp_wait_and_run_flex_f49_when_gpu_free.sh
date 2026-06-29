#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

WAIT_LOG="outputs/logs/flex_f49_wait_gpu_20260607.log"
RUN_SCRIPT="scripts/tmp_run_flex_f49_nods_alllora_target32_from_f42_chain.sh"
MAX_USED_MB="${MAX_USED_MB:-60000}"
CHECK_EVERY_SEC="${CHECK_EVERY_SEC:-300}"
ALLOW_CONCURRENT_AFTER_VAL_STEP="${ALLOW_CONCURRENT_AFTER_VAL_STEP:-}"
STAGE2_LOG="outputs/action_expert/stage2_200k_more2ep_b3_nt16_lowmem_eval_20260605/train_log.jsonl"

mkdir -p outputs/logs

event() {
  local payload="$1"
  printf '%s\n' "${payload}" | tee -a "${WAIT_LOG}"
}

event "{\"event\":\"f49_wait_start\",\"max_used_mb\":${MAX_USED_MB},\"check_every_sec\":${CHECK_EVERY_SEC},\"allow_concurrent_after_val_step\":\"${ALLOW_CONCURRENT_AFTER_VAL_STEP}\"}"

while true; do
  used_mb="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  stage2_tmux_alive=0
  stage2_train_alive=0
  stage2_last_val_step=0
  if tmux has-session -t stage2_ae_b3_lowmem_eval 2>/dev/null; then
    stage2_tmux_alive=1
  fi
  for proc in /proc/[0-9]*; do
    cmdline="$(tr '\0' ' ' 2>/dev/null < "${proc}/cmdline" || true)"
    case "${cmdline}" in
      *84_train_student_ae28_official.py*stage2_200k_more2ep_b3_nt16_lowmem_eval_20260605*)
        stage2_train_alive=1
        break
        ;;
    esac
  done
  if [[ -n "${ALLOW_CONCURRENT_AFTER_VAL_STEP}" && -s "${STAGE2_LOG}" ]]; then
    stage2_last_val_step="$(
      .venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/action_expert/stage2_200k_more2ep_b3_nt16_lowmem_eval_20260605/train_log.jsonl")
last = 0
for line in p.open():
    try:
        o = json.loads(line)
    except Exception:
        continue
    if o.get("event") == "val_eval":
        last = int(o.get("step") or 0)
print(last)
PY
    )"
  fi
  event "{\"event\":\"f49_wait_poll\",\"used_mb\":${used_mb:-0},\"stage2_tmux_alive\":${stage2_tmux_alive},\"stage2_train_alive\":${stage2_train_alive},\"stage2_last_val_step\":${stage2_last_val_step}}"
  if [[ "${stage2_train_alive}" == "0" && "${used_mb:-999999}" -lt "${MAX_USED_MB}" ]]; then
    break
  fi
  if [[ -n "${ALLOW_CONCURRENT_AFTER_VAL_STEP}" && "${stage2_last_val_step:-0}" -ge "${ALLOW_CONCURRENT_AFTER_VAL_STEP}" && "${used_mb:-999999}" -lt "${MAX_USED_MB}" ]]; then
    break
  fi
  sleep "${CHECK_EVERY_SEC}"
done

event "{\"event\":\"f49_wait_done_starting_run\",\"run_script\":\"${RUN_SCRIPT}\"}"
exec bash "${RUN_SCRIPT}"
