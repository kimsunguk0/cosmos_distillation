#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_PID="${TRAIN_PID:-907}"
STAGE2_DIR="${STAGE2_DIR:-outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603}"
MIN_FREE_MIB="${MIN_FREE_MIB:-22000}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-60}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-outputs/action_expert/post_stage2_minade6_queue_${RUN_TAG}}"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/watch.log"

{
  echo "{\"event\":\"post_stage2_wait_start\",\"time\":\"$(date -Is)\",\"train_pid\":${TRAIN_PID},\"stage2_dir\":\"${STAGE2_DIR}\",\"min_free_mib\":${MIN_FREE_MIB}}"

  while [ -e "/proc/${TRAIN_PID}" ]; do
    latest_step="$(
      .venv/bin/python - <<'PY' 2>/dev/null || true
import json
from pathlib import Path
p=Path("outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603/train_log.jsonl")
step=None
if p.exists():
    for line in p.read_text().splitlines():
        if line.strip():
            r=json.loads(line)
            if r.get("event")=="train_step":
                step=r.get("step")
print(step if step is not None else "")
PY
    )"
    echo "{\"event\":\"train_still_running\",\"time\":\"$(date -Is)\",\"latest_step\":\"${latest_step}\"}"
    sleep "${CHECK_INTERVAL_SEC}"
  done

  echo "{\"event\":\"train_pid_exited\",\"time\":\"$(date -Is)\",\"train_pid\":${TRAIN_PID}}"
  while [ ! -f "${STAGE2_DIR}/final.pt" ]; do
    echo "{\"event\":\"waiting_for_final_checkpoint\",\"time\":\"$(date -Is)\",\"path\":\"${STAGE2_DIR}/final.pt\"}"
    sleep "${CHECK_INTERVAL_SEC}"
  done

  while true; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    echo "{\"event\":\"vram_check\",\"time\":\"$(date -Is)\",\"free_mib\":${free_mib}}"
    if [ "${free_mib}" -ge "${MIN_FREE_MIB}" ]; then
      break
    fi
    sleep "${CHECK_INTERVAL_SEC}"
  done

  echo "{\"event\":\"stage2_minade6_eval_start\",\"time\":\"$(date -Is)\"}"
  RUN_TAG="${RUN_TAG}_stage2" bash scripts/launch_stage2_200k_best_minade6_eval.sh
  echo "{\"event\":\"stage2_minade6_eval_done\",\"time\":\"$(date -Is)\"}"

  echo "{\"event\":\"q3_minade6_eval_start\",\"time\":\"$(date -Is)\"}"
  RUN_TAG="${RUN_TAG}_q3" bash scripts/launch_q3_minade6_temp_sweep_seed1042.sh
  echo "{\"event\":\"q3_minade6_eval_done\",\"time\":\"$(date -Is)\"}"
} >> "$LOG_PATH" 2>&1
