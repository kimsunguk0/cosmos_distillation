#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="outputs/action_expert/q3_e2e_val512_n6_temp1_seed42_evalbase1042_20260604"
mkdir -p "$OUT_DIR"

MIN_FREE_MIB="${MIN_FREE_MIB:-22000}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-60}"
LOG_PATH="$OUT_DIR/wait_then_run.log"

{
  echo "{\"event\":\"wait_start\",\"min_free_mib\":${MIN_FREE_MIB},\"check_interval_sec\":${CHECK_INTERVAL_SEC},\"time\":\"$(date -Is)\"}"
  while true; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    echo "{\"event\":\"vram_check\",\"free_mib\":${free_mib},\"time\":\"$(date -Is)\"}"
    if [ "${free_mib}" -ge "${MIN_FREE_MIB}" ]; then
      break
    fi
    sleep "${CHECK_INTERVAL_SEC}"
  done

  echo "{\"event\":\"q3_temp1_eval_launch\",\"time\":\"$(date -Is)\"}"
  bash scripts/launch_q3_e2e_val512_n6_temp1_seed1042.sh
  echo "{\"event\":\"q3_temp1_eval_done\",\"time\":\"$(date -Is)\"}"
} >> "$LOG_PATH" 2>&1
