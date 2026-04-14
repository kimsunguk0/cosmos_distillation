#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/pm97/workspace/sukim/cosmos_distillation"
PYTHON_BIN="/home/pm97/workspace/kjhong/alpamayo_100/ar1_venv/bin/python"
SUMMARY_JSON="$PROJECT_ROOT/outputs/reports/priority_next2_40_run.json"
LOG_PATH="$PROJECT_ROOT/outputs/logs/priority_next2_40_retry_v2.log"
PID_FILE="$PROJECT_ROOT/outputs/logs/priority_next2_40_retry_v2.pid"
DONE_FILE="$PROJECT_ROOT/outputs/logs/priority_next2_40_retry_v2.done"

mkdir -p "$(dirname "$LOG_PATH")"
rm -f "$DONE_FILE"

cleanup() {
  rm -f "$PID_FILE"
}
trap cleanup EXIT

attempt=1
while true; do
  {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] attempt=$attempt start"
  } >>"$LOG_PATH"

  (
    cd "$PROJECT_ROOT"
    export HF_HUB_DISABLE_XET=1
    export PYTHONPATH="$PROJECT_ROOT/.vendor:$PROJECT_ROOT"
    exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/03_download_priority_chunks.py" \
      --manifest-path "$PROJECT_ROOT/data/processed/strict_download_subset_manifest.parquet" \
      --selected-chunks-json "$SUMMARY_JSON" \
      --summary-json "$SUMMARY_JSON" \
      --max-workers 8
  ) >>"$LOG_PATH" 2>&1 &

  child_pid=$!
  echo "$child_pid" >"$PID_FILE"
  wait "$child_pid"
  rc=$?

  {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] attempt=$attempt exit_code=$rc"
  } >>"$LOG_PATH"

  rm -f "$PID_FILE"

  if [ "$rc" -eq 0 ]; then
    date '+%Y-%m-%d %H:%M:%S' >"$DONE_FILE"
    exit 0
  fi

  attempt=$((attempt + 1))
  sleep 30
done
