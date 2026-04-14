#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/pm97/workspace/sukim/cosmos_distillation"
PYTHON_BIN="/home/pm97/workspace/kjhong/alpamayo_100/ar1_venv/bin/python"
SUMMARY_JSON="$PROJECT_ROOT/outputs/reports/priority_next2_40_run.json"
OUTPUT_JSON="$PROJECT_ROOT/outputs/reports/priority_next2_40_progress.json"
LOG_PATH="$PROJECT_ROOT/outputs/logs/priority_next2_40_retry_v2.log"
PID_FILE="$PROJECT_ROOT/outputs/logs/priority_next2_40_retry_v2.pid"
DONE_FILE="$PROJECT_ROOT/outputs/logs/priority_next2_40_retry_v2.done"
WATCH_LOG="$PROJECT_ROOT/outputs/logs/priority_next2_40_progress_watch_v2.log"

mkdir -p "$(dirname "$WATCH_LOG")"

while true; do
  (
    cd "$PROJECT_ROOT"
    export PYTHONPATH="$PROJECT_ROOT/.vendor:$PROJECT_ROOT"
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/03_report_priority_progress.py" \
      --summary-json "$SUMMARY_JSON" \
      --output-json "$OUTPUT_JSON" \
      --log-path "$LOG_PATH" \
      --pid-file "$PID_FILE" \
      --pid-cmd-substring "03_download_priority_chunks.py"
  ) >>"$WATCH_LOG" 2>&1 || true

  if [ -f "$DONE_FILE" ] && [ ! -f "$PID_FILE" ]; then
    break
  fi
  sleep 60
done
