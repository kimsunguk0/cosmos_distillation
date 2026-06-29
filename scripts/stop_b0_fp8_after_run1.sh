#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

LOG="logs/b0_fp8_gkd_20260618/queue.log"
WATCH="/tmp/b0_fp8_stop_after_run1.tmux.log"

echo "$(date -Is) tmux_watcher_start" >> "$WATCH"

while true; do
  if rg -q "done run1_20k_fp8_old_recipe_offpolicy" "$LOG" 2>/dev/null; then
    echo "$(date -Is) watcher_stop_before_old_run2" >> "$LOG"
    echo "$(date -Is) tmux_watcher_kill" >> "$WATCH"
    tmux kill-session -t b0_fp8_gkd_20260618 2>/dev/null || true
    exit 0
  fi
  sleep 2
done
