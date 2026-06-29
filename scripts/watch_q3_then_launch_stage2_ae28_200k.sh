#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

Q3_DIR="outputs/action_expert/q3_cosine_cooldown_from_q2best_s26000_2k_20260603_0505"
Q3_SUMMARY="${Q3_DIR}/summary.json"
WATCH_LOG="${Q3_DIR}/stage2_after_q3_watcher.log"

echo "===== WATCH_Q3_THEN_STAGE2 START $(date -Is) =====" | tee -a "$WATCH_LOG"
echo "q3_summary=${Q3_SUMMARY}" | tee -a "$WATCH_LOG"

while true; do
  if [[ -f "$Q3_SUMMARY" ]]; then
    status="$(.venv/bin/python -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); print(d.get("status", ""))' "$Q3_SUMMARY")"
    if [[ "$status" == "ok" ]]; then
      best="$(.venv/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); b=d.get("best_eval") or {}; print(b.get("ade_mean_m", ""))' "$Q3_SUMMARY")"
      echo "Q3 completed ok at $(date -Is); best_val_ade=${best}" | tee -a "$WATCH_LOG"
      break
    fi
    if [[ "$status" == "failed" ]]; then
      echo "Q3 failed; not launching Stage2. See ${Q3_SUMMARY}" | tee -a "$WATCH_LOG"
      exit 1
    fi
  fi
  sleep 60
done

echo "Launching Stage2 200k at $(date -Is)" | tee -a "$WATCH_LOG"
bash scripts/launch_stage2_ae28_200k.sh
