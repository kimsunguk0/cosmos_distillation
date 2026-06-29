#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

METRICS="outputs/checkpoints/b0_fp8_vit_step006250_20260618/run2_20k_fp8vit_late_onpolicy_from_step006250_val512_b8/metrics.jsonl"
LOG_PATH="logs/b0_fp8_vit_step006250_20260618/run2_onpolicy_activation_watch.log"
TRAIN_TMUX="b0_fp8_vit_run2_late_onpolicy_b8_20260618"

mkdir -p logs/b0_fp8_vit_step006250_20260618
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"onpolicy_watch_start\",\"time\":\"$(date -Is)\",\"metrics\":\"${METRICS}\"}"

while true; do
  set +e
  .venv/bin/python - <<'PY'
import json
import sys
from pathlib import Path

p = Path("outputs/checkpoints/b0_fp8_vit_step006250_20260618/run2_20k_fp8vit_late_onpolicy_from_step006250_val512_b8/metrics.jsonl")
rows = []
if p.exists():
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("phase") == "train":
                rows.append(row)
if not rows:
    print(json.dumps({"event": "onpolicy_wait", "reason": "no_train_rows"}), flush=True)
    sys.exit(3)

last = rows[-1]
activated = [
    row for row in rows
    if int(row.get("global_step", 0) or 0) >= 875
    and float((row.get("logs") or {}).get("scheduled_sampling_replaced", 0.0) or 0.0) > 0
]
last_logs = last.get("logs") or {}
status = {
    "event": "onpolicy_status",
    "step": int(last.get("global_step", 0) or 0),
    "ss_p": float(last_logs.get("scheduled_sampling_probability", 0.0) or 0.0),
    "ss_replaced": float(last_logs.get("scheduled_sampling_replaced", 0.0) or 0.0),
    "ss_candidates": float(last_logs.get("scheduled_sampling_candidates", 0.0) or 0.0),
    "activated_rows": len(activated),
    "first_activated_step": int(activated[0].get("global_step", 0)) if activated else None,
    "max_replaced_after_875": max(
        [
            float((row.get("logs") or {}).get("scheduled_sampling_replaced", 0.0) or 0.0)
            for row in rows
            if int(row.get("global_step", 0) or 0) >= 875
        ]
        or [0.0]
    ),
}
print(json.dumps(status), flush=True)
if activated:
    sys.exit(0)
if int(last.get("global_step", 0) or 0) >= 1050:
    sys.exit(2)
sys.exit(3)
PY
  rc=$?
  set -e
  if [[ "${rc}" == "0" ]]; then
    echo "{\"event\":\"onpolicy_activation_confirmed\",\"time\":\"$(date -Is)\"}"
    exit 0
  fi
  if [[ "${rc}" == "2" ]]; then
    echo "{\"event\":\"onpolicy_activation_failed\",\"time\":\"$(date -Is)\",\"reason\":\"no_replacement_by_step_1050\"}"
    exit 2
  fi
  if ! tmux has-session -t "${TRAIN_TMUX}" 2>/dev/null; then
    echo "{\"event\":\"onpolicy_watch_train_session_missing\",\"time\":\"$(date -Is)\",\"tmux\":\"${TRAIN_TMUX}\"}"
    exit 3
  fi
  sleep 60
done
