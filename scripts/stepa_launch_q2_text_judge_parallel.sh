#!/usr/bin/env bash
set -euo pipefail

JUDGE_DIR="${1:-data/vqa_q2_stepa_pilot50k/teacher_q2_t0p60_parallel_merged/text_judge_gpt55_medium}"
PARALLEL="${PARALLEL:-4}"
TIMEOUT_S="${TIMEOUT_S:-900}"
RETRIES="${RETRIES:-1}"
SESSION_PREFIX="${SESSION_PREFIX:-stepa_q2_text_judge}"

cd "$(dirname "$0")/.."

manifest="$JUDGE_DIR/judge_manifest.json"
if [[ ! -f "$manifest" ]]; then
  echo "missing manifest: $manifest" >&2
  exit 1
fi

shard_count="$(
  .venv/bin/python - "$manifest" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(len(payload["shards"]))
PY
)"

if [[ "$shard_count" -eq 0 ]]; then
  echo "no shards in $manifest"
  exit 0
fi

per_worker=$(( (shard_count + PARALLEL - 1) / PARALLEL ))
echo "judge_dir=$JUDGE_DIR shard_count=$shard_count parallel=$PARALLEL per_worker=$per_worker"

for worker in $(seq 0 $((PARALLEL - 1))); do
  start=$((worker * per_worker))
  if [[ "$start" -ge "$shard_count" ]]; then
    break
  fi
  count="$per_worker"
  session="${SESSION_PREFIX}_p${worker}"
  log="$JUDGE_DIR/${session}.log"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "skip existing tmux session $session"
    continue
  fi
  cmd=".venv/bin/python -u scripts/stepa_run_q2_text_judge_shards.py --judge-dir '$JUDGE_DIR' --start-shard '$start' --max-shards '$count' --timeout-s '$TIMEOUT_S' --retries '$RETRIES' >> '$log' 2>&1"
  tmux new-session -d -s "$session" "$cmd"
  echo "launched $session start=$start count=$count log=$log"
done
