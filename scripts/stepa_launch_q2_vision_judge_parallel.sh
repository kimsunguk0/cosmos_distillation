#!/usr/bin/env bash
set -euo pipefail

AUDIT_DIR="${1:-outputs/stepa_q2_vision_audit_pilot50k}"
PARALLEL="${PARALLEL:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TIMEOUT_S="${TIMEOUT_S:-300}"
RETRIES="${RETRIES:-1}"
SESSION_PREFIX="${SESSION_PREFIX:-stepa_q2_vision_judge}"

cd "$(dirname "$0")/.."

manifest="$AUDIT_DIR/manifest.jsonl"
summary_file="$AUDIT_DIR/summary.json"
if [[ ! -f "$manifest" ]]; then
  echo "missing manifest: $manifest" >&2
  exit 1
fi
if [[ ! -f "$summary_file" ]]; then
  echo "missing completed render summary: $summary_file" >&2
  exit 1
fi

sample_count="$(wc -l < "$manifest")"
if [[ "$sample_count" -eq 0 ]]; then
  echo "no samples in $manifest"
  exit 0
fi

per_worker=$(( (sample_count + PARALLEL - 1) / PARALLEL ))
result_dir="$AUDIT_DIR/vision_judge_results"
summary_dir="$AUDIT_DIR/vision_judge_summaries"
mkdir -p "$result_dir" "$summary_dir"

if find "$result_dir" -name '*.jsonl' -type f -print -quit | grep -q .; then
  stale_count="$(find "$result_dir" -name '*.jsonl' -type f ! -newer "$summary_file" | wc -l)"
  if [[ "$stale_count" -gt 0 ]]; then
    echo "clearing stale vision judge outputs older than $summary_file"
    rm -f "$result_dir"/*.jsonl "$summary_dir"/*.json 2>/dev/null || true
  fi
fi

echo "audit_dir=$AUDIT_DIR sample_count=$sample_count parallel=$PARALLEL per_worker=$per_worker"

for worker in $(seq 0 $((PARALLEL - 1))); do
  start=$((worker * per_worker))
  if [[ "$start" -ge "$sample_count" ]]; then
    break
  fi
  count="$per_worker"
  session="${SESSION_PREFIX}_p${worker}"
  result="$result_dir/vision_results_p${worker}.jsonl"
  summary="$summary_dir/vision_summary_p${worker}.json"
  log="$AUDIT_DIR/${session}.log"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "skip existing tmux session $session"
    continue
  fi
  cmd=".venv/bin/python -u scripts/stepa_run_q2_vision_judge.py --audit-dir '$AUDIT_DIR' --output-jsonl '$result' --summary-json '$summary' --start-index '$start' --max-samples '$count' --batch-size '$BATCH_SIZE' --timeout-s '$TIMEOUT_S' --retries '$RETRIES' >> '$log' 2>&1"
  tmux new-session -d -s "$session" "$cmd"
  echo "launched $session start=$start count=$count result=$result log=$log"
done
