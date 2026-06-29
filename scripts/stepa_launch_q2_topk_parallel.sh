#!/usr/bin/env bash
set -euo pipefail

INPUT_JSONL="${1:?usage: stepa_launch_q2_topk_parallel.sh <input_jsonl> <output_root>}"
OUTPUT_ROOT="${2:?usage: stepa_launch_q2_topk_parallel.sh <input_jsonl> <output_root>}"
PARALLEL="${PARALLEL:-4}"
TOPK="${TOPK:-32}"
SESSION_PREFIX="${SESSION_PREFIX:-stepa_q2_topk}"
SUMMARY_EVERY="${SUMMARY_EVERY:-25}"

cd "$(dirname "$0")/.."

if [[ ! -f "$INPUT_JSONL" ]]; then
  echo "missing input jsonl: $INPUT_JSONL" >&2
  exit 1
fi

row_count="$(wc -l < "$INPUT_JSONL")"
if [[ "$row_count" -eq 0 ]]; then
  echo "no rows in $INPUT_JSONL" >&2
  exit 0
fi

per_worker=$(( (row_count + PARALLEL - 1) / PARALLEL ))
mkdir -p "$OUTPUT_ROOT"
echo "input=$INPUT_JSONL rows=$row_count output_root=$OUTPUT_ROOT parallel=$PARALLEL per_worker=$per_worker topk=$TOPK"

for worker in $(seq 0 $((PARALLEL - 1))); do
  start=$((worker * per_worker))
  if [[ "$start" -ge "$row_count" ]]; then
    break
  fi
  count="$per_worker"
  shard_root="$OUTPUT_ROOT/shard_$(printf '%02d' "$worker")"
  session="${SESSION_PREFIX}_p${worker}"
  log="$OUTPUT_ROOT/${session}.log"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "skip existing tmux session $session"
    continue
  fi
  cmd=".venv/bin/python -u scripts/stepa_extract_q2_topk.py --input-jsonl '$INPUT_JSONL' --output-root '$shard_root' --start-index '$start' --limit '$count' --topk '$TOPK' --summary-every '$SUMMARY_EVERY' >> '$log' 2>&1"
  tmux new-session -d -s "$session" "$cmd"
  echo "launched $session start=$start count=$count root=$shard_root log=$log"
done
