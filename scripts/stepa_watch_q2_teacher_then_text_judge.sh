#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INTERVAL_S="${INTERVAL_S:-7200}"
PARALLEL_TEXT_JUDGE="${PARALLEL_TEXT_JUDGE:-4}"
SHARD_SIZE="${SHARD_SIZE:-250}"

TEACHER_ROOTS=(
  "data/vqa_q2_stepa_pilot50k/teacher_q2_t0p60"
  "data/vqa_q2_stepa_pilot50k/teacher_q2_t0p60_shard01"
  "data/vqa_q2_stepa_pilot50k/teacher_q2_t0p60_shard02"
  "data/vqa_q2_stepa_pilot50k/teacher_q2_t0p60_shard03"
)
TEACHER_SESSIONS=(
  "stepa_q2_teacher_s0"
  "stepa_q2_teacher_s1"
  "stepa_q2_teacher_s2"
  "stepa_q2_teacher_s3"
)
TEACHER_STARTS=(0 12501 25001 37501)
TEACHER_LIMITS=(12501 12500 12500 12500)
TEACHER_EXPECTED=(12501 12500 12500 12500)

MERGED_ROOT="data/vqa_q2_stepa_pilot50k/teacher_q2_t0p60_parallel_merged"
JUDGE_DIR="$MERGED_ROOT/text_judge_gpt55_medium"
LOG="data/vqa_q2_stepa_pilot50k/stepa_teacher_to_text_judge_watch.log"

mkdir -p "$(dirname "$LOG")"

log_state() {
  {
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
    for i in "${!TEACHER_ROOTS[@]}"; do
      root="${TEACHER_ROOTS[$i]}"
      records=0
      accept=0
      reject=0
      [[ -f "$root/teacher_records.jsonl" ]] && records="$(wc -l < "$root/teacher_records.jsonl")"
      [[ -f "$root/q2_hard_gate_accept.jsonl" ]] && accept="$(wc -l < "$root/q2_hard_gate_accept.jsonl")"
      [[ -f "$root/q2_hard_gate_reject.jsonl" ]] && reject="$(wc -l < "$root/q2_hard_gate_reject.jsonl")"
      echo "$root records=$records/${TEACHER_EXPECTED[$i]} accept=$accept reject=$reject"
    done
    nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits || true
  } | tee -a "$LOG"
}

teachers_running() {
  for session in "${TEACHER_SESSIONS[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

teacher_records_count() {
  local root="$1"
  if [[ -f "$root/teacher_records.jsonl" ]]; then
    wc -l < "$root/teacher_records.jsonl"
  else
    echo 0
  fi
}

teacher_processed_count() {
  local root="$1"
  .venv/bin/python -c "import json
from pathlib import Path
root=Path('$root')
done=set()
for name in ('teacher_records.jsonl','q2_hard_gate_reject.jsonl'):
    path=root/name
    if not path.exists():
        continue
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                sample_id=json.loads(line).get('sample_id')
            except json.JSONDecodeError:
                continue
            if sample_id:
                done.add(str(sample_id))
print(len(done))"
}

teachers_complete() {
  for i in "${!TEACHER_ROOTS[@]}"; do
    local count
    count="$(teacher_processed_count "${TEACHER_ROOTS[$i]}")"
    if [[ "$count" -lt "${TEACHER_EXPECTED[$i]}" ]]; then
      return 1
    fi
  done
  return 0
}

relaunch_missing_teacher_sessions() {
  for i in "${!TEACHER_ROOTS[@]}"; do
    local session="${TEACHER_SESSIONS[$i]}"
    local root="${TEACHER_ROOTS[$i]}"
    local count
    count="$(teacher_processed_count "$root")"
    if [[ "$count" -ge "${TEACHER_EXPECTED[$i]}" ]]; then
      continue
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
      continue
    fi
    echo "relaunching $session root=$root records=$count/${TEACHER_EXPECTED[$i]}" | tee -a "$LOG"
    tmux new-session -d -s "$session" \
      ".venv/bin/python -u scripts/stepa_run_q2_teacher_from_candidates.py --candidate-jsonl data/vqa_q2_stepa_pilot50k/q2_candidates_all.jsonl --output-root '$root' --start-index '${TEACHER_STARTS[$i]}' --limit '${TEACHER_LIMITS[$i]}' --batch-size 4 --summary-every 100"
  done
}

while true; do
  log_state
  if teachers_complete; then
    break
  fi
  relaunch_missing_teacher_sessions
  sleep "$INTERVAL_S"
done

echo "teacher shards complete; merging and launching text judge" | tee -a "$LOG"
.venv/bin/python -u scripts/stepa_merge_q2_teacher_shards.py \
  --input-root "${TEACHER_ROOTS[0]}" \
  --input-root "${TEACHER_ROOTS[1]}" \
  --input-root "${TEACHER_ROOTS[2]}" \
  --input-root "${TEACHER_ROOTS[3]}" \
  --output-root "$MERGED_ROOT" | tee -a "$LOG"

rm -rf "$JUDGE_DIR"

.venv/bin/python -u scripts/stepa_build_q2_text_judge_shards.py \
  --teacher-dir "$MERGED_ROOT" \
  --output-dir "$JUDGE_DIR" \
  --shard-size "$SHARD_SIZE" | tee -a "$LOG"

PARALLEL="$PARALLEL_TEXT_JUDGE" SESSION_PREFIX="stepa_q2_text_judge" \
  bash scripts/stepa_launch_q2_text_judge_parallel.sh "$JUDGE_DIR" | tee -a "$LOG"

echo "text judge launched at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$LOG"
