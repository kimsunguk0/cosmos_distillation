#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INTERVAL_S="${INTERVAL_S:-600}"
VISION_PARALLEL="${VISION_PARALLEL:-4}"
VISION_BATCH_SIZE="${VISION_BATCH_SIZE:-4}"
VISION_TIMEOUT_S="${VISION_TIMEOUT_S:-300}"
VISION_RETRIES="${VISION_RETRIES:-1}"
CELL_WIDTH="${CELL_WIDTH:-480}"
RENDER_WORKERS="${RENDER_WORKERS:-1}"
RENDER_RESUME="${RENDER_RESUME:-0}"

MERGED_ROOT="${MERGED_ROOT:-data/vqa_q2_stepa_pilot50k/teacher_q2_t0p60_parallel_merged}"
JUDGE_DIR="${JUDGE_DIR:-$MERGED_ROOT/text_judge_gpt55_medium}"
TEXT_MERGED_DIR="${TEXT_MERGED_DIR:-$JUDGE_DIR/merged}"
VISION_AUDIT_DIR="${VISION_AUDIT_DIR:-outputs/stepa_q2_vision_audit_pilot50k}"
LOG="${LOG:-data/vqa_q2_stepa_pilot50k/stepa_text_to_vision_watch.log}"

mkdir -p "$(dirname "$LOG")"

text_sessions_running() {
  if tmux ls 2>/dev/null | rg -q '^stepa_q2_text_judge_p[0-9]+'; then
    return 0
  fi
  return 1
}

expected_text_rows() {
  local manifest="$JUDGE_DIR/judge_manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo 0
    return
  fi
  .venv/bin/python - "$manifest" <<'PY'
import json, sys
payload=json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(sum(int(s.get("count", 0)) for s in payload.get("shards", [])))
PY
}

actual_text_rows() {
  local dir="$JUDGE_DIR/judge_results"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return
  fi
  find "$dir" -name '*.jsonl' -print0 | xargs -0r cat | wc -l
}

log_state() {
  local expected actual
  expected="$(expected_text_rows)"
  actual="$(actual_text_rows)"
  {
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
    echo "judge_dir=$JUDGE_DIR expected_text_rows=$expected actual_text_rows=$actual"
    tmux ls 2>/dev/null | rg 'stepa_q2_text_judge|stepa_q2_vision_judge' || true
  } | tee -a "$LOG"
}

while true; do
  log_state
  expected="$(expected_text_rows)"
  actual="$(actual_text_rows)"
  if [[ "$expected" -gt 0 && "$actual" -ge "$expected" ]] && ! text_sessions_running; then
    break
  fi
  if [[ "$expected" -gt 0 && "$actual" -lt "$expected" ]] && ! text_sessions_running; then
    echo "text judge incomplete without active sessions; relaunching missing/skipped shards" | tee -a "$LOG"
    PARALLEL=4 SESSION_PREFIX="stepa_q2_text_judge" \
      bash scripts/stepa_launch_q2_text_judge_parallel.sh "$JUDGE_DIR" | tee -a "$LOG"
  fi
  sleep "$INTERVAL_S"
done

echo "text judge complete; merging text decisions" | tee -a "$LOG"
.venv/bin/python -u scripts/stepa_merge_q2_text_judge.py \
  --teacher-dir "$MERGED_ROOT" \
  --judge-results "$JUDGE_DIR/judge_results" \
  --output-dir "$TEXT_MERGED_DIR" | tee -a "$LOG"

echo "rendering vision audit sheets" | tee -a "$LOG"
if [[ "$RENDER_RESUME" != "1" ]]; then
  rm -rf "$VISION_AUDIT_DIR"
fi
render_extra_args=()
if [[ "$RENDER_RESUME" == "1" ]]; then
  render_extra_args+=(--resume)
fi
.venv/bin/python -u scripts/stepa_render_q2_contact_sheets.py \
  --input-jsonl "$TEXT_MERGED_DIR/q2_text_judged_all.jsonl" \
  --output-dir "$VISION_AUDIT_DIR" \
  --cell-width "$CELL_WIDTH" \
  --workers "$RENDER_WORKERS" \
  "${render_extra_args[@]}" | tee -a "$LOG"

echo "launching vision judge" | tee -a "$LOG"
PARALLEL="$VISION_PARALLEL" BATCH_SIZE="$VISION_BATCH_SIZE" TIMEOUT_S="$VISION_TIMEOUT_S" RETRIES="$VISION_RETRIES" \
  SESSION_PREFIX="stepa_q2_vision_judge" bash scripts/stepa_launch_q2_vision_judge_parallel.sh "$VISION_AUDIT_DIR" | tee -a "$LOG"

echo "vision judge launched at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$LOG"
