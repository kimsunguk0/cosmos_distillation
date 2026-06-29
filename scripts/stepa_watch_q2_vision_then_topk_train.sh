#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INTERVAL_S="${INTERVAL_S:-900}"
TOPK_PARALLEL="${TOPK_PARALLEL:-4}"
TOPK_K="${TOPK_K:-32}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-2000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-8}"
TRAIN_LR="${TRAIN_LR:-1e-5}"

TEXT_MERGED_JSONL="${TEXT_MERGED_JSONL:-data/vqa_q2_stepa_pilot50k/teacher_q2_t0p60_parallel_merged/text_judge_gpt55_medium/merged/q2_text_judged_all.jsonl}"
VISION_AUDIT_DIR="${VISION_AUDIT_DIR:-outputs/stepa_q2_vision_audit_pilot50k}"
FINAL_ROOT="${FINAL_ROOT:-data/vqa_q2_stepa_pilot50k/q2_final_judged}"
TOPK_ROOT="${TOPK_ROOT:-data/vqa_q2_stepa_pilot50k/q2_final_judged/teacher_topk32_train}"
TOPK_MERGED_ROOT="${TOPK_MERGED_ROOT:-data/vqa_q2_stepa_pilot50k/q2_final_judged/teacher_topk32_train_merged}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-outputs/checkpoints/stepa_q2_vqa_fullft_pilot50k}"
SANITY_OUTPUT_DIR="${SANITY_OUTPUT_DIR:-outputs/stepa_q2_fullft_pilot50k_sanity}"
LOG="${LOG:-data/vqa_q2_stepa_pilot50k/stepa_vision_to_train_watch.log}"

mkdir -p "$(dirname "$LOG")"

vision_sessions_running() {
  if tmux ls 2>/dev/null | rg -q '^stepa_q2_vision_judge_p[0-9]+'; then
    return 0
  fi
  return 1
}

topk_sessions_running() {
  if tmux ls 2>/dev/null | rg -q '^stepa_q2_topk_p[0-9]+'; then
    return 0
  fi
  return 1
}

expected_vision_rows() {
  local manifest="$VISION_AUDIT_DIR/manifest.jsonl"
  local summary="$VISION_AUDIT_DIR/summary.json"
  if [[ ! -f "$manifest" || ! -f "$summary" ]]; then
    echo 0
    return
  fi
  .venv/bin/python - "$summary" "$manifest" <<'PY'
import json
import sys
from pathlib import Path

summary = json.load(open(sys.argv[1], "r", encoding="utf-8"))
rendered = int(summary.get("rendered", 0))
manifest_rows = sum(1 for _ in Path(sys.argv[2]).open("rb"))
print(min(rendered, manifest_rows))
PY
}

actual_vision_rows() {
  local dir="$VISION_AUDIT_DIR/vision_judge_results"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return
  fi
  find "$dir" -name '*.jsonl' -print0 | xargs -0r cat | wc -l
}

train_rows() {
  local path="$FINAL_ROOT/q2_text_vision_judged_train.jsonl"
  if [[ ! -f "$path" ]]; then
    echo 0
    return
  fi
  wc -l < "$path"
}

topk_rows() {
  local path="$TOPK_MERGED_ROOT/records_with_topk_train.jsonl"
  if [[ ! -f "$path" ]]; then
    echo 0
    return
  fi
  wc -l < "$path"
}

log_state() {
  {
    echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
    echo "vision expected=$(expected_vision_rows) actual=$(actual_vision_rows)"
    echo "final train_rows=$(train_rows) topk_rows=$(topk_rows)"
    tmux ls 2>/dev/null | rg 'stepa_q2_vision_judge|stepa_q2_topk|stepa_q2_fullft' || true
    nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits || true
  } | tee -a "$LOG"
}

while true; do
  log_state
  expected="$(expected_vision_rows)"
  actual="$(actual_vision_rows)"
  if [[ "$expected" -gt 0 && "$actual" -ge "$expected" ]] && ! vision_sessions_running; then
    break
  fi
  if [[ "$expected" -gt 0 && "$actual" -lt "$expected" ]] && ! vision_sessions_running; then
    echo "vision judge incomplete without active sessions; relaunching missing/skipped partitions" | tee -a "$LOG"
    PARALLEL=4 BATCH_SIZE="${VISION_BATCH_SIZE:-4}" TIMEOUT_S=300 RETRIES=1 SESSION_PREFIX="stepa_q2_vision_judge" \
      bash scripts/stepa_launch_q2_vision_judge_parallel.sh "$VISION_AUDIT_DIR" | tee -a "$LOG"
  fi
  sleep "$INTERVAL_S"
done

echo "vision judge complete; merging final judged data" | tee -a "$LOG"
rm -rf "$FINAL_ROOT"
.venv/bin/python -u scripts/stepa_merge_q2_vision_judge.py \
  --text-judged-jsonl "$TEXT_MERGED_JSONL" \
  --vision-results "$VISION_AUDIT_DIR/vision_judge_results/*.jsonl" \
  --output-root "$FINAL_ROOT" \
  --require-usable | tee -a "$LOG"

train_jsonl="$FINAL_ROOT/q2_text_vision_judged_train.jsonl"
train_count="$(wc -l < "$train_jsonl")"
if [[ "$train_count" -le 0 ]]; then
  echo "no final train rows; not launching top-k/train" | tee -a "$LOG"
  exit 1
fi

echo "launching train top-k extraction for $train_count rows" | tee -a "$LOG"
rm -rf "$TOPK_ROOT" "$TOPK_MERGED_ROOT"
PARALLEL="$TOPK_PARALLEL" TOPK="$TOPK_K" SESSION_PREFIX="stepa_q2_topk" \
  bash scripts/stepa_launch_q2_topk_parallel.sh "$train_jsonl" "$TOPK_ROOT" | tee -a "$LOG"

while true; do
  log_state
  if ! topk_sessions_running; then
    break
  fi
  sleep "$INTERVAL_S"
done

echo "top-k shards finished; merging" | tee -a "$LOG"
.venv/bin/python -u scripts/stepa_merge_q2_topk_shards.py \
  --input-root "$TOPK_ROOT/shard_*" \
  --output-root "$TOPK_MERGED_ROOT" | tee -a "$LOG"

topk_train_jsonl="$TOPK_MERGED_ROOT/records_with_topk_train.jsonl"
topk_count="$(wc -l < "$topk_train_jsonl")"
if [[ "$topk_count" -le 0 ]]; then
  echo "no top-k train rows; not launching training" | tee -a "$LOG"
  exit 1
fi

echo "running training sanity check on $topk_count top-k rows" | tee -a "$LOG"
.venv/bin/python -u scripts/stepa_train_q2_fullft.py \
  --train-jsonl "$topk_train_jsonl" \
  --val-jsonl "$FINAL_ROOT/q2_text_vision_judged_val.jsonl" \
  --output-dir "$SANITY_OUTPUT_DIR" \
  --batch-size 1 \
  --max-train-samples 8 \
  --max-steps 1 \
  --sanity-only | tee -a "$LOG"

if tmux has-session -t stepa_q2_fullft_pilot50k 2>/dev/null; then
  echo "training session already exists: stepa_q2_fullft_pilot50k" | tee -a "$LOG"
  exit 0
fi

echo "launching Step 6 full fine-tune" | tee -a "$LOG"
train_cmd=".venv/bin/python -u scripts/stepa_train_q2_fullft.py --train-jsonl '$topk_train_jsonl' --val-jsonl '$FINAL_ROOT/q2_text_vision_judged_val.jsonl' --output-dir '$TRAIN_OUTPUT_DIR' --batch-size '$TRAIN_BATCH_SIZE' --grad-accum-steps '$TRAIN_GRAD_ACCUM' --max-steps '$TRAIN_MAX_STEPS' --learning-rate '$TRAIN_LR' --lambda-kl 1.0 --kd-temperature 1.5 --optimizer adamw8bit --gradient-checkpointing --log-every 10 >> '$TRAIN_OUTPUT_DIR/train.log' 2>&1"
mkdir -p "$TRAIN_OUTPUT_DIR"
tmux new-session -d -s stepa_q2_fullft_pilot50k "$train_cmd"
echo "Step 6 launched at $(date -u '+%Y-%m-%dT%H:%M:%SZ') session=stepa_q2_fullft_pilot50k output=$TRAIN_OUTPUT_DIR" | tee -a "$LOG"
