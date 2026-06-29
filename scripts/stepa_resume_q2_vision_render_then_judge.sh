#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TEXT_MERGED_JSONL="${TEXT_MERGED_JSONL:-data/vqa_q2_stepa_pilot50k/teacher_q2_t0p60_parallel_merged/text_judge_gpt55_medium/merged/q2_text_judged_all.jsonl}"
VISION_AUDIT_DIR="${VISION_AUDIT_DIR:-outputs/stepa_q2_vision_audit_pilot50k}"
LOG="${LOG:-data/vqa_q2_stepa_pilot50k/stepa_resume_vision_render_then_judge.log}"
CELL_WIDTH="${CELL_WIDTH:-480}"
RENDER_WORKERS="${RENDER_WORKERS:-8}"
RENDER_PROGRESS_EVERY="${RENDER_PROGRESS_EVERY:-250}"
VISION_PARALLEL="${VISION_PARALLEL:-4}"
VISION_BATCH_SIZE="${VISION_BATCH_SIZE:-4}"
VISION_TIMEOUT_S="${VISION_TIMEOUT_S:-300}"
VISION_RETRIES="${VISION_RETRIES:-1}"

mkdir -p "$(dirname "$LOG")"

{
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') resume vision render ====="
  echo "input=$TEXT_MERGED_JSONL"
  echo "audit_dir=$VISION_AUDIT_DIR"
  echo "render_workers=$RENDER_WORKERS cell_width=$CELL_WIDTH"
} | tee -a "$LOG"

.venv/bin/python -u scripts/stepa_render_q2_contact_sheets.py \
  --input-jsonl "$TEXT_MERGED_JSONL" \
  --output-dir "$VISION_AUDIT_DIR" \
  --cell-width "$CELL_WIDTH" \
  --workers "$RENDER_WORKERS" \
  --resume \
  --progress-every "$RENDER_PROGRESS_EVERY" | tee -a "$LOG"

{
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') launch vision judge ====="
  echo "audit_dir=$VISION_AUDIT_DIR"
  echo "parallel=$VISION_PARALLEL batch_size=$VISION_BATCH_SIZE"
} | tee -a "$LOG"

PARALLEL="$VISION_PARALLEL" \
BATCH_SIZE="$VISION_BATCH_SIZE" \
TIMEOUT_S="$VISION_TIMEOUT_S" \
RETRIES="$VISION_RETRIES" \
SESSION_PREFIX="stepa_q2_vision_judge" \
  bash scripts/stepa_launch_q2_vision_judge_parallel.sh "$VISION_AUDIT_DIR" | tee -a "$LOG"

echo "vision judge launch command finished at $(date -u '+%Y-%m-%dT%H:%M:%SZ')" | tee -a "$LOG"
