#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/checkpoints/no_nav_bp3_200k_epoch/$RUN_ID}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT_DIR/outputs/reports/no_nav_distill/${RUN_ID}_checkpoint_suite}"
CORPUS_JSONL="${CORPUS_JSONL:-$ROOT_DIR/data/corpus/no_nav_teacher_pair_300chunks.jsonl}"
VAL_EVAL_SAMPLES="${VAL_EVAL_SAMPLES:-204}"
TRAIN_EVAL_SAMPLES="${TRAIN_EVAL_SAMPLES:-64}"
PREFILL_QC_SAMPLES="${PREFILL_QC_SAMPLES:-128}"

mkdir -p "$REPORT_ROOT"

checkpoint_names=(
  step_003125
  step_006250
  step_009375
  step_012500
  final
)

for name in "${checkpoint_names[@]}"; do
  checkpoint_dir="$OUTPUT_DIR/$name"
  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "[skip] missing checkpoint $checkpoint_dir"
    continue
  fi

  echo "[eval] free-run teacher-reference decode $name"
  CHECKPOINT_DIR="$checkpoint_dir" \
  CORPUS_JSONL="$CORPUS_JSONL" \
  REPORT_PREFIX="$REPORT_ROOT/${name}" \
  VAL_EVAL_SAMPLES="$VAL_EVAL_SAMPLES" \
  TRAIN_EVAL_SAMPLES="$TRAIN_EVAL_SAMPLES" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}" \
  GEOMETRY_REFERENCE=teacher \
  bash "$ROOT_DIR/scripts/39_eval_h200_checkpoint.sh" \
    > "$REPORT_ROOT/${name}.decode.log" 2>&1

  echo "[eval] true prefill hidden QC $name"
  "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/55_probe_no_nav_prefill_hidden_qc.py" \
    --num-samples "$PREFILL_QC_SAMPLES" \
    --student-checkpoint "${name}=${checkpoint_dir}" \
    --report-name "${RUN_ID}_${name}_prefill_qc.json" \
    --markdown-name "${RUN_ID}_${name}_prefill_qc.md" \
    > "$REPORT_ROOT/${name}.prefill_qc.log" 2>&1
done

echo "[eval] checkpoint suite done report_root=$REPORT_ROOT"
