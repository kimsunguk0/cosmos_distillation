#!/usr/bin/env bash
# Run the post-SFT checkpoint sweep and pin B0_GT_SFT_BODY_BEST.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

CKPT_ROOT="${CKPT_ROOT:-$ROOT_DIR/outputs/checkpoints/stage_t1_sft_reweight_off_continue5k}"
CORPUS_JSONL="${CORPUS_JSONL:-$ROOT_DIR/data/corpus/distill_v3_2_959.jsonl}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT_DIR/outputs/reports/b0_sft_body_best_sweep}"
VAL_SAMPLES="${VAL_SAMPLES:-0}"        # 0 means full split
TRAIN_SAMPLES="${TRAIN_SAMPLES:-64}"   # train-gap probe
WAIT_FOR_TRAIN="${WAIT_FOR_TRAIN:-0}"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$WAIT_FOR_TRAIN" == "1" ]]; then
  echo "[wait] waiting for active SFT continuation to finish..."
  while pgrep -f "09_train_distill.py .*stage_t1_sft_reweight_off_continue5k" >/dev/null; do
    sleep 60
  done
fi

mkdir -p "$REPORT_ROOT"

declare -a candidates=()
for name in step_003016 step_003770 step_004524 final; do
  if [[ -d "$CKPT_ROOT/$name" ]]; then
    candidates+=("$CKPT_ROOT/$name")
  fi
done

if [[ "${#candidates[@]}" -eq 0 ]]; then
  echo "[error] no candidate checkpoints found under $CKPT_ROOT" >&2
  exit 1
fi

declare -a summaries=()
for ckpt in "${candidates[@]}"; do
  name="$(basename "$ckpt")"
  echo "[eval] checkpoint=$name val_samples=$VAL_SAMPLES"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/25_decode_checkpoint_overlays.py" \
    --corpus-jsonl "$CORPUS_JSONL" \
    --checkpoint-dir "$ckpt" \
    --split val \
    --num-samples "$VAL_SAMPLES" \
    --output-dir "$REPORT_ROOT/${name}_val_overlays" \
    --summary-json "$REPORT_ROOT/${name}_val_summary.json"
  summaries+=("$REPORT_ROOT/${name}_val_summary.json")

  echo "[eval] checkpoint=$name train_samples=$TRAIN_SAMPLES"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/25_decode_checkpoint_overlays.py" \
    --corpus-jsonl "$CORPUS_JSONL" \
    --checkpoint-dir "$ckpt" \
    --split train \
    --num-samples "$TRAIN_SAMPLES" \
    --skip-overlays \
    --output-dir "$REPORT_ROOT/${name}_train_probe" \
    --summary-json "$REPORT_ROOT/${name}_train_summary.json"
  summaries+=("$REPORT_ROOT/${name}_train_summary.json")
done

"$PYTHON_BIN" "$ROOT_DIR/scripts/26_select_best_sft_checkpoint.py" \
  "${summaries[@]}" \
  --force

echo "[done] B0 selection written to:"
echo "  $ROOT_DIR/outputs/checkpoints/B0_GT_SFT_BODY_BEST"
echo "  $ROOT_DIR/outputs/reports/B0_GT_SFT_BODY_BEST_selection.json"
echo "  $ROOT_DIR/outputs/reports/B0_GT_SFT_BODY_BEST_selection.md"
