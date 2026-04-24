#!/usr/bin/env bash
# Continue the clean reweight-off GT-SFT baseline in the main repo.
#
# Default behavior:
# - init from the imported 5k reweight-off checkpoint
# - train 5k more optimizer steps, for 10k total SFT steps
# - save every epoch (~754 optimizer steps on the 959-sample corpus)
# - select the final candidate later by validation decode ADE/FDE
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

CORPUS_JSONL="${CORPUS_JSONL:-$ROOT_DIR/data/corpus/distill_v3_2_959.jsonl}"
STAGE_CONFIG="${STAGE_CONFIG:-$ROOT_DIR/configs/train/stage_t1_sft_lora_reweight_off.yaml}"
INIT_CKPT="${INIT_CKPT:-$ROOT_DIR/outputs/checkpoints/stage_t1_sft_reweight_off_5k/final}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/checkpoints/stage_t1_sft_reweight_off_continue5k}"
SUMMARY_JSON="${SUMMARY_JSON:-$ROOT_DIR/outputs/reports/stage_t1_sft_reweight_off_continue5k.json}"
MAX_STEPS="${MAX_STEPS:-5000}"
LOG_EVERY="${LOG_EVERY:-25}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LOGFILE="$ROOT_DIR/outputs/logs/stage_t1_sft_reweight_off_continue5k.log"
mkdir -p "$(dirname "$LOGFILE")"
echo "[launch] main-repo SFT reweight-off continuation CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES init=$INIT_CKPT max_steps=$MAX_STEPS" | tee "$LOGFILE"

"$PYTHON_BIN" "$ROOT_DIR/scripts/09_train_distill.py" \
  --corpus-jsonl "$CORPUS_JSONL" \
  --stage-config "$STAGE_CONFIG" \
  --init-checkpoint-dir "$INIT_CKPT" \
  --max-steps "$MAX_STEPS" \
  --eval-every-epochs 999 \
  --save-every-epochs 1 \
  --early-stop-patience 9999 \
  --early-stop-stage stage_b \
  --log-every-steps "$LOG_EVERY" \
  --batch-size 1 \
  --multi-gpu off \
  --output-dir "$OUT_DIR" \
  --summary-json "$SUMMARY_JSON" 2>&1 | tee -a "$LOGFILE"

echo "[done] log=$LOGFILE metrics=$OUT_DIR/metrics.jsonl"
