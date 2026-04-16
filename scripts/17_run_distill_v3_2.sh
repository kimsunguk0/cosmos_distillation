#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

CORPUS_JSONL="${CORPUS_JSONL:-$ROOT_DIR/data/corpus/distill_v3_2.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MULTI_GPU="${MULTI_GPU:-auto}"

STAGE_A1_EPOCHS="${STAGE_A1_EPOCHS:-1.0}"
STAGE_A2_EPOCHS="${STAGE_A2_EPOCHS:-1.0}"
STAGE_B_EPOCHS="${STAGE_B_EPOCHS:-4.0}"
STAGE_A0_EPOCHS="${STAGE_A0_EPOCHS:-1.0}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-0.5}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1.0}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-1}"

STAGE_A0_OUT="${STAGE_A0_OUT:-$ROOT_DIR/outputs/checkpoints/stage_a0_v3_2}"
STAGE_A1_OUT="${STAGE_A1_OUT:-$ROOT_DIR/outputs/checkpoints/stage_a1_v3_2}"
STAGE_A2_OUT="${STAGE_A2_OUT:-$ROOT_DIR/outputs/checkpoints/stage_a_v3_2}"
STAGE_B_OUT="${STAGE_B_OUT:-$ROOT_DIR/outputs/checkpoints/stage_b_v3_2}"

echo "[distill] stage A0 traj-only: epochs=${STAGE_A0_EPOCHS}, batch_size=${BATCH_SIZE}, multi_gpu=${MULTI_GPU}"
PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$ROOT_DIR/scripts/09_train_distill.py" \
  --corpus-jsonl "$CORPUS_JSONL" \
  --stage-config "$ROOT_DIR/configs/train/stage_a0_traj_only.yaml" \
  --epochs "$STAGE_A0_EPOCHS" \
  --eval-every-epochs "$EVAL_EVERY_EPOCHS" \
  --save-every-epochs "$SAVE_EVERY_EPOCHS" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --early-stop-stage stage_b \
  --log-every-steps "$LOG_EVERY_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --multi-gpu "$MULTI_GPU" \
  --output-dir "$STAGE_A0_OUT" \
  --summary-json "$ROOT_DIR/outputs/reports/stage_a0_v3_2.json"

echo "[distill] stage A1 warmup: epochs=${STAGE_A1_EPOCHS}, batch_size=${BATCH_SIZE}, multi_gpu=${MULTI_GPU}"
PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$ROOT_DIR/scripts/09_train_distill.py" \
  --corpus-jsonl "$CORPUS_JSONL" \
  --stage-config "$ROOT_DIR/configs/train/stage_a1_warmup.yaml" \
  --epochs "$STAGE_A1_EPOCHS" \
  --eval-every-epochs "$EVAL_EVERY_EPOCHS" \
  --save-every-epochs "$SAVE_EVERY_EPOCHS" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --early-stop-stage stage_b \
  --log-every-steps "$LOG_EVERY_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --multi-gpu "$MULTI_GPU" \
  --init-checkpoint-dir "$STAGE_A0_OUT/final" \
  --output-dir "$STAGE_A1_OUT" \
  --summary-json "$ROOT_DIR/outputs/reports/stage_a1_v3_2.json"

echo "[distill] stage A2 refine: epochs=${STAGE_A2_EPOCHS}, batch_size=${BATCH_SIZE}, multi_gpu=${MULTI_GPU}"
PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$ROOT_DIR/scripts/09_train_distill.py" \
  --corpus-jsonl "$CORPUS_JSONL" \
  --stage-config "$ROOT_DIR/configs/train/stage_a2_refine.yaml" \
  --epochs "$STAGE_A2_EPOCHS" \
  --eval-every-epochs "$EVAL_EVERY_EPOCHS" \
  --save-every-epochs "$SAVE_EVERY_EPOCHS" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --early-stop-stage stage_b \
  --log-every-steps "$LOG_EVERY_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --multi-gpu "$MULTI_GPU" \
  --init-checkpoint-dir "$STAGE_A1_OUT/final" \
  --output-dir "$STAGE_A2_OUT" \
  --summary-json "$ROOT_DIR/outputs/reports/stage_a_v3_2.json"

echo "[distill] stage B: epochs=${STAGE_B_EPOCHS}, batch_size=${BATCH_SIZE}, multi_gpu=${MULTI_GPU}"
PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$ROOT_DIR/scripts/09_train_distill.py" \
  --corpus-jsonl "$CORPUS_JSONL" \
  --stage-config "$ROOT_DIR/configs/train/stage_b.yaml" \
  --epochs "$STAGE_B_EPOCHS" \
  --eval-every-epochs "$EVAL_EVERY_EPOCHS" \
  --save-every-epochs "$SAVE_EVERY_EPOCHS" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --early-stop-stage stage_b \
  --log-every-steps "$LOG_EVERY_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --multi-gpu "$MULTI_GPU" \
  --init-checkpoint-dir "$STAGE_A2_OUT/final" \
  --output-dir "$STAGE_B_OUT" \
  --summary-json "$ROOT_DIR/outputs/reports/stage_b_v3_2.json"

echo "[distill] done"
