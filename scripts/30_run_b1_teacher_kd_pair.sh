#!/usr/bin/env bash
set -euo pipefail

# Launch the two B1 teacher-KD continuations from the fixed B0 SFT checkpoint.
# Override with env vars, e.g. MAX_STEPS=3000 MULTI_GPU=off bash scripts/30_run_b1_teacher_kd_pair.sh

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/checkpoints/B0_GT_SFT_BODY_BEST}"
MAX_STEPS="${MAX_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MULTI_GPU="${MULTI_GPU:-auto}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-0.5}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-0.5}"

run_stage() {
  local name="$1"
  local corpus="$2"
  local config="$3"
  local output_dir="outputs/checkpoints/${name}_from_b0"
  local summary_json="outputs/reports/${name}_from_b0.json"

  "${PYTHON_BIN}" scripts/09_train_distill.py \
    --corpus-jsonl "${corpus}" \
    --stage-config "${config}" \
    --init-checkpoint-dir "${INIT_CHECKPOINT}" \
    --output-dir "${output_dir}" \
    --summary-json "${summary_json}" \
    --batch-size "${BATCH_SIZE}" \
    --max-steps "${MAX_STEPS}" \
    --eval-every-epochs "${EVAL_EVERY_EPOCHS}" \
    --save-every-epochs "${SAVE_EVERY_EPOCHS}" \
    --log-every-steps "${LOG_EVERY_STEPS}" \
    --multi-gpu "${MULTI_GPU}"
}

run_stage \
  "stage_b1b_abs_quality_teacher_kd" \
  "data/corpus/distill_v3_2_959_b1b_abs_quality.jsonl" \
  "configs/train/stage_b1b_abs_quality_teacher_kd.yaml"

run_stage \
  "stage_b1d_horizon_teacher_kd" \
  "data/corpus/distill_v3_2_959_b1d_horizon.jsonl" \
  "configs/train/stage_b1d_horizon_teacher_kd.yaml"
