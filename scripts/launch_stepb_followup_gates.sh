#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/pm97/workspace/sukim/distillation/cosmos_distillation}"
cd "${ROOT}"

RUN="${RUN:-0}"
RUN_ID="${RUN_ID:-followup_$(date -u +%Y%m%dT%H%M%SZ)}"

STUDENT="${STUDENT:-outputs/checkpoints/stepa_q2_vqa_fullft_repaired_v1_bs8_e1/step_003488}"
CORPUS_20K="${CORPUS_20K:-data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl}"

LPRIME_CKPT="${LPRIME_CKPT:-outputs/checkpoints/stepb_ladder_20k/r0_lprime_lora_lr1e4/best_decode}"
FULLFT_CKPT="${FULLFT_CKPT:-outputs/checkpoints/stepb_ladder_20k/r0_f_fullft_lr2e5/best_decode}"

OUT_ROOT="${OUT_ROOT:-outputs/checkpoints/stepb_followup/${RUN_ID}}"
REPORT_ROOT="${REPORT_ROOT:-outputs/reports/stepb_followup/${RUN_ID}}"

BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LOG_EVERY="${LOG_EVERY:-25}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-1.0}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-0.3}"
MAX_KEEP_CHECKPOINTS="${MAX_KEEP_CHECKPOINTS:-4}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-999}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

mkdir -p "${OUT_ROOT}" "${REPORT_ROOT}"
STATUS_JSONL="${REPORT_ROOT}/queue_status.jsonl"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

status_line() {
  local name="$1"
  local state="$2"
  local exit_code="$3"
  local log_path="$4"
  printf '{"time_utc":"%s","run_id":"%s","name":"%s","state":"%s","exit_code":%s,"log":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${RUN_ID}" "${name}" "${state}" "${exit_code}" "${log_path}" \
    >> "${STATUS_JSONL}"
}

run_cmd() {
  local name="$1"
  shift
  local log_path="${REPORT_ROOT}/${name}.log"
  printf '\n# %s\n' "${name}"
  printf '%q ' "$@"
  printf '\n'
  status_line "${name}" "planned" 0 "${log_path}"
  if [[ "${RUN}" != "1" ]]; then
    return 0
  fi

  status_line "${name}" "started" 0 "${log_path}"
  "$@" > "${log_path}" 2>&1
  local exit_code=$?
  if [[ "${exit_code}" == "0" ]]; then
    status_line "${name}" "completed" 0 "${log_path}"
  else
    status_line "${name}" "failed" "${exit_code}" "${log_path}"
    printf '[queue] %s failed with exit_code=%s; continuing\n' "${name}" "${exit_code}" >&2
  fi
  return 0
}

run_train() {
  local name="$1"
  local config="$2"
  local disable_lora="$3"
  shift 3

  local cmd=(
    .venv/bin/python -u scripts/09_train_distill.py
    --corpus-jsonl "${CORPUS_20K}"
    --student-model "${STUDENT}"
    --stage-config "${config}"
    --output-dir "${OUT_ROOT}/${name}"
    --summary-json "${REPORT_ROOT}/${name}_summary.json"
    --batch-size "${BATCH_SIZE}"
    --epochs 3.0
    --num-workers "${NUM_WORKERS}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    --pin-memory
    --persistent-workers
    --skip-asset-check
    --eval-every-epochs "${EVAL_EVERY_EPOCHS}"
    --save-every-epochs "${SAVE_EVERY_EPOCHS}"
    --max-keep-checkpoints "${MAX_KEEP_CHECKPOINTS}"
    --grad-clip-norm "${GRAD_CLIP_NORM}"
    --early-stop-stage stage_a
    --early-stop-patience "${EARLY_STOP_PATIENCE}"
    --log-every-steps "${LOG_EVERY}"
    --skip-final-save
  )
  if [[ "${disable_lora}" == "1" ]]; then
    cmd+=(--disable-lora)
  fi
  run_cmd "${name}" "${cmd[@]}"
}

run_cmd \
  freeze_val512_sample_ids \
  .venv/bin/python scripts/stepb_freeze_eval_sample_ids.py \
  --corpus-jsonl "${CORPUS_20K}" \
  --split val \
  --num-samples 512 \
  --output-json "${REPORT_ROOT}/val512_sample_ids.json"

run_cmd \
  val512_lprime_lora_lr1e4 \
  .venv/bin/python -u scripts/70_eval_checkpoint_free_run.py \
  --corpus-jsonl "${CORPUS_20K}" \
  --checkpoint-dir "${LPRIME_CKPT}" \
  --split val \
  --num-samples 512 \
  --max-new-tokens 256 \
  --metric-name free_run_geometry_score \
  --summary-json "${REPORT_ROOT}/val512_lprime_lora_lr1e4_summary.json" \
  --device cuda

run_cmd \
  val512_fullft_lr2e5 \
  .venv/bin/python -u scripts/70_eval_checkpoint_free_run.py \
  --corpus-jsonl "${CORPUS_20K}" \
  --checkpoint-dir "${FULLFT_CKPT}" \
  --split val \
  --num-samples 512 \
  --max-new-tokens 256 \
  --metric-name free_run_geometry_score \
  --summary-json "${REPORT_ROOT}/val512_fullft_lr2e5_summary.json" \
  --device cuda

run_cmd \
  val512_fullft_vs_lprime_ci \
  .venv/bin/python scripts/stepb_compare_decode_summaries.py \
  --baseline-summary "${REPORT_ROOT}/val512_lprime_lora_lr1e4_summary.json" \
  --candidate-summary "${REPORT_ROOT}/val512_fullft_lr2e5_summary.json" \
  --baseline-name lprime_lora_lr1e4 \
  --candidate-name fullft_lr2e5 \
  --bootstrap 5000 \
  --seed 42 \
  --output-json "${REPORT_ROOT}/val512_fullft_vs_lprime_ci.json" \
  --output-md "${REPORT_ROOT}/val512_fullft_vs_lprime_ci.md"

run_cmd \
  vision_ablation_fullft_normal \
  .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  --corpus-jsonl "${CORPUS_20K}" \
  --checkpoint-dir "${FULLFT_CKPT}" \
  --split val \
  --num-samples 256 \
  --prompt-mode joint \
  --target-mode joint \
  --image-prompt-style camera_labeled \
  --prompt-text-style official_alpamayo \
  --fuse-history-tokens \
  --geometry-reference teacher \
  --image-ablation normal \
  --max-new-tokens 256 \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --samples-per-row 1 \
  --output-dir "${REPORT_ROOT}/vision_ablation_fullft_normal" \
  --summary-json "${REPORT_ROOT}/vision_ablation_fullft_normal_summary.json" \
  --skip-overlays

run_cmd \
  vision_ablation_fullft_black \
  .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  --corpus-jsonl "${CORPUS_20K}" \
  --checkpoint-dir "${FULLFT_CKPT}" \
  --split val \
  --num-samples 256 \
  --prompt-mode joint \
  --target-mode joint \
  --image-prompt-style camera_labeled \
  --prompt-text-style official_alpamayo \
  --fuse-history-tokens \
  --geometry-reference teacher \
  --image-ablation black \
  --max-new-tokens 256 \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --samples-per-row 1 \
  --output-dir "${REPORT_ROOT}/vision_ablation_fullft_black" \
  --summary-json "${REPORT_ROOT}/vision_ablation_fullft_black_summary.json" \
  --skip-overlays

run_cmd \
  vision_ablation_black_vs_normal_ci \
  .venv/bin/python scripts/stepb_compare_decode_summaries.py \
  --baseline-summary "${REPORT_ROOT}/vision_ablation_fullft_normal_summary.json" \
  --candidate-summary "${REPORT_ROOT}/vision_ablation_fullft_black_summary.json" \
  --baseline-name fullft_normal \
  --candidate-name fullft_black \
  --bootstrap 5000 \
  --seed 42 \
  --output-json "${REPORT_ROOT}/vision_ablation_black_vs_normal_ci.json" \
  --output-md "${REPORT_ROOT}/vision_ablation_black_vs_normal_ci.md"

run_train \
  r0_f_fullft_lr3e5_20k \
  configs/train/stepb_ladder_r0_f_fullft_lr3e5.yaml \
  1

run_train \
  r0_lprime_lora_lr2e4_20k \
  configs/train/stepb_ladder_r0_lprime_lora_lr2e4.yaml \
  0

printf '\n[queue] run_id=%s run=%s status=%s\n' "${RUN_ID}" "${RUN}" "${STATUS_JSONL}"
