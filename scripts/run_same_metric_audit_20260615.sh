#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

AUDIT_DIR="outputs/same_metric_audit_20260615"
CORPUS="data/corpus/benchmark_semantic_val_cap50_seed42.jsonl"
BACKBONE_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
LOG="${AUDIT_DIR}/run.log"

mkdir -p "${AUDIT_DIR}"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "${LOG}"
}

run_backbone_decode() {
  local tag="$1"
  local samples_per_row="$2"
  local out_dir="${AUDIT_DIR}/${tag}"
  local summary="${out_dir}/summary.json"

  if [[ -s "${summary}" ]]; then
    log "skip_existing backbone tag=${tag} summary=${summary}"
    return
  fi

  log "start backbone tag=${tag} samples_per_row=${samples_per_row}"
  .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
    --corpus-jsonl "${CORPUS}" \
    --checkpoint-dir "${BACKBONE_CKPT}" \
    --split val \
    --num-samples 0 \
    --prompt-mode joint \
    --target-mode joint \
    --image-prompt-style camera_labeled \
    --prompt-text-style official_alpamayo \
    --fuse-history-tokens \
    --geometry-reference gt \
    --batch-size 16 \
    --samples-per-row "${samples_per_row}" \
    --temperature 0.6 \
    --top-p 0.98 \
    --seed 42 \
    --max-new-tokens 256 \
    --skip-overlays \
    --output-dir "${out_dir}" \
    --summary-json "${summary}" \
    2>&1 | tee -a "${LOG}"
  log "done backbone tag=${tag}"
}

run_ae28_benchmark() {
  local tag="$1"
  local num_paths="$2"
  local selection="$3"
  local out_dir="outputs/benchmarks/${tag}"
  local summary="${out_dir}/student_noflex_ae28/summary.json"

  if [[ -s "${summary}" ]]; then
    log "skip_existing ae28 tag=${tag} summary=${summary}"
  else
    log "start ae28 tag=${tag} num_paths=${num_paths} selection=${selection}"
    .venv/bin/python -u scripts/benchmark_4models.py \
      --corpus-jsonl "${CORPUS}" \
      --split val \
      --num-samples 0 \
      --model student_noflex_ae28 \
      --output-dir "${out_dir}" \
      --batch-size 4 \
      --student-batch-size 8 \
      --eval-num-paths "${num_paths}" \
      --eval-temperature 0.6 \
      --eval-selection-method "${selection}" \
      --teacher-top-p 0.98 \
      --seed 42 \
      2>&1 | tee -a "${LOG}"
    log "done ae28 tag=${tag}"
  fi

  log "summarize ae28 tag=${tag}"
  .venv/bin/python scripts/summarize_teacher_selection_methods.py \
    --benchmark-root "${out_dir}" \
    --model-key student_noflex_ae28 \
    --tag "${tag}" \
    --output-json "${AUDIT_DIR}/${tag}_selection_summary.json" \
    --output-md "${AUDIT_DIR}/${tag}_selection_summary.md" \
    2>&1 | tee -a "${LOG}"
}

log "audit_start corpus=${CORPUS}"

run_backbone_decode "backbone_step006250_semantic_val806_official_t06_topp098_n1" 1
run_backbone_decode "backbone_step006250_semantic_val806_official_t06_topp098_n6" 6

run_ae28_benchmark "student_noflex_ae28_semantic_val806_official_t06_n1_20260615" 1 single
run_ae28_benchmark "student_noflex_ae28_semantic_val806_official_t06_n6_20260615" 6 mean_traj

log "audit_done"
