#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

AUDIT_DIR="outputs/same_metric_audit_20260615"
CORPUS="data/corpus/benchmark_semantic_val_cap50_seed42.jsonl"
AE28_N6_SUMMARY="outputs/benchmarks/student_noflex_ae28_semantic_val806_official_t06_n6_20260615/student_noflex_ae28/summary.json"
LOG="${AUDIT_DIR}/run_10b_backbone_discrete.log"

mkdir -p "${AUDIT_DIR}"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "${LOG}"
}

wait_for_ae28_n6() {
  log "wait_for ae28_n6_summary=${AE28_N6_SUMMARY}"
  while [[ ! -s "${AE28_N6_SUMMARY}" ]]; do
    sleep 60
  done
  log "detected ae28_n6_summary=${AE28_N6_SUMMARY}"
}

run_10b_backbone() {
  local tag="$1"
  local samples_per_row="$2"
  local out_dir="${AUDIT_DIR}/${tag}"
  local summary="${out_dir}/summary.json"

  if [[ -s "${summary}" ]]; then
    log "skip_existing 10b_backbone tag=${tag} summary=${summary}"
    return
  fi

  log "start 10b_backbone tag=${tag} samples_per_row=${samples_per_row}"
  .venv/bin/python -u scripts/eval_10b_backbone_discrete.py \
    --corpus-jsonl "${CORPUS}" \
    --split val \
    --num-samples 0 \
    --samples-per-row "${samples_per_row}" \
    --temperature 0.6 \
    --top-p 0.98 \
    --top-k 0 \
    --seed 42 \
    --max-new-tokens 256 \
    --output-dir "${out_dir}" \
    --summary-json "${summary}" \
    2>&1 | tee -a "${LOG}"
  log "done 10b_backbone tag=${tag}"
}

log "10b_backbone_discrete_audit_start corpus=${CORPUS}"
wait_for_ae28_n6
run_10b_backbone "teacher10b_backbone_discrete_semantic_val806_official_t06_topp098_n1" 1
run_10b_backbone "teacher10b_backbone_discrete_semantic_val806_official_t06_topp098_n6" 6
log "10b_backbone_discrete_audit_done"
