#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

PY=".venv/bin/python"
CORPUS="data/corpus/benchmark_semantic_val_cap50_seed42.jsonl"
AUDIT_DIR="outputs/teacher_native_audit_20260615"
mkdir -p "$AUDIT_DIR"

log_event() {
  printf '%s %s\n' "$(date -Is)" "$*" >> "$AUDIT_DIR/audit.log"
}

summarize_run() {
  local tag="$1"
  local benchmark_root="$2"
  local methods="$3"

  log_event "summarize_start tag=$tag benchmark_root=$benchmark_root"
  "$PY" scripts/summarize_teacher_selection_methods.py \
    --benchmark-root "$benchmark_root" \
    --model-key teacher10b \
    --tag "$tag" \
    --methods "$methods" \
    --output-json "$AUDIT_DIR/${tag}_summary.json" \
    --output-md "$AUDIT_DIR/${tag}_summary.md" \
    >> "$AUDIT_DIR/audit.log" 2>&1
  log_event "summarize_done tag=$tag"
}

run_teacher() {
  local tag="$1"
  local out_dir="$2"
  local top_p="$3"
  local temperature="$4"
  local num_paths="$5"
  local selection="$6"
  local methods="$7"

  mkdir -p "$out_dir"
  log_event "benchmark_start tag=$tag out_dir=$out_dir top_p=$top_p temperature=$temperature num_paths=$num_paths"
  "$PY" -u scripts/benchmark_4models.py \
    --corpus-jsonl "$CORPUS" \
    --output-dir "$out_dir" \
    --model teacher10b \
    --split val \
    --num-samples 0 \
    --batch-size 4 \
    --student-batch-size 8 \
    --eval-num-paths "$num_paths" \
    --eval-temperature "$temperature" \
    --eval-selection-method "$selection" \
    --teacher-top-p "$top_p" \
    --teacher-top-k 0 \
    --seed 42 \
    > "$out_dir/run.log" 2>&1
  log_event "benchmark_done tag=$tag"
  summarize_run "$tag" "$out_dir" "$methods"
}

log_event "audit_start"

summarize_run \
  "report138_topp095_temp085_n6_existing" \
  "outputs/benchmarks/semantic_val806_4models_20260612" \
  "first_path,mean_traj,medoid,oracle_best"

run_teacher \
  "report138_topp095_temp085_n1" \
  "outputs/benchmarks/teacher_native_report138_topp095_temp085_n1_20260615" \
  "0.95" \
  "0.85" \
  "1" \
  "single" \
  "single,first_path,mean_traj,medoid,oracle_best"

run_teacher \
  "official_topp098_temp06_n1" \
  "outputs/benchmarks/teacher_native_official_topp098_temp06_n1_20260615" \
  "0.98" \
  "0.6" \
  "1" \
  "single" \
  "single,first_path,mean_traj,medoid,oracle_best"

run_teacher \
  "official_topp098_temp06_n6" \
  "outputs/benchmarks/teacher_native_official_topp098_temp06_n6_20260615" \
  "0.98" \
  "0.6" \
  "6" \
  "mean_traj" \
  "first_path,mean_traj,medoid,oracle_best"

log_event "audit_done"
