#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/pm97/workspace/sukim/cosmos_distillation"
PYTHON_BIN="${PYTHON_BIN:-/home/pm97/workspace/kjhong/alpamayo_100/ar1_venv/bin/python}"
export PYTHONPATH="${ROOT}/.vendor${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"

echo "[pipeline] $(date -Is) teacher262 full pipeline start"

echo "[step] teacher cache request generation 262"
"${PYTHON_BIN}" scripts/05_generate_teacher_text_cache.py \
  --manifest-path data/processed/teacher_262_manifest.parquet \
  --supervision-jsonl data/processed/supervision_records_v2_262/records.jsonl \
  --canonical-root data/processed/canonical_samples \
  --cache-root data/teacher_cache/text_v2_262 \
  --summary-json outputs/reports/teacher_text_cache_summary_teacher262_v2.json

echo "[step] teacher text inference overwrite 262"
"${PYTHON_BIN}" scripts/05_run_teacher_text_inference.py \
  --teacher-index-jsonl data/teacher_cache/text_v2_262/index.jsonl \
  --cache-root data/teacher_cache/text_v2_262 \
  --overwrite \
  --summary-json outputs/reports/teacher_text_inference_teacher262_full.json

echo "[step] teacher signal cache 262"
"${PYTHON_BIN}" scripts/05_extract_teacher_signal_cache.py \
  --teacher-index-jsonl data/teacher_cache/text_v2_262/index.jsonl \
  --cache-root data/teacher_cache/text_v2_262 \
  --fields teacher_long_cot teacher_short_reason teacher_answer \
  --summary-json outputs/reports/teacher_signal_cache_teacher262_full.json

echo "[step] consistency gate 262"
"${PYTHON_BIN}" scripts/07_run_consistency_gate.py \
  --manifest-path data/processed/teacher_262_manifest.parquet \
  --supervision-jsonl data/processed/supervision_records_v2_262/records.jsonl \
  --canonical-root data/processed/canonical_samples \
  --teacher-index-jsonl data/teacher_cache/text_v2_262/index.jsonl \
  --output-parquet data/teacher_cache/diagnostics/consistency_matrix_teacher262_v2.parquet \
  --summary-md outputs/reports/consistency_summary_teacher262_v2.md

echo "[step] corpus build 262"
"${PYTHON_BIN}" scripts/08_build_multitask_corpus.py \
  --manifest-path data/processed/teacher_262_manifest.parquet \
  --supervision-jsonl data/processed/supervision_records_v2_262/records.jsonl \
  --teacher-index-jsonl data/teacher_cache/text_v2_262/index.jsonl \
  --consistency-parquet data/teacher_cache/diagnostics/consistency_matrix_teacher262_v2.parquet \
  --output-jsonl data/corpus/strict_human_long_cot_262.jsonl \
  --summary-json outputs/reports/corpus_summary_teacher262_v2.json

echo "[step] smoke gate snapshot 262"
"${PYTHON_BIN}" scripts/11_validate_smoke_gates.py \
  --manifest-path data/processed/teacher_262_manifest.parquet \
  --teacher-index-jsonl data/teacher_cache/text_v2_262/index.jsonl \
  --corpus-jsonl data/corpus/strict_human_long_cot_262.jsonl \
  --num-samples 10 \
  --summary-json outputs/reports/smoke_gate_summary_teacher262_full.json

echo "[step] stage-b train 262 with-lora"
"${PYTHON_BIN}" scripts/09_train_distill.py \
  --corpus-jsonl data/corpus/strict_human_long_cot_262.jsonl \
  --stage-config configs/train/stage_b_tuned.yaml \
  --output-dir outputs/checkpoints/train_stage_b_teacher262_full \
  --summary-json outputs/reports/train_stage_b_teacher262_full.json \
  --batch-size 1 \
  --max-train-samples 233 \
  --max-steps 120

echo "[step] eval 262"
"${PYTHON_BIN}" scripts/10_eval_distill.py \
  --corpus-jsonl data/corpus/strict_human_long_cot_262.jsonl \
  --teacher-index-jsonl data/teacher_cache/text_v2_262/index.jsonl \
  --consistency-parquet data/teacher_cache/diagnostics/consistency_matrix_teacher262_v2.parquet \
  --summary-json outputs/reports/eval_summary_teacher262_full.json

echo "[pipeline] $(date -Is) teacher262 full pipeline done"
