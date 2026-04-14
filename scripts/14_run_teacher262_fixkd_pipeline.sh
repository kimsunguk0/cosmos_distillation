#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/pm97/workspace/sukim/cosmos_distillation"
PYTHON_BIN="${PYTHON_BIN:-/home/pm97/workspace/kjhong/alpamayo_100/ar1_venv/bin/python}"
export PYTHONPATH="${ROOT}/.vendor${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"

echo "[pipeline] $(date -Is) teacher262 fixkd pipeline start"

echo "[step] teacher signal cache 262 full"
"${PYTHON_BIN}" scripts/05_extract_teacher_signal_cache.py \
  --teacher-index-jsonl data/teacher_cache/text_v2_262/index.jsonl \
  --cache-root data/teacher_cache/text_v2_262 \
  --fields teacher_long_cot teacher_short_reason teacher_answer \
  --overwrite \
  --summary-json outputs/reports/teacher_signal_cache_teacher262_fixkd_full.json

echo "[step] corpus build 262 fixkd"
"${PYTHON_BIN}" scripts/08_build_multitask_corpus.py \
  --manifest-path data/processed/teacher_262_manifest.parquet \
  --supervision-jsonl data/processed/supervision_records_v2_262/records.jsonl \
  --teacher-index-jsonl data/teacher_cache/text_v2_262/index.jsonl \
  --consistency-parquet data/teacher_cache/diagnostics/consistency_matrix_teacher262_v2.parquet \
  --output-jsonl data/corpus/strict_human_long_cot_262_fixkd.jsonl \
  --summary-json outputs/reports/corpus_summary_teacher262_fixkd.json

echo "[step] smoke gate 262 fixkd"
"${PYTHON_BIN}" scripts/11_validate_smoke_gates.py \
  --manifest-path data/processed/teacher_262_manifest.parquet \
  --teacher-index-jsonl data/teacher_cache/text_v2_262/index.jsonl \
  --corpus-jsonl data/corpus/strict_human_long_cot_262_fixkd.jsonl \
  --num-samples 10 \
  --summary-json outputs/reports/smoke_gate_summary_teacher262_fixkd.json

echo "[step] stage-b train 262 fixkd with-lora"
"${PYTHON_BIN}" scripts/09_train_distill.py \
  --corpus-jsonl data/corpus/strict_human_long_cot_262_fixkd.jsonl \
  --stage-config configs/train/stage_b_tuned.yaml \
  --output-dir outputs/checkpoints/train_stage_b_teacher262_fixkd_full \
  --summary-json outputs/reports/train_stage_b_teacher262_fixkd_full.json \
  --batch-size 1 \
  --max-train-samples 233 \
  --max-steps 120

echo "[step] eval 262 fixkd"
"${PYTHON_BIN}" scripts/10_eval_distill.py \
  --corpus-jsonl data/corpus/strict_human_long_cot_262_fixkd.jsonl \
  --teacher-index-jsonl data/teacher_cache/text_v2_262/index.jsonl \
  --consistency-parquet data/teacher_cache/diagnostics/consistency_matrix_teacher262_v2.parquet \
  --summary-json outputs/reports/eval_summary_teacher262_fixkd_full.json

echo "[pipeline] $(date -Is) teacher262 fixkd pipeline done"
