#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/31_local_env.sh"

CORPUS_JSONL="${CORPUS_JSONL:-$ROOT_DIR/data/corpus/human_coc_local.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-$ROOT_DIR/outputs/reports/human_coc_local_corpus_summary.json}"

"$VENV_PYTHON" "$ROOT_DIR/scripts/08_build_multitask_corpus.py" \
  --event-manifest "$COSMOS_DATA_ROOT/state/event_manifest.parquet" \
  --semantic-gate-parquet "$COSMOS_DATA_ROOT/state/split_semantic_gate.parquet" \
  --teacher-index-jsonl "$COSMOS_DATA_ROOT/teacher_cache/text/index.jsonl" \
  --materialized-root "$COSMOS_DATA_ROOT/materialized" \
  --allow-missing-teacher \
  --output-jsonl "$CORPUS_JSONL" \
  --summary-json "$SUMMARY_JSON"
