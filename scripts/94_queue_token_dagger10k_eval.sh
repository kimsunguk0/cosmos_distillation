#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN_ID="${RUN_ID:?RUN_ID is required}"
CKPT_ROOT="outputs/checkpoints/no_nav_token_dagger10k/${RUN_ID}"
CKPT_DIR="${CKPT_ROOT}/final"
CORPUS_JSONL="${CORPUS_JSONL:-data/corpus/no_nav_teacher_pair_300chunks_semantic_balanced_50k.jsonl}"
REPORT_DIR="outputs/reports/no_nav_distill/${RUN_ID}_eval"
POLL_SEC="${POLL_SEC:-60}"

mkdir -p "${REPORT_DIR}"

echo "[queue-eval] run_id=${RUN_ID}"
echo "[queue-eval] waiting checkpoint=${CKPT_DIR}"
while [[ ! -f "${CKPT_DIR}/checkpoint_manifest.json" ]]; do
  echo "[queue-eval] $(date -Is) waiting final checkpoint..."
  sleep "${POLL_SEC}"
done

echo "[queue-eval] final checkpoint ready; free-run val64"
.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  --corpus-jsonl "${CORPUS_JSONL}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --split val \
  --num-samples 64 \
  --prompt-mode joint \
  --target-mode joint \
  --image-prompt-style camera_labeled \
  --prompt-text-style official_alpamayo \
  --fuse-history-tokens \
  --geometry-reference teacher \
  --max-new-tokens 256 \
  --batch-size 4 \
  --skip-overlays \
  --disable-failure-tags \
  --output-dir "${REPORT_DIR}/free_run_val64" \
  --summary-json "${REPORT_DIR}/free_run_val64_summary.json"

echo "[queue-eval] teacher-forced Test B val64"
.venv/bin/python -u scripts/82_eval_test_b_teacher_forced.py \
  --corpus-jsonl "${CORPUS_JSONL}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --checkpoint-name "${RUN_ID}_final" \
  --split val \
  --num-samples 64 \
  --batch-size 8 \
  --image-prompt-style camera_labeled \
  --prompt-text-style official_alpamayo \
  --fuse-history-tokens \
  --summary-json "${REPORT_DIR}/test_b_val64_summary.json" \
  --samples-jsonl "${REPORT_DIR}/test_b_val64_samples.jsonl"

echo "[queue-eval] done report_dir=${REPORT_DIR}"
