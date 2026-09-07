#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/pm97/workspace/sukim/distillation/cosmos_distillation}"
cd "${ROOT}"

CORPUS="${CORPUS:-data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl}"
STUDENT="${STUDENT:-outputs/checkpoints/stepa_q2_vqa_fullft_repaired_v1_bs8_e1/step_003488}"
OUT_ROOT="${OUT_ROOT:-outputs/checkpoints/stepb_ladder_20k}"
REPORT_ROOT="${REPORT_ROOT:-outputs/reports/stepb_ladder_20k}"
WAIT_RUN="${WAIT_RUN:-r0_f_fullft_lr5e6}"
NEXT_RUN="${NEXT_RUN:-r0_f_fullft_lr2e5}"
NEXT_CONFIG="${NEXT_CONFIG:-configs/train/stepb_ladder_r0_f_fullft_lr2e5.yaml}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-3}"
LOG_EVERY="${LOG_EVERY:-25}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
POLL_SEC="${POLL_SEC:-300}"
TARGET_STEPS="${TARGET_STEPS:-7500}"

mkdir -p "${OUT_ROOT}" "${REPORT_ROOT}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

LOG_PATH="${REPORT_ROOT}/${NEXT_RUN}_after_${WAIT_RUN}_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_PATH}") 2>&1

is_wait_run_complete() {
  .venv/bin/python - "${OUT_ROOT}/${WAIT_RUN}/metrics.jsonl" "${TARGET_STEPS}" <<'PY'
import json
import sys
from pathlib import Path

metrics = Path(sys.argv[1])
target = int(sys.argv[2])
if not metrics.exists():
    raise SystemExit(1)

max_step = 0
for line in metrics.read_text().splitlines():
    try:
        row = json.loads(line)
    except Exception:
        continue
    if row.get("phase") == "train":
        try:
            max_step = max(max_step, int(row.get("global_step") or 0))
        except Exception:
            pass

raise SystemExit(0 if max_step >= target else 1)
PY
}

echo "[queue] waiting for ${WAIT_RUN} to reach step ${TARGET_STEPS}"
while ! is_wait_run_complete; do
  echo "[queue][$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${WAIT_RUN} not complete yet; sleeping ${POLL_SEC}s"
  sleep "${POLL_SEC}"
done

echo "[queue][$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${WAIT_RUN} complete; starting ${NEXT_RUN}"

cmd=(
  .venv/bin/python -u scripts/09_train_distill.py
  --corpus-jsonl "${CORPUS}"
  --student-model "${STUDENT}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --pin-memory
  --persistent-workers
  --skip-asset-check
  --eval-every-epochs 1.0
  --save-every-epochs 0.3
  --skip-final-save
  --grad-clip-norm 1.0
  --early-stop-stage stage_a
  --early-stop-patience 999
  --log-every-steps "${LOG_EVERY}"
  --stage-config "${NEXT_CONFIG}"
  --output-dir "${OUT_ROOT}/${NEXT_RUN}"
  --summary-json "${REPORT_ROOT}/${NEXT_RUN}_summary.json"
  --disable-lora
)

printf '[queue] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
