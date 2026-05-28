#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/pm97/workspace/sukim/distillation/cosmos_distillation"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
WAIT_PID="${1:-9785}"
REQUIRE_SUMMARY_JSON="${2:-}"
PORT="${DAGGER_DASHBOARD_PORT:-8797}"

CORPUS="${ROOT_DIR}/data/corpus/no_nav_teacher_pair_300chunks_semantic_balanced_50k.jsonl"
CONFIG="${ROOT_DIR}/configs/train/stage_bp3_no_nav_camera_labeled_dagger50k_semantic.yaml"
INIT_CKPT="${ROOT_DIR}/outputs/checkpoints/no_nav_camera_labeled_official_200k/no_nav_bp3_camera_labeled_official_200k_from_20kbest_20260514_092509/best_decode"
BASE_MODEL="/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b"
LOG_DIR="${ROOT_DIR}/outputs/logs"
REPORT_DIR="${ROOT_DIR}/outputs/reports/no_nav_distill"
CKPT_ROOT="${ROOT_DIR}/outputs/checkpoints/no_nav_camera_labeled_official_200k"

mkdir -p "${LOG_DIR}" "${REPORT_DIR}" "${CKPT_ROOT}"

if [[ ! -f "${CORPUS}" ]]; then
  echo "[queue][error] missing corpus: ${CORPUS}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "[queue][error] missing config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${INIT_CKPT}" ]]; then
  echo "[queue][error] missing init checkpoint: ${INIT_CKPT}" >&2
  exit 1
fi

echo "[queue] waiting for PID ${WAIT_PID} to exit before DAgger run"
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  date +"[queue] %Y-%m-%d %H:%M:%S still waiting for PID ${WAIT_PID}"
  sleep 60
done

if [[ -n "${REQUIRE_SUMMARY_JSON}" ]]; then
  echo "[queue] checking required upstream summary: ${REQUIRE_SUMMARY_JSON}"
  if [[ ! -f "${REQUIRE_SUMMARY_JSON}" ]]; then
    echo "[queue][error] required upstream summary is missing: ${REQUIRE_SUMMARY_JSON}" >&2
    exit 2
  fi
  UPSTREAM_STATUS="$("${PYTHON_BIN}" - "${REQUIRE_SUMMARY_JSON}" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload.get("status", ""))
PY
)"
  if [[ "${UPSTREAM_STATUS}" != "ok" ]]; then
    echo "[queue][error] upstream status is not ok: ${UPSTREAM_STATUS}" >&2
    exit 3
  fi
  echo "[queue] upstream status ok"
fi

RUN_ID="no_nav_official12500_dagger_semantic50k_p15_prefix32_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${CKPT_ROOT}/${RUN_ID}"
TRAIN_LOG="${LOG_DIR}/${RUN_ID}.log"
DASH_LOG="${LOG_DIR}/${RUN_ID}_dashboard_${PORT}.log"
SUMMARY_JSON="${REPORT_DIR}/${RUN_ID}_summary.json"

echo "[queue] starting ${RUN_ID}"
echo "[queue] output=${OUT_DIR}"
echo "[queue] log=${TRAIN_LOG}"

cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export COSMOS_DATALOADER_NUM_WORKERS="${COSMOS_DATALOADER_NUM_WORKERS:-8}"
export COSMOS_DATALOADER_PREFETCH_FACTOR="${COSMOS_DATALOADER_PREFETCH_FACTOR:-2}"
export COSMOS_DATALOADER_PIN_MEMORY="${COSMOS_DATALOADER_PIN_MEMORY:-1}"
export COSMOS_DATALOADER_PERSISTENT_WORKERS="${COSMOS_DATALOADER_PERSISTENT_WORKERS:-1}"

"${PYTHON_BIN}" scripts/09_train_distill.py \
  --corpus-jsonl "${CORPUS}" \
  --stage-config "${CONFIG}" \
  --student-model "${BASE_MODEL}" \
  --init-checkpoint-dir "${INIT_CKPT}" \
  --batch-size 16 \
  --max-steps 3125 \
  --eval-every-epochs 0.5 \
  --save-every-epochs 0.5 \
  --early-stop-patience 0 \
  --multi-gpu off \
  --num-workers "${COSMOS_DATALOADER_NUM_WORKERS}" \
  --prefetch-factor "${COSMOS_DATALOADER_PREFETCH_FACTOR}" \
  --pin-memory \
  --persistent-workers \
  --skip-asset-check \
  --log-every-steps 10 \
  --output-dir "${OUT_DIR}" \
  --summary-json "${SUMMARY_JSON}" \
  > "${TRAIN_LOG}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${LOG_DIR}/${RUN_ID}.pid"
echo "[queue] train_pid=${TRAIN_PID}"

"${PYTHON_BIN}" scripts/58_serve_no_nav_bp3_dashboard.py \
  --run-id "${RUN_ID}" \
  --output-dir "${OUT_DIR}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  > "${DASH_LOG}" 2>&1 &

DASH_PID=$!
echo "${DASH_PID}" > "${LOG_DIR}/${RUN_ID}_dashboard_${PORT}.pid"
echo "[queue] dashboard_pid=${DASH_PID} port=${PORT}"

wait "${TRAIN_PID}"
STATUS=$?
echo "[queue] train exited status=${STATUS}"
exit "${STATUS}"
