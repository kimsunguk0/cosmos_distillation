#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/pm97/workspace/sukim/distillation/cosmos_distillation"
PY="/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/.venv/bin/python"
OUT="${OUT:-outputs/eval/vlm_cap_gap_human_ood_20260701_full300}"
MANIFEST="${MANIFEST:-${OUT}/manifest.jsonl}"
TASKS="${TASKS:-T1_pos,T1_neg,T2,T3,T4}"
SAMPLE_N="${SAMPLE_N:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
EXPECTED_ROWS="${EXPECTED_ROWS:-9000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BATCH_SIZE_32B="${BATCH_SIZE_32B:-8}"
BACKGROUND_MODELS="${BACKGROUND_MODELS:-32b}"
LOG_DIR="${OUT}/logs"

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

row_count() {
  local model="$1"
  local path="${OUT}/runs/${model}/predictions.jsonl"
  if [[ -f "${path}" ]]; then
    wc -l < "${path}"
  else
    echo 0
  fi
}

start_model_session() {
  local model="$1"
  local session="vlm_gap_full300_${model}"
  local log="${LOG_DIR}/${model}_parallel.log"
  local batch_size="${BATCH_SIZE}"
  if [[ "${model}" == "32b" ]]; then
    batch_size="${BATCH_SIZE_32B}"
  fi
  if [[ "$(row_count "${model}")" -ge "${EXPECTED_ROWS}" ]]; then
    echo "[$(date -u +%Y%m%dT%H%M%SZ)] ${model} already complete rows=$(row_count "${model}")"
    return
  fi
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "[$(date -u +%Y%m%dT%H%M%SZ)] ${model} session already running rows=$(row_count "${model}")"
    return
  fi
  echo "[$(date -u +%Y%m%dT%H%M%SZ)] launching ${model} rows=$(row_count "${model}")"
  tmux new-session -d -s "${session}" \
    "cd ${ROOT} && ${PY} scripts/vlm_cap_gap_eval.py run-model \
      --output-dir ${OUT} \
      --manifest ${MANIFEST} \
      --model-key ${model} \
      --limit-clips 300 \
      --tasks ${TASKS} \
      --sample-n ${SAMPLE_N} \
      --sample-temperature 0.7 \
      --max-new-tokens ${MAX_NEW_TOKENS} \
      --batch-size ${batch_size} \
      --log-every 100 \
      2>&1 | tee -a ${log}"
}

restart_32b_for_phase2_batch() {
  local restart_marker="${OUT}/locks/32b_phase2_batch_restart.done"
  mkdir -p "${OUT}/locks"
  if [[ "${BATCH_SIZE_32B}" -le 1 ]]; then
    return
  fi
  if [[ "$(row_count 32b)" -ge "${EXPECTED_ROWS}" ]]; then
    return
  fi
  if [[ -f "${restart_marker}" ]]; then
    return
  fi
  if tmux has-session -t vlm_gap_full300_32b 2>/dev/null; then
    echo "[$(date -u +%Y%m%dT%H%M%SZ)] restarting 32b with batch_size=${BATCH_SIZE_32B} for phase2"
    tmux kill-session -t vlm_gap_full300_32b || true
    sleep 10
  fi
  touch "${restart_marker}"
}

wait_for_models() {
  local models=("$@")
  while true; do
    local all_done=1
    local bg_model
    for bg_model in ${BACKGROUND_MODELS}; do
      start_model_session "${bg_model}"
    done
    for model in "${models[@]}"; do
      local rows
      rows="$(row_count "${model}")"
      if [[ "${rows}" -lt "${EXPECTED_ROWS}" ]]; then
        all_done=0
        start_model_session "${model}"
      fi
    done
    echo "[$(date -u +%Y%m%dT%H%M%SZ)] rows: 2b=$(row_count 2b) 8b=$(row_count 8b) 32b=$(row_count 32b)"
    if [[ "${all_done}" -eq 1 ]]; then
      break
    fi
    sleep "${POLL_SECONDS}"
  done
}

echo "[$(date -u +%Y%m%dT%H%M%SZ)] phase1: run 2b and 8b in parallel"
start_model_session 2b
start_model_session 8b
wait_for_models 2b 8b

echo "[$(date -u +%Y%m%dT%H%M%SZ)] phase2: run 32b alone"
restart_32b_for_phase2_batch
start_model_session 32b
wait_for_models 32b

while tmux has-session -t vlm_gap_full300_32b 2>/dev/null; do
  echo "[$(date -u +%Y%m%dT%H%M%SZ)] 32b rows complete; waiting for session cleanup"
  sleep 60
done

bash scripts/run_vlm_cap_gap_postprocess.sh \
  2>&1 | tee -a "${LOG_DIR}/postprocess_scheduler.log"

echo "[$(date -u +%Y%m%dT%H%M%SZ)] DONE ${OUT}"
