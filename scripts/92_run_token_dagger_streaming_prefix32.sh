#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN_ID="${RUN_ID:-no_nav_token_dagger_stream_prefix32_b16_$(date +%Y%m%d_%H%M%S)}"
SHARD_SIZE="${SHARD_SIZE:-1000}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PREFIX_TOKENS="${PREFIX_TOKENS:-32}"

BASE_CORPUS="data/corpus/no_nav_teacher_pair_300chunks_semantic_balanced_50k.jsonl"
BASE_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_200k/no_nav_bp3_camera_labeled_official_200k_from_20kbest_20260514_092509/best_decode"
CONFIG="configs/train/stage_bp3_no_nav_token_dagger_prefix32.yaml"
SHARD_DIR="data/corpus/no_nav_token_dagger_stream_prefix32_b16/${RUN_ID}"
REPORT_DIR="outputs/reports/no_nav_distill/${RUN_ID}"
CKPT_ROOT="outputs/checkpoints/no_nav_token_dagger_stream/${RUN_ID}"
mkdir -p "${SHARD_DIR}" "${REPORT_DIR}" "${CKPT_ROOT}"

CURRENT_CKPT="${BASE_CKPT}"
SEED_PARTIAL="data/corpus/no_nav_token_dagger50k_prefix32_b16.jsonl"
SEED_LINES=0
if [[ -s "${SEED_PARTIAL}" ]]; then
  SEED_LINES="$(wc -l < "${SEED_PARTIAL}" | tr -d ' ')"
fi

echo "[stream] run_id=${RUN_ID} shard_size=${SHARD_SIZE} total=${TOTAL_SAMPLES} seed_lines=${SEED_LINES}"

start=0
while [[ "${start}" -lt "${TOTAL_SAMPLES}" ]]; do
  end=$(( start + SHARD_SIZE ))
  if [[ "${end}" -gt "${TOTAL_SAMPLES}" ]]; then
    end="${TOTAL_SAMPLES}"
  fi
  shard_tag="$(printf "%06d_%06d" "${start}" "${end}")"
  shard_path="${SHARD_DIR}/shard_${shard_tag}.jsonl"
  shard_report="${REPORT_DIR}/shard_${shard_tag}.json"
  train_out="${CKPT_ROOT}/shard_${shard_tag}"
  train_summary="${REPORT_DIR}/train_${shard_tag}_summary.json"

  if [[ ! -s "${shard_path}" ]]; then
    if [[ "${start}" -eq 0 && "${SEED_LINES}" -gt 0 ]]; then
      seed_use="${SEED_LINES}"
      if [[ "${seed_use}" -gt "${end}" ]]; then
        seed_use="${end}"
      fi
      seed_path="${SHARD_DIR}/shard_${shard_tag}.seed_${seed_use}.jsonl"
      head -n "${seed_use}" "${SEED_PARTIAL}" > "${seed_path}"
      remain=$(( end - seed_use ))
      if [[ "${remain}" -gt 0 ]]; then
        cont_path="${SHARD_DIR}/shard_${shard_tag}.cont_${seed_use}_${end}.jsonl"
        echo "[stream] build continuation ${seed_use}->${end} using ckpt=${CURRENT_CKPT}"
        .venv/bin/python -u scripts/90_build_token_dagger_corpus.py \
          --corpus-jsonl "${BASE_CORPUS}" \
          --student-checkpoint-dir "${CURRENT_CKPT}" \
          --teacher-model-path /home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B \
          --alpamayo-src /home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/src \
          --split train \
          --start-index "${seed_use}" \
          --max-samples "${remain}" \
          --prefix-tokens "${PREFIX_TOKENS}" \
          --batch-size "${BATCH_SIZE}" \
          --log-every 160 \
          --flush-every 1 \
          --output-jsonl "${cont_path}" \
          --report-json "${shard_report%.json}.cont.json"
        cat "${seed_path}" "${cont_path}" > "${shard_path}"
      else
        cp "${seed_path}" "${shard_path}"
      fi
    else
      count=$(( end - start ))
      echo "[stream] build shard ${start}->${end} using ckpt=${CURRENT_CKPT}"
      .venv/bin/python -u scripts/90_build_token_dagger_corpus.py \
        --corpus-jsonl "${BASE_CORPUS}" \
        --student-checkpoint-dir "${CURRENT_CKPT}" \
        --teacher-model-path /home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B \
        --alpamayo-src /home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/src \
        --split train \
        --start-index "${start}" \
        --max-samples "${count}" \
        --prefix-tokens "${PREFIX_TOKENS}" \
        --batch-size "${BATCH_SIZE}" \
        --log-every 160 \
        --flush-every 1 \
        --output-jsonl "${shard_path}" \
        --report-json "${shard_report}"
    fi
  else
    echo "[stream] shard exists, skip build: ${shard_path}"
  fi

  echo "[stream] train shard ${shard_tag} from ${CURRENT_CKPT}"
  .venv/bin/python -u scripts/09_train_distill.py \
    --corpus-jsonl "${shard_path}" \
    --stage-config "${CONFIG}" \
    --init-checkpoint-dir "${CURRENT_CKPT}" \
    --batch-size "${BATCH_SIZE}" \
    --max-train-samples "$(( end - start ))" \
    --max-steps "$(( (end - start + BATCH_SIZE - 1) / BATCH_SIZE ))" \
    --eval-every-epochs 0 \
    --save-every-epochs 1 \
    --skip-asset-check \
    --output-dir "${train_out}" \
    --summary-json "${train_summary}" \
    --log-every-steps 20
  CURRENT_CKPT="${train_out}/final"
  echo "${CURRENT_CKPT}" > "${CKPT_ROOT}/latest_checkpoint.txt"
  start="${end}"
done

echo "[stream] done run_id=${RUN_ID} latest=${CURRENT_CKPT}"
