#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/pm97/workspace/sukim/distillation/cosmos_distillation}"
cd "${ROOT}"

PY="${PY:-.venv/bin/python}"
RUN_ID="${RUN_ID:-r1_signcheck_$(date -u +%Y%m%dT%H%M%SZ)}"

CORPUS_VAL="${CORPUS_VAL:-data/corpus/val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl}"
CORPUS_20K="${CORPUS_20K:-data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl}"
REF_FROZEN="${REF_FROZEN:-outputs/references/backbone_eval_v1/frozen_val512_teacher_greedy_ref_v1/rows.jsonl}"

STEPA_INIT="${STEPA_INIT:-outputs/checkpoints/stepa_q2_vqa_fullft_repaired_v1_bs8_e1/step_003488}"
R0F20K_CKPT="${R0F20K_CKPT:-outputs/checkpoints/stepb_followup/followup_20260711_023134/r0_f_fullft_lr3e5_20k/best_decode}"
FULLFT200K_CKPT="${FULLFT200K_CKPT:-outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/fullft_lr3e5_200k_e1/best_decode}"
LORA200K_CKPT="${LORA200K_CKPT:-outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/lprime_lora_lr2e4_200k_e1/best_decode}"

OUT_ROOT="${OUT_ROOT:-outputs/checkpoints/stepb_r1_signcheck/${RUN_ID}}"
REPORT_ROOT="${REPORT_ROOT:-outputs/reports/stepb_r1_signcheck/${RUN_ID}}"
BASELINE_DIR="${REPORT_ROOT}/baseline"

BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LOG_EVERY="${LOG_EVERY:-25}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_R1="${RUN_R1:-1}"

mkdir -p "${OUT_ROOT}" "${REPORT_ROOT}" "${BASELINE_DIR}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

run_pg() {
  local tag="$1"
  local ckpt="$2"
  local summary="${BASELINE_DIR}/${tag}_pg_teacher_ref_summary.json"
  if [[ -s "${summary}" ]]; then
    log "{\"event\":\"skip_existing\",\"task\":\"pg\",\"tag\":\"${tag}\",\"summary\":\"${summary}\"}"
    return
  fi
  log "{\"event\":\"start\",\"task\":\"pg\",\"tag\":\"${tag}\",\"checkpoint\":\"${ckpt}\"}"
  "${PY}" -u scripts/70_eval_checkpoint_free_run.py \
    --corpus-jsonl "${CORPUS_VAL}" \
    --checkpoint-dir "${ckpt}" \
    --split val \
    --num-samples 512 \
    --max-new-tokens 320 \
    --no-do-sample \
    --temperature 1.0 \
    --top-p 1.0 \
    --prompt-mode joint \
    --target-mode joint \
    --image-prompt-style camera_labeled \
    --prompt-text-style official_alpamayo \
    --fuse-history-tokens \
    --reference-id teacher_greedy_ref_v1 \
    --reference-jsonl "${REF_FROZEN}" \
    --reference-token-field selected_traj_tokens \
    --reference-xyz-field selected_xyz \
    --summary-json "${summary}" \
    --device cuda
  log "{\"event\":\"done\",\"task\":\"pg\",\"tag\":\"${tag}\",\"summary\":\"${summary}\"}"
}

run_test_b() {
  local tag="r0f20k"
  local summary="${BASELINE_DIR}/${tag}_testb_summary.json"
  local samples="${BASELINE_DIR}/${tag}_testb_samples.jsonl"
  if [[ -s "${summary}" && -s "${samples}" ]]; then
    log "{\"event\":\"skip_existing\",\"task\":\"test_b\",\"summary\":\"${summary}\",\"samples\":\"${samples}\"}"
  else
    log "{\"event\":\"start\",\"task\":\"test_b\",\"tag\":\"${tag}\"}"
    "${PY}" -u scripts/82_eval_test_b_teacher_forced.py \
      --corpus-jsonl "${CORPUS_VAL}" \
      --checkpoint-dir "${R0F20K_CKPT}" \
      --checkpoint-name r0_f_fullft_lr3e5_20k \
      --split val \
      --num-samples 512 \
      --batch-size "${EVAL_BATCH_SIZE}" \
      --image-prompt-style camera_labeled \
      --prompt-text-style official_alpamayo \
      --fuse-history-tokens \
      --summary-json "${summary}" \
      --samples-jsonl "${samples}" \
      --save-token-sequences \
      --device cuda
    log "{\"event\":\"done\",\"task\":\"test_b\",\"summary\":\"${summary}\",\"samples\":\"${samples}\"}"
  fi

  local matched_dir="${BASELINE_DIR}/${tag}_matched_argmax"
  if [[ -s "${matched_dir}/summary.json" ]]; then
    log "{\"event\":\"skip_existing\",\"task\":\"matched_argmax\",\"summary\":\"${matched_dir}/summary.json\"}"
  else
    log "{\"event\":\"start\",\"task\":\"matched_argmax\",\"tag\":\"${tag}\"}"
    "${PY}" scripts/audit_val512_matched_argmax_vs_teacher_top1.py \
      --corpus-jsonl "${CORPUS_VAL}" \
      --model-samples r0_f_20k "${samples}" \
      --output-dir "${matched_dir}"
    log "{\"event\":\"done\",\"task\":\"matched_argmax\",\"summary\":\"${matched_dir}/summary.json\"}"
  fi
}

run_clean_bucket() {
  local summary="${BASELINE_DIR}/r0f20k_pg_teacher_ref_summary.json"
  local out_dir="${BASELINE_DIR}/r0f20k_clean_bucket"
  if [[ -s "${out_dir}/summary.json" ]]; then
    log "{\"event\":\"skip_existing\",\"task\":\"clean_bucket\",\"summary\":\"${out_dir}/summary.json\"}"
    return
  fi
  log "{\"event\":\"start\",\"task\":\"clean_bucket\"}"
  "${PY}" scripts/audit_val512_bin_distance_and_ade_strata.py \
    --corpus-jsonl "${CORPUS_VAL}" \
    --current-10b-rows-jsonl "${REF_FROZEN}" \
    --fullft-summary "${summary}" \
    --lora-summary "${summary}" \
    --student-model /home/pm97/workspace/sukim/base_weights/Cosmos-Reason2-2B \
    --output-dir "${out_dir}"
  log "{\"event\":\"done\",\"task\":\"clean_bucket\",\"summary\":\"${out_dir}/summary.json\"}"
}

run_ps6() {
  local tag="r0f20k_ps6"
  local summary="${BASELINE_DIR}/${tag}_summary.json"
  if [[ -s "${summary}" ]]; then
    log "{\"event\":\"skip_existing\",\"task\":\"ps6\",\"summary\":\"${summary}\"}"
    return
  fi
  log "{\"event\":\"start\",\"task\":\"ps6\",\"tag\":\"${tag}\"}"
  "${PY}" -u scripts/25_decode_checkpoint_overlays.py \
    --corpus-jsonl "${CORPUS_VAL}" \
    --checkpoint-dir "${R0F20K_CKPT}" \
    --split val \
    --num-samples 512 \
    --prompt-mode joint \
    --target-mode joint \
    --image-prompt-style camera_labeled \
    --prompt-text-style official_alpamayo \
    --fuse-history-tokens \
    --geometry-reference gt \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --samples-per-row 6 \
    --temperature 1.0 \
    --top-p 1.0 \
    --max-new-tokens 320 \
    --seed 42 \
    --device cuda \
    --output-dir "${BASELINE_DIR}/${tag}" \
    --summary-json "${summary}" \
    --skip-overlays
  log "{\"event\":\"done\",\"task\":\"ps6\",\"summary\":\"${summary}\"}"
}

write_baseline_index() {
  "${PY}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["BASELINE_DIR"])
paths = {
    "r0f20k_pg": root / "r0f20k_pg_teacher_ref_summary.json",
    "fullft200k_pg": root / "fullft200k_pg_teacher_ref_summary.json",
    "lora200k_pg": root / "lora200k_pg_teacher_ref_summary.json",
    "r0f20k_testb": root / "r0f20k_testb_summary.json",
    "r0f20k_matched": root / "r0f20k_matched_argmax" / "summary.json",
    "r0f20k_clean": root / "r0f20k_clean_bucket" / "summary.json",
    "r0f20k_ps6": root / "r0f20k_ps6_summary.json",
}

def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def pg_metrics(data):
    if not data:
        return None
    return {
        "n": data.get("num_samples"),
        "ade_vs_teacher_ref": data.get("avg_student_vs_geometry_reference_ade_m"),
        "fde_vs_teacher_ref": data.get("avg_student_vs_geometry_reference_fde_m"),
        "bad_geometry_rate": data.get("bad_geometry_rate"),
        "missing_geometry_reference_count": data.get("missing_geometry_reference_count"),
        "excluded_geometry_reference_count": data.get("excluded_geometry_reference_count"),
        "truncation_count": data.get("truncation_count"),
        "incomplete_traj_count": data.get("incomplete_traj_count"),
        "avg_unique_traj_ids": data.get("avg_unique_traj_ids"),
    }

matched = load(paths["r0f20k_matched"])
clean = load(paths["r0f20k_clean"])
ps6 = load(paths["r0f20k_ps6"])
out = {
    "run_id": os.environ["RUN_ID"],
    "reference_id": "teacher_greedy_ref_v1",
    "reference_jsonl": os.environ["REF_FROZEN"],
    "artifacts": {key: str(path) for key, path in paths.items()},
    "metrics": {
        "r0f20k_pg": pg_metrics(load(paths["r0f20k_pg"])),
        "fullft200k_pg": pg_metrics(load(paths["fullft200k_pg"])),
        "lora200k_pg": pg_metrics(load(paths["lora200k_pg"])),
        "r0f20k_matched_argmax": ((matched or {}).get("models") or {}).get("r0_f_20k"),
        "r0f20k_clean_bucket": ((clean or {}).get("models") or {}).get("fullft"),
        "r0f20k_ps6": {
            "n": (ps6 or {}).get("num_samples"),
            "minADE6_gt": (ps6 or {}).get("minADE6@6.4s_m"),
            "avg_fde_for_selected_minade": (ps6 or {}).get("avg_fde_m"),
            "avg_unique_traj_ids": (ps6 or {}).get("avg_unique_traj_ids"),
        },
    },
}
path = root / "baseline_index.json"
path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"event": "baseline_index_written", "path": str(path), "metrics": out["metrics"]}, indent=2), flush=True)
PY
}

run_r1_train() {
  local name="r1_fullft_lr3e5_tailkl_tau1_20k"
  local log_path="${REPORT_ROOT}/${name}.log"
  if [[ -s "${REPORT_ROOT}/${name}_summary.json" ]]; then
    log "{\"event\":\"skip_existing\",\"task\":\"r1_train\",\"summary\":\"${REPORT_ROOT}/${name}_summary.json\"}"
    return
  fi
  log "{\"event\":\"start\",\"task\":\"r1_train\",\"name\":\"${name}\",\"log\":\"${log_path}\"}"
  "${PY}" -u scripts/09_train_distill.py \
    --corpus-jsonl "${CORPUS_20K}" \
    --student-model "${STEPA_INIT}" \
    --stage-config configs/train/stepb_ladder_r1_tailkl.yaml \
    --output-dir "${OUT_ROOT}/${name}" \
    --summary-json "${REPORT_ROOT}/${name}_summary.json" \
    --batch-size "${BATCH_SIZE}" \
    --epochs 3.0 \
    --num-workers "${NUM_WORKERS}" \
    --prefetch-factor 2 \
    --pin-memory \
    --persistent-workers \
    --skip-asset-check \
    --eval-every-epochs 0.2 \
    --save-every-epochs 0.3 \
    --max-keep-checkpoints 4 \
    --grad-clip-norm 1.0 \
    --early-stop-stage stage_a \
    --early-stop-patience 4 \
    --log-every-steps "${LOG_EVERY}" \
    --disable-lora \
    > "${log_path}" 2>&1
  log "{\"event\":\"done\",\"task\":\"r1_train\",\"summary\":\"${REPORT_ROOT}/${name}_summary.json\"}"
}

log "{\"event\":\"run_start\",\"run_id\":\"${RUN_ID}\",\"report_root\":\"${REPORT_ROOT}\",\"out_root\":\"${OUT_ROOT}\"}"
"${PY}" scripts/check_stepb_weekend_ready.py
"${PY}" scripts/test_kd_tail_bucket.py

if [[ "${RUN_BASELINE}" == "1" ]]; then
  export RUN_ID BASELINE_DIR REF_FROZEN
  run_pg r0f20k "${R0F20K_CKPT}"
  run_pg fullft200k "${FULLFT200K_CKPT}"
  run_pg lora200k "${LORA200K_CKPT}"
  run_test_b
  run_clean_bucket
  run_ps6
  write_baseline_index
fi

if [[ "${RUN_R1}" == "1" ]]; then
  run_r1_train
fi

log "{\"event\":\"run_done\",\"run_id\":\"${RUN_ID}\",\"report_root\":\"${REPORT_ROOT}\",\"out_root\":\"${OUT_ROOT}\"}"
