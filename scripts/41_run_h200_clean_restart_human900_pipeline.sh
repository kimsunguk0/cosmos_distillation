#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/31_local_env.sh"

CORPUS_JSONL="${CORPUS_JSONL:-$ROOT_DIR/data/corpus/human_coc_teacher_full.jsonl}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT_DIR/outputs/checkpoints/h200_clean_restart_human900}"
REPORT_ROOT="${REPORT_ROOT:-$ROOT_DIR/outputs/reports/h200_clean_restart_human900}"
mkdir -p "$CHECKPOINT_ROOT" "$REPORT_ROOT"

stage_has_outputs() {
  local output_dir="$1"
  local report_prefix="$2"
  [[ -d "$output_dir/final" ]] || return 1
  [[ -f "${report_prefix}_val204_decode_summary.json" ]] || return 1
  [[ -f "${report_prefix}_train64_decode_summary.json" ]] || return 1
}

stage_outputs_current() {
  local output_dir="$1"
  local report_prefix="$2"
  local init_checkpoint="${3:-}"
  stage_has_outputs "$output_dir" "$report_prefix" || return 1

  if [[ "${FORCE_RERUN_STAGES:-0}" == "1" ]]; then
    return 1
  fi

  if [[ -z "$init_checkpoint" ]]; then
    return 0
  fi

  local train_config="$output_dir/final/train_config.json"
  [[ -f "$train_config" ]] || return 1
  "$VENV_PYTHON" - "$train_config" "$init_checkpoint" <<'PY'
import json
import sys
from pathlib import Path

train_config = Path(sys.argv[1])
expected = Path(sys.argv[2]).expanduser().resolve()
payload = json.loads(train_config.read_text(encoding="utf-8"))
actual_raw = payload.get("init_checkpoint_dir") or payload.get("args", {}).get("init_checkpoint_dir")
if not actual_raw:
    raise SystemExit(1)
actual = Path(actual_raw).expanduser().resolve()
raise SystemExit(0 if actual == expected else 1)
PY
}

stage0_overfit_has_outputs() {
  local output_dir="$1"
  local train_summary_json="$2"
  local decode_summary_json="$3"
  [[ -d "$output_dir/final" ]] || return 1
  [[ -f "$train_summary_json" ]] || return 1
  [[ -f "$decode_summary_json" ]] || return 1
}

select_best() {
  local selection_name="$1"
  shift
  "$VENV_PYTHON" "$ROOT_DIR/scripts/26_select_best_sft_checkpoint.py" \
    "$@" \
    --link-path "$CHECKPOINT_ROOT/${selection_name}" \
    --selection-json "$REPORT_ROOT/${selection_name}_selection.json" \
    --selection-md "$REPORT_ROOT/${selection_name}_selection.md" \
    --force
}

run_stage() {
  local stage_name="$1"
  local stage_config="$2"
  local init_checkpoint="${3:-}"
  local batch_size="$4"
  local output_dir="$CHECKPOINT_ROOT/$stage_name"
  local summary_json="$REPORT_ROOT/${stage_name}_train_summary.json"
  local report_prefix="$REPORT_ROOT/$stage_name"

  if stage_outputs_current "$output_dir" "$report_prefix" "$init_checkpoint"; then
    echo "[skip] $stage_name already has decode summaries"
    return
  fi

  local -a env_vars=(
    "STAGE_CONFIG=$stage_config"
    "CORPUS_JSONL=$CORPUS_JSONL"
    "OUTPUT_DIR=$output_dir"
    "SUMMARY_JSON=$summary_json"
    "REPORT_PREFIX=$report_prefix"
    "BATCH_SIZE=$batch_size"
    "VAL_EVAL_SAMPLES=204"
    "TRAIN_EVAL_SAMPLES=64"
  )

  if [[ -n "$init_checkpoint" ]]; then
    env_vars+=("INIT_CHECKPOINT_DIR=$init_checkpoint")
  fi

  env "${env_vars[@]}" "$ROOT_DIR/scripts/40_train_eval_h200_stage.sh"
}

echo "[pipeline] rebuild strict teacher corpus"
"$ROOT_DIR/scripts/34_build_local_teacher_corpus.sh"

echo "[pipeline] stage0 token-row gate"
"$VENV_PYTHON" "$ROOT_DIR/scripts/35_stage0_token_row_gate.py" \
  --corpus-jsonl "$CORPUS_JSONL" \
  --stage-config "$ROOT_DIR/configs/train/stage_h200_clean_human900_stage0.yaml" \
  --batch-size 4 \
  --max-train-samples 4 \
  --summary-json "$REPORT_ROOT/stage0_token_row_gate.json"

if ! stage0_overfit_has_outputs \
  "$CHECKPOINT_ROOT/stage0_overfit4" \
  "$REPORT_ROOT/stage0_overfit4_train_summary.json" \
  "$REPORT_ROOT/stage0_overfit4_decode_summary.json"; then
  echo "[pipeline] stage0 overfit4"
  OUTPUT_DIR="$CHECKPOINT_ROOT/stage0_overfit4" \
  SUMMARY_JSON="$REPORT_ROOT/stage0_overfit4_train_summary.json" \
  DECODE_OUTPUT_DIR="$REPORT_ROOT/stage0_overfit4_decode" \
  DECODE_SUMMARY_JSON="$REPORT_ROOT/stage0_overfit4_decode_summary.json" \
  MAX_STEPS="${STAGE0_OVERFIT4_MAX_STEPS:-200}" \
  TRAIN_SAMPLES=4 \
  BATCH_SIZE=4 \
  "$ROOT_DIR/scripts/37_run_h200_stage0_overfit4.sh"
else
  echo "[skip] stage0 overfit4"
fi

if ! stage0_overfit_has_outputs \
  "$CHECKPOINT_ROOT/stage0_overfit64" \
  "$REPORT_ROOT/stage0_overfit64_train_summary.json" \
  "$REPORT_ROOT/stage0_overfit64_decode_summary.json"; then
  echo "[pipeline] stage0 overfit64"
  OUTPUT_DIR="$CHECKPOINT_ROOT/stage0_overfit64" \
  SUMMARY_JSON="$REPORT_ROOT/stage0_overfit64_train_summary.json" \
  DECODE_OUTPUT_DIR="$REPORT_ROOT/stage0_overfit64_decode" \
  DECODE_SUMMARY_JSON="$REPORT_ROOT/stage0_overfit64_decode_summary.json" \
  MAX_STEPS="${STAGE0_OVERFIT64_MAX_STEPS:-1200}" \
  SAVE_EVERY_EPOCHS="${STAGE0_OVERFIT64_SAVE_EVERY_EPOCHS:-10}" \
  LOG_EVERY_STEPS="${STAGE0_OVERFIT64_LOG_EVERY_STEPS:-10}" \
  INIT_CHECKPOINT_DIR="${STAGE0_OVERFIT64_INIT_CHECKPOINT_DIR:-}" \
  TRAIN_SAMPLES=64 \
  "$ROOT_DIR/scripts/38_run_h200_stage0_overfit64.sh"
else
  echo "[skip] stage0 overfit64"
fi

STAGE0_BEST="${STAGE0_BEST:-$CHECKPOINT_ROOT/stage0_overfit64/final}"
run_stage "stage1_r64" "$ROOT_DIR/configs/train/stage_h200_clean_human900_stage1.yaml" "$STAGE0_BEST" "${FULL_STAGE_BATCH_SIZE:-32}"
run_stage "stage1_r128" "$ROOT_DIR/configs/train/stage_h200_clean_human900_stage1_r128.yaml" "$STAGE0_BEST" "${FULL_STAGE_BATCH_SIZE:-32}"
select_best "H200_STAGE1_BEST" \
  "$REPORT_ROOT/stage1_r64_val204_decode_summary.json" \
  "$REPORT_ROOT/stage1_r64_train64_decode_summary.json" \
  "$REPORT_ROOT/stage1_r128_val204_decode_summary.json" \
  "$REPORT_ROOT/stage1_r128_train64_decode_summary.json"
STAGE1_BEST="$(readlink -f "$CHECKPOINT_ROOT/H200_STAGE1_BEST")"

run_stage "stage2_kd020" "$ROOT_DIR/configs/train/stage_h200_clean_human900_stage2_kd020.yaml" "$STAGE1_BEST" "${FULL_STAGE_BATCH_SIZE:-32}"
run_stage "stage2_kd030" "$ROOT_DIR/configs/train/stage_h200_clean_human900_stage2_kd030.yaml" "$STAGE1_BEST" "${FULL_STAGE_BATCH_SIZE:-32}"
select_best "H200_STAGE2_BEST" \
  "$REPORT_ROOT/stage2_kd020_val204_decode_summary.json" \
  "$REPORT_ROOT/stage2_kd020_train64_decode_summary.json" \
  "$REPORT_ROOT/stage2_kd030_val204_decode_summary.json" \
  "$REPORT_ROOT/stage2_kd030_train64_decode_summary.json"
STAGE2_BEST="$(readlink -f "$CHECKPOINT_ROOT/H200_STAGE2_BEST")"

run_stage "stage3_ce015" "$ROOT_DIR/configs/train/stage_h200_clean_human900_stage3a_teacherce015.yaml" "$STAGE2_BEST" "${FULL_STAGE_BATCH_SIZE:-32}"
run_stage "stage3_ce020" "$ROOT_DIR/configs/train/stage_h200_clean_human900_stage3b_teacherce020.yaml" "$STAGE2_BEST" "${FULL_STAGE_BATCH_SIZE:-32}"
select_best "H200_STAGE3_BEST" \
  "$REPORT_ROOT/stage3_ce015_val204_decode_summary.json" \
  "$REPORT_ROOT/stage3_ce015_train64_decode_summary.json" \
  "$REPORT_ROOT/stage3_ce020_val204_decode_summary.json" \
  "$REPORT_ROOT/stage3_ce020_train64_decode_summary.json"

select_best "H200_STAGE23_BEST" \
  "$REPORT_ROOT/stage2_kd020_val204_decode_summary.json" \
  "$REPORT_ROOT/stage2_kd020_train64_decode_summary.json" \
  "$REPORT_ROOT/stage2_kd030_val204_decode_summary.json" \
  "$REPORT_ROOT/stage2_kd030_train64_decode_summary.json" \
  "$REPORT_ROOT/stage3_ce015_val204_decode_summary.json" \
  "$REPORT_ROOT/stage3_ce015_train64_decode_summary.json" \
  "$REPORT_ROOT/stage3_ce020_val204_decode_summary.json" \
  "$REPORT_ROOT/stage3_ce020_train64_decode_summary.json"
STAGE23_BEST="$(readlink -f "$CHECKPOINT_ROOT/H200_STAGE23_BEST")"

run_stage "stage4_hidden003" "$ROOT_DIR/configs/train/stage_h200_clean_human900_stage4a_hidden003.yaml" "$STAGE23_BEST" "${FULL_STAGE_BATCH_SIZE:-32}"
run_stage "stage4_hidden005" "$ROOT_DIR/configs/train/stage_h200_clean_human900_stage4b_hidden005.yaml" "$STAGE23_BEST" "${FULL_STAGE_BATCH_SIZE:-32}"
select_best "H200_STAGE4_BEST" \
  "$REPORT_ROOT/stage4_hidden003_val204_decode_summary.json" \
  "$REPORT_ROOT/stage4_hidden003_train64_decode_summary.json" \
  "$REPORT_ROOT/stage4_hidden005_val204_decode_summary.json" \
  "$REPORT_ROOT/stage4_hidden005_train64_decode_summary.json"

select_best "H200_STABLE_BEST" \
  "$REPORT_ROOT/stage2_kd020_val204_decode_summary.json" \
  "$REPORT_ROOT/stage2_kd020_train64_decode_summary.json" \
  "$REPORT_ROOT/stage2_kd030_val204_decode_summary.json" \
  "$REPORT_ROOT/stage2_kd030_train64_decode_summary.json" \
  "$REPORT_ROOT/stage3_ce015_val204_decode_summary.json" \
  "$REPORT_ROOT/stage3_ce015_train64_decode_summary.json" \
  "$REPORT_ROOT/stage3_ce020_val204_decode_summary.json" \
  "$REPORT_ROOT/stage3_ce020_train64_decode_summary.json" \
  "$REPORT_ROOT/stage4_hidden003_val204_decode_summary.json" \
  "$REPORT_ROOT/stage4_hidden003_train64_decode_summary.json" \
  "$REPORT_ROOT/stage4_hidden005_val204_decode_summary.json" \
  "$REPORT_ROOT/stage4_hidden005_train64_decode_summary.json"
STABLE_BEST="$(readlink -f "$CHECKPOINT_ROOT/H200_STABLE_BEST")"

run_stage "stage6_last4" "$ROOT_DIR/configs/train/stage_h200_clean_human900_stage6a_last4.yaml" "$STABLE_BEST" "${STAGE6_LAST4_BATCH_SIZE:-16}"
run_stage "stage6_last8" "$ROOT_DIR/configs/train/stage_h200_clean_human900_stage6b_last8.yaml" "$STABLE_BEST" "${STAGE6_LAST8_BATCH_SIZE:-12}"
select_best "H200_STAGE6_BEST" \
  "$REPORT_ROOT/stage6_last4_val204_decode_summary.json" \
  "$REPORT_ROOT/stage6_last4_train64_decode_summary.json" \
  "$REPORT_ROOT/stage6_last8_val204_decode_summary.json" \
  "$REPORT_ROOT/stage6_last8_train64_decode_summary.json"

run_stage "stage7_ss010" "$ROOT_DIR/configs/train/stage_h200_clean_human900_stage7a_ss010.yaml" "$STABLE_BEST" "${FULL_STAGE_BATCH_SIZE:-32}"
run_stage "stage7_ss015" "$ROOT_DIR/configs/train/stage_h200_clean_human900_stage7b_ss015.yaml" "$STABLE_BEST" "${FULL_STAGE_BATCH_SIZE:-32}"
select_best "H200_STAGE7_BEST" \
  "$REPORT_ROOT/stage7_ss010_val204_decode_summary.json" \
  "$REPORT_ROOT/stage7_ss010_train64_decode_summary.json" \
  "$REPORT_ROOT/stage7_ss015_val204_decode_summary.json" \
  "$REPORT_ROOT/stage7_ss015_train64_decode_summary.json"

select_best "H200_OVERALL_BEST" \
  "$REPORT_ROOT/stage2_kd020_val204_decode_summary.json" \
  "$REPORT_ROOT/stage2_kd020_train64_decode_summary.json" \
  "$REPORT_ROOT/stage2_kd030_val204_decode_summary.json" \
  "$REPORT_ROOT/stage2_kd030_train64_decode_summary.json" \
  "$REPORT_ROOT/stage3_ce015_val204_decode_summary.json" \
  "$REPORT_ROOT/stage3_ce015_train64_decode_summary.json" \
  "$REPORT_ROOT/stage3_ce020_val204_decode_summary.json" \
  "$REPORT_ROOT/stage3_ce020_train64_decode_summary.json" \
  "$REPORT_ROOT/stage4_hidden003_val204_decode_summary.json" \
  "$REPORT_ROOT/stage4_hidden003_train64_decode_summary.json" \
  "$REPORT_ROOT/stage4_hidden005_val204_decode_summary.json" \
  "$REPORT_ROOT/stage4_hidden005_train64_decode_summary.json" \
  "$REPORT_ROOT/stage6_last4_val204_decode_summary.json" \
  "$REPORT_ROOT/stage6_last4_train64_decode_summary.json" \
  "$REPORT_ROOT/stage6_last8_val204_decode_summary.json" \
  "$REPORT_ROOT/stage6_last8_train64_decode_summary.json" \
  "$REPORT_ROOT/stage7_ss010_val204_decode_summary.json" \
  "$REPORT_ROOT/stage7_ss010_train64_decode_summary.json" \
  "$REPORT_ROOT/stage7_ss015_val204_decode_summary.json" \
  "$REPORT_ROOT/stage7_ss015_train64_decode_summary.json"

echo "[pipeline] completed"
