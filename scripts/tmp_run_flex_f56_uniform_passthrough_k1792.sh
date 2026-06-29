#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

CORPUS="data/corpus/flex_heldout256_stage2val_seed42.jsonl"
B0_TRAJONLY="outputs/reports/b0_step006250_flexheldout16_decode_trajonly_summary.json"
F0_K1792="outputs/checkpoints/flex_f54_f0_perimage_k1792_camtime_from_b0_20260607"
RUN_NAME="flex_f56_uniform_passthrough_k1792_f0_step006250_heldout16"
LOG_PATH="outputs/logs/${RUN_NAME}.log"

mkdir -p outputs/logs outputs/reports
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"f56_start\",\"checkpoint\":\"${F0_K1792}\",\"selection\":\"uniform\"}"

.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  --corpus-jsonl "${CORPUS}" \
  --split val \
  --num-samples 16 \
  --max-new-tokens 160 \
  --prompt-mode joint \
  --target-mode traj_only \
  --image-prompt-style camera_labeled \
  --prompt-text-style official_alpamayo \
  --fuse-history-tokens \
  --geometry-reference teacher \
  --batch-size 1 \
  --samples-per-row 1 \
  --skip-overlays \
  --disable-failure-tags \
  --checkpoint-dir "${F0_K1792}" \
  --flex-passthrough-image-slots \
  --flex-selection-strategy uniform \
  --output-dir "outputs/reports/${RUN_NAME}_decode_trajonly" \
  --summary-json "outputs/reports/${RUN_NAME}_decode_trajonly_summary.json"

.venv/bin/python -u scripts/109_compare_decode_to_free_run_targets.py \
  --decode-summary "outputs/reports/${RUN_NAME}_decode_trajonly_summary.json" \
  --target-summary "${B0_TRAJONLY}" \
  --corpus-jsonl "${CORPUS}" \
  --split val \
  --num-samples 16 \
  --summary-json "outputs/reports/${RUN_NAME}_b0_trajonly_parity_summary.json"

.venv/bin/python - <<'PY'
import json
from pathlib import Path

run_name = "flex_f56_uniform_passthrough_k1792_f0_step006250_heldout16"
decode_path = Path(f"outputs/reports/{run_name}_decode_trajonly_summary.json")
parity_path = Path(f"outputs/reports/{run_name}_b0_trajonly_parity_summary.json")
decode = json.loads(decode_path.read_text())
parity = json.loads(parity_path.read_text())
rows = decode.get("samples", [])
summary = {
    "event": "f56_done",
    "decode_summary": str(decode_path),
    "parity_summary": str(parity_path),
    "free_run_ade_m": sum(float(r.get("ade_m", 0.0)) for r in rows) / max(len(rows), 1),
    "free_run_fde_m": sum(float(r.get("fde_m", 0.0)) for r in rows) / max(len(rows), 1),
    "b0_parity_ade_m": float(parity.get("avg_target_ade_m", float("nan"))),
    "b0_parity_fde_m": float(parity.get("avg_target_fde_m", float("nan"))),
    "b0_token_match": float(parity.get("avg_target_token_match_rate", float("nan"))),
    "unique_tokens": float(parity.get("avg_generated_unique_token_count", float("nan"))),
    "max_same_token_run": float(parity.get("avg_generated_max_same_token_run", float("nan"))),
}
print(json.dumps(summary, ensure_ascii=True))
PY
