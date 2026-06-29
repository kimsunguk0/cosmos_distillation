#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

CORPUS="data/corpus/flex_heldout256_stage2val_seed42.jsonl"
F62_CKPT="outputs/checkpoints/flex_f62_selector_passthrough_lora4_from_f0_overfit16_s1000_20260608/final"
B0_TRAJONLY="outputs/reports/b0_step006250_flexheldout256_decode_trajonly_summary.json"
RUN_NAME="flex_f63_selector_passthrough_lora4_f62_heldout256_trajonly_20260608"
LOG_PATH="outputs/logs/${RUN_NAME}.log"

mkdir -p outputs/logs outputs/reports
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"f63_start\",\"checkpoint\":\"${F62_CKPT}\",\"corpus\":\"${CORPUS}\"}"

COMMON_DECODE_ARGS=(
  --corpus-jsonl "${CORPUS}"
  --split val
  --num-samples 256
  --max-new-tokens 160
  --prompt-mode joint
  --target-mode traj_only
  --image-prompt-style camera_labeled
  --prompt-text-style official_alpamayo
  --fuse-history-tokens
  --geometry-reference teacher
  --batch-size 1
  --samples-per-row 1
  --skip-overlays
  --disable-failure-tags
  --checkpoint-dir "${F62_CKPT}"
  --preserve-flex-positions
  --flex-selection-strategy uniform
  --flex-passthrough-image-slots
  --flex-scene-deepstack
)

for MODE in normal camera_shuffle black; do
  MODE_ARGS=()
  if [[ "${MODE}" != "normal" ]]; then
    MODE_ARGS=(--image-ablation "${MODE}")
  fi
  SUMMARY="outputs/reports/${RUN_NAME}_${MODE}_decode_summary.json"
  echo "{\"event\":\"f63_decode_start\",\"mode\":\"${MODE}\"}"
  .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
    "${COMMON_DECODE_ARGS[@]}" \
    "${MODE_ARGS[@]}" \
    --output-dir "outputs/reports/${RUN_NAME}_${MODE}_decode" \
    --summary-json "${SUMMARY}"
done

.venv/bin/python -u scripts/109_compare_decode_to_free_run_targets.py \
  --decode-summary "outputs/reports/${RUN_NAME}_normal_decode_summary.json" \
  --target-summary "${B0_TRAJONLY}" \
  --corpus-jsonl "${CORPUS}" \
  --split val \
  --num-samples 256 \
  --summary-json "outputs/reports/${RUN_NAME}_normal_b0_trajonly_parity_summary.json"

.venv/bin/python - <<'PY'
import json
from pathlib import Path

run = "flex_f63_selector_passthrough_lora4_f62_heldout256_trajonly_20260608"
rows = []
for mode in ("normal", "camera_shuffle", "black"):
    path = Path(f"outputs/reports/{run}_{mode}_decode_summary.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "mode": mode,
            "ade": float(data.get("avg_ade_m", float("nan"))),
            "fde": float(data.get("avg_fde_m", float("nan"))),
            "token_match_gt": float(data.get("avg_token_match_rate", float("nan"))),
            "unique": float(data.get("avg_unique_traj_ids", float("nan"))),
            "max_run": float(data.get("avg_max_same_token_run", float("nan"))),
            "path": str(path),
        }
    )
parity = json.loads(Path(f"outputs/reports/{run}_normal_b0_trajonly_parity_summary.json").read_text(encoding="utf-8"))
print(
    json.dumps(
        {
            "event": "f63_done",
            "decode_rows": rows,
            "normal_b0_parity": {
                "ade": float(parity.get("avg_target_ade_m", float("nan"))),
                "fde": float(parity.get("avg_target_fde_m", float("nan"))),
                "token": float(parity.get("avg_target_token_match_rate", float("nan"))),
            },
        },
        ensure_ascii=True,
    ),
    flush=True,
)
PY
