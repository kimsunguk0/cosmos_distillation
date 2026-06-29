#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

CORPUS="data/corpus/flex_heldout256_stage2val_seed42.jsonl"
B0_TRAJONLY="outputs/reports/b0_step006250_flexheldout16_decode_trajonly_summary.json"
F0_K1792="outputs/checkpoints/flex_f54_f0_perimage_k1792_camtime_from_b0_20260607"
RUN_ROOT="flex_f58_mrope_passthrough_k1792_f0_step006250_heldout16"
LOG_PATH="outputs/logs/${RUN_ROOT}.log"

mkdir -p outputs/logs outputs/reports
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"f58_start\",\"checkpoint\":\"${F0_K1792}\"}"

for SELECTION in first uniform; do
  for DEEPSTACK in off on; do
    RUN_NAME="${RUN_ROOT}_${SELECTION}_ds${DEEPSTACK}"
    DEEPSTACK_ARGS=()
    if [[ "${DEEPSTACK}" == "on" ]]; then
      DEEPSTACK_ARGS=(--flex-scene-deepstack)
    fi
    echo "{\"event\":\"f58_decode_start\",\"selection\":\"${SELECTION}\",\"deepstack\":\"${DEEPSTACK}\"}"

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
      --flex-selection-strategy "${SELECTION}" \
      --preserve-flex-positions \
      "${DEEPSTACK_ARGS[@]}" \
      --output-dir "outputs/reports/${RUN_NAME}_decode_trajonly" \
      --summary-json "outputs/reports/${RUN_NAME}_decode_trajonly_summary.json"

    .venv/bin/python -u scripts/109_compare_decode_to_free_run_targets.py \
      --decode-summary "outputs/reports/${RUN_NAME}_decode_trajonly_summary.json" \
      --target-summary "${B0_TRAJONLY}" \
      --corpus-jsonl "${CORPUS}" \
      --split val \
      --num-samples 16 \
      --summary-json "outputs/reports/${RUN_NAME}_b0_trajonly_parity_summary.json"
  done
done

.venv/bin/python - <<'PY'
import json
from pathlib import Path

run_root = "flex_f58_mrope_passthrough_k1792_f0_step006250_heldout16"
results = []
for selection in ("first", "uniform"):
    for deepstack in ("off", "on"):
        name = f"{run_root}_{selection}_ds{deepstack}"
        decode_path = Path(f"outputs/reports/{name}_decode_trajonly_summary.json")
        parity_path = Path(f"outputs/reports/{name}_b0_trajonly_parity_summary.json")
        decode = json.loads(decode_path.read_text())
        parity = json.loads(parity_path.read_text())
        rows = decode.get("samples", [])
        results.append(
            {
                "selection": selection,
                "deepstack": deepstack,
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
        )
print(json.dumps({"event": "f58_done", "results": results}, ensure_ascii=True))
PY
