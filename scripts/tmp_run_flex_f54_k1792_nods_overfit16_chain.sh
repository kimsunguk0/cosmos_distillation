#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

CORPUS="data/corpus/flex_heldout256_stage2val_seed42.jsonl"
B0_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
B0_TRAJONLY="outputs/reports/b0_step006250_flexheldout16_decode_trajonly_summary.json"
F0_K1792="outputs/checkpoints/flex_f54_f0_perimage_k1792_camtime_from_b0_20260607"
RUN_NAME="flex_f54_k1792_nods_overfit16_from_f0_s3000_lr5e7_20260607"
LOG_PATH="outputs/logs/${RUN_NAME}_chain.log"

mkdir -p outputs/logs outputs/reports
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"f54_chain_start\",\"run_name\":\"${RUN_NAME}\",\"corpus\":\"${CORPUS}\",\"target\":\"${B0_TRAJONLY}\"}"

if [[ ! -d "${F0_K1792}" ]]; then
  echo "{\"event\":\"f54_create_f0_start\",\"output_dir\":\"${F0_K1792}\"}"
  .venv/bin/python -u scripts/103_make_flex_untrained_checkpoint.py \
    --base-checkpoint-dir "${B0_CKPT}" \
    --output-dir "${F0_K1792}" \
    --tokens-per-image 112 \
    --expected-images-per-sample 16 \
    --input-hidden-size 2048 \
    --hidden-size 1024 \
    --num-layers 2 \
    --num-heads 8 \
    --mlp-ratio 4.0 \
    --dropout 0.0 \
    --use-camera-time-embeddings \
    --max-camera-types 16 \
    --compression-mode per_image \
    --seed 20260607
fi

.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl "${CORPUS}" \
  --teacher-checkpoint-dir "${B0_CKPT}" \
  --student-checkpoint-dir "${F0_K1792}" \
  --output-dir "outputs/checkpoints/${RUN_NAME}" \
  --split val \
  --max-train-samples 16 \
  --prompt-mode-override joint \
  --target-mode-override traj_only \
  --image-ablations normal \
  --paired-ablation none \
  --max-steps 3000 \
  --batch-size 1 \
  --learning-rate 5e-7 \
  --weight-decay 0.0 \
  --grad-clip-norm 5.0 \
  --log-every 100 \
  --save-every 500 \
  --seed 42 \
  --traj-kl-weight 0.0 \
  --text-kl-weight 0.0 \
  --format-kl-weight 0.0 \
  --boundary-cos-weight 0.0 \
  --boundary-norm-weight 0.0 \
  --boundary-mse-weight 0.0 \
  --pairwise-boundary-delta-cos-weight 0.0 \
  --pairwise-boundary-delta-norm-weight 0.0 \
  --pairwise-traj-logprob-delta-weight 0.0 \
  --pairwise-free-run-margin-weight 0.0 \
  --free-run-token-targets normal="${B0_TRAJONLY}" \
  --free-run-token-ce-weight 1.0 \
  --free-run-token-ce-modes normal \
  --free-run-end-token-ce-weight 0.05 \
  --prefix-token-ce-weight 0.0 \
  --traj-state-cos-weight 1.0 \
  --traj-state-norm-weight 0.10 \
  --traj-state-mse-weight 0.001 \
  --free-run-token-force-context \
  --free-run-token-context-source target \
  --cache-teacher-targets \
  --cache-collated-batches \
  --train-flex \
  --unfreeze-lora-last-n-layers 4 \
  --unfreeze-multimodal-projector \
  --summary-json "outputs/reports/${RUN_NAME}_train_summary.json"

COMMON_DECODE_ARGS=(
  --corpus-jsonl "${CORPUS}"
  --split val
  --num-samples 16
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
)

for CKPT_NAME in step_000500 step_001000 step_001500 step_002000 step_002500 step_003000 final; do
  CKPT_DIR="outputs/checkpoints/${RUN_NAME}/${CKPT_NAME}"
  if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "{\"event\":\"f54_decode_skip_missing\",\"checkpoint\":\"${CKPT_DIR}\"}"
    continue
  fi
  SAFE_NAME="${CKPT_NAME//_/-}"
  DECODE_SUMMARY="outputs/reports/${RUN_NAME}_${SAFE_NAME}_decode_trajonly_summary.json"
  PARITY_SUMMARY="outputs/reports/${RUN_NAME}_${SAFE_NAME}_b0_trajonly_parity_summary.json"
  echo "{\"event\":\"f54_decode_start\",\"checkpoint\":\"${CKPT_DIR}\"}"
  .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
    "${COMMON_DECODE_ARGS[@]}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --output-dir "outputs/reports/${RUN_NAME}_${SAFE_NAME}_decode_trajonly" \
    --summary-json "${DECODE_SUMMARY}"

  .venv/bin/python -u scripts/109_compare_decode_to_free_run_targets.py \
    --decode-summary "${DECODE_SUMMARY}" \
    --target-summary "${B0_TRAJONLY}" \
    --corpus-jsonl "${CORPUS}" \
    --split val \
    --num-samples 16 \
    --summary-json "${PARITY_SUMMARY}"
done

.venv/bin/python - <<'PY'
import datetime as _dt
import json
from pathlib import Path

run_name = "flex_f54_k1792_nods_overfit16_from_f0_s3000_lr5e7_20260607"
report_path = Path("reports/119-flex-f54-k1792-nods-overfit16-gate.md")
rows = []
for summary_path in sorted(Path("outputs/reports").glob(f"{run_name}_*_b0_trajonly_parity_summary.json")):
    data = json.loads(summary_path.read_text())
    label = summary_path.name.removeprefix(f"{run_name}_").removesuffix("_b0_trajonly_parity_summary.json")
    rows.append(
        {
            "label": label,
            "token": float(data.get("avg_target_token_match_rate", float("nan"))),
            "ade": float(data.get("avg_target_ade_m", float("nan"))),
            "fde": float(data.get("avg_target_fde_m", float("nan"))),
            "unique": float(data.get("avg_generated_unique_token_count", float("nan"))),
            "max_run": float(data.get("avg_generated_max_same_token_run", float("nan"))),
            "path": str(summary_path),
        }
    )
best = min(rows, key=lambda row: row["ade"]) if rows else None
lines = [
    "# 119 - FLEX F54 K1792 No-DeepStack 16-Sample Gate",
    "",
    f"Auto-generated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "## Purpose",
    "",
    "Test whether K896 is over-compressed by increasing retained visual tokens from 56/image to 112/image.",
    "",
    "## Settings",
    "",
    "- F0 source: B0 step_006250 + untrained per-image FLEX",
    "- tokens per image: 112",
    "- scene tokens: 1792",
    "- original visual tokens: 2880",
    "- actual compressed DeepStack: off",
    "- samples: 16 held-out val rows",
    "- trainable: FLEX scene encoder + last4 LoRA + multimodal projector",
    "- LR: 5e-7",
    "- steps: 3000",
    "- loss: B0 free-run token CE + trajectory-state alignment",
    "",
    "## Results",
    "",
    "| Checkpoint | Token match | ADE m | FDE m | Unique tokens | Max same-token run |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['label']} | {row['token']:.3f} | {row['ade']:.3f} | {row['fde']:.3f} | {row['unique']:.2f} | {row['max_run']:.2f} |"
    )
lines += ["", "## Decision", ""]
if best is None:
    lines.append("No parity summaries were produced.")
else:
    if best["ade"] < 0.5:
        verdict = "PASS: K1792 can preserve B0 16-sample parity well enough to try 32 samples."
    elif best["ade"] < 0.9:
        verdict = "Partial: K1792 is better than failed K896 gates but not stable enough yet."
    else:
        verdict = "FAIL: increasing K to 1792 does not fix 16-sample parity."
    lines.append(
        f"Best checkpoint: `{best['label']}` with ADE `{best['ade']:.3f} m`, FDE `{best['fde']:.3f} m`, token match `{best['token']:.3f}`."
    )
    lines.append("")
    lines.append(verdict)
    lines.append("")
    lines.append(f"Best artifact: `{best['path']}`")
report_path.write_text("\n".join(lines) + "\n")
print(json.dumps({"event": "f54_report_updated", "report": str(report_path), "best": best}))
PY

echo "{\"event\":\"f54_chain_done\",\"run_name\":\"${RUN_NAME}\"}"
