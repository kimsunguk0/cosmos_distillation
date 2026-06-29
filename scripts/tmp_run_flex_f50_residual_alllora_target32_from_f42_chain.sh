#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

CORPUS="data/corpus/flex_heldout256_stage2val_seed42.jsonl"
B0_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
F42_CKPT="outputs/checkpoints/flex_f42_scene_deepstack_long_state_from_f41_overfit16_s8000_lr2e7_20260607/step_002000"
B0_TRAJONLY="outputs/reports/b0_step006250_flexheldout256_decode_trajonly_summary.json"
RUN_NAME="flex_f50_residualslots_alllora_target32_from_f42_s8000_lr2e7_20260607"
LOG_PATH="outputs/logs/${RUN_NAME}_chain.log"

mkdir -p outputs/logs outputs/reports
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"f50_chain_start\",\"run_name\":\"${RUN_NAME}\",\"corpus\":\"${CORPUS}\",\"target\":\"${B0_TRAJONLY}\"}"

.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl "${CORPUS}" \
  --teacher-checkpoint-dir "${B0_CKPT}" \
  --student-checkpoint-dir "${F42_CKPT}" \
  --output-dir "outputs/checkpoints/${RUN_NAME}" \
  --split val \
  --max-train-samples 32 \
  --prompt-mode-override joint \
  --target-mode-override traj_only \
  --image-ablations normal \
  --paired-ablation none \
  --max-steps 8000 \
  --batch-size 1 \
  --learning-rate 2e-7 \
  --weight-decay 0.0 \
  --grad-clip-norm 5.0 \
  --log-every 100 \
  --save-every 2000 \
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
  --flex-residual-image-slots \
  --flex-residual-scale 1.0 \
  --train-flex \
  --unfreeze-all-lora \
  --unfreeze-multimodal-projector \
  --summary-json "outputs/reports/${RUN_NAME}_train_summary.json"

F50_CKPT="outputs/checkpoints/${RUN_NAME}/final"

.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  --corpus-jsonl "${CORPUS}" \
  --split val \
  --num-samples 32 \
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
  --flex-residual-image-slots \
  --flex-residual-scale 1.0 \
  --checkpoint-dir "${F50_CKPT}" \
  --output-dir "outputs/reports/${RUN_NAME}_final_decode_trajonly" \
  --summary-json "outputs/reports/${RUN_NAME}_final_decode_trajonly_summary.json"

.venv/bin/python -u scripts/109_compare_decode_to_free_run_targets.py \
  --decode-summary "outputs/reports/${RUN_NAME}_final_decode_trajonly_summary.json" \
  --target-summary "${B0_TRAJONLY}" \
  --corpus-jsonl "${CORPUS}" \
  --split val \
  --num-samples 32 \
  --summary-json "outputs/reports/${RUN_NAME}_final_b0_trajonly_parity_summary.json"

.venv/bin/python - <<'PY'
import datetime as _dt
import json
from pathlib import Path

run_name = "flex_f50_residualslots_alllora_target32_from_f42_s8000_lr2e7_20260607"
summary_path = Path(f"outputs/reports/{run_name}_final_b0_trajonly_parity_summary.json")
report_path = Path("reports/115-flex-f50-residual-slots-32-structure-gate.md")
summary = json.loads(summary_path.read_text())

token = float(summary.get("avg_target_token_match_rate", float("nan")))
ade = float(summary.get("avg_target_ade_m", float("nan")))
fde = float(summary.get("avg_target_fde_m", float("nan")))
unique = float(summary.get("avg_generated_unique_token_count", float("nan")))
max_run = float(summary.get("avg_generated_max_same_token_run", float("nan")))

if ade < 0.8:
    verdict = "residual-slot PASS: with original visual positions preserved, FLEX can match B0 on 32 samples."
elif ade < 2.333:
    verdict = "partial improvement: residual slots help, but do not recover F42-like 32-sample parity."
else:
    verdict = "residual-slot FAIL: preserving original visual positions is not enough."

block = f"""

## F50 Result

Auto-appended: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

| Checkpoint | Token match | ADE m | FDE m | Unique tokens | Max same-token run |
|---|---:|---:|---:|---:|---:|
| final / step 8000 | {token:.3f} | {ade:.3f} | {fde:.3f} | {unique:.2f} | {max_run:.2f} |

Decision: {verdict}

Artifact:

`{summary_path}`
"""

text = report_path.read_text() if report_path.exists() else ""
if "## F50 Result" not in text:
    report_path.write_text(text.rstrip() + block + "\n")
else:
    before = text.split("## F50 Result", 1)[0].rstrip()
    report_path.write_text(before + block + "\n")
print(json.dumps({
    "event": "f50_report_updated",
    "report": str(report_path),
    "summary": str(summary_path),
    "token_match": token,
    "ade_m": ade,
    "fde_m": fde,
    "verdict": verdict,
}))
PY

echo "{\"event\":\"f50_chain_done\",\"run_name\":\"${RUN_NAME}\"}"
