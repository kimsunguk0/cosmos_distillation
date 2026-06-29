#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

K1024_RUN="${K1024_RUN:-mlflex_k1024_bp3_20k_e3_b16_fast_20260609_fast}"
K1024_DIR="${K1024_DIR:-outputs/checkpoints/${K1024_RUN}}"
K1024_SUMMARY="${K1024_SUMMARY:-outputs/reports/${K1024_RUN}_summary.json}"
K1024_TMUX="${K1024_TMUX:-mlflex_k1024_bp3_fast}"

K512_BASE_DIR="${K512_BASE_DIR:-outputs/checkpoints/mlflex_k512_bp3_20k_e3_b16_20260608/final}"
K512_RUN="${K512_RUN:-mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024}"
K512_DIR="${K512_DIR:-outputs/checkpoints/${K512_RUN}}"
K512_SUMMARY="${K512_SUMMARY:-outputs/reports/${K512_RUN}_summary.json}"

CORPUS="${CORPUS:-data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl}"
SELECTED_JSON="${SELECTED_JSON:-outputs/reports/mlflex_k512_bp3_eval512_seed42_selected_ids.json}"
CONFIG_K512="${CONFIG_K512:-configs/train/stage_mlflex_k512_bp3_hidden_gc_20k_e3.yaml}"
LOG_PATH="${LOG_PATH:-outputs/logs/watch_k1024_eval_then_k512_continue_20260609.log}"
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-2}"
DECODE_SEED="${DECODE_SEED:-97}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-8}"
TRAIN_PREFETCH_FACTOR="${TRAIN_PREFETCH_FACTOR:-2}"
TRAIN_LOG_EVERY_STEPS="${TRAIN_LOG_EVERY_STEPS:-100}"
K512_LR="${K512_LR:-1e-5}"
K512_EPOCHS="${K512_EPOCHS:-3.0}"

K1024_GREEDY_SUMMARY="${K1024_GREEDY_SUMMARY:-outputs/reports/${K1024_RUN}_val512_trajonly_gt_greedy_summary.json}"
K1024_N6_SUMMARY="${K1024_N6_SUMMARY:-outputs/reports/${K1024_RUN}_val512_trajonly_gt_n6_summary.json}"
K1024_COMBINED_JSON="${K1024_COMBINED_JSON:-outputs/reports/${K1024_RUN}_val512_trajonly_gt_minade6_summary.json}"
K1024_COMBINED_MD="${K1024_COMBINED_MD:-outputs/reports/${K1024_RUN}_val512_trajonly_gt_minade6_summary.md}"

mkdir -p outputs/logs outputs/reports outputs/checkpoints
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"watch_start\",\"time\":\"$(date -Is)\",\"k1024_run\":\"${K1024_RUN}\",\"k512_run\":\"${K512_RUN}\"}"

if [[ ! -f "${SELECTED_JSON}" ]]; then
  echo "{\"event\":\"selected_ids_missing\",\"time\":\"$(date -Is)\",\"path\":\"${SELECTED_JSON}\"}"
  exit 1
fi

while [[ ! -d "${K1024_DIR}/final" || ! -f "${K1024_SUMMARY}" ]]; do
  if ! tmux has-session -t "${K1024_TMUX}" 2>/dev/null; then
    echo "{\"event\":\"k1024_missing_final_and_tmux_dead\",\"time\":\"$(date -Is)\",\"k1024_dir\":\"${K1024_DIR}\",\"summary\":\"${K1024_SUMMARY}\",\"tmux\":\"${K1024_TMUX}\"}"
    exit 2
  fi
  .venv/bin/python - <<'PY'
import json
from pathlib import Path

p = Path("outputs/checkpoints/mlflex_k1024_bp3_20k_e3_b16_fast_20260609_fast/metrics.jsonl")
last = None
if p.exists():
    for line in p.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("phase") == "train":
            last = row.get("global_step")
print(json.dumps({"event": "waiting_k1024", "last_train_step": last}))
PY
  sleep 120
done

echo "{\"event\":\"k1024_ready\",\"time\":\"$(date -Is)\",\"checkpoint\":\"${K1024_DIR}/final\",\"summary\":\"${K1024_SUMMARY}\"}"

run_decode() {
  local label="$1"
  local samples_per_row="$2"
  local output_dir="$3"
  local summary_json="$4"

  echo "{\"event\":\"decode_start\",\"time\":\"$(date -Is)\",\"label\":\"${label}\",\"samples_per_row\":${samples_per_row},\"summary\":\"${summary_json}\"}"
  .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
    --corpus-jsonl "${CORPUS}" \
    --checkpoint-dir "${K1024_DIR}/final" \
    --split val \
    --selected-json "${SELECTED_JSON}" \
    --prompt-mode joint \
    --target-mode traj_only \
    --image-prompt-style camera_labeled \
    --prompt-text-style official_alpamayo \
    --fuse-history-tokens \
    --geometry-reference gt \
    --max-new-tokens 160 \
    --batch-size "${DECODE_BATCH_SIZE}" \
    --samples-per-row "${samples_per_row}" \
    --temperature 1.0 \
    --top-p 1.0 \
    --seed "${DECODE_SEED}" \
    --preserve-flex-positions \
    --flex-selection-strategy uniform \
    --flex-scene-deepstack \
    --skip-overlays \
    --disable-failure-tags \
    --output-dir "${output_dir}" \
    --summary-json "${summary_json}"
  echo "{\"event\":\"decode_done\",\"time\":\"$(date -Is)\",\"label\":\"${label}\",\"summary\":\"${summary_json}\"}"
}

if [[ ! -f "${K1024_COMBINED_JSON}" || "${FORCE_REEVAL:-0}" == "1" ]]; then
  run_decode \
    "k1024_final_greedy" \
    1 \
    "outputs/reports/${K1024_RUN}_val512_trajonly_gt_greedy" \
    "${K1024_GREEDY_SUMMARY}"

  run_decode \
    "k1024_final_n6" \
    6 \
    "outputs/reports/${K1024_RUN}_val512_trajonly_gt_n6" \
    "${K1024_N6_SUMMARY}"

  .venv/bin/python - <<PY
import json
from pathlib import Path

greedy_path = Path("${K1024_GREEDY_SUMMARY}")
n6_path = Path("${K1024_N6_SUMMARY}")
combined_json = Path("${K1024_COMBINED_JSON}")
combined_md = Path("${K1024_COMBINED_MD}")
greedy = json.loads(greedy_path.read_text(encoding="utf-8"))
n6 = json.loads(n6_path.read_text(encoding="utf-8"))

def metric(data, *keys):
    for key in keys:
        value = data.get(key)
        if value is not None:
            return float(value)
    return None

row = {
    "model": "K1024 final",
    "checkpoint_dir": greedy.get("checkpoint_dir"),
    "num_samples": greedy.get("num_samples"),
    "ade@6.4s_m": metric(greedy, "ade@6.4s_m", "avg_ade_m"),
    "fde@6.4s_m": metric(greedy, "avg_fde_m"),
    "minADE6@6.4s_m": metric(n6, "minADE6@6.4s_m", "avg_ade_m"),
    "minFDE6_selected_by_ADE@6.4s_m": metric(n6, "avg_fde_m"),
    "greedy_summary": str(greedy_path),
    "n6_summary": str(n6_path),
}
summary = {
    "eval_set": {
        "corpus_jsonl": "${CORPUS}",
        "selected_json": "${SELECTED_JSON}",
        "split": "val",
        "num_samples": 512,
        "geometry_reference": "gt",
        "prompt_mode": "joint",
        "target_mode": "traj_only",
        "max_new_tokens": 160,
        "seed": int("${DECODE_SEED}"),
    },
    "rows": [row],
}
combined_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def fmt(value):
    return "n/a" if value is None else f"{value:.4f}"

lines = [
    "# K1024 Final Decode Eval",
    "",
    f"- checkpoint: `{row['checkpoint_dir']}`",
    f"- eval set: `{summary['eval_set']['selected_json']}`",
    "- reference: GT future geometry",
    "- mode: `prompt_mode=joint`, `target_mode=traj_only`",
    "",
    "| model | ADE@6.4s | FDE@6.4s | minADE6@6.4s | minFDE6 selected by ADE |",
    "|---|---:|---:|---:|---:|",
    f"| {row['model']} | {fmt(row['ade@6.4s_m'])} | {fmt(row['fde@6.4s_m'])} | {fmt(row['minADE6@6.4s_m'])} | {fmt(row['minFDE6_selected_by_ADE@6.4s_m'])} |",
    "",
    f"- combined JSON: `{combined_json}`",
]
combined_md.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(json.dumps({"event": "k1024_decode_summary_done", "json": str(combined_json), "md": str(combined_md), "row": row}))
PY
else
  echo "{\"event\":\"k1024_decode_skip_exists\",\"time\":\"$(date -Is)\",\"summary\":\"${K1024_COMBINED_JSON}\"}"
fi

echo "{\"event\":\"k512_continue_start\",\"time\":\"$(date -Is)\",\"init\":\"${K512_BASE_DIR}\",\"output_dir\":\"${K512_DIR}\",\"lr\":\"${K512_LR}\",\"epochs\":\"${K512_EPOCHS}\"}"

.venv/bin/python -u scripts/09_train_distill.py \
  --corpus-jsonl "${CORPUS}" \
  --stage-config "${CONFIG_K512}" \
  --student-model /home/pm97/workspace/sukim/base_weights/cosmos-reason-2b \
  --init-checkpoint-dir "${K512_BASE_DIR}" \
  --max-train-samples 20000 \
  --max-val-samples 512 \
  --batch-size 16 \
  --epochs "${K512_EPOCHS}" \
  --learning-rate "${K512_LR}" \
  --eval-every-epochs 0.5 \
  --save-every-epochs 1.0 \
  --num-workers "${TRAIN_NUM_WORKERS}" \
  --prefetch-factor "${TRAIN_PREFETCH_FACTOR}" \
  --pin-memory \
  --persistent-workers \
  --log-every-steps "${TRAIN_LOG_EVERY_STEPS}" \
  --output-dir "${K512_DIR}" \
  --summary-json "${K512_SUMMARY}"

echo "{\"event\":\"k512_continue_done\",\"time\":\"$(date -Is)\",\"summary\":\"${K512_SUMMARY}\",\"output_dir\":\"${K512_DIR}\"}"
