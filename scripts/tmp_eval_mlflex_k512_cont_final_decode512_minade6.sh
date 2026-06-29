#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN="mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024"
CORPUS="data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl"
SELECTED_JSON="outputs/reports/mlflex_k512_bp3_eval512_seed42_selected_ids.json"
FLEX_CKPT="outputs/checkpoints/${RUN}/final"
LOG_PATH="outputs/logs/decode_${RUN}_val512_minade6.log"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEED="${SEED:-97}"

GREEDY_SUMMARY="outputs/reports/${RUN}_val512_trajonly_gt_greedy_summary.json"
N6_SUMMARY="outputs/reports/${RUN}_val512_trajonly_gt_n6_summary.json"
COMBINED_JSON="outputs/reports/${RUN}_val512_trajonly_gt_minade6_summary.json"
COMBINED_MD="outputs/reports/${RUN}_val512_trajonly_gt_minade6_summary.md"

mkdir -p outputs/logs outputs/reports
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"decode_eval_launch\",\"time\":\"$(date -Is)\",\"checkpoint\":\"${FLEX_CKPT}\",\"corpus\":\"${CORPUS}\",\"selected_json\":\"${SELECTED_JSON}\",\"batch_size\":${BATCH_SIZE},\"seed\":${SEED}}"

if [[ ! -f "${SELECTED_JSON}" ]]; then
  .venv/bin/python - <<'PY'
import json
from pathlib import Path

corpus = Path("data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl")
out = Path("outputs/reports/mlflex_k512_bp3_eval512_seed42_selected_ids.json")
ids = []
val_rows = 0
with corpus.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "val":
            continue
        val_rows += 1
        if len(ids) < 512:
            ids.append(str(row.get("sample_id")))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(ids, indent=2), encoding="utf-8")
print(json.dumps({"event": "selected_ids_ready", "val_rows": val_rows, "selected": len(ids), "path": str(out)}))
PY
fi

run_decode() {
  local label="$1"
  local samples_per_row="$2"
  local output_dir="$3"
  local summary_json="$4"

  echo "{\"event\":\"decode_start\",\"time\":\"$(date -Is)\",\"label\":\"${label}\",\"samples_per_row\":${samples_per_row},\"summary\":\"${summary_json}\"}"
  .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
    --corpus-jsonl "${CORPUS}" \
    --checkpoint-dir "${FLEX_CKPT}" \
    --split val \
    --selected-json "${SELECTED_JSON}" \
    --prompt-mode joint \
    --target-mode traj_only \
    --image-prompt-style camera_labeled \
    --prompt-text-style official_alpamayo \
    --fuse-history-tokens \
    --geometry-reference gt \
    --max-new-tokens 160 \
    --batch-size "${BATCH_SIZE}" \
    --samples-per-row "${samples_per_row}" \
    --temperature 1.0 \
    --top-p 1.0 \
    --seed "${SEED}" \
    --preserve-flex-positions \
    --flex-selection-strategy uniform \
    --flex-scene-deepstack \
    --skip-overlays \
    --disable-failure-tags \
    --output-dir "${output_dir}" \
    --summary-json "${summary_json}"
  echo "{\"event\":\"decode_done\",\"time\":\"$(date -Is)\",\"label\":\"${label}\",\"summary\":\"${summary_json}\"}"
}

run_decode \
  "${RUN}_greedy" \
  1 \
  "outputs/reports/${RUN}_val512_trajonly_gt_greedy" \
  "${GREEDY_SUMMARY}"

run_decode \
  "${RUN}_n6" \
  6 \
  "outputs/reports/${RUN}_val512_trajonly_gt_n6" \
  "${N6_SUMMARY}"

.venv/bin/python - <<'PY'
import json
from pathlib import Path

run = "mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024"
greedy_path = Path(f"outputs/reports/{run}_val512_trajonly_gt_greedy_summary.json")
n6_path = Path(f"outputs/reports/{run}_val512_trajonly_gt_n6_summary.json")
combined_json = Path(f"outputs/reports/{run}_val512_trajonly_gt_minade6_summary.json")
combined_md = Path(f"outputs/reports/{run}_val512_trajonly_gt_minade6_summary.md")
greedy = json.loads(greedy_path.read_text(encoding="utf-8"))
n6 = json.loads(n6_path.read_text(encoding="utf-8"))

def metric(data, *keys):
    for key in keys:
        value = data.get(key)
        if value is not None:
            return float(value)
    return None

row = {
    "model": "K512 continuation final",
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
        "corpus_jsonl": "data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl",
        "selected_json": "outputs/reports/mlflex_k512_bp3_eval512_seed42_selected_ids.json",
        "split": "val",
        "num_samples": 512,
        "geometry_reference": "gt",
        "prompt_mode": "joint",
        "target_mode": "traj_only",
        "max_new_tokens": 160,
        "seed": 97,
    },
    "rows": [row],
}
combined_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def fmt(value):
    return "n/a" if value is None else f"{value:.4f}"

lines = [
    "# K512 Continuation Final Decode Eval",
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
combined_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"event": "combined_summary_done", "json": str(combined_json), "md": str(combined_md), "row": row}))
PY

echo "{\"event\":\"decode_eval_done\",\"time\":\"$(date -Is)\",\"combined_json\":\"${COMBINED_JSON}\",\"combined_md\":\"${COMBINED_MD}\"}"
