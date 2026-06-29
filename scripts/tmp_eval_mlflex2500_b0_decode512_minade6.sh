#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN_TAG="${RUN_TAG:-20260608}"
CORPUS="data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl"
SELECTED_JSON="outputs/reports/mlflex_k512_bp3_eval512_seed42_selected_ids.json"
FLEX_CKPT="outputs/checkpoints/mlflex_k512_bp3_20k_e3_b16_20260608/step_002500"
B0_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
LOG_PATH="outputs/logs/decode_mlflex2500_b0_val512_minade6_${RUN_TAG}.log"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEED="${SEED:-97}"

FLEX_GREEDY_SUMMARY="outputs/reports/mlflex_k512_bp3_step2500_val512_trajonly_gt_greedy_summary.json"
FLEX_N6_SUMMARY="outputs/reports/mlflex_k512_bp3_step2500_val512_trajonly_gt_n6_summary.json"
B0_GREEDY_SUMMARY="outputs/reports/b0_step006250_val512_trajonly_gt_greedy_summary.json"
B0_N6_SUMMARY="outputs/reports/b0_step006250_val512_trajonly_gt_n6_summary.json"
COMBINED_JSON="outputs/reports/mlflex_k512_bp3_step2500_vs_b0_val512_trajonly_gt_minade6_summary.json"
COMBINED_MD="outputs/reports/mlflex_k512_bp3_step2500_vs_b0_val512_trajonly_gt_minade6_summary.md"

mkdir -p outputs/logs outputs/reports
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"decode_eval_launch\",\"time\":\"$(date -Is)\",\"corpus\":\"${CORPUS}\",\"selected_json\":\"${SELECTED_JSON}\",\"batch_size\":${BATCH_SIZE},\"seed\":${SEED}}"

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

run_decode() {
  local label="$1"
  local checkpoint="$2"
  local samples_per_row="$3"
  local output_dir="$4"
  local summary_json="$5"
  shift 5

  echo "{\"event\":\"decode_start\",\"time\":\"$(date -Is)\",\"label\":\"${label}\",\"checkpoint\":\"${checkpoint}\",\"samples_per_row\":${samples_per_row},\"summary\":\"${summary_json}\"}"
  .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
    --corpus-jsonl "${CORPUS}" \
    --checkpoint-dir "${checkpoint}" \
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
    --skip-overlays \
    --disable-failure-tags \
    "$@" \
    --output-dir "${output_dir}" \
    --summary-json "${summary_json}"
  echo "{\"event\":\"decode_done\",\"time\":\"$(date -Is)\",\"label\":\"${label}\",\"summary\":\"${summary_json}\"}"
}

run_decode \
  "flex_step2500_greedy" \
  "${FLEX_CKPT}" \
  1 \
  "outputs/reports/mlflex_k512_bp3_step2500_val512_trajonly_gt_greedy" \
  "${FLEX_GREEDY_SUMMARY}" \
  --preserve-flex-positions \
  --flex-selection-strategy uniform \
  --flex-scene-deepstack

run_decode \
  "flex_step2500_n6" \
  "${FLEX_CKPT}" \
  6 \
  "outputs/reports/mlflex_k512_bp3_step2500_val512_trajonly_gt_n6" \
  "${FLEX_N6_SUMMARY}" \
  --preserve-flex-positions \
  --flex-selection-strategy uniform \
  --flex-scene-deepstack

run_decode \
  "b0_step006250_greedy" \
  "${B0_CKPT}" \
  1 \
  "outputs/reports/b0_step006250_val512_trajonly_gt_greedy" \
  "${B0_GREEDY_SUMMARY}"

run_decode \
  "b0_step006250_n6" \
  "${B0_CKPT}" \
  6 \
  "outputs/reports/b0_step006250_val512_trajonly_gt_n6" \
  "${B0_N6_SUMMARY}"

.venv/bin/python - <<'PY'
import json
from pathlib import Path

paths = {
    "flex_step2500_greedy": Path("outputs/reports/mlflex_k512_bp3_step2500_val512_trajonly_gt_greedy_summary.json"),
    "flex_step2500_n6": Path("outputs/reports/mlflex_k512_bp3_step2500_val512_trajonly_gt_n6_summary.json"),
    "b0_step006250_greedy": Path("outputs/reports/b0_step006250_val512_trajonly_gt_greedy_summary.json"),
    "b0_step006250_n6": Path("outputs/reports/b0_step006250_val512_trajonly_gt_n6_summary.json"),
}

loaded = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}

def metric(data, *keys):
    for key in keys:
        value = data.get(key)
        if value is not None:
            return float(value)
    return None

rows = []
for label, prefix in (("FLEX step_002500", "flex_step2500"), ("B0 step_006250", "b0_step006250")):
    greedy = loaded[f"{prefix}_greedy"]
    n6 = loaded[f"{prefix}_n6"]
    rows.append(
        {
            "model": label,
            "checkpoint_dir": greedy.get("checkpoint_dir"),
            "num_samples": greedy.get("num_samples"),
            "ade@6.4s_m": metric(greedy, "ade@6.4s_m", "avg_ade_m"),
            "fde@6.4s_m": metric(greedy, "avg_fde_m"),
            "minADE6@6.4s_m": metric(n6, "minADE6@6.4s_m", "avg_ade_m"),
            "minFDE6_selected_by_ADE@6.4s_m": metric(n6, "avg_fde_m"),
            "greedy_summary": str(paths[f"{prefix}_greedy"]),
            "n6_summary": str(paths[f"{prefix}_n6"]),
        }
    )

flex, b0 = rows[0], rows[1]
deltas = {
    "flex_minus_b0_ade@6.4s_m": None if flex["ade@6.4s_m"] is None or b0["ade@6.4s_m"] is None else flex["ade@6.4s_m"] - b0["ade@6.4s_m"],
    "flex_minus_b0_minADE6@6.4s_m": None if flex["minADE6@6.4s_m"] is None or b0["minADE6@6.4s_m"] is None else flex["minADE6@6.4s_m"] - b0["minADE6@6.4s_m"],
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
    "rows": rows,
    "deltas": deltas,
}
json_path = Path("outputs/reports/mlflex_k512_bp3_step2500_vs_b0_val512_trajonly_gt_minade6_summary.json")
md_path = Path("outputs/reports/mlflex_k512_bp3_step2500_vs_b0_val512_trajonly_gt_minade6_summary.md")
json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def fmt(value):
    return "n/a" if value is None else f"{value:.4f}"

lines = [
    "# MLFLEX step_002500 vs B0 decode eval",
    "",
    "- eval set: first 512 val ids from `no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl`",
    "- reference: GT future geometry",
    "- mode: `prompt_mode=joint`, `target_mode=traj_only`",
    "- ADE: greedy 1 trajectory, 6.4s horizon",
    "- minADE6@6.4s: best ADE among 6 sampled trajectories at temperature 1.0, top_p 1.0",
    "",
    "| model | ADE@6.4s | FDE@6.4s | minADE6@6.4s | minFDE6 selected by ADE |",
    "|---|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['model']} | {fmt(row['ade@6.4s_m'])} | {fmt(row['fde@6.4s_m'])} | "
        f"{fmt(row['minADE6@6.4s_m'])} | {fmt(row['minFDE6_selected_by_ADE@6.4s_m'])} |"
    )
lines.extend(
    [
        "",
        f"- FLEX - B0 ADE@6.4s: {fmt(deltas['flex_minus_b0_ade@6.4s_m'])}",
        f"- FLEX - B0 minADE6@6.4s: {fmt(deltas['flex_minus_b0_minADE6@6.4s_m'])}",
        "",
        f"- combined JSON: `{json_path}`",
    ]
)
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"event": "combined_summary_done", "json": str(json_path), "md": str(md_path), "deltas": deltas}))
PY

echo "{\"event\":\"decode_eval_done\",\"time\":\"$(date -Is)\",\"combined_json\":\"${COMBINED_JSON}\",\"combined_md\":\"${COMBINED_MD}\"}"
