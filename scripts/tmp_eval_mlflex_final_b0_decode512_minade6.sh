#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN_TAG="${RUN_TAG:-20260608_final}"
CORPUS="data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl"
SELECTED_JSON="outputs/reports/mlflex_k512_bp3_eval512_seed42_selected_ids.json"
FLEX_CKPT="outputs/checkpoints/mlflex_k512_bp3_20k_e3_b16_20260608/final"
B0_GREEDY_SUMMARY="outputs/reports/b0_step006250_val512_trajonly_gt_greedy_summary.json"
B0_N6_SUMMARY="outputs/reports/b0_step006250_val512_trajonly_gt_n6_summary.json"
LOG_PATH="outputs/logs/decode_mlflex_final_b0_val512_minade6_${RUN_TAG}.log"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEED="${SEED:-97}"

FLEX_GREEDY_SUMMARY="outputs/reports/mlflex_k512_bp3_final_val512_trajonly_gt_greedy_summary.json"
FLEX_N6_SUMMARY="outputs/reports/mlflex_k512_bp3_final_val512_trajonly_gt_n6_summary.json"
COMBINED_JSON="outputs/reports/mlflex_k512_bp3_final_vs_b0_val512_trajonly_gt_minade6_summary.json"
COMBINED_MD="outputs/reports/mlflex_k512_bp3_final_vs_b0_val512_trajonly_gt_minade6_summary.md"

mkdir -p outputs/logs outputs/reports
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"decode_eval_launch\",\"time\":\"$(date -Is)\",\"checkpoint\":\"${FLEX_CKPT}\",\"corpus\":\"${CORPUS}\",\"selected_json\":\"${SELECTED_JSON}\",\"batch_size\":${BATCH_SIZE},\"seed\":${SEED}}"

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
  "flex_final_greedy" \
  1 \
  "outputs/reports/mlflex_k512_bp3_final_val512_trajonly_gt_greedy" \
  "${FLEX_GREEDY_SUMMARY}"

run_decode \
  "flex_final_n6" \
  6 \
  "outputs/reports/mlflex_k512_bp3_final_val512_trajonly_gt_n6" \
  "${FLEX_N6_SUMMARY}"

.venv/bin/python - <<'PY'
import json
from pathlib import Path

paths = {
    "flex_final_greedy": Path("outputs/reports/mlflex_k512_bp3_final_val512_trajonly_gt_greedy_summary.json"),
    "flex_final_n6": Path("outputs/reports/mlflex_k512_bp3_final_val512_trajonly_gt_n6_summary.json"),
    "b0_step006250_greedy": Path("outputs/reports/b0_step006250_val512_trajonly_gt_greedy_summary.json"),
    "b0_step006250_n6": Path("outputs/reports/b0_step006250_val512_trajonly_gt_n6_summary.json"),
    "flex_step2500_greedy": Path("outputs/reports/mlflex_k512_bp3_step2500_val512_trajonly_gt_greedy_summary.json"),
    "flex_step2500_n6": Path("outputs/reports/mlflex_k512_bp3_step2500_val512_trajonly_gt_n6_summary.json"),
}
loaded = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items() if path.exists()}

def metric(data, *keys):
    for key in keys:
        value = data.get(key)
        if value is not None:
            return float(value)
    return None

def row(label, greedy_key, n6_key):
    greedy = loaded[greedy_key]
    n6 = loaded[n6_key]
    return {
        "model": label,
        "checkpoint_dir": greedy.get("checkpoint_dir"),
        "num_samples": greedy.get("num_samples"),
        "ade@6.4s_m": metric(greedy, "ade@6.4s_m", "avg_ade_m"),
        "fde@6.4s_m": metric(greedy, "avg_fde_m"),
        "minADE6@6.4s_m": metric(n6, "minADE6@6.4s_m", "avg_ade_m"),
        "minFDE6_selected_by_ADE@6.4s_m": metric(n6, "avg_fde_m"),
        "greedy_summary": str(paths[greedy_key]),
        "n6_summary": str(paths[n6_key]),
    }

rows = [
    row("FLEX final", "flex_final_greedy", "flex_final_n6"),
    row("FLEX step_002500", "flex_step2500_greedy", "flex_step2500_n6"),
    row("B0 step_006250", "b0_step006250_greedy", "b0_step006250_n6"),
]
by_model = {item["model"]: item for item in rows}
final = by_model["FLEX final"]
step2500 = by_model["FLEX step_002500"]
b0 = by_model["B0 step_006250"]

def delta(left, right, key):
    if left.get(key) is None or right.get(key) is None:
        return None
    return left[key] - right[key]

deltas = {
    "final_minus_b0_ade@6.4s_m": delta(final, b0, "ade@6.4s_m"),
    "final_minus_b0_minADE6@6.4s_m": delta(final, b0, "minADE6@6.4s_m"),
    "final_minus_step2500_ade@6.4s_m": delta(final, step2500, "ade@6.4s_m"),
    "final_minus_step2500_minADE6@6.4s_m": delta(final, step2500, "minADE6@6.4s_m"),
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
json_path = Path("outputs/reports/mlflex_k512_bp3_final_vs_b0_val512_trajonly_gt_minade6_summary.json")
md_path = Path("outputs/reports/mlflex_k512_bp3_final_vs_b0_val512_trajonly_gt_minade6_summary.md")
json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def fmt(value):
    return "n/a" if value is None else f"{value:.4f}"

lines = [
    "# MLFLEX final vs B0 decode eval",
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
for item in rows:
    lines.append(
        f"| {item['model']} | {fmt(item['ade@6.4s_m'])} | {fmt(item['fde@6.4s_m'])} | "
        f"{fmt(item['minADE6@6.4s_m'])} | {fmt(item['minFDE6_selected_by_ADE@6.4s_m'])} |"
    )
lines.extend(
    [
        "",
        f"- final - B0 ADE@6.4s: {fmt(deltas['final_minus_b0_ade@6.4s_m'])}",
        f"- final - B0 minADE6@6.4s: {fmt(deltas['final_minus_b0_minADE6@6.4s_m'])}",
        f"- final - step2500 ADE@6.4s: {fmt(deltas['final_minus_step2500_ade@6.4s_m'])}",
        f"- final - step2500 minADE6@6.4s: {fmt(deltas['final_minus_step2500_minADE6@6.4s_m'])}",
        "",
        f"- combined JSON: `{json_path}`",
    ]
)
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"event": "combined_summary_done", "json": str(json_path), "md": str(md_path), "deltas": deltas}))
PY

echo "{\"event\":\"decode_eval_done\",\"time\":\"$(date -Is)\",\"combined_json\":\"${COMBINED_JSON}\",\"combined_md\":\"${COMBINED_MD}\"}"
