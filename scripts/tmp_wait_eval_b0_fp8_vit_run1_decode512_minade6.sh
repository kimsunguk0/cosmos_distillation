#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN="${RUN:-b0_fp8_vit_step006250_20260618/run1_20k_fp8vit_from_step006250_offpolicy_val512_b8}"
RUN_LABEL="${RUN_LABEL:-b0_fp8_vit_run1_offpolicy_final}"
TRAIN_TMUX="${TRAIN_TMUX:-b0_fp8_vit_run1_b8_20260618}"
CKPT_ROOT="outputs/checkpoints/${RUN}"
CKPT_DIR="${CKPT_ROOT}/final"
TRAIN_SUMMARY="outputs/reports/b0_fp8_vit_step006250_20260618/run1_20k_fp8vit_from_step006250_offpolicy_val512_b8_summary.json"

CORPUS="${CORPUS:-data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl}"
SELECTED_JSON="${SELECTED_JSON:-outputs/reports/b0_fp8_vit_step006250_20260618/val512_seed42_selected_ids.json}"
LOG_PATH="${LOG_PATH:-logs/b0_fp8_vit_step006250_20260618/eval_final_decode512_qat_fp8vit.log}"
BATCH_SIZE="${BATCH_SIZE:-2}"
QAT_CALIB_SAMPLES="${QAT_CALIB_SAMPLES:-128}"
QAT_CALIB_BATCH_SIZE="${QAT_CALIB_BATCH_SIZE:-2}"
SEED="${SEED:-97}"

GREEDY_SUMMARY="outputs/reports/b0_fp8_vit_step006250_20260618/${RUN_LABEL}_val512_trajonly_gt_greedy_qat_fp8vit_summary.json"
N6_SUMMARY="outputs/reports/b0_fp8_vit_step006250_20260618/${RUN_LABEL}_val512_trajonly_gt_n6_qat_fp8vit_summary.json"
COMBINED_JSON="outputs/reports/b0_fp8_vit_step006250_20260618/${RUN_LABEL}_val512_trajonly_gt_minade6_qat_fp8vit_summary.json"
COMBINED_MD="outputs/reports/b0_fp8_vit_step006250_20260618/${RUN_LABEL}_val512_trajonly_gt_minade6_qat_fp8vit_summary.md"

mkdir -p logs/b0_fp8_vit_step006250_20260618 outputs/reports/b0_fp8_vit_step006250_20260618
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"wait_eval_start\",\"time\":\"$(date -Is)\",\"checkpoint\":\"${CKPT_DIR}\",\"train_tmux\":\"${TRAIN_TMUX}\"}"

while [[ ! -f "${CKPT_DIR}/checkpoint_manifest.json" ]]; do
  if ! tmux has-session -t "${TRAIN_TMUX}" 2>/dev/null && [[ ! -f "${TRAIN_SUMMARY}" ]]; then
    echo "{\"event\":\"train_missing_before_final\",\"time\":\"$(date -Is)\",\"checkpoint\":\"${CKPT_DIR}\",\"summary\":\"${TRAIN_SUMMARY}\"}"
    exit 2
  fi
  .venv/bin/python - <<PY
import json
from pathlib import Path
p = Path("${CKPT_ROOT}") / "metrics.jsonl"
last = None
if p.exists():
    for line in p.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("phase") == "train":
            last = row.get("global_step")
print(json.dumps({"event": "waiting_final", "last_train_step": last}))
PY
  sleep 120
done

echo "{\"event\":\"checkpoint_ready\",\"time\":\"$(date -Is)\",\"checkpoint\":\"${CKPT_DIR}\"}"

if [[ ! -f "${SELECTED_JSON}" ]]; then
  .venv/bin/python - <<PY
import json
from pathlib import Path
corpus = Path("${CORPUS}")
out = Path("${SELECTED_JSON}")
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
  PATH="/home/pm97/workspace/sukim/distillation/cosmos_distillation/.venv/bin:${PATH}" \
    .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
      --corpus-jsonl "${CORPUS}" \
      --checkpoint-dir "${CKPT_DIR}" \
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
      --qat-quantization fp8_pcpt_vit \
      --qat-calib-samples "${QAT_CALIB_SAMPLES}" \
      --qat-calib-batch-size "${QAT_CALIB_BATCH_SIZE}" \
      --output-dir "${output_dir}" \
      --summary-json "${summary_json}"
  echo "{\"event\":\"decode_done\",\"time\":\"$(date -Is)\",\"label\":\"${label}\",\"summary\":\"${summary_json}\"}"
}

run_decode \
  "${RUN_LABEL}_greedy_qat_fp8vit" \
  1 \
  "outputs/reports/b0_fp8_vit_step006250_20260618/${RUN_LABEL}_val512_trajonly_gt_greedy_qat_fp8vit" \
  "${GREEDY_SUMMARY}"

run_decode \
  "${RUN_LABEL}_n6_qat_fp8vit" \
  6 \
  "outputs/reports/b0_fp8_vit_step006250_20260618/${RUN_LABEL}_val512_trajonly_gt_n6_qat_fp8vit" \
  "${N6_SUMMARY}"

.venv/bin/python - <<PY
import json
from pathlib import Path

greedy_path = Path("${GREEDY_SUMMARY}")
n6_path = Path("${N6_SUMMARY}")
combined_json = Path("${COMBINED_JSON}")
combined_md = Path("${COMBINED_MD}")
greedy = json.loads(greedy_path.read_text(encoding="utf-8"))
n6 = json.loads(n6_path.read_text(encoding="utf-8"))

def metric(data, *keys):
    for key in keys:
        value = data.get(key)
        if value is not None:
            return float(value)
    return None

row = {
    "model": "${RUN_LABEL}",
    "checkpoint_dir": greedy.get("checkpoint_dir"),
    "num_samples": greedy.get("num_samples"),
    "ade@6.4s_m": metric(greedy, "ade@6.4s_m", "avg_ade_m"),
    "fde@6.4s_m": metric(greedy, "avg_fde_m"),
    "minADE6@6.4s_m": metric(n6, "minADE6@6.4s_m", "avg_ade_m"),
    "minFDE6_selected_by_ADE@6.4s_m": metric(n6, "avg_fde_m"),
    "greedy_summary": str(greedy_path),
    "n6_summary": str(n6_path),
    "qat": n6.get("qat") or greedy.get("qat"),
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
        "seed": int("${SEED}"),
        "qat_quantization": "fp8_pcpt_vit",
        "qat_calib_samples": int("${QAT_CALIB_SAMPLES}"),
    },
    "rows": [row],
}
combined_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def fmt(value):
    return "n/a" if value is None else f"{value:.4f}"

lines = [
    "# B0 FP8 ViT Run1 Final Decode Eval",
    "",
    f"- checkpoint: `{row['checkpoint_dir']}`",
    f"- eval set: `{summary['eval_set']['selected_json']}`",
    "- reference: GT future geometry",
    "- mode: `prompt_mode=joint`, `target_mode=traj_only`",
    "- quantization: `fp8_pcpt_vit`",
    "",
    "| model | ADE@6.4s | FDE@6.4s | minADE6@6.4s | minFDE6 selected by ADE |",
    "|---|---:|---:|---:|---:|",
    f"| {row['model']} | {fmt(row['ade@6.4s_m'])} | {fmt(row['fde@6.4s_m'])} | {fmt(row['minADE6@6.4s_m'])} | {fmt(row['minFDE6_selected_by_ADE@6.4s_m'])} |",
    "",
    f"- combined JSON: `{combined_json}`",
]
combined_md.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(json.dumps({"event": "combined_summary_done", "json": str(combined_json), "md": str(combined_md), "row": row}))
PY

echo "{\"event\":\"wait_eval_done\",\"time\":\"$(date -Is)\",\"combined_json\":\"${COMBINED_JSON}\",\"combined_md\":\"${COMBINED_MD}\"}"
