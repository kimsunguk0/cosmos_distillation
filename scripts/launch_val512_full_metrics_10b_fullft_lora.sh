#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/pm97/workspace/sukim/distillation/cosmos_distillation"
PY="$ROOT/.venv/bin/python"
BASE_RUN_ID="${BASE_RUN_ID:-val512_greedy_10b_fullft_lora_20260715_0148}"
BASE_OUT="$ROOT/outputs/benchmarks/$BASE_RUN_ID"
RUN_ID="${RUN_ID:-val512_full_metrics_10b_fullft_lora_$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="$ROOT/outputs/benchmarks/$RUN_ID"
CORPUS="$ROOT/data/corpus/val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl"

FULLFT_CKPT="$ROOT/outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/fullft_lr3e5_200k_e1/best_decode"
LORA_CKPT="$ROOT/outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/lprime_lora_lr2e4_200k_e1/best_decode"

mkdir -p "$OUT_ROOT"
export BASE_OUT OUT_ROOT CORPUS

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

wait_for_file() {
  local path="$1"
  log "{\"event\":\"wait_for_file\",\"path\":\"$path\"}"
  while [[ ! -s "$path" ]]; do
    sleep 60
  done
  log "{\"event\":\"file_ready\",\"path\":\"$path\"}"
}

run_10b_vlm_discrete() {
  local tag="$1"
  local samples="$2"
  local out_dir="$OUT_ROOT/$tag"
  local summary="$OUT_ROOT/${tag}_summary.json"
  if [[ -s "$summary" ]]; then
    log "{\"event\":\"skip_existing\",\"tag\":\"$tag\",\"summary\":\"$summary\"}"
    return
  fi
  log "{\"event\":\"start\",\"tag\":\"$tag\",\"samples_per_row\":$samples}"
  "$PY" -u scripts/eval_10b_backbone_discrete.py \
    --corpus-jsonl "$CORPUS" \
    --split val \
    --num-samples 0 \
    --samples-per-row "$samples" \
    --temperature 0.6 \
    --top-p 0.98 \
    --top-k 0 \
    --seed 42 \
    --max-new-tokens 256 \
    --device cuda:0 \
    --dtype bfloat16 \
    --output-dir "$out_dir" \
    --summary-json "$summary"
  log "{\"event\":\"done\",\"tag\":\"$tag\",\"summary\":\"$summary\"}"
}

run_student_n6() {
  local tag="$1"
  local ckpt="$2"
  local out_dir="$OUT_ROOT/$tag"
  local summary="$OUT_ROOT/${tag}_summary.json"
  if [[ -s "$summary" ]]; then
    log "{\"event\":\"skip_existing\",\"tag\":\"$tag\",\"summary\":\"$summary\"}"
    return
  fi
  log "{\"event\":\"start\",\"tag\":\"$tag\",\"samples_per_row\":6}"
  "$PY" -u scripts/25_decode_checkpoint_overlays.py \
    --corpus-jsonl "$CORPUS" \
    --checkpoint-dir "$ckpt" \
    --split val \
    --num-samples 0 \
    --prompt-mode joint \
    --target-mode joint \
    --image-prompt-style camera_labeled \
    --prompt-text-style official_alpamayo \
    --fuse-history-tokens \
    --geometry-reference gt \
    --batch-size 4 \
    --samples-per-row 6 \
    --temperature 1.0 \
    --top-p 1.0 \
    --max-new-tokens 256 \
    --seed 42 \
    --device cuda \
    --output-dir "$out_dir" \
    --summary-json "$summary" \
    --skip-overlays
  log "{\"event\":\"done\",\"tag\":\"$tag\",\"summary\":\"$summary\"}"
}

run_10b_ae_n6() {
  local tag="teacher10b_ae_n6"
  local summary="$OUT_ROOT/$tag/summary.json"
  if [[ -s "$summary" ]]; then
    log "{\"event\":\"skip_existing\",\"tag\":\"$tag\",\"summary\":\"$summary\"}"
    return
  fi
  log "{\"event\":\"start\",\"tag\":\"$tag\",\"eval_num_paths\":6}"
  "$PY" -u scripts/benchmark_4models.py \
    --corpus-jsonl "$CORPUS" \
    --output-dir "$OUT_ROOT/$tag" \
    --model teacher10b \
    --split val \
    --num-samples 0 \
    --batch-size 4 \
    --device cuda:0 \
    --dtype bfloat16 \
    --attn-implementation sdpa \
    --eval-num-paths 6 \
    --eval-temperature 0.6 \
    --eval-selection-method single \
    --teacher-decoding-mode sampling \
    --teacher-top-p 0.98 \
    --teacher-top-k 0 \
    --teacher-max-new-tokens 192 \
    --seed 42
  log "{\"event\":\"done\",\"tag\":\"$tag\",\"summary\":\"$summary\"}"
}

write_combined() {
  "$PY" - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["BASE_OUT"])
root = Path(os.environ["OUT_ROOT"])

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def metric_mean(summary, key):
    return ((summary.get("metrics") or {}).get(key) or {}).get("mean")

teacher_ae_greedy = load(base / "teacher10b" / "summary.json")
teacher_ae_n6 = load(root / "teacher10b_ae_n6" / "teacher10b" / "summary.json")
teacher_vlm_n1 = load(root / "teacher10b_vlm_discrete_n1_summary.json")
teacher_vlm_n6 = load(root / "teacher10b_vlm_discrete_n6_summary.json")
fullft_greedy = load(base / "fullft_lr3e5_200k_best_greedy_summary.json")
lora_greedy = load(base / "lprime_lora_lr2e4_200k_best_greedy_summary.json")
fullft_n6 = load(root / "fullft_lr3e5_200k_best_n6_summary.json")
lora_n6 = load(root / "lprime_lora_lr2e4_200k_best_n6_summary.json")

rows = [
    {
        "model": "Alpamayo-1.5-10B VLM discrete only",
        "n": teacher_vlm_n1["num_samples"],
        "ADE@6.4s_m": teacher_vlm_n1["ade@6.4s_m"],
        "FDE@6.4s_m": teacher_vlm_n1["avg_fde_m"],
        "minADE6@6.4s_m": teacher_vlm_n6["minADE6@6.4s_m"],
        "minFDE6@6.4s_m": metric_mean(teacher_vlm_n6, "minfde6_gt_m"),
        "greedy_summary": str(root / "teacher10b_vlm_discrete_n1_summary.json"),
        "n6_summary": str(root / "teacher10b_vlm_discrete_n6_summary.json"),
    },
    {
        "model": "Alpamayo-1.5-10B + Action Expert",
        "n": teacher_ae_greedy["count"],
        "ADE@6.4s_m": metric_mean(teacher_ae_greedy, "ade_gt_m"),
        "FDE@6.4s_m": metric_mean(teacher_ae_greedy, "fde_gt_m"),
        "minADE6@6.4s_m": metric_mean(teacher_ae_n6, "minade6_gt_m"),
        "minFDE6@6.4s_m": metric_mean(teacher_ae_n6, "minfde6_gt_m"),
        "greedy_summary": str(base / "teacher10b" / "summary.json"),
        "n6_summary": str(root / "teacher10b_ae_n6" / "teacher10b" / "summary.json"),
    },
    {
        "model": "FullFT 3e-5 200K best",
        "n": fullft_greedy["num_samples"],
        "ADE@6.4s_m": fullft_greedy["ade@6.4s_m"],
        "FDE@6.4s_m": fullft_greedy["avg_fde_m"],
        "minADE6@6.4s_m": fullft_n6["minADE6@6.4s_m"],
        "FDE_of_minADE6_candidate_m": fullft_n6.get("avg_fde_m"),
        "greedy_summary": str(base / "fullft_lr3e5_200k_best_greedy_summary.json"),
        "n6_summary": str(root / "fullft_lr3e5_200k_best_n6_summary.json"),
    },
    {
        "model": "LoRA 2e-4 200K best",
        "n": lora_greedy["num_samples"],
        "ADE@6.4s_m": lora_greedy["ade@6.4s_m"],
        "FDE@6.4s_m": lora_greedy["avg_fde_m"],
        "minADE6@6.4s_m": lora_n6["minADE6@6.4s_m"],
        "FDE_of_minADE6_candidate_m": lora_n6.get("avg_fde_m"),
        "greedy_summary": str(base / "lprime_lora_lr2e4_200k_best_greedy_summary.json"),
        "n6_summary": str(root / "lprime_lora_lr2e4_200k_best_n6_summary.json"),
    },
]

out = root / "combined_val512_ade_minade6_summary.json"
out.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"event": "combined_summary_written", "path": str(out), "rows": rows}, indent=2, ensure_ascii=False), flush=True)
PY
}

log "{\"event\":\"run_start\",\"run_id\":\"$RUN_ID\",\"out_root\":\"$OUT_ROOT\",\"base_out\":\"$BASE_OUT\"}"
wait_for_file "$BASE_OUT/lprime_lora_lr2e4_200k_best_greedy_summary.json"

run_10b_vlm_discrete "teacher10b_vlm_discrete_n1" 1
run_10b_vlm_discrete "teacher10b_vlm_discrete_n6" 6
run_10b_ae_n6
run_student_n6 "fullft_lr3e5_200k_best_n6" "$FULLFT_CKPT"
run_student_n6 "lprime_lora_lr2e4_200k_best_n6" "$LORA_CKPT"
write_combined

log "{\"event\":\"run_done\",\"run_id\":\"$RUN_ID\",\"out_root\":\"$OUT_ROOT\"}"
