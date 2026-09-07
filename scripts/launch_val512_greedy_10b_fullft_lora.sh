#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/pm97/workspace/sukim/distillation/cosmos_distillation"
PY="$ROOT/.venv/bin/python"
RUN_ID="${RUN_ID:-val512_greedy_10b_fullft_lora_$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="$ROOT/outputs/benchmarks/$RUN_ID"
CORPUS="$ROOT/data/corpus/val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl"
export OUT_ROOT

FULLFT_CKPT="$ROOT/outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/fullft_lr3e5_200k_e1/best_decode"
LORA_CKPT="$ROOT/outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/lprime_lora_lr2e4_200k_e1/best_decode"

mkdir -p "$OUT_ROOT"

echo "{\"event\":\"run_start\",\"run_id\":\"$RUN_ID\",\"out_root\":\"$OUT_ROOT\",\"corpus\":\"$CORPUS\"}"

"$PY" -u scripts/benchmark_4models.py \
  --corpus-jsonl "$CORPUS" \
  --output-dir "$OUT_ROOT" \
  --model teacher10b \
  --split val \
  --num-samples 0 \
  --batch-size 4 \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --eval-num-paths 1 \
  --eval-temperature 1.0 \
  --eval-selection-method single \
  --teacher-decoding-mode greedy \
  --teacher-top-p 1.0 \
  --teacher-top-k 0 \
  --teacher-max-new-tokens 192 \
  --seed 42

"$PY" -u scripts/25_decode_checkpoint_overlays.py \
  --corpus-jsonl "$CORPUS" \
  --checkpoint-dir "$FULLFT_CKPT" \
  --split val \
  --num-samples 0 \
  --prompt-mode joint \
  --target-mode joint \
  --image-prompt-style camera_labeled \
  --prompt-text-style official_alpamayo \
  --fuse-history-tokens \
  --geometry-reference gt \
  --batch-size 4 \
  --samples-per-row 1 \
  --max-new-tokens 256 \
  --seed 42 \
  --device cuda \
  --output-dir "$OUT_ROOT/fullft_lr3e5_200k_best_greedy" \
  --summary-json "$OUT_ROOT/fullft_lr3e5_200k_best_greedy_summary.json" \
  --skip-overlays

"$PY" -u scripts/25_decode_checkpoint_overlays.py \
  --corpus-jsonl "$CORPUS" \
  --checkpoint-dir "$LORA_CKPT" \
  --split val \
  --num-samples 0 \
  --prompt-mode joint \
  --target-mode joint \
  --image-prompt-style camera_labeled \
  --prompt-text-style official_alpamayo \
  --fuse-history-tokens \
  --geometry-reference gt \
  --batch-size 4 \
  --samples-per-row 1 \
  --max-new-tokens 256 \
  --seed 42 \
  --device cuda \
  --output-dir "$OUT_ROOT/lprime_lora_lr2e4_200k_best_greedy" \
  --summary-json "$OUT_ROOT/lprime_lora_lr2e4_200k_best_greedy_summary.json" \
  --skip-overlays

"$PY" - <<'PY'
import json
from pathlib import Path
import os

root = Path(os.environ["OUT_ROOT"])

def teacher_row():
    path = root / "teacher10b" / "summary.json"
    d = json.loads(path.read_text())
    m = d["metrics"]
    return {
        "model": "Alpamayo-1.5-10B VLM",
        "n": d["count"],
        "ADE@6.4s": m["ade_gt_m"]["mean"],
        "FDE@6.4s": m["fde_gt_m"]["mean"],
        "minADE6@6.4s": m["minade6_gt_m"]["mean"],
        "minFDE6@6.4s": m["minfde6_gt_m"]["mean"],
        "summary": str(path),
    }

def student_row(label, name):
    path = root / name
    d = json.loads(path.read_text())
    return {
        "model": label,
        "n": d["num_samples"],
        "ADE@6.4s": d["ade@6.4s_m"],
        "FDE@6.4s": d["avg_fde_m"],
        "minADE6@6.4s": d["minADE6@6.4s_m"],
        "minFDE6@6.4s": None,
        "summary": str(path),
    }

rows = [
    teacher_row(),
    student_row("FullFT 3e-5 200K best", "fullft_lr3e5_200k_best_greedy_summary.json"),
    student_row("LoRA 2e-4 200K best", "lprime_lora_lr2e4_200k_best_greedy_summary.json"),
]
out = root / "greedy_metrics_summary.json"
out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
print(json.dumps({"event": "combined_greedy_summary_written", "path": str(out), "rows": rows}, indent=2))
PY

echo "{\"event\":\"run_done\",\"run_id\":\"$RUN_ID\",\"out_root\":\"$OUT_ROOT\"}"
