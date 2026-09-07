#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/pm97/workspace/sukim/distillation/cosmos_distillation"
PY="$ROOT/.venv/bin/python"
RUN_ID="${RUN_ID:-val512_oracle_cot_prefix_vs_teacher_$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="$ROOT/outputs/benchmarks/$RUN_ID"
CORPUS="$ROOT/data/corpus/val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl"

FULLFT_CKPT="$ROOT/outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/fullft_lr3e5_200k_e1/best_decode"
LORA_CKPT="$ROOT/outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/lprime_lora_lr2e4_200k_e1/best_decode"

mkdir -p "$OUT_ROOT"
export OUT_ROOT

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

run_eval() {
  local tag="$1"
  local ckpt="$2"
  local out_dir="$OUT_ROOT/$tag"
  local summary="$OUT_ROOT/${tag}_summary.json"
  if [[ -s "$summary" ]]; then
    log "{\"event\":\"skip_existing\",\"tag\":\"$tag\",\"summary\":\"$summary\"}"
    return
  fi
  log "{\"event\":\"start\",\"tag\":\"$tag\"}"
  "$PY" -u scripts/25_decode_checkpoint_overlays.py \
    --corpus-jsonl "$CORPUS" \
    --checkpoint-dir "$ckpt" \
    --split val \
    --num-samples 0 \
    --prompt-mode joint \
    --target-mode traj_only \
    --oracle-cot-prefix \
    --image-prompt-style camera_labeled \
    --prompt-text-style official_alpamayo \
    --fuse-history-tokens \
    --geometry-reference teacher \
    --batch-size 4 \
    --samples-per-row 1 \
    --max-new-tokens 132 \
    --seed 42 \
    --device cuda \
    --output-dir "$out_dir" \
    --summary-json "$summary" \
    --skip-overlays
  log "{\"event\":\"done\",\"tag\":\"$tag\",\"summary\":\"$summary\"}"
}

write_combined() {
  "$PY" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
rows = []
for label, name in [
    ("FullFT 3e-5 200K, teacher-CoT-prefix traj-only greedy", "fullft_teacher_cot_prefix_trajonly_greedy_summary.json"),
    ("LoRA 2e-4 200K, teacher-CoT-prefix traj-only greedy", "lora_teacher_cot_prefix_trajonly_greedy_summary.json"),
]:
    path = root / name
    d = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "model": label,
        "n": d["num_samples"],
        "ADE_vs_teacher_discrete_m": d["avg_ade_m"],
        "FDE_vs_teacher_discrete_m": d["avg_fde_m"],
        "token_match_rate": d["avg_token_match_rate"],
        "unique_traj_ids": d["avg_unique_traj_ids"],
        "summary": str(path),
    })
out = root / "combined_oracle_cot_prefix_vs_teacher_summary.json"
out.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"event": "combined_summary_written", "path": str(out), "rows": rows}, indent=2, ensure_ascii=False), flush=True)
PY
}

log "{\"event\":\"run_start\",\"run_id\":\"$RUN_ID\",\"out_root\":\"$OUT_ROOT\"}"
run_eval "fullft_teacher_cot_prefix_trajonly_greedy" "$FULLFT_CKPT"
run_eval "lora_teacher_cot_prefix_trajonly_greedy" "$LORA_CKPT"
write_combined
log "{\"event\":\"run_done\",\"run_id\":\"$RUN_ID\",\"out_root\":\"$OUT_ROOT\"}"
