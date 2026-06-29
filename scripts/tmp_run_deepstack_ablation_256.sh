#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

OUT_ROOT="outputs/reports/deepstack_ablation_256"
mkdir -p "${OUT_ROOT}"

CORPUS="data/corpus/flex_heldout256_stage2val_seed42.jsonl"
B0_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
B0_MODEL="/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b"
AE_CKPT="outputs/action_expert/q3_cosine_cooldown_from_q2best_s26000_2k_20260603_0505/best.pt"
TEACHER="/home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B"

echo "{\"event\":\"deepstack_ablation_256_start\",\"ts\":\"$(date -Is)\"}" | tee "${OUT_ROOT}/run_status.jsonl"

echo "{\"event\":\"public10b_start\",\"ts\":\"$(date -Is)\"}" | tee -a "${OUT_ROOT}/run_status.jsonl"
.venv/bin/python scripts/110_eval_public10b_deepstack_ablation.py \
  --corpus-jsonl "${CORPUS}" \
  --checkpoint-path "${TEACHER}" \
  --split val \
  --num-samples 256 \
  --batch-size 2 \
  --modes on,off \
  --output-dir outputs/reports/public10b_deepstack_ablation_256 \
  --attn-implementation sdpa \
  --device cuda:0 \
  > "${OUT_ROOT}/public10b_onoff.log" 2>&1
echo "{\"event\":\"public10b_done\",\"ts\":\"$(date -Is)\"}" | tee -a "${OUT_ROOT}/run_status.jsonl"

echo "{\"event\":\"b0_discrete_on_start\",\"ts\":\"$(date -Is)\"}" | tee -a "${OUT_ROOT}/run_status.jsonl"
.venv/bin/python scripts/25_decode_checkpoint_overlays.py \
  --corpus-jsonl "${CORPUS}" \
  --checkpoint-dir "${B0_CKPT}" \
  --student-model "${B0_MODEL}" \
  --split val \
  --num-samples 256 \
  --prompt-mode joint \
  --target-mode traj_only \
  --image-prompt-style camera_labeled \
  --prompt-text-style official_alpamayo \
  --fuse-history-tokens \
  --geometry-reference teacher \
  --max-new-tokens 129 \
  --batch-size 2 \
  --skip-overlays \
  --output-dir "${OUT_ROOT}/b0_2b_discrete_dson_decode_trajonly" \
  --summary-json "${OUT_ROOT}/b0_2b_discrete_dson_summary.json" \
  > "${OUT_ROOT}/b0_2b_discrete_dson.log" 2>&1
echo "{\"event\":\"b0_discrete_on_done\",\"ts\":\"$(date -Is)\"}" | tee -a "${OUT_ROOT}/run_status.jsonl"

echo "{\"event\":\"b0_discrete_off_start\",\"ts\":\"$(date -Is)\"}" | tee -a "${OUT_ROOT}/run_status.jsonl"
.venv/bin/python scripts/25_decode_checkpoint_overlays.py \
  --corpus-jsonl "${CORPUS}" \
  --checkpoint-dir "${B0_CKPT}" \
  --student-model "${B0_MODEL}" \
  --split val \
  --num-samples 256 \
  --prompt-mode joint \
  --target-mode traj_only \
  --image-prompt-style camera_labeled \
  --prompt-text-style official_alpamayo \
  --fuse-history-tokens \
  --geometry-reference teacher \
  --max-new-tokens 129 \
  --batch-size 2 \
  --disable-qwen-deepstack \
  --skip-overlays \
  --output-dir "${OUT_ROOT}/b0_2b_discrete_dsoff_decode_trajonly" \
  --summary-json "${OUT_ROOT}/b0_2b_discrete_dsoff_summary.json" \
  > "${OUT_ROOT}/b0_2b_discrete_dsoff.log" 2>&1
echo "{\"event\":\"b0_discrete_off_done\",\"ts\":\"$(date -Is)\"}" | tee -a "${OUT_ROOT}/run_status.jsonl"

COMMON_AE_ARGS=(
  --ckpt-path "${AE_CKPT}"
  --corpus-jsonl "${CORPUS}"
  --split val
  --num-samples 256
  --eval-samples 256
  --eval-batch-size 2
  --eval-num-paths 6
  --eval-temperature 1.0
  --eval-selection-method mean_traj
  --eval-vectorize-paths
  --eval-path-batch-size 6
  --eval-log-rows 0
  --eval-seed-mode fixed
  --student-checkpoint-dir "${B0_CKPT}"
  --student-model "${B0_MODEL}"
  --teacher-checkpoint-path "${TEACHER}"
  --teacher-load-device cpu
  --device cuda:0
  --student-dtype bfloat16
  --ae-dtype bfloat16
  --attn-implementation sdpa
  --prefix-mode student_free
  --ae-init-mode student_backbone_init
  --target-source teacher
  --max-new-tokens 192
  --stage2-attention-mode official_none
)

echo "{\"event\":\"b0_ae_on_start\",\"ts\":\"$(date -Is)\"}" | tee -a "${OUT_ROOT}/run_status.jsonl"
.venv/bin/python scripts/85_eval_ae28_best_of_n.py "${COMMON_AE_ARGS[@]}" \
  > "${OUT_ROOT}/b0_2b_ae_dson.log" 2>&1
echo "{\"event\":\"b0_ae_on_done\",\"ts\":\"$(date -Is)\"}" | tee -a "${OUT_ROOT}/run_status.jsonl"

echo "{\"event\":\"b0_ae_off_start\",\"ts\":\"$(date -Is)\"}" | tee -a "${OUT_ROOT}/run_status.jsonl"
.venv/bin/python scripts/85_eval_ae28_best_of_n.py "${COMMON_AE_ARGS[@]}" \
  --disable-student-deepstack \
  > "${OUT_ROOT}/b0_2b_ae_dsoff.log" 2>&1
echo "{\"event\":\"b0_ae_off_done\",\"ts\":\"$(date -Is)\"}" | tee -a "${OUT_ROOT}/run_status.jsonl"

echo "{\"event\":\"aggregate_start\",\"ts\":\"$(date -Is)\"}" | tee -a "${OUT_ROOT}/run_status.jsonl"
.venv/bin/python - <<'PY'
import json
from pathlib import Path

out = Path("outputs/reports/deepstack_ablation_256")

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def last_eval(log_path):
    last = None
    for line in Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "eval":
            last = payload
    if last is None:
        raise RuntimeError(f"No eval JSON found in {log_path}")
    return last

public = load("outputs/reports/public10b_deepstack_ablation_256/summary.json")
b0_on = load(out / "b0_2b_discrete_dson_summary.json")
b0_off = load(out / "b0_2b_discrete_dsoff_summary.json")
ae_on = last_eval(out / "b0_2b_ae_dson.log")
ae_off = last_eval(out / "b0_2b_ae_dsoff.log")

summary = {
    "status": "ok",
    "corpus_jsonl": "data/corpus/flex_heldout256_stage2val_seed42.jsonl",
    "num_samples": 256,
    "public10b": public,
    "b0_2b_discrete": {
        "deepstack_on": b0_on,
        "deepstack_off": b0_off,
        "delta_off_minus_on": {
            "avg_ade_m": b0_off["avg_ade_m"] - b0_on["avg_ade_m"],
            "avg_fde_m": b0_off["avg_fde_m"] - b0_on["avg_fde_m"],
            "avg_unique_traj_ids": b0_off["avg_unique_traj_ids"] - b0_on["avg_unique_traj_ids"],
            "avg_token_match_rate": b0_off["avg_token_match_rate"] - b0_on["avg_token_match_rate"],
        },
    },
    "b0_2b_action_expert": {
        "ckpt_path": "outputs/action_expert/q3_cosine_cooldown_from_q2best_s26000_2k_20260603_0505/best.pt",
        "eval_num_paths": 6,
        "eval_selection_method": "mean_traj",
        "target_source": "teacher",
        "deepstack_on": ae_on,
        "deepstack_off": ae_off,
        "delta_off_minus_on": {
            "ade_mean_m": ae_off["ade_mean_m"] - ae_on["ade_mean_m"],
            "fde_mean_m": ae_off["fde_mean_m"] - ae_on["fde_mean_m"],
            "minade_at_6_mean_m": ae_off.get("minade_at_6_mean_m", ae_off.get("minade_at_n_mean_m")) - ae_on.get("minade_at_6_mean_m", ae_on.get("minade_at_n_mean_m")),
            "minfde_at_6_mean_m": ae_off.get("minfde_at_6_mean_m", ae_off.get("minfde_at_n_mean_m")) - ae_on.get("minfde_at_6_mean_m", ae_on.get("minfde_at_n_mean_m")),
        },
    },
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
print(json.dumps({"event": "aggregate_written", "path": str(out / "summary.json")}))
PY
echo "{\"event\":\"deepstack_ablation_256_done\",\"ts\":\"$(date -Is)\"}" | tee -a "${OUT_ROOT}/run_status.jsonl"
