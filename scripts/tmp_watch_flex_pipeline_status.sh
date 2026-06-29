#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

WATCH_LOG="outputs/logs/flex_pipeline_watch_20260607.log"
INTERVAL_SEC="${INTERVAL_SEC:-600}"

mkdir -p outputs/logs

while true; do
  .venv/bin/python - <<'PY' | tee -a "outputs/logs/flex_pipeline_watch_20260607.log"
import json
import subprocess
import time
from pathlib import Path

stage_log = Path("outputs/action_expert/stage2_200k_more2ep_b3_nt16_lowmem_eval_20260605/train_log.jsonl")
f49_log = Path("outputs/logs/flex_f49_nods_alllora_target32_from_f42_s8000_lr2e7_20260607_chain.log")
f49_summary = Path("outputs/reports/flex_f49_nods_alllora_target32_from_f42_s8000_lr2e7_20260607_final_b0_trajonly_parity_summary.json")
f50_summary = Path("outputs/reports/flex_f50_residualslots_alllora_target32_from_f42_s8000_lr2e7_20260607_final_b0_trajonly_parity_summary.json")

def last_events(path: Path):
    steps = []
    vals = []
    trains = []
    if not path.exists():
        return None, None, None
    for line in path.open():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        event = obj.get("event")
        if event == "train_step":
            steps.append(obj)
        elif event == "val_eval":
            vals.append(obj)
        elif event == "train_eval":
            trains.append(obj)
    return (steps[-1] if steps else None), (vals[-1] if vals else None), (trains[-1] if trains else None)

def metric_summary(path: Path):
    if not path.exists():
        return None
    obj = json.loads(path.read_text())
    return {
        "token_match": obj.get("avg_target_token_match_rate"),
        "ade_m": obj.get("avg_target_ade_m"),
        "fde_m": obj.get("avg_target_fde_m"),
        "unique": obj.get("avg_generated_unique_token_count"),
        "max_same_run": obj.get("avg_generated_max_same_token_run"),
    }

def f49_train_progress(path: Path):
    if not path.exists():
        return None
    last = None
    train_done = False
    for line in path.open(errors="ignore"):
        if "flex_parity_train_done" in line:
            train_done = True
        if "flex_parity_train_step" not in line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        last = obj
    if last is None:
        return {"done": train_done, "last_step": None}
    metrics = last.get("metrics") or {}
    return {
        "done": train_done,
        "last_step": last.get("step"),
        "loss": metrics.get("loss"),
        "token_acc": metrics.get("free_run_token_acc"),
        "traj_state_cos": metrics.get("traj_state_cos"),
        "grad_norm": metrics.get("grad_norm"),
    }

def cmd(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""

last_step, last_val, last_train = last_events(stage_log)
gpu = cmd([
    "nvidia-smi",
    "--query-gpu=memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
])
tmux_raw = cmd(["tmux", "ls"])
tmux = "\n".join(
    line
    for line in tmux_raw.splitlines()
    if any(marker in line for marker in ("stage2", "flex_f49", "flex_f50"))
)
row = {
    "event": "flex_pipeline_watch",
    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    "gpu": gpu,
    "tmux": tmux.splitlines(),
    "stage2_last_step": {
        key: last_step.get(key)
        for key in ("step", "loss", "elapsed_sec", "pred_v_abs_mean", "target_v_abs_mean")
    } if last_step else None,
    "stage2_remaining_to_75000": 75000 - int(last_step["step"]) if last_step and "step" in last_step else None,
    "stage2_last_val": {
        key: last_val.get(key)
        for key in ("step", "ade_mean_m", "minade_at_6_mean_m", "fde_mean_m")
    } if last_val else None,
    "stage2_last_train_eval": {
        key: last_train.get(key)
        for key in ("step", "ade_mean_m", "minade_at_6_mean_m", "fde_mean_m")
    } if last_train else None,
    "f49_train": f49_train_progress(f49_log),
    "f49_summary": metric_summary(f49_summary),
    "f50_summary": metric_summary(f50_summary),
}
print(json.dumps(row, ensure_ascii=True))
PY
  if [[ -s "outputs/reports/flex_f50_residualslots_alllora_target32_from_f42_s8000_lr2e7_20260607_final_b0_trajonly_parity_summary.json" ]]; then
    break
  fi
  if [[ -s "outputs/reports/flex_f49_nods_alllora_target32_from_f42_s8000_lr2e7_20260607_final_b0_trajonly_parity_summary.json" ]]; then
    ade="$(
      .venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/reports/flex_f49_nods_alllora_target32_from_f42_s8000_lr2e7_20260607_final_b0_trajonly_parity_summary.json")
print(float(json.loads(p.read_text()).get("avg_target_ade_m", 999999.0)))
PY
    )"
    skip="$(
      .venv/bin/python - <<PY
print(1 if float("${ade}") < 0.8 else 0)
PY
    )"
    if [[ "${skip}" == "1" ]]; then
      break
    fi
  fi
  sleep "${INTERVAL_SEC}"
done
