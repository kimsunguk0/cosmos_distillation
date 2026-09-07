#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


CORPUS_JSONL = Path(
    "/home/pm97/workspace/sukim/distillation/cosmos_distillation/data/corpus/"
    "val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl"
)
DEFAULT_OUT_JSON = Path(
    "/home/pm97/workspace/sukim/distillation/cosmos_distillation/outputs/reports/"
    "r1_pilot_horizon_20260728/pilot_summary.json"
)
MIN_T0_US = 4_800_000
NUM_WAYPOINTS = 64
WAYPOINT_DT_SEC = 0.1
METRIC_NAMES = ("ADE_le2s", "ADE_gt2s", "ADE_full", "FDE_2s", "FDE_6p4s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pilot Alpamayo-R1 horizon ADE/FDE eval on filtered val512 clips."
    )
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    return parser.parse_args()


def parse_sample_id(sample_id):
    clip_id = sample_id.split("__", 1)[0]
    t0_us = int(sample_id.split("t0_", 1)[1])
    return clip_id, t0_us


def load_selected_samples(num_samples):
    selected = []
    with CORPUS_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["sample_id"]
            clip_id, t0_us = parse_sample_id(sample_id)
            if t0_us < MIN_T0_US:
                continue
            selected.append(
                {
                    "sample_id": sample_id,
                    "clip_id": clip_id,
                    "t0_us": t0_us,
                }
            )
            if len(selected) >= num_samples:
                break
    return selected


def load_model_and_processor():
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        device_map="auto",
        # load_in_4bit=True removed to test original precision
    )
    processor = helper.get_processor(model.tokenizer)
    return model, processor


def build_model_inputs(processor, data, device):
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }

    model_inputs = helper.to_device(model_inputs, device)
    return model_inputs


def run_inference(model, processor, data, device):
    model_inputs = build_model_inputs(processor, data, device)
    device_type = torch.device(device).type

    torch.cuda.manual_seed_all(42)
    with torch.autocast(device_type, dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    return pred_xyz, pred_rot, extra


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def compute_horizon_metrics(pred_xyz, ego_future_xyz):
    pred_np = as_numpy(pred_xyz)
    gt_np = as_numpy(ego_future_xyz)

    # Mirror test_inference.py batch/sample indexing, but use xyz when both
    # tensors expose z; otherwise fall back to xy for distance computation.
    pred_traj = pred_np[0, 0, 0, :NUM_WAYPOINTS, :]
    gt_traj = gt_np[0, 0, :NUM_WAYPOINTS, :]
    if pred_traj.shape[0] < NUM_WAYPOINTS or gt_traj.shape[0] < NUM_WAYPOINTS:
        raise ValueError(
            f"expected at least {NUM_WAYPOINTS} waypoints, got "
            f"pred={pred_traj.shape[0]} gt={gt_traj.shape[0]}"
        )

    if pred_traj.shape[-1] >= 3 and gt_traj.shape[-1] >= 3:
        dims = 3
        metric_dimensionality = "3D"
    else:
        dims = 2
        metric_dimensionality = "2D"

    dist = np.linalg.norm(pred_traj[:, :dims] - gt_traj[:, :dims], axis=-1)
    return {
        "metric_dimensionality": metric_dimensionality,
        "ADE_le2s": float(dist[:20].mean()),
        "ADE_gt2s": float(dist[20:NUM_WAYPOINTS].mean()),
        "ADE_full": float(dist[:NUM_WAYPOINTS].mean()),
        "FDE_2s": float(dist[19]),
        "FDE_6p4s": float(dist[63]),
    }


def first_cot_text(extra):
    cot = extra["cot"]
    if isinstance(cot, (list, tuple)):
        cot = cot[0] if cot else ""
    return str(cot)


def aggregate_metrics(rows):
    if not rows:
        return {name: None for name in METRIC_NAMES}
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in METRIC_NAMES
    }


def print_summary(summary):
    print("Alpamayo-R1 pilot horizon eval")
    print(
        f"selected={summary['n_selected']} "
        f"success={summary['n_success']} fail={summary['n_fail']}"
    )
    print(
        f"metric_dimensionality={summary['metric_dimensionality']} "
        f"waypoint_dt_sec={summary['waypoint_dt_sec']}"
    )
    print()
    print(f"{'metric':<12} {'mean_m':>12}")
    print(f"{'-' * 12} {'-' * 12}")
    for name in METRIC_NAMES:
        value = summary["aggregates"][name]
        rendered = "n/a" if value is None else f"{value:.6f}"
        print(f"{name:<12} {rendered:>12}")
    print()
    print(f"Wrote JSON: {summary['out_json']}")


def main():
    args = parse_args()
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")

    samples = load_selected_samples(args.num_samples)
    model, processor = load_model_and_processor()

    per_sample_rows = []
    failure_reasons = []
    cot_examples = []

    for sample in samples:
        try:
            data = load_physical_aiavdataset(sample["clip_id"], t0_us=sample["t0_us"])
            pred_xyz, pred_rot, extra = run_inference(model, processor, data, args.device)
            metrics = compute_horizon_metrics(pred_xyz, data["ego_future_xyz"])
            row = {
                "sample_id": sample["sample_id"],
                "clip_id": sample["clip_id"],
                "t0_us": sample["t0_us"],
                **metrics,
            }
            per_sample_rows.append(row)
            if len(cot_examples) < 3:
                cot_examples.append(
                    {
                        "sample_id": sample["sample_id"],
                        "cot": first_cot_text(extra),
                    }
                )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {sample['sample_id']}: {message}")
            failure_reasons.append(
                {
                    "sample_id": sample["sample_id"],
                    "clip_id": sample["clip_id"],
                    "t0_us": sample["t0_us"],
                    "reason": message,
                }
            )

    dimensionalities = sorted(
        {row["metric_dimensionality"] for row in per_sample_rows}
    )
    metric_dimensionality = (
        dimensionalities[0]
        if len(dimensionalities) == 1
        else "mixed" if dimensionalities else "unavailable"
    )
    aggregates = aggregate_metrics(per_sample_rows)

    summary = {
        "corpus_jsonl": str(CORPUS_JSONL),
        "min_t0_us": MIN_T0_US,
        "num_requested": args.num_samples,
        "n_selected": len(samples),
        "n_success": len(per_sample_rows),
        "n_fail": len(failure_reasons),
        "waypoint_dt_sec": WAYPOINT_DT_SEC,
        "metric_dimensionality": metric_dimensionality,
        "metric_dimensionalities": dimensionalities,
        "per_sample": per_sample_rows,
        "aggregates": aggregates,
        "failure_reasons": failure_reasons,
        "cot_examples": cot_examples,
        "out_json": str(args.out_json),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print_summary(summary)


if __name__ == "__main__":
    main()
