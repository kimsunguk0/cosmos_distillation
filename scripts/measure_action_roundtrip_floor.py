#!/usr/bin/env python3
"""Measure the teacher action-space roundtrip floor for cached trajectories."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# This script is intentionally CPU-only. Hide CUDA before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn.functional as F

# Prevent imported helper modules from probing CUDA availability in this CPU-only run.
torch.cuda.is_available = lambda: False  # type: ignore[method-assign]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUKIM_ROOT = PROJECT_ROOT.parents[1]
ALPAMAYO_SRC = SUKIM_ROOT / "alpamayo_repo" / "alpamayo1.5" / "src"
VIS_ROOT = SUKIM_ROOT / "visualization"
for path in (PROJECT_ROOT, SUKIM_ROOT, ALPAMAYO_SRC, VIS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.inference.checkpoint_eval import load_ego_history_rot  # noqa: E402
from src.training.collator import load_ego_history_xyz  # noqa: E402


STAGE1_SCRIPT = PROJECT_ROOT / "scripts" / "51_train_stage1_ae28_teacher_kv_scale.py"
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl"
DEFAULT_TEACHER = SUKIM_ROOT / "base_weights" / "Alpamayo-1.5-10B"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "action_roundtrip_floor_20260724"


def load_stage1_module() -> Any:
    spec = importlib.util.spec_from_file_location("stage1_ae28_teacher_kv_scale", STAGE1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {STAGE1_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=163840)
    parser.add_argument("--max-pixels", type=int, default=196608)
    return parser.parse_args()


def assert_cpu_only(device: torch.device) -> None:
    if device.type != "cpu":
        raise ValueError(f"This script is CPU-only; got device={device}")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized; aborting CPU-only measurement.")


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def normalize_history_rot(rot: np.ndarray) -> np.ndarray:
    arr = np.asarray(rot, dtype=np.float32)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-2:] != (3, 3):
        raise ValueError(f"Expected ego_history_rot as [T,3,3] after squeeze, got shape={arr.shape}")
    return arr


def moving_average_xyz(xyz: torch.Tensor, window: int) -> torch.Tensor:
    width = int(window)
    if width <= 1:
        return xyz.clone()
    if width > int(xyz.shape[-2]):
        raise ValueError(f"--smooth-window={width} exceeds waypoint count={xyz.shape[-2]}")
    left = width // 2
    right = width - 1 - left
    channels_first = xyz.transpose(1, 2)
    padded = F.pad(channels_first, (left, right), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=width, stride=1).transpose(1, 2)


def select_items_with_rows(stage1: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    scanned = 0
    for row in stage1.iter_jsonl(args.corpus_jsonl):
        scanned += 1
        if args.split and row.get("split") != args.split:
            continue
        raw_path = stage1.resolve_raw_json(row)
        sample_dir = Path(str((row.get("input") or {}).get("materialized_sample_path") or ""))
        if raw_path is None or not sample_dir.exists():
            continue
        items.append(
            {
                "sample_id": str(row["sample_id"]),
                "row": row,
                "sample_dir": str(sample_dir),
                "raw_json": str(raw_path),
                "clip_id": str(row.get("clip_id") or ""),
                "chunk_id": str(row.get("chunk_id") or ""),
            }
        )
        if len(items) >= int(args.num_samples):
            break
    if not items:
        raise RuntimeError("No val samples with raw teacher trajectory outputs were found.")
    print(
        json.dumps(
            {
                "event": "select_items_done",
                "selected_count": len(items),
                "scanned_count": scanned,
                "corpus_jsonl": str(args.corpus_jsonl),
            }
        ),
        flush=True,
    )
    return items


def iter_batches(items: list[dict[str, Any]], batch_size: int):
    width = max(int(batch_size), 1)
    for index in range(0, len(items), width):
        yield items[index : index + width]


def build_metric_batch(stage1: Any, batch_items: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    histories_xyz: list[np.ndarray] = []
    histories_rot: list[np.ndarray] = []
    target_xyz_np: list[np.ndarray] = []
    target_rot_np: list[np.ndarray] = []
    for item in batch_items:
        row = item["row"]
        histories_xyz.append(load_ego_history_xyz(row, PROJECT_ROOT).astype(np.float32))
        histories_rot.append(normalize_history_rot(load_ego_history_rot(row, PROJECT_ROOT)))
        xyz, rot = stage1.raw_teacher_pred(Path(item["raw_json"]))
        target_xyz_np.append(xyz.astype(np.float32))
        target_rot_np.append(rot.astype(np.float32))
    return {
        "sample_ids": [item["sample_id"] for item in batch_items],
        "target_xyz": torch.from_numpy(np.stack(target_xyz_np, axis=0)).to(device=device, dtype=torch.float32),
        "target_rot": torch.from_numpy(np.stack(target_rot_np, axis=0)).to(device=device, dtype=torch.float32),
        "ego_history_xyz": torch.from_numpy(np.stack(histories_xyz, axis=0)).to(device=device, dtype=torch.float32),
        "ego_history_rot": torch.from_numpy(np.stack(histories_rot, axis=0)).to(device=device, dtype=torch.float32),
    }


def load_teacher_action_space(stage1: Any, args: argparse.Namespace) -> tuple[Any, tuple[int, ...], dict[str, Any]]:
    print(
        json.dumps(
            {
                "event": "load_teacher_start",
                "checkpoint": str(args.teacher_checkpoint_path),
                "device": "cpu",
                "attn_implementation": str(args.attn_implementation),
            }
        ),
        flush=True,
    )
    teacher_model, processor, config, config_path, runtime = stage1.load_model_and_processor(
        checkpoint_path=args.teacher_checkpoint_path,
        dtype=stage1.torch_dtype_from_name(args.dtype),
        device="cpu",
        config_json=None,
        runtime_support=None,
        attn_implementation=args.attn_implementation,
        min_pixels=int(args.min_pixels),
        max_pixels=int(args.max_pixels),
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)
    if hasattr(teacher_model, "expert"):
        stage1.force_attention(teacher_model.expert, "sdpa" if args.attn_implementation != "eager" else "eager")
    action_space = teacher_model.action_space.to("cpu").eval()
    action_dims = tuple(action_space.get_action_space_dims())

    # Free teacher VLM weights from memory; only action_space is used below.
    if hasattr(teacher_model, "vlm"):
        delattr(teacher_model, "vlm")
    del processor
    gc.collect()
    metadata = {
        "config_path": str(config_path),
        "runtime_support_path": str(runtime) if runtime is not None else None,
        "action_space_class": type(action_space).__name__,
        "action_space_dims": list(action_dims),
        "teacher_config_attn_implementation": getattr(config, "attn_implementation", None),
    }
    print(json.dumps({"event": "load_teacher_done", **metadata}), flush=True)
    return action_space, action_dims, metadata


def add_metric(metrics: dict[str, list[float]], row: dict[str, Any], key: str, value: float) -> None:
    scalar = float(value)
    row[key] = scalar
    metrics.setdefault(key, []).append(scalar)


def measure_batch(
    *,
    stage1: Any,
    action_space: Any,
    action_dims: tuple[int, ...],
    batch: dict[str, Any],
    smooth_window: int,
    metrics: dict[str, list[float]],
) -> list[dict[str, Any]]:
    target_xyz = batch["target_xyz"]
    target_rot = batch["target_rot"]
    ego_history_xyz = batch["ego_history_xyz"]
    ego_history_rot = batch["ego_history_rot"]
    smoothed_xyz = moving_average_xyz(target_xyz, int(smooth_window))

    with torch.inference_mode():
        raw_action = action_space.traj_to_action(
            ego_history_xyz,
            ego_history_rot,
            target_xyz,
            target_rot,
        )
        smoothed_action = action_space.traj_to_action(
            ego_history_xyz,
            ego_history_rot,
            smoothed_xyz,
            target_rot,
        )
        if tuple(raw_action.shape[1:]) != action_dims:
            raise AssertionError(f"raw_action shape={tuple(raw_action.shape)} expected (B, *{action_dims})")
        if tuple(smoothed_action.shape[1:]) != action_dims:
            raise AssertionError(f"smoothed_action shape={tuple(smoothed_action.shape)} expected (B, *{action_dims})")
        raw_rt_xyz, _raw_rt_rot = action_space.action_to_traj(
            raw_action,
            ego_history_xyz,
            ego_history_rot,
        )
        smoothed_rt_xyz, _smoothed_rt_rot = action_space.action_to_traj(
            smoothed_action,
            ego_history_xyz,
            ego_history_rot,
        )

    target_xyz_np = target_xyz.detach().cpu().numpy()
    smoothed_xyz_np = smoothed_xyz.detach().cpu().numpy()
    raw_rt_xyz_np = raw_rt_xyz.detach().float().cpu().numpy()
    smoothed_rt_xyz_np = smoothed_rt_xyz.detach().float().cpu().numpy()
    raw_action_np = raw_action.detach().float().cpu().numpy()
    smoothed_action_np = smoothed_action.detach().float().cpu().numpy()

    rows: list[dict[str, Any]] = []
    for row_index, sample_id in enumerate(batch["sample_ids"]):
        row: dict[str, Any] = {"sample_id": sample_id}
        raw_ade, raw_fde = stage1.ade_fde(raw_rt_xyz_np[row_index], target_xyz_np[row_index])
        smoothed_ade, smoothed_fde = stage1.ade_fde(smoothed_rt_xyz_np[row_index], smoothed_xyz_np[row_index])
        smooth_delta_ade, smooth_delta_fde = stage1.ade_fde(smoothed_xyz_np[row_index], target_xyz_np[row_index])
        add_metric(metrics, row, "raw_roundtrip_ade_m", raw_ade)
        add_metric(metrics, row, "raw_roundtrip_fde_m", raw_fde)
        add_metric(metrics, row, "smoothed_roundtrip_ade_m", smoothed_ade)
        add_metric(metrics, row, "smoothed_roundtrip_fde_m", smoothed_fde)
        add_metric(metrics, row, "smoothed_input_vs_raw_ade_m", smooth_delta_ade)
        add_metric(metrics, row, "smoothed_input_vs_raw_fde_m", smooth_delta_fde)
        add_metric(metrics, row, "raw_action_accel_abs_mean", np.abs(raw_action_np[row_index, :, 0]).mean())
        add_metric(metrics, row, "raw_action_curvature_abs_mean", np.abs(raw_action_np[row_index, :, 1]).mean())
        add_metric(metrics, row, "smoothed_action_accel_abs_mean", np.abs(smoothed_action_np[row_index, :, 0]).mean())
        add_metric(
            metrics,
            row,
            "smoothed_action_curvature_abs_mean",
            np.abs(smoothed_action_np[row_index, :, 1]).mean(),
        )
        rows.append(row)
    return rows


def print_human_summary(summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]

    def fmt_metric(name: str) -> str:
        stats = metrics[name]
        return f"mean={stats['mean']:.6f} p50={stats['p50']:.6f} p95={stats['p95']:.6f}"

    print("Action roundtrip floor (CPU)", flush=True)
    print(f"summary_json: {summary['summary_json']}", flush=True)
    print(
        f"samples: {summary['selected_count']}  batch_size: {summary['args']['batch_size']}  "
        f"smooth_window: {summary['args']['smooth_window']}",
        flush=True,
    )
    print(f"raw roundtrip ADE: {fmt_metric('raw_roundtrip_ade_m')}", flush=True)
    print(f"raw roundtrip FDE: {fmt_metric('raw_roundtrip_fde_m')}", flush=True)
    print(f"smoothed roundtrip ADE: {fmt_metric('smoothed_roundtrip_ade_m')}", flush=True)
    print(f"smoothed roundtrip FDE: {fmt_metric('smoothed_roundtrip_fde_m')}", flush=True)
    print(f"raw accel |mean|: {fmt_metric('raw_action_accel_abs_mean')}", flush=True)
    print(f"smoothed accel |mean|: {fmt_metric('smoothed_action_accel_abs_mean')}", flush=True)
    print(f"raw curvature |mean|: {fmt_metric('raw_action_curvature_abs_mean')}", flush=True)
    print(f"smoothed curvature |mean|: {fmt_metric('smoothed_action_curvature_abs_mean')}", flush=True)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    if int(args.num_samples) <= 0:
        raise ValueError("--num-samples must be positive")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if int(args.smooth_window) <= 0:
        raise ValueError("--smooth-window must be positive")
    device = torch.device(args.device)
    assert_cpu_only(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    started = time.perf_counter()
    stage1 = load_stage1_module()
    items = select_items_with_rows(stage1, args)
    action_space, action_dims, teacher_metadata = load_teacher_action_space(stage1, args)
    assert_cpu_only(device)

    metrics: dict[str, list[float]] = {}
    rows: list[dict[str, Any]] = []
    for batch_index, batch_items in enumerate(iter_batches(items, int(args.batch_size))):
        batch = build_metric_batch(stage1, batch_items, device)
        batch_rows = measure_batch(
            stage1=stage1,
            action_space=action_space,
            action_dims=action_dims,
            batch=batch,
            smooth_window=int(args.smooth_window),
            metrics=metrics,
        )
        for row in batch_rows:
            row["batch_index"] = batch_index
        rows.extend(batch_rows)
        print(
            json.dumps({"event": "batch_done", "batch_index": batch_index, "batch_size": len(batch_items)}),
            flush=True,
        )
        assert_cpu_only(device)

    summary = {
        "status": "ok",
        "summary_json": str(summary_path),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "created_at_unix": time.time(),
        "stage1_module": {
            "script_path": str(STAGE1_SCRIPT),
            "module_name": "stage1_ae28_teacher_kv_scale",
            "import_line": (
                'spec = importlib.util.spec_from_file_location("stage1_ae28_teacher_kv_scale", STAGE1_SCRIPT)'
            ),
        },
        "teacher": teacher_metadata,
        "device": {
            "requested": str(args.device),
            "actual": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch_cuda_initialized": bool(torch.cuda.is_initialized()),
        },
        "args": {
            "corpus_jsonl": str(args.corpus_jsonl),
            "teacher_checkpoint_path": str(args.teacher_checkpoint_path),
            "output_dir": str(args.output_dir),
            "split": str(args.split),
            "num_samples": int(args.num_samples),
            "batch_size": int(args.batch_size),
            "smooth_window": int(args.smooth_window),
            "dtype": str(args.dtype),
            "attn_implementation": str(args.attn_implementation),
        },
        "selected_count": len(items),
        "metrics": {key: summarize(values) for key, values in sorted(metrics.items())},
        "rows": rows,
        "notes": {
            "raw_roundtrip": "pred_xyz/pred_rot -> traj_to_action -> action_to_traj; ADE/FDE vs raw pred_xyz.",
            "smoothed_roundtrip": (
                "moving-average pred_xyz with edge replication -> traj_to_action using raw pred_rot "
                "-> action_to_traj; ADE/FDE vs smoothed pred_xyz."
            ),
            "history_source": (
                "ego_history_xyz from src.training.collator.load_ego_history_xyz; "
                "ego_history_rot from src.inference.checkpoint_eval.load_ego_history_rot then normalize_history_rot."
            ),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    assert_cpu_only(device)
    print_human_summary(summary)


if __name__ == "__main__":
    main()
