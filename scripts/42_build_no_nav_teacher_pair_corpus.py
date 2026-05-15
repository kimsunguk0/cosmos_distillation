#!/usr/bin/env python3
"""Build a no-nav Alpamayo teacher-pair corpus for Cosmos distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MANIFESTS = (
    Path("/home/pm97/workspace/dataset/distill_dataset/teacher_cache/no_nav/text/manifest/no_nav_teacher_infer_manifest.parquet"),
    Path("/home/pm97/workspace/dataset/distill_dataset/reports/no_nav/next50_after_250/manifest/no_nav_teacher_infer_manifest.parquet"),
)
DEFAULT_NAV_TEXT = "Navigation guidance: No navigation instruction is provided."
IMAGE_NAMES = tuple(f"cam{camera_idx}_f{frame_idx}.png" for camera_idx in range(4) for frame_idx in range(4))
FAST_PATH_REPLACEMENTS = (
    ("/data/materialized", "/home/pm97/workspace/dataset/distill_dataset/materialized"),
    ("/data/teacher_cache", "/home/pm97/workspace/dataset/distill_dataset/teacher_cache"),
)
READ_COLUMNS = (
    "sample_id",
    "clip_id",
    "chunk_id",
    "sample_idx_in_clip",
    "sample_time_sec",
    "sample_timestamp_us",
    "split",
    "materialized_dir",
    "inference_status",
    "cot_nonempty",
    "teacher_long_cot",
    "future_tokens_nonempty",
    "teacher_future_token_count",
    "teacher_future_invalid_i3000_plus_count",
    "teacher_future_token_ids",
    "teacher_future_token_ids_path",
    "teacher_text_topk_valid",
    "teacher_text_topk_ids_path",
    "teacher_text_topk_logprobs_path",
    "teacher_text_token_ids_path",
    "teacher_text_topk_num_positions",
    "teacher_text_topk_k",
    "teacher_text_topk_position_type",
    "teacher_text_topk_source",
    "teacher_cot_num_tokens",
    "topk_nonempty",
    "teacher_traj_topk_ids_path",
    "teacher_traj_topk_logprobs_path",
    "teacher_traj_hidden_path",
    "teacher_output_json_path",
    "teacher_output_json_hash",
    "request_json_path",
    "request_json_hash",
    "teacher_cot_end_hidden_path",
    "teacher_traj_start_hidden_path",
    "teacher_action_pre_hidden_path",
    "nav_text",
    "trajectory_frame",
    "trajectory_axis_convention",
    "trajectory_units",
    "trajectory_horizon_sec",
    "trajectory_dt_sec",
    "prompt_hash",
    "full_prompt_hash",
    "teacher_run_id",
    "model_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-parquet",
        action="append",
        type=Path,
        default=None,
        help="Teacher inference manifest parquet. Repeat to concatenate multiple runs.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "no_nav_teacher_pair_300chunks_summary.json",
    )
    parser.add_argument(
        "--reported-output-jsonl",
        type=Path,
        default=None,
        help="Path to record in the summary when building through a temporary output file.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--split-policy",
        choices=("hash_clip", "manifest"),
        default="hash_clip",
        help="Use deterministic clip-level hashing or the manifest split column.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--split-salt", default="no_nav_teacher_pair_v1")
    parser.add_argument(
        "--allow-missing-text-topk",
        action="store_true",
        help="Keep samples even when text top-k artifacts are missing.",
    )
    parser.add_argument(
        "--allow-missing-traj-topk",
        action="store_true",
        help="Keep samples even when trajectory top-k artifacts are missing.",
    )
    parser.add_argument(
        "--skip-image-stat",
        action="store_true",
        help="Do not stat all 16 image files while building the corpus.",
    )
    parser.add_argument(
        "--inline-image-paths",
        action="store_true",
        help="Store all 16 absolute image paths in each JSONL row instead of the compact materialized layout reference.",
    )
    parser.add_argument(
        "--inline-future-tokens",
        action="store_true",
        help="Store the 128 future token ids inline instead of relying on traj_future_token_ids_path.",
    )
    return parser.parse_args()


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def as_bool(value: Any, default: bool = False) -> bool:
    value = clean_value(value)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "ready", "ok"}
    return bool(value)


def as_str(value: Any, default: str | None = None) -> str | None:
    value = clean_value(value)
    if value is None:
        return default
    return str(value)


def as_int(value: Any, default: int | None = None) -> int | None:
    value = clean_value(value)
    if value is None:
        return default
    return int(value)


def resolve_path(raw_path: Any) -> Path | None:
    raw_path = clean_value(raw_path)
    if raw_path in (None, ""):
        return None
    path_str = str(raw_path)
    for old_prefix, new_prefix in FAST_PATH_REPLACEMENTS:
        if path_str.startswith(old_prefix):
            path_str = path_str.replace(old_prefix, new_prefix, 1)
            break
    return Path(path_str).expanduser()


def path_str(raw_path: Any) -> str | None:
    path = resolve_path(raw_path)
    return str(path) if path is not None else None


def normalize_token_ids(raw_tokens: Any) -> list[int] | None:
    raw_tokens = clean_value(raw_tokens)
    if raw_tokens is None:
        return None
    if isinstance(raw_tokens, str):
        try:
            raw_tokens = json.loads(raw_tokens)
        except json.JSONDecodeError:
            return None
    try:
        token_ids = np.asarray(raw_tokens, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if token_ids.shape[0] != 128:
        return None
    return [int(value) for value in token_ids.tolist()]


def load_future_tokens(row: dict[str, Any]) -> tuple[list[int] | None, str | None]:
    inline = normalize_token_ids(row.get("teacher_future_token_ids"))
    token_path = path_str(row.get("teacher_future_token_ids_path"))
    if inline is not None:
        return inline, token_path
    resolved = resolve_path(row.get("teacher_future_token_ids_path"))
    if resolved is None or not resolved.exists():
        return None, token_path
    try:
        token_ids = np.load(resolved, mmap_mode="r").astype(np.int64).reshape(-1)
    except Exception:  # noqa: BLE001
        return None, token_path
    if token_ids.shape[0] != 128:
        return None, token_path
    return [int(value) for value in token_ids.tolist()], token_path


def deterministic_split(row: dict[str, Any], *, policy: str, val_fraction: float, salt: str) -> str:
    if policy == "manifest":
        split = as_str(row.get("split"), "train")
        if split in {"train", "val"}:
            return split
    clip_id = as_str(row.get("clip_id"), as_str(row.get("sample_id"), ""))
    digest = hashlib.sha1(f"{salt}:{clip_id}".encode("utf-8")).digest()
    ratio = int.from_bytes(digest[:8], "big") / float(2**64)
    return "val" if ratio < float(val_fraction) else "train"


def image_paths_for_materialized(materialized_dir: Path, *, check_exists: bool) -> list[str] | None:
    image_dir = materialized_dir / "images"
    paths = [image_dir / name for name in IMAGE_NAMES]
    if check_exists and not all(path.exists() for path in paths):
        discovered = sorted(image_dir.glob("cam*_f*.png"))
        if len(discovered) != 16:
            return None
        paths = discovered
    return [str(path) for path in paths]


def read_manifest_frame(manifest_path: Path) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(manifest_path)
        available = set(parquet_file.schema.names)
        columns = [column for column in READ_COLUMNS if column in available]
        return pd.read_parquet(manifest_path, columns=columns)
    except Exception:  # noqa: BLE001
        frame = pd.read_parquet(manifest_path)
        keep_columns = [column for column in READ_COLUMNS if column in frame.columns]
        return frame[keep_columns] if keep_columns else frame


def reject_reason(row: dict[str, Any], args: argparse.Namespace) -> str | None:
    status = as_str(row.get("inference_status"), "ready")
    if status != "ready":
        return f"inference_status:{status}"
    if not as_bool(row.get("cot_nonempty"), default=bool(as_str(row.get("teacher_long_cot"), ""))):
        return "empty_cot"
    if not as_bool(row.get("future_tokens_nonempty"), default=True):
        return "empty_future_tokens"
    if as_int(row.get("teacher_future_token_count"), 128) != 128:
        return "future_token_count_not_128"
    if as_int(row.get("teacher_future_invalid_i3000_plus_count"), 0) not in (None, 0):
        return "future_token_i3000_plus"
    if not args.allow_missing_text_topk:
        if not as_bool(row.get("teacher_text_topk_valid"), default=False):
            return "missing_text_topk"
        if not path_str(row.get("teacher_text_topk_ids_path")) or not path_str(row.get("teacher_text_topk_logprobs_path")):
            return "missing_text_topk_paths"
    if not args.allow_missing_traj_topk:
        if not as_bool(row.get("topk_nonempty"), default=True):
            return "missing_traj_topk"
        if not path_str(row.get("teacher_traj_topk_ids_path")):
            return "missing_traj_topk_path"
    return None


def build_record(row: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, str | None]:
    reason = reject_reason(row, args)
    if reason is not None:
        return None, reason

    sample_id = as_str(row.get("sample_id"))
    materialized_dir_raw = path_str(row.get("materialized_dir"))
    if not sample_id or not materialized_dir_raw:
        return None, "missing_identity_or_materialized_dir"
    materialized_dir = Path(materialized_dir_raw)
    if not args.skip_image_stat and not materialized_dir.exists():
        return None, "missing_materialized_dir"
    image_paths: list[str] = []
    if args.inline_image_paths or not args.skip_image_stat:
        resolved_image_paths = image_paths_for_materialized(materialized_dir, check_exists=not args.skip_image_stat)
        if not resolved_image_paths:
            return None, "missing_images"
        image_paths = resolved_image_paths

    future_token_ids_path = path_str(row.get("teacher_future_token_ids_path"))
    if not future_token_ids_path:
        return None, "missing_future_token_path"
    future_token_ids: list[int] = []
    if args.inline_future_tokens:
        loaded_future_token_ids, future_token_ids_path = load_future_tokens(row)
        if loaded_future_token_ids is None:
            return None, "missing_or_invalid_future_tokens"
        if min(loaded_future_token_ids) < 0 or max(loaded_future_token_ids) >= 3000:
            return None, "future_tokens_outside_0_2999"
        future_token_ids = loaded_future_token_ids

    cot_text = as_str(row.get("teacher_long_cot"), "")
    if not cot_text:
        return None, "empty_cot"

    nav_text = as_str(row.get("nav_text"), DEFAULT_NAV_TEXT) or DEFAULT_NAV_TEXT
    split = deterministic_split(row, policy=args.split_policy, val_fraction=args.val_fraction, salt=args.split_salt)
    chunk_id = as_str(row.get("chunk_id"))
    prompt_question = (
        f"{nav_text}\n"
        "Explain the causal driving context, then emit the ego future trajectory tokens."
    )
    input_block = {
        "materialized_sample_path": str(materialized_dir),
        "metadata_path": str(materialized_dir / "metadata.json"),
        "ego_history_path": str(materialized_dir / "ego" / "ego_history_xyz.npy"),
        "question": prompt_question,
        "conditioning_mode": "no_nav",
        "nav_available": False,
        "nav_text": nav_text,
        "camera_count": 4,
        "image_count": 16,
        "num_frames_per_camera": 4,
        "image_layout": "materialized_4x4_png",
        "image_names": list(IMAGE_NAMES),
    }
    if args.inline_image_paths:
        input_block["image_paths"] = image_paths

    text_topk_ids_path = path_str(row.get("teacher_text_topk_ids_path"))
    text_topk_logprobs_path = path_str(row.get("teacher_text_topk_logprobs_path"))
    text_token_ids_path = path_str(row.get("teacher_text_token_ids_path"))
    traj_topk_path = path_str(row.get("teacher_traj_topk_ids_path") or row.get("teacher_traj_topk_logprobs_path"))
    traj_hidden_path = path_str(row.get("teacher_traj_hidden_path"))

    record = {
        "sample_id": sample_id,
        "clip_id": as_str(row.get("clip_id")),
        "chunk_id": chunk_id,
        "split": split,
        "input": input_block,
        "hard_target": {
            "cot_text": cot_text,
            "traj_future_token_ids": future_token_ids if args.inline_future_tokens else [],
            "traj_future_token_ids_path": future_token_ids_path,
            "traj_token_count": 128,
            "traj_waypoint_count": 64,
            "traj_frame": as_str(row.get("trajectory_frame"), "ego_at_sample_time"),
            "traj_axis_convention": as_str(row.get("trajectory_axis_convention"), "x_forward_y_left_z_up"),
            "traj_units": as_str(row.get("trajectory_units"), "meters"),
            "traj_horizon_sec": float(clean_value(row.get("trajectory_horizon_sec")) or 6.4),
            "traj_dt_sec": float(clean_value(row.get("trajectory_dt_sec")) or 0.1),
            "source": "alpamayo15_no_nav_teacher_future_tokens",
        },
        "teacher_target": {
            "cot_text": cot_text,
            "source": "alpamayo15_no_nav",
            "topk_ids_path": text_topk_ids_path,
            "topk_logprobs_path": text_topk_logprobs_path,
            "target_token_ids_path": text_token_ids_path,
            "target_token_count": as_int(row.get("teacher_text_topk_num_positions"), as_int(row.get("teacher_cot_num_tokens"), 0)),
            "teacher_text_topk_k": as_int(row.get("teacher_text_topk_k"), 8),
            "teacher_text_topk_position_type": as_str(row.get("teacher_text_topk_position_type"), "cot_tokens_plus_boundaries"),
            "teacher_text_topk_source": as_str(row.get("teacher_text_topk_source"), "generation_scores"),
            "teacher_text_topk_valid": as_bool(row.get("teacher_text_topk_valid"), default=bool(text_topk_ids_path)),
            "pooled_hidden_path": None,
            "teacher_quality_multiplier": 1.0,
            "teacher_view_weight": 0.0,
        },
        "teacher_traj_target": {
            "token_ids_path": future_token_ids_path,
            "topk_logits_path": traj_topk_path,
            "topk_ids_path": traj_topk_path,
            "topk_logprobs_path": traj_topk_path,
            "hidden_path": traj_hidden_path,
            "hidden_source": "final_lm_pre_head",
            "hidden_position_type": "traj_body_128",
            "quality_multiplier": 1.0,
            "valid": True,
            "status": "ready",
        },
        "teacher_cache": {
            "text_raw_json_path": path_str(row.get("teacher_output_json_path")),
            "text_output_hash": as_str(row.get("teacher_output_json_hash")),
            "request_json_path": path_str(row.get("request_json_path")),
            "request_json_hash": as_str(row.get("request_json_hash")),
            "boundary_hidden_paths": {
                "cot_end": path_str(row.get("teacher_cot_end_hidden_path")),
                "traj_start": path_str(row.get("teacher_traj_start_hidden_path")),
                "action_pre": path_str(row.get("teacher_action_pre_hidden_path")),
            },
        },
        "gate": {
            "teacher_view_allowed": False,
            "teacher_view_weight": 0.0,
            "action_aux_allowed": False,
            "action_aux_weight": 0.0,
            "teacher_vs_gt_motion": "teacher_pair_no_nav",
        },
        "weights": {
            "hard_cot_ce": 1.0,
            "traj_ce": 1.0,
            "teacher_logit_kd": 1.0 if text_topk_ids_path and text_topk_logprobs_path else 0.0,
            "teacher_traj_topk_kd": 1.0 if traj_topk_path else 0.0,
            "teacher_traj_hidden_align": 1.0 if traj_hidden_path else 0.0,
        },
        "provenance": {
            "hard_text": "alpamayo15_no_nav_teacher_cot",
            "soft_text": "alpamayo15_no_nav_text_topk",
            "traj_target": "alpamayo15_no_nav_teacher_future_tokens",
            "teacher_gt_joint_pair_forbidden": False,
            "source_manifest": as_str(row.get("_source_manifest")),
        },
        "metadata": {
            "sample_idx_in_clip": as_int(row.get("sample_idx_in_clip")),
            "sample_time_sec": clean_value(row.get("sample_time_sec")),
            "sample_timestamp_us": as_int(row.get("sample_timestamp_us")),
            "prompt_hash": as_str(row.get("prompt_hash")),
            "full_prompt_hash": as_str(row.get("full_prompt_hash")),
            "teacher_run_id": as_str(row.get("teacher_run_id")),
            "model_path": as_str(row.get("model_path")),
        },
    }
    return record, None


def main() -> None:
    args = parse_args()
    manifest_paths = args.manifest_parquet or list(DEFAULT_MANIFESTS)
    skip_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    chunk_counts: Counter[str] = Counter()
    accepted = 0
    input_rows = 0

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output_handle:
        for manifest_path in manifest_paths:
            if not manifest_path.exists():
                skip_counts[f"missing_manifest:{manifest_path}"] += 1
                continue
            frame = read_manifest_frame(manifest_path)
            frame["_source_manifest"] = str(manifest_path)
            input_rows += int(len(frame))
            for row in frame.to_dict(orient="records"):
                record, reason = build_record(row, args)
                if record is None:
                    skip_counts[str(reason or "unknown")] += 1
                    continue
                output_handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
                accepted += 1
                split_counts[str(record["split"])] += 1
                chunk_counts[str(record.get("chunk_id") or "unknown")] += 1
                if args.max_records is not None and accepted >= int(args.max_records):
                    break
            if args.max_records is not None and accepted >= int(args.max_records):
                break

    summary = {
        "manifest_paths": [str(path) for path in manifest_paths],
        "output_jsonl": str(args.reported_output_jsonl or args.output_jsonl),
        "input_rows": input_rows,
        "accepted_records": accepted,
        "skip_counts": dict(skip_counts.most_common()),
        "split_counts": dict(split_counts.most_common()),
        "unique_chunks": len(chunk_counts),
        "chunk_count_min": min(chunk_counts.values()) if chunk_counts else None,
        "chunk_count_max": max(chunk_counts.values()) if chunk_counts else None,
        "split_policy": args.split_policy,
        "val_fraction": args.val_fraction,
        "allow_missing_text_topk": bool(args.allow_missing_text_topk),
        "allow_missing_traj_topk": bool(args.allow_missing_traj_topk),
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
