#!/usr/bin/env python3
"""Build full Step A Q2 candidate manifests from all PhysicalAI AV chunks.

This script does not run Alpamayo. It enumerates clips, chooses three
4cam x 1frame t0 slots, resolves camera frame indices, and writes candidate
rows that can later be passed to Q2 teacher generation/gating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vqa.q2_stepa import Q2_OFFICIAL


DEFAULT_DATASET_ROOT = Path("/home/pm97/workspace/dataset/physical_ai_av_ood_dataset")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "vqa_q2_stepa_full"
MAX_FRAME_DELTA_US = 80_000
CAMERA_SPECS = [
    ("cross_left", "camera_cross_left_120fov", 0, "Front left camera"),
    ("front_wide", "camera_front_wide_120fov", 1, "Front camera"),
    ("cross_right", "camera_cross_right_120fov", 2, "Front right camera"),
    ("front_tele", "camera_front_tele_30fov", 6, "Front telephoto camera"),
]
SLOT_OFFSETS_US = {
    "early": 5_500_000,
    "middle": 10_500_000,
    "late": 15_500_000,
}
SLOT_JITTER_RANGE_US = 1_500_000


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def stable_int(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)


def split_for_clip(clip_id: str, *, val_percent: int, test_percent: int) -> str:
    bucket = stable_int(clip_id) % 10_000
    test_cut = int(test_percent) * 100
    val_cut = test_cut + int(val_percent) * 100
    if bucket < test_cut:
        return "test"
    if bucket < val_cut:
        return "val"
    return "train"


def chunk_from_path(path: Path) -> int:
    match = re.search(r"chunk_(\d{4})", path.name)
    if not match:
        raise ValueError(f"Cannot parse chunk id from {path}")
    return int(match.group(1))


def list_egomotion_members(dataset_root: Path) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    for zip_path in sorted((dataset_root / "labels" / "egomotion").glob("egomotion.chunk_*.zip")):
        chunk = chunk_from_path(zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            for member in sorted(zf.namelist()):
                base = Path(member).name
                if base.endswith(".egomotion.parquet"):
                    clip_id = base[: -len(".egomotion.parquet")]
                    rows.append((chunk, clip_id, member))
    return rows


def camera_zip_path(dataset_root: Path, feature: str, chunk: int) -> Path:
    return dataset_root / "camera" / feature / f"{feature}.chunk_{int(chunk):04d}.zip"


def read_parquet_from_zip(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as handle:
        return pd.read_parquet(handle)


def nearest_indices(source_timestamps_us: np.ndarray, target_timestamps_us: np.ndarray) -> np.ndarray:
    target_timestamps_us = target_timestamps_us.astype(np.int64)
    source_timestamps_us = source_timestamps_us.astype(np.int64)
    right = np.searchsorted(source_timestamps_us, target_timestamps_us, side="left")
    right = np.clip(right, 0, len(source_timestamps_us) - 1)
    left = np.clip(right - 1, 0, len(source_timestamps_us) - 1)
    right_dist = np.abs(source_timestamps_us[right] - target_timestamps_us)
    left_dist = np.abs(source_timestamps_us[left] - target_timestamps_us)
    return np.where(left_dist <= right_dist, left, right)


def jittered_slot_t0(clip_id: str, slot: str, clip_start_us: int, clip_end_us: int) -> int:
    center = int(clip_start_us) + int(SLOT_OFFSETS_US[slot])
    jitter_seed = stable_int(f"{clip_id}:{slot}")
    jitter = (jitter_seed % (2 * SLOT_JITTER_RANGE_US + 1)) - SLOT_JITTER_RANGE_US
    t0 = center + int(jitter)
    margin = 200_000
    return int(np.clip(t0, int(clip_start_us) + margin, int(clip_end_us) - margin))


def load_camera_timestamps(dataset_root: Path, clip_id: str, chunk: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for _alias, feature, _camera_index, _display in CAMERA_SPECS:
        zip_path = camera_zip_path(dataset_root, feature, chunk)
        member = f"{clip_id}.{feature}.timestamps.parquet"
        df = read_parquet_from_zip(zip_path, member)
        timestamps = df["timestamp"].to_numpy(dtype=np.int64)
        if timestamps.size < 2:
            raise ValueError(f"too few timestamps for {clip_id} {feature}")
        out[feature] = np.sort(timestamps)
    return out


def make_frame_plan(
    *,
    dataset_root: Path,
    clip_id: str,
    chunk: int,
    t0_us: int,
    timestamps_by_feature: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], bool, int]:
    frame_plan: list[dict[str, Any]] = []
    max_delta = 0
    ok = True
    for _alias, feature, camera_index, display_name in CAMERA_SPECS:
        timestamps = timestamps_by_feature[feature]
        target = np.asarray([int(t0_us)], dtype=np.int64)
        frame_indices = nearest_indices(timestamps, target)
        selected = timestamps[frame_indices]
        deltas = np.abs(selected - target)
        max_delta = max(max_delta, int(deltas.max()))
        if int(deltas.max()) > MAX_FRAME_DELTA_US:
            ok = False
        frame_plan.append(
            {
                "feature": feature,
                "camera_index": int(camera_index),
                "display_name": display_name,
                "target_timestamps_us": [int(v) for v in target.tolist()],
                "frame_indices": [int(v) for v in frame_indices.tolist()],
                "frame_timestamps_us": [int(v) for v in selected.tolist()],
                "selected_delta_us": [int(v) for v in deltas.tolist()],
            }
        )
    return frame_plan, ok, max_delta


def build_candidates(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.time()
    members = list_egomotion_members(args.dataset_root)
    candidates: list[dict[str, Any]] = []
    seen_clip_ids: set[str] = set()
    counters: Counter[str] = Counter()
    chunk_counts: Counter[int] = Counter()
    clip_index = 0
    for member_index, (chunk, clip_id, ego_member) in enumerate(members, start=1):
        counters["egomotion_members"] += 1
        if args.deduplicate_clip_id and clip_id in seen_clip_ids:
            counters["skipped_duplicate_clip_id"] += 1
            continue
        seen_clip_ids.add(clip_id)
        try:
            ts_by_feature = load_camera_timestamps(args.dataset_root, clip_id, chunk)
            common_start = max(int(ts.min()) for ts in ts_by_feature.values())
            common_end = min(int(ts.max()) for ts in ts_by_feature.values())
            if common_end - common_start < 18_000_000:
                counters["skipped_short_common_window"] += 1
                continue
            clip_index += 1
            chunk_counts[int(chunk)] += 1
            split = split_for_clip(
                clip_id,
                val_percent=int(args.val_percent),
                test_percent=int(args.test_percent),
            )
            for slot in ("early", "middle", "late"):
                t0_us = jittered_slot_t0(clip_id, slot, common_start, common_end)
                frame_plan, ok, max_delta = make_frame_plan(
                    dataset_root=args.dataset_root,
                    clip_id=clip_id,
                    chunk=chunk,
                    t0_us=t0_us,
                    timestamps_by_feature=ts_by_feature,
                )
                if not ok:
                    counters[f"skipped_{slot}_frame_delta"] += 1
                    continue
                candidates.append(
                    {
                        "sample_id": f"{clip_index:05d}_{clip_id}_{slot}_Q2",
                        "base_sample_id": f"{clip_index:05d}_{clip_id}_{slot}",
                        "task": "stepa_q2_vqa_candidate",
                        "family": "Q2",
                        "stage": "1B-pre",
                        "qid": "Q2_official",
                        "image_profile": "4cam_x1",
                        "split": split,
                        "dataset_root": str(args.dataset_root),
                        "clip_id": clip_id,
                        "clip_index": int(clip_index),
                        "chunk": int(chunk),
                        "egomotion_member": ego_member,
                        "slot": slot,
                        "t0_us": int(t0_us),
                        "camera_aliases": ["cross_left", "front_wide", "cross_right", "front_tele"],
                        "camera_indices": [0, 1, 2, 6],
                        "frames_per_camera": 1,
                        "frame_offsets_us": [0],
                        "image_frames_shape_before_flatten": [4, 1, 3, 1080, 1920],
                        "image_frames_shape_after_flatten": [4, 3, 1080, 1920],
                        "frame_plan": frame_plan,
                        "max_frame_delta_us": int(max_delta),
                        "question": Q2_OFFICIAL,
                    }
                )
                counters[f"candidate_{slot}"] += 1
            counters["clips_selected"] += 1
        except Exception as exc:  # noqa: BLE001
            counters["skipped_error"] += 1
            if len(counters) < 0:  # keeps lint quiet for no-op branch
                print(exc)
        if args.limit_clips and counters["clips_selected"] >= int(args.limit_clips):
            break
        if args.progress_every and member_index % int(args.progress_every) == 0:
            print(
                json.dumps(
                    {
                        "event": "candidate_progress",
                        "members_seen": member_index,
                        "clips_selected": counters["clips_selected"],
                        "candidates": len(candidates),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
    summary = {
        "created_at": utc_now(),
        "dataset_root": str(args.dataset_root),
        "deduplicate_clip_id": bool(args.deduplicate_clip_id),
        "slot_offsets_us": SLOT_OFFSETS_US,
        "slot_jitter_range_us": SLOT_JITTER_RANGE_US,
        "max_frame_delta_us": MAX_FRAME_DELTA_US,
        "split_policy": {
            "unit": "clip_id",
            "train_percent": 100 - int(args.val_percent) - int(args.test_percent),
            "val_percent": int(args.val_percent),
            "test_percent": int(args.test_percent),
        },
        "counts": dict(counters),
        "chunks_with_selected_clips": len(chunk_counts),
        "selected_clips_per_chunk": {
            "min": min(chunk_counts.values()) if chunk_counts else 0,
            "max": max(chunk_counts.values()) if chunk_counts else 0,
        },
        "split_counts": dict(Counter(row["split"] for row in candidates)),
        "split_clip_counts": {
            split: len({row["clip_id"] for row in candidates if row["split"] == split})
            for split in ("train", "val", "test")
        },
        "slot_counts": dict(Counter(row["slot"] for row in candidates)),
        "elapsed_sec": round(time.time() - started, 3),
    }
    return candidates, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--val-percent", type=int, default=5)
    parser.add_argument("--test-percent", type=int, default=5)
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument("--deduplicate-clip-id", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    candidates, summary = build_candidates(args)
    all_path = args.output_root / "q2_candidates_all.jsonl"
    write_jsonl(all_path, candidates)
    for split in ("train", "val", "test"):
        write_jsonl(args.output_root / f"q2_candidates_{split}.jsonl", [row for row in candidates if row["split"] == split])
    write_json(args.output_root / "candidate_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

