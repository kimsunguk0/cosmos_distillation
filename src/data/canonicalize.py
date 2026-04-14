"""Canonical sample materialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.local_dataset import (
    CAMERA_NAMES,
    decode_video_frames,
    interpolate_egomotion,
    load_camera_timestamps,
    load_egomotion,
    localize_trajectory,
    nearest_timestamp_indices,
)
from src.data.schema_versions import CANONICAL_LOCALIZATION_VERSION


@dataclass(slots=True)
class CanonicalSampleSpec:
    sample_id: str
    output_dir: Path
    frame_offsets_sec: tuple[float, ...]
    ego_history_steps: int
    ego_future_steps: int


def history_timestamps_us(t0_us: int, ego_history_steps: int, frequency_hz: int) -> np.ndarray:
    """Return history sample timestamps ending at t0."""
    dt_us = int(1_000_000 / frequency_hz)
    return np.arange(t0_us - (ego_history_steps - 1) * dt_us, t0_us + 1, dt_us, dtype=np.int64)


def future_timestamps_us(t0_us: int, ego_future_steps: int, frequency_hz: int) -> np.ndarray:
    """Return future sample timestamps starting one step after t0."""
    dt_us = int(1_000_000 / frequency_hz)
    return np.arange(t0_us + dt_us, t0_us + (ego_future_steps + 1) * dt_us, dt_us, dtype=np.int64)


def image_timestamps_us(t0_us: int, frame_offsets_sec: tuple[float, ...]) -> np.ndarray:
    """Return image timestamps for the configured frame offsets."""
    return np.asarray([int(t0_us + offset * 1_000_000) for offset in frame_offsets_sec], dtype=np.int64)


def _save_numpy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def materialize_sample(
    row: pd.Series,
    dataset_root: Path,
    sample_root: Path,
    *,
    frame_offsets_sec: tuple[float, ...] = (-0.3, -0.2, -0.1, 0.0),
    ego_history_steps: int = 16,
    ego_future_steps: int = 64,
    frequency_hz: int = 10,
    extract_images: bool = True,
    reuse_existing_frames: bool = False,
) -> dict[str, Any]:
    """Materialize one canonical sample directory from local zip chunks."""
    sample_id = row["sample_id"]
    clip_id = row["clip_uuid"]
    chunk = int(row["chunk"])
    t0_us = int(row["t0_us"])
    output_dir = sample_root / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    hist_ts = history_timestamps_us(t0_us, ego_history_steps, frequency_hz)
    fut_ts = future_timestamps_us(t0_us, ego_future_steps, frequency_hz)
    img_ts = image_timestamps_us(t0_us, frame_offsets_sec)

    ego_df = load_egomotion(dataset_root, clip_id, chunk)
    hist_xyz_world, hist_quat_world = interpolate_egomotion(ego_df, hist_ts)
    fut_xyz_world, fut_quat_world = interpolate_egomotion(ego_df, fut_ts)

    # The explicit t0 anchor is the last history pose, which lands exactly on t0.
    anchor_xyz_world = hist_xyz_world[-1].copy()
    anchor_quat_world = hist_quat_world[-1].copy()
    hist_xyz_local, hist_rot_local = localize_trajectory(
        hist_xyz_world,
        hist_quat_world,
        anchor_xyz_world=anchor_xyz_world,
        anchor_quat_xyzw=anchor_quat_world,
    )
    fut_xyz_local, fut_rot_local = localize_trajectory(
        fut_xyz_world,
        fut_quat_world,
        anchor_xyz_world=anchor_xyz_world,
        anchor_quat_xyzw=anchor_quat_world,
    )

    _save_numpy(output_dir / "ego_history_xyz.npy", hist_xyz_local.astype(np.float32))
    _save_numpy(output_dir / "ego_history_rot.npy", hist_rot_local.astype(np.float32))
    _save_numpy(output_dir / "ego_future_xyz.npy", fut_xyz_local.astype(np.float32))
    _save_numpy(output_dir / "ego_future_rot.npy", fut_rot_local.astype(np.float32))
    _save_numpy(output_dir / "ego_history_xyz_local_t0.npy", hist_xyz_local.astype(np.float32))
    _save_numpy(output_dir / "ego_history_rot_local_t0.npy", hist_rot_local.astype(np.float32))
    _save_numpy(output_dir / "ego_future_xyz_local_t0.npy", fut_xyz_local.astype(np.float32))
    _save_numpy(output_dir / "ego_future_rot_local_t0.npy", fut_rot_local.astype(np.float32))
    _save_numpy(output_dir / "ego_history_xyz_world.npy", hist_xyz_world.astype(np.float32))
    _save_numpy(output_dir / "ego_history_quat_world.npy", hist_quat_world.astype(np.float32))
    _save_numpy(output_dir / "ego_future_xyz_world.npy", fut_xyz_world.astype(np.float32))
    _save_numpy(output_dir / "ego_future_quat_world.npy", fut_quat_world.astype(np.float32))
    _save_numpy(output_dir / "ego_anchor_xyz_world.npy", anchor_xyz_world.astype(np.float32))
    _save_numpy(output_dir / "ego_anchor_quat_world.npy", anchor_quat_world.astype(np.float32))

    rel_timestamp_payload: dict[str, list[float]] = {}
    abs_timestamp_payload: dict[str, list[int]] = {}
    decoder_status: dict[str, str] = {}

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for camera_name in CAMERA_NAMES:
        camera_ts_df = load_camera_timestamps(dataset_root, clip_id, chunk, camera_name)
        source_timestamps = camera_ts_df["timestamp"].to_numpy(dtype=np.int64)
        selected_indices = nearest_timestamp_indices(source_timestamps, img_ts).tolist()
        selected_timestamps = source_timestamps[selected_indices]

        rel_timestamp_payload[camera_name] = [round((int(ts) - t0_us) / 1_000_000, 6) for ts in selected_timestamps]
        abs_timestamp_payload[camera_name] = [int(ts) for ts in selected_timestamps]

        if not extract_images:
            decoder_status[camera_name] = "skipped"
            continue
        expected_paths = [frames_dir / f"{camera_name}_t{rel_sec:+.1f}.jpg" for rel_sec in frame_offsets_sec]
        if reuse_existing_frames and all(path.exists() for path in expected_paths):
            decoder_status[camera_name] = "decoded"
            continue
        try:
            images = decode_video_frames(
                dataset_root=dataset_root,
                clip_id=clip_id,
                chunk=chunk,
                camera_name=camera_name,
                frame_indices=selected_indices,
            )
            for rel_sec, image in zip(frame_offsets_sec, images):
                image.save(frames_dir / f"{camera_name}_t{rel_sec:+.1f}.jpg", format="JPEG", quality=95)
            decoder_status[camera_name] = "decoded"
        except Exception as exc:  # noqa: BLE001 - persist decoder failures in sample metadata
            decoder_status[camera_name] = f"failed:{exc}"

    _save_json(output_dir / "rel_timestamps.json", rel_timestamp_payload)
    _save_json(output_dir / "abs_timestamps.json", abs_timestamp_payload)

    sample_meta = {
        "sample_id": sample_id,
        "clip_uuid": clip_id,
        "chunk": chunk,
        "t0_us": t0_us,
        "frame_offsets_sec": list(frame_offsets_sec),
        "ego_history_steps": ego_history_steps,
        "ego_future_steps": ego_future_steps,
        "frequency_hz": frequency_hz,
        "event_cluster": row.get("event_cluster"),
        "decoder_status": decoder_status,
        "canonical_localization_version": CANONICAL_LOCALIZATION_VERSION,
        "anchor_pose": {
            "source": "history_last_step_at_t0",
            "anchor_xyz_world_path": "ego_anchor_xyz_world.npy",
            "anchor_quat_world_path": "ego_anchor_quat_world.npy",
        },
    }
    _save_json(output_dir / "sample_meta.json", sample_meta)
    return sample_meta
