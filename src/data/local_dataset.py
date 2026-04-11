"""Local PhysicalAI AV readers that work directly from downloaded zip chunks."""

from __future__ import annotations

import io
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp


CAMERA_NAMES = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]


@dataclass(slots=True)
class LocalDatasetPaths:
    root: Path

    def camera_zip(self, camera_name: str, chunk: int) -> Path:
        return self.root / "camera" / camera_name / f"{camera_name}.chunk_{int(chunk):04d}.zip"

    def egomotion_zip(self, chunk: int) -> Path:
        return self.root / "labels" / "egomotion" / f"egomotion.chunk_{int(chunk):04d}.zip"


def camera_member_names(clip_id: str, camera_name: str) -> dict[str, str]:
    """Return the zip member names for a camera feature."""
    return {
        "video": f"{clip_id}.{camera_name}.mp4",
        "timestamps": f"{clip_id}.{camera_name}.timestamps.parquet",
        "blurred_boxes": f"{clip_id}.{camera_name}.blurred_boxes.parquet",
    }


def egomotion_member_name(clip_id: str) -> str:
    """Return the egomotion parquet member name."""
    return f"{clip_id}.egomotion.parquet"


def load_camera_timestamps(dataset_root: Path, clip_id: str, chunk: int, camera_name: str) -> pd.DataFrame:
    """Read the frame timestamps parquet for one camera from a chunk zip."""
    zip_path = LocalDatasetPaths(dataset_root).camera_zip(camera_name, chunk)
    member = camera_member_names(clip_id, camera_name)["timestamps"]
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as handle:
        return pd.read_parquet(handle)


def load_egomotion(dataset_root: Path, clip_id: str, chunk: int) -> pd.DataFrame:
    """Read the egomotion parquet for one clip from a chunk zip."""
    zip_path = LocalDatasetPaths(dataset_root).egomotion_zip(chunk)
    member = egomotion_member_name(clip_id)
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as handle:
        return pd.read_parquet(handle)


def _quaternion_xyzw(egomotion_df: pd.DataFrame) -> np.ndarray:
    return egomotion_df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)


def interpolate_egomotion(egomotion_df: pd.DataFrame, target_timestamps_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate xyz and quaternion values onto arbitrary timestamps."""
    source_timestamps = egomotion_df["timestamp"].to_numpy(dtype=np.int64)
    xyz = egomotion_df[["x", "y", "z"]].to_numpy(dtype=np.float64)

    xyz_interp = np.stack(
        [
            np.interp(target_timestamps_us, source_timestamps, xyz[:, axis])
            for axis in range(xyz.shape[1])
        ],
        axis=1,
    )

    rotations = Rotation.from_quat(_quaternion_xyzw(egomotion_df))
    slerp = Slerp(source_timestamps.astype(np.float64), rotations)
    quat_interp = slerp(target_timestamps_us.astype(np.float64)).as_quat()
    return xyz_interp, quat_interp


def localize_trajectory(
    xyz_world: np.ndarray,
    quat_world_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform world-frame poses into the local t0 ego frame."""
    t0_xyz = xyz_world[-1].copy()
    t0_rot = Rotation.from_quat(quat_world_xyzw[-1].copy())
    t0_rot_inv = t0_rot.inv()

    xyz_local = t0_rot_inv.apply(xyz_world - t0_xyz)
    rot_local = (t0_rot_inv * Rotation.from_quat(quat_world_xyzw)).as_matrix()
    return xyz_local, rot_local


def nearest_timestamp_indices(source_timestamps_us: np.ndarray, target_timestamps_us: np.ndarray) -> np.ndarray:
    """Map target timestamps to nearest source indices."""
    target_timestamps_us = target_timestamps_us.astype(np.int64)
    source_timestamps_us = source_timestamps_us.astype(np.int64)
    right = np.searchsorted(source_timestamps_us, target_timestamps_us, side="left")
    right = np.clip(right, 0, len(source_timestamps_us) - 1)
    left = np.clip(right - 1, 0, len(source_timestamps_us) - 1)

    right_dist = np.abs(source_timestamps_us[right] - target_timestamps_us)
    left_dist = np.abs(source_timestamps_us[left] - target_timestamps_us)
    return np.where(left_dist <= right_dist, left, right)


def decode_video_frames(
    dataset_root: Path,
    clip_id: str,
    chunk: int,
    camera_name: str,
    frame_indices: list[int],
) -> list[Image.Image]:
    """Decode selected frame indices from a camera mp4 using PyAV when available."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required for frame extraction but is not installed.") from exc

    zip_path = LocalDatasetPaths(dataset_root).camera_zip(camera_name, chunk)
    member = camera_member_names(clip_id, camera_name)["video"]
    wanted = set(frame_indices)
    decoded: dict[int, Image.Image] = {}

    with zipfile.ZipFile(zip_path) as zf, tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(zf.read(member))
        tmp.flush()
        with av.open(tmp.name) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    decoded[index] = frame.to_image()
                if len(decoded) == len(wanted):
                    break

    missing = [index for index in frame_indices if index not in decoded]
    if missing:
        raise RuntimeError(f"Failed to decode frame indices {missing} for {clip_id} {camera_name}")
    return [decoded[index] for index in frame_indices]
