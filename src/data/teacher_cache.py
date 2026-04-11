"""Teacher cache request scaffolding and readiness helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.local_dataset import CAMERA_NAMES
from src.data.teacher_prompts import get_prompt


CAMERA_NAME_TO_INDEX = {
    "camera_cross_left_120fov": 0,
    "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_front_tele_30fov": 6,
}


@dataclass(slots=True)
class TeacherSampleReadiness:
    sample_id: str
    status: str
    blockers: list[str] = field(default_factory=list)
    sample_dir: Path | None = None
    frame_paths: dict[str, list[str]] = field(default_factory=dict)
    sample_meta: dict[str, Any] = field(default_factory=dict)


def load_jsonl_by_key(path: Path, *, key: str = "sample_id") -> dict[str, dict[str, Any]]:
    """Load a JSONL file into a dict keyed by the requested field."""
    result: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            value = record.get(key)
            if value is None:
                continue
            result[str(value)] = record
    return result


def prompt_bundle_for_sample(
    *,
    sample_id: str,
    canonical_sample_path: Path,
    prompt_names: list[str],
    question: str,
) -> dict[str, Any]:
    """Build a teacher request bundle without leaking hard labels into the prompt."""
    sample_meta_path = canonical_sample_path / "sample_meta.json"
    sample_meta = json.loads(sample_meta_path.read_text(encoding="utf-8")) if sample_meta_path.exists() else {}

    return {
        "sample_id": sample_id,
        "canonical_sample_path": str(canonical_sample_path),
        "camera_names": list(CAMERA_NAMES),
        "camera_indices": [CAMERA_NAME_TO_INDEX[name] for name in CAMERA_NAMES],
        "question": question,
        "prompt_families": [
            {
                "name": prompt_name,
                "prompt": get_prompt(prompt_name),
            }
            for prompt_name in prompt_names
        ],
        "inputs": {
            "ego_history_xyz_path": str(canonical_sample_path / "ego_history_xyz.npy"),
            "ego_history_rot_path": str(canonical_sample_path / "ego_history_rot.npy"),
            "ego_future_xyz_path": str(canonical_sample_path / "ego_future_xyz.npy"),
            "ego_future_rot_path": str(canonical_sample_path / "ego_future_rot.npy"),
            "relative_timestamps_path": str(canonical_sample_path / "rel_timestamps.json"),
            "absolute_timestamps_path": str(canonical_sample_path / "abs_timestamps.json"),
            "frame_dir": str(canonical_sample_path / "frames"),
        },
        "policy": {
            "teacher_is_not_gt": True,
            "reasoning_gt_pairing_forbidden_in_v1": True,
            "trajectory_target_disabled_in_v1": True,
        },
        "sample_meta": sample_meta,
    }


def _frame_paths_for_sample(sample_dir: Path, frame_offsets_sec: list[float]) -> dict[str, list[str]]:
    frame_dir = sample_dir / "frames"
    result: dict[str, list[str]] = {}
    for camera_name in CAMERA_NAMES:
        camera_paths = [
            str(frame_dir / f"{camera_name}_t{float(offset):+.1f}.jpg") for offset in frame_offsets_sec
        ]
        result[camera_name] = camera_paths
    return result


def inspect_teacher_sample(sample_id: str, canonical_root: Path) -> TeacherSampleReadiness:
    """Inspect whether a canonical sample is ready for teacher generation."""
    sample_dir = canonical_root / sample_id
    if not sample_dir.exists():
        return TeacherSampleReadiness(
            sample_id=sample_id,
            status="awaiting_canonical_sample",
            blockers=["canonical_sample_missing"],
            sample_dir=sample_dir,
        )

    sample_meta_path = sample_dir / "sample_meta.json"
    if not sample_meta_path.exists():
        return TeacherSampleReadiness(
            sample_id=sample_id,
            status="awaiting_sample_meta",
            blockers=["sample_meta_missing"],
            sample_dir=sample_dir,
        )

    sample_meta = json.loads(sample_meta_path.read_text(encoding="utf-8"))
    frame_offsets_sec = list(sample_meta.get("frame_offsets_sec", [-0.3, -0.2, -0.1, 0.0]))
    frame_paths = _frame_paths_for_sample(sample_dir, frame_offsets_sec)

    blockers: list[str] = []
    required_files = [
        sample_dir / "ego_history_xyz.npy",
        sample_dir / "ego_history_rot.npy",
        sample_dir / "ego_future_xyz.npy",
        sample_dir / "ego_future_rot.npy",
        sample_dir / "rel_timestamps.json",
        sample_dir / "abs_timestamps.json",
    ]
    if any(not path.exists() for path in required_files):
        blockers.append("trajectory_or_timestamp_artifacts_missing")

    decoder_status = sample_meta.get("decoder_status", {})
    missing_frames: list[str] = []
    for camera_name, camera_paths in frame_paths.items():
        if any(not Path(path).exists() for path in camera_paths):
            missing_frames.append(camera_name)
        elif str(decoder_status.get(camera_name, "")).startswith("failed:"):
            blockers.append(f"decoder_failed:{camera_name}")

    if missing_frames:
        blockers.append("image_frames_missing")

    status = "ready" if not blockers else "blocked"
    return TeacherSampleReadiness(
        sample_id=sample_id,
        status=status,
        blockers=blockers,
        sample_dir=sample_dir,
        frame_paths=frame_paths,
        sample_meta=sample_meta,
    )


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist a JSON payload with parent creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
