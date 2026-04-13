"""Teacher cache request scaffolding and readiness helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.consistency import grade_action_pair
from src.data.local_dataset import CAMERA_NAMES
from src.data.parsers import action_record_from_text, critical_object_records_from_text
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


def extract_json_object(text: str | None) -> dict[str, Any] | None:
    """Best-effort JSON extraction from a model answer string."""
    if not text:
        return None
    candidate = str(text).strip()
    candidates = [candidate]
    left = candidate.find("{")
    right = candidate.rfind("}")
    if left != -1 and right != -1 and right > left:
        candidates.append(candidate[left : right + 1])
    for attempt in candidates:
        try:
            value = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def summarize_json_candidate(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize parse status and field coverage for a JSON payload."""
    if payload is None:
        return {
            "is_valid": False,
            "field_count": 0,
            "has_meta_action": False,
            "has_answer": False,
        }
    answer = payload.get("answer") or payload.get("final_answer")
    return {
        "is_valid": True,
        "field_count": len(payload),
        "has_meta_action": bool(payload.get("meta_action")),
        "has_answer": bool(answer),
    }


def selection_score_from_outputs(
    *,
    long_cot: str | None,
    json_payload: dict[str, Any] | None,
    meta_action: str | None,
    answer: str | None,
) -> float:
    """Score candidate teacher outputs for downstream use."""
    score = 0.0
    if long_cot:
        score += 0.25
    summary = summarize_json_candidate(json_payload)
    if summary["is_valid"]:
        score += 0.4
        score += min(summary["field_count"], 6) * 0.03
    if meta_action:
        score += 0.15
    if answer:
        score += 0.1
    return round(min(score, 1.0), 4)


def normalize_teacher_action_class(
    *,
    meta_action: str | None,
    answer: str | None,
    short_reason: str | None,
    long_cot: str | None,
) -> dict[str, Any]:
    """Derive a normalized teacher action-class record from available text fields."""
    for text, source in (
        (meta_action, "teacher_meta_action"),
        (answer, "teacher_answer"),
        (short_reason, "teacher_short_reason"),
        (long_cot, "teacher_long_cot"),
    ):
        record = action_record_from_text(text)
        if record and record["value"] != "unknown":
            record["source_field"] = source
            return record
    fallback = action_record_from_text(meta_action or answer or short_reason or long_cot)
    if fallback is None:
        return {
            "value": "unknown",
            "confidence": 0.0,
            "method": "keyword_parser_v1",
            "source_field": "missing",
        }
    fallback["source_field"] = "fallback"
    return fallback


def build_hallucination_flags(
    *,
    long_cot: str | None,
    json_payload: dict[str, Any] | None,
    meta_action: str | None,
    answer: str | None,
) -> list[str]:
    """Emit conservative heuristic flags for under-specified teacher outputs."""
    flags: list[str] = []
    if json_payload is None:
        flags.append("structured_json_parse_failed")
    if not answer:
        flags.append("answer_missing")
    elif not any(char.isalpha() for char in str(answer)):
        flags.append("answer_nonlinguistic")
    if not meta_action:
        flags.append("meta_action_missing")
    if long_cot and len(long_cot.split()) < 4:
        flags.append("cot_too_short")
    if not critical_object_records_from_text(long_cot or answer or ""):
        flags.append("critical_objects_missing")
    return flags


def pair_level_to_score(level: str | None) -> float:
    """Convert pair levels into a numeric consistency score."""
    mapping = {
        "pass": 1.0,
        "soft_pass": 0.75,
        "soft_fail": 0.25,
        "hard_fail": 0.0,
    }
    return mapping.get(str(level), 0.0)


def compare_teacher_to_reference(teacher_action: str | None, reference_action: str | None) -> dict[str, Any] | None:
    """Grade a teacher action class against a reference action class."""
    if not teacher_action or not reference_action:
        return None
    pair = grade_action_pair(teacher_action, reference_action)
    return {
        "teacher_action_class": pair.reason_action_class,
        "reference_action_class": pair.path_action_class,
        "consistency_level": pair.consistency_level,
        "consistency_score": pair_level_to_score(pair.consistency_level),
        "notes": pair.notes,
    }
