"""Supervision-record builders for hard human and weak derived fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.parsers import action_record_from_text, critical_object_records_from_text
from src.data.path_semantics import (
    extract_path_semantics,
    stop_profile_from_semantics,
    turn_direction_from_semantics,
)
from src.data.schema_versions import CANONICAL_LOCALIZATION_VERSION, PATH_SEMANTICS_VERSION


def _json_list(raw_value: str | None) -> list[Any]:
    if not raw_value:
        return []
    return json.loads(raw_value)


def build_hard_human_record(row: pd.Series) -> dict[str, Any]:
    """Build the hard-human section for a sample."""
    return {
        "clip_uuid": row["clip_uuid"],
        "event_cluster": row["event_cluster"],
        "human_coc": row.get("human_refined_coc"),
        "keyframes": _json_list(row.get("keyframes_json")),
        "keyframe_timestamps_us": _json_list(row.get("keyframe_timestamps_us_json")),
        "ego_future_path_path": f"data/processed/canonical_samples/{row['sample_id']}/ego_future_xyz.npy",
    }


def _gt_path_records(sample_id: str, canonical_root: Path) -> dict[str, Any]:
    sample_dir = canonical_root / sample_id
    future_path = sample_dir / "ego_future_xyz.npy"
    sample_meta_path = sample_dir / "sample_meta.json"
    if not future_path.exists() or not sample_meta_path.exists():
        return {
            "action_class_from_gt_path": None,
            "turn_direction_from_gt_path": None,
            "stop_profile_from_gt_path": None,
        }
    sample_meta = json.loads(sample_meta_path.read_text(encoding="utf-8"))
    if sample_meta.get("canonical_localization_version") != CANONICAL_LOCALIZATION_VERSION:
        return {
            "action_class_from_gt_path": None,
            "turn_direction_from_gt_path": None,
            "stop_profile_from_gt_path": None,
        }
    semantics = extract_path_semantics(np.load(future_path))
    return {
        "action_class_from_gt_path": {
            "value": semantics.action_class,
            "confidence": round(float(semantics.confidence), 4),
            "method": PATH_SEMANTICS_VERSION,
            "metrics": {
                "total_displacement_m": round(float(semantics.total_displacement_m), 4),
                "near_zero_speed_steps": int(semantics.near_zero_speed_steps),
                "heading_delta_deg": round(float(semantics.heading_delta_deg), 4),
                "lateral_offset_delta_m": round(float(semantics.lateral_offset_delta_m), 4),
                "first_stop_step": semantics.first_stop_step,
                "reaccelerates_after_stop": bool(semantics.reaccelerates_after_stop),
                "initial_speed_mps": round(float(semantics.initial_speed_mps), 4),
                "final_speed_mps": round(float(semantics.final_speed_mps), 4),
            },
        },
        "turn_direction_from_gt_path": turn_direction_from_semantics(semantics),
        "stop_profile_from_gt_path": stop_profile_from_semantics(semantics),
    }


def build_weak_derived_record(row: pd.Series, canonical_root: Path) -> dict[str, Any]:
    """Build weak labels derived from human CoC text and GT future path semantics."""
    human_coc = row.get("human_refined_coc")
    meta_action = action_record_from_text(human_coc)
    critical_objects = critical_object_records_from_text(human_coc)
    gt_records = _gt_path_records(str(row["sample_id"]), canonical_root)
    return {
        "meta_action_from_human": meta_action,
        "action_class_from_gt_path": gt_records["action_class_from_gt_path"],
        "turn_direction_from_gt_path": gt_records["turn_direction_from_gt_path"],
        "stop_profile_from_gt_path": gt_records["stop_profile_from_gt_path"],
        "critical_objects_from_human_parser": critical_objects,
    }


def build_supervision_record(row: pd.Series, canonical_root: Path) -> dict[str, Any]:
    """Build the supervision payload for a manifest row."""
    return {
        "sample_id": row["sample_id"],
        "provenance": {
            "hard": "human",
            "weak": "derived_from_human_text",
            "path_semantics_version": PATH_SEMANTICS_VERSION,
        },
        "hard_human": build_hard_human_record(row),
        "weak_derived": build_weak_derived_record(row, canonical_root),
    }
