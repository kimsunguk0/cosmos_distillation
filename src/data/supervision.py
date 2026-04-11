"""Supervision-record builders for hard human and weak derived fields."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.data.parsers import action_record_from_text, critical_object_records_from_text


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


def build_weak_derived_record(row: pd.Series) -> dict[str, Any]:
    """Build weak labels derived from human CoC text only."""
    human_coc = row.get("human_refined_coc")
    meta_action = action_record_from_text(human_coc)
    critical_objects = critical_object_records_from_text(human_coc)
    return {
        "meta_action_from_human": meta_action,
        "action_class_from_gt_path": None,
        "turn_direction_from_gt_path": None,
        "stop_profile_from_gt_path": None,
        "critical_objects_from_human_parser": critical_objects,
    }


def build_supervision_record(row: pd.Series) -> dict[str, Any]:
    """Build the supervision payload for a manifest row."""
    return {
        "sample_id": row["sample_id"],
        "provenance": {
            "hard": "human",
            "weak": "derived_from_human_text",
        },
        "hard_human": build_hard_human_record(row),
        "weak_derived": build_weak_derived_record(row),
    }
