"""Consistency gate helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.path_semantics import extract_path_semantics


ACTION_CLASSES = (
    "lane_keep",
    "slow_down",
    "stop",
    "yield",
    "follow_lead",
    "left_turn",
    "right_turn",
    "creep",
    "creep_then_go",
    "change_lane_left",
    "change_lane_right",
    "parked_or_blocked_wait",
    "unknown",
)

SOFT_PASS_PAIRS = {
    ("yield", "creep_then_go"),
    ("creep_then_go", "yield"),
}

HARD_FAIL_PAIRS = {
    ("stop", "lane_keep"),
    ("lane_keep", "stop"),
    ("right_turn", "left_turn"),
    ("left_turn", "right_turn"),
}


@dataclass(slots=True)
class PairCheck:
    reason_action_class: str
    path_action_class: str
    consistency_level: str
    is_consistent: bool
    notes: list[str]


def normalize_action_class(value: str | None) -> str:
    """Map an action class into the supported taxonomy."""
    if not value:
        return "unknown"
    value = value.strip().lower()
    return value if value in ACTION_CLASSES else "unknown"


def grade_action_pair(reason_action_class: str, path_action_class: str) -> PairCheck:
    """Grade a pair using a conservative default policy."""
    left = normalize_action_class(reason_action_class)
    right = normalize_action_class(path_action_class)

    if left == right:
        return PairCheck(left, right, "pass", True, [])
    if (left, right) in SOFT_PASS_PAIRS:
        return PairCheck(left, right, "soft_pass", True, [])
    if (left, right) in HARD_FAIL_PAIRS:
        return PairCheck(
            left,
            right,
            "hard_fail",
            False,
            [f"reason={left} conflicts with path={right}"],
        )
    return PairCheck(
        left,
        right,
        "soft_fail",
        False,
        [f"reason={left} does not cleanly align with path={right}"],
    )


def load_gt_path_action_class(sample_root: Path, sample_id: str) -> tuple[str | None, dict | None]:
    """Load the GT path action class from a materialized canonical sample if it exists."""
    future_path = sample_root / sample_id / "ego_future_xyz.npy"
    if not future_path.exists():
        return None, None
    semantics = extract_path_semantics(np.load(future_path))
    return semantics.action_class, {
        "confidence": semantics.confidence,
        "total_displacement_m": semantics.total_displacement_m,
        "near_zero_speed_steps": semantics.near_zero_speed_steps,
        "heading_delta_deg": semantics.heading_delta_deg,
        "lateral_offset_delta_m": semantics.lateral_offset_delta_m,
        "first_stop_step": semantics.first_stop_step,
        "reaccelerates_after_stop": semantics.reaccelerates_after_stop,
        "initial_speed_mps": semantics.initial_speed_mps,
        "final_speed_mps": semantics.final_speed_mps,
    }


def summarize_pair_levels(values: pd.Series) -> dict[str, int]:
    """Summarize pair-check labels including missing values."""
    counter = Counter("missing" if pd.isna(value) else str(value) for value in values)
    return dict(counter)
