"""Simple GT path semantics extraction for consistency checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.schema_versions import PATH_SEMANTICS_VERSION


@dataclass(slots=True)
class PathSemantics:
    action_class: str
    confidence: float
    total_displacement_m: float
    near_zero_speed_steps: int
    heading_delta_deg: float
    lateral_offset_delta_m: float
    first_stop_step: int | None
    reaccelerates_after_stop: bool
    initial_speed_mps: float
    final_speed_mps: float


def turn_direction_from_semantics(semantics: PathSemantics) -> dict[str, object]:
    """Convert path semantics into a compact turn-direction record."""
    if semantics.action_class == "left_turn":
        value = "left"
        confidence = semantics.confidence
    elif semantics.action_class == "right_turn":
        value = "right"
        confidence = semantics.confidence
    else:
        value = "straight_or_none"
        confidence = max(0.4, min(0.75, 1.0 - abs(semantics.heading_delta_deg) / 90.0))
    return {
        "value": value,
        "confidence": round(float(confidence), 4),
        "method": PATH_SEMANTICS_VERSION,
    }


def stop_profile_from_semantics(semantics: PathSemantics) -> dict[str, object]:
    """Convert path semantics into a compact stop-profile record."""
    if semantics.action_class == "stop":
        value = "full_stop"
        confidence = semantics.confidence
    elif semantics.action_class == "creep_then_go":
        value = "stop_then_go"
        confidence = semantics.confidence
    elif semantics.action_class == "creep":
        value = "creep"
        confidence = semantics.confidence
    elif semantics.action_class == "slow_down":
        value = "rolling_slow"
        confidence = semantics.confidence
    else:
        value = "no_stop_signature"
        confidence = 0.55
    return {
        "value": value,
        "confidence": round(float(confidence), 4),
        "method": PATH_SEMANTICS_VERSION,
    }


def _heading_degrees(vectors_xy: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(vectors_xy[:, 1], vectors_xy[:, 0]))


def extract_path_semantics(
    ego_future_xyz: np.ndarray,
    *,
    frequency_hz: int = 10,
) -> PathSemantics:
    """Infer a coarse action class from a local-frame future path."""
    xy = np.asarray(ego_future_xyz, dtype=np.float64)[:, :2]
    if len(xy) < 3:
        return PathSemantics("unknown", 0.0, 0.0, 0, 0.0, 0.0, None, False, 0.0, 0.0)

    deltas = np.diff(xy, axis=0)
    step_dist = np.linalg.norm(deltas, axis=1)
    speeds = step_dist * float(frequency_hz)
    total_displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    lateral_offset_delta = float(xy[-1, 1] - xy[0, 1])

    non_trivial = np.where(step_dist > 0.05)[0]
    if len(non_trivial) >= 2:
        heading = _heading_degrees(deltas[non_trivial])
        heading_delta = float(heading[-1] - heading[0])
    else:
        heading_delta = 0.0

    near_zero_mask = speeds < 0.3
    near_zero_steps = int(near_zero_mask.sum())
    first_stop_step = int(np.argmax(near_zero_mask)) if near_zero_mask.any() else None
    reaccelerates_after_stop = False
    if first_stop_step is not None and first_stop_step + 1 < len(speeds):
        reaccelerates_after_stop = bool(np.any(speeds[first_stop_step + 1 :] > 0.8))

    initial_speed = float(np.mean(speeds[: min(5, len(speeds))]))
    final_speed = float(np.mean(speeds[-min(5, len(speeds)) :]))

    action_class = "lane_keep"
    confidence = 0.55

    if near_zero_steps >= max(6, len(speeds) // 5):
        action_class = "creep_then_go" if reaccelerates_after_stop else "stop"
        confidence = 0.85 if action_class == "stop" else 0.8
    elif abs(heading_delta) >= 30.0:
        action_class = "left_turn" if heading_delta > 0 else "right_turn"
        confidence = 0.8
    elif abs(lateral_offset_delta) >= 2.0:
        action_class = "change_lane_left" if lateral_offset_delta > 0 else "change_lane_right"
        confidence = 0.75
    elif total_displacement < 4.0 and initial_speed < 1.5:
        action_class = "creep"
        confidence = 0.7
    elif final_speed < max(0.5, initial_speed * 0.5):
        action_class = "slow_down"
        confidence = 0.65

    return PathSemantics(
        action_class=action_class,
        confidence=confidence,
        total_displacement_m=total_displacement,
        near_zero_speed_steps=near_zero_steps,
        heading_delta_deg=heading_delta,
        lateral_offset_delta_m=lateral_offset_delta,
        first_stop_step=first_stop_step,
        reaccelerates_after_stop=reaccelerates_after_stop,
        initial_speed_mps=initial_speed,
        final_speed_mps=final_speed,
    )
