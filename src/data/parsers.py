"""Lightweight text parsers used before richer implementations land."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ExtractedAction:
    value: str
    confidence: float
    method: str


@dataclass(slots=True)
class ExtractedObject:
    object_type: str
    why: str
    confidence: float
    method: str


def extract_action_from_text(text: str) -> ExtractedAction:
    """Very small keyword-based parser for bootstrap tests."""
    lowered = text.lower()
    if (
        "nudge left" in lowered
        or "slight left" in lowered
        or "slightly left" in lowered
        or "move left" in lowered
        or "shift left" in lowered
        or "veer left" in lowered
    ):
        return ExtractedAction("nudge_left", 0.82, "keyword_parser_v2")
    if (
        "nudge right" in lowered
        or "slight right" in lowered
        or "slightly right" in lowered
        or "move right" in lowered
        or "shift right" in lowered
        or "veer right" in lowered
    ):
        return ExtractedAction("nudge_right", 0.82, "keyword_parser_v2")
    if (
        ("blocked" in lowered or "blocking" in lowered or "encroach" in lowered or "encroaching" in lowered)
        and "right" in lowered
        and any(token in lowered for token in ("left", "avoid", "around", "nudge", "shift", "move"))
    ):
        return ExtractedAction("nudge_left", 0.88, "keyword_parser_v2")
    if (
        ("blocked" in lowered or "blocking" in lowered or "encroach" in lowered or "encroaching" in lowered)
        and "left" in lowered
        and any(token in lowered for token in ("right", "avoid", "around", "nudge", "shift", "move"))
    ):
        return ExtractedAction("nudge_right", 0.88, "keyword_parser_v2")
    if "lane change" in lowered and "left" in lowered:
        return ExtractedAction("change_lane_left", 0.8, "keyword_parser_v1")
    if "lane change" in lowered and "right" in lowered:
        return ExtractedAction("change_lane_right", 0.8, "keyword_parser_v1")
    if "left" in lowered and "turn" in lowered:
        return ExtractedAction("left_turn", 0.7, "keyword_parser_v1")
    if "right" in lowered and "turn" in lowered:
        return ExtractedAction("right_turn", 0.7, "keyword_parser_v1")
    if "follow" in lowered and "lead vehicle" in lowered:
        return ExtractedAction("follow_lead", 0.75, "keyword_parser_v1")
    if (
        "keep lane" in lowered
        or "stay in lane" in lowered
        or "maintain lane" in lowered
        or ("lane is clear" in lowered and "ahead" in lowered)
        or ("no lead vehicle" in lowered and "ahead" in lowered)
        or ("lane is clear" in lowered and "no lead vehicle" in lowered)
    ):
        return ExtractedAction("keep_lane", 0.86, "keyword_parser_v2")
    if "creep" in lowered:
        return ExtractedAction("creep", 0.7, "keyword_parser_v1")
    if "yield" in lowered:
        return ExtractedAction("yield", 0.7, "keyword_parser_v1")
    if "stop" in lowered:
        return ExtractedAction("stop", 0.7, "keyword_parser_v1")
    if "slow" in lowered:
        return ExtractedAction("slow_down", 0.6, "keyword_parser_v1")
    return ExtractedAction("unknown", 0.0, "keyword_parser_v1")


def extract_critical_objects_from_text(text: str) -> list[ExtractedObject]:
    """Find a small set of scene-critical objects from human or teacher text."""
    lowered = text.lower()
    results: list[ExtractedObject] = []

    keyword_map: list[tuple[str, str, str, float]] = [
        ("pedestrian", "pedestrian", "Pedestrians are mentioned as a decision driver.", 0.85),
        ("cyclist", "cyclist", "Cyclists or micromobility actors are mentioned.", 0.8),
        ("lead vehicle", "lead_vehicle", "The lead vehicle influences ego behavior.", 0.8),
        ("car ahead", "lead_vehicle", "A vehicle ahead influences ego behavior.", 0.75),
        ("stop sign", "stop_sign", "A stop sign is explicitly referenced.", 0.9),
        ("construction", "construction_zone", "Construction or work-zone context is referenced.", 0.75),
        ("work zone", "construction_zone", "Construction or work-zone context is referenced.", 0.75),
        ("traffic", "traffic", "Other traffic is mentioned as a decision factor.", 0.6),
    ]
    seen: set[str] = set()
    for needle, object_type, why, confidence in keyword_map:
        if needle in lowered and object_type not in seen:
            results.append(
                ExtractedObject(
                    object_type=object_type,
                    why=why,
                    confidence=confidence,
                    method="keyword_object_parser_v1",
                )
            )
            seen.add(object_type)
    return results


def action_record_from_text(text: str | None) -> dict[str, Any] | None:
    """Return a serializable action record or None when text is missing."""
    if not text:
        return None
    parsed = extract_action_from_text(text)
    return {"value": parsed.value, "confidence": parsed.confidence, "method": parsed.method}


def critical_object_records_from_text(text: str | None) -> list[dict[str, Any]]:
    """Return serializable critical-object records."""
    if not text:
        return []
    return [
        {
            "type": item.object_type,
            "why": item.why,
            "confidence": item.confidence,
            "method": item.method,
        }
        for item in extract_critical_objects_from_text(text)
    ]
