"""Reusable selection helpers for reasoning-subset manifests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


REQUIRED_CAMERA_COLUMNS = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]
REQUIRED_SENSOR_COLUMNS = REQUIRED_CAMERA_COLUMNS + ["egomotion"]


def parse_event_payload(raw_events: str | float | None) -> tuple[list[dict], str | None, list[int], list[int]]:
    """Parse the reasoning events column into normalized fields."""
    if raw_events is None or (isinstance(raw_events, float) and math.isnan(raw_events)):
        return [], None, [], []

    events = json.loads(raw_events)
    cots = [event.get("cot", "").strip() for event in events if event.get("cot")]
    keyframes = [
        int(event["event_start_frame"])
        for event in events
        if event.get("event_start_frame") is not None
    ]
    timestamps_us = [
        int(event["event_start_timestamp"])
        for event in events
        if event.get("event_start_timestamp") is not None
    ]
    human_coc = "\n".join(cots) if cots else None
    return events, human_coc, keyframes, timestamps_us


def load_merged_reasoning_table(project_root: Path) -> pd.DataFrame:
    """Merge reasoning rows with clip, collection, and presence metadata."""
    root = project_root / "data" / "raw" / "physical_ai_av"
    reasoning = pd.read_parquet(root / "reasoning" / "ood_reasoning.parquet").reset_index()
    reasoning.insert(0, "reasoning_row_id", range(len(reasoning)))

    clip_index = pd.read_parquet(root / "clip_index.parquet").reset_index()
    clip_index.insert(0, "clip_index_row_id", range(len(clip_index)))

    data_collection = pd.read_parquet(root / "metadata" / "data_collection.parquet").reset_index()
    data_collection.insert(0, "data_collection_row_id", range(len(data_collection)))

    presence_path = root / "metadata" / "feature_presence.parquet"
    if not presence_path.exists():
        presence_path = root / "metadata" / "sensor_presence.parquet"
    presence = pd.read_parquet(presence_path).reset_index()
    presence.insert(0, "feature_presence_row_id", range(len(presence)))

    merged = reasoning.merge(
        clip_index,
        on="clip_id",
        how="left",
        suffixes=("_reasoning", "_clip_index"),
    )
    merged = merged.merge(
        data_collection,
        on="clip_id",
        how="left",
        suffixes=("", "_data_collection"),
    )
    merged = merged.merge(
        presence,
        on="clip_id",
        how="left",
        suffixes=("", "_presence"),
    )

    merged["split"] = merged["split_reasoning"]
    merged["has_required_sensors"] = merged[REQUIRED_SENSOR_COLUMNS].fillna(False).all(axis=1)
    parsed = merged["events"].apply(parse_event_payload)
    merged["parsed_events"] = parsed.map(lambda item: item[0])
    merged["human_refined_coc"] = parsed.map(lambda item: item[1])
    merged["keyframes"] = parsed.map(lambda item: item[2])
    merged["keyframe_timestamps_us"] = parsed.map(lambda item: item[3])
    merged["events_present"] = merged["human_refined_coc"].notna()
    merged["num_events"] = merged["parsed_events"].map(len)
    merged["t0_us"] = merged["keyframe_timestamps_us"].map(
        lambda values: int(values[len(values) // 2]) if values else None
    )
    merged["primary_keyframe"] = merged["keyframes"].map(
        lambda values: int(values[len(values) // 2]) if values else None
    )
    return merged


def proportional_cluster_sample(frame: pd.DataFrame, target_count: int) -> pd.DataFrame:
    """Take a deterministic event-cluster-balanced subset."""
    if len(frame) <= target_count:
        return frame.sort_values("clip_id").copy()

    group_sizes = frame.groupby("event_cluster").size().sort_index()
    raw = group_sizes / group_sizes.sum() * target_count
    base = raw.apply(math.floor).astype(int)
    remainder = target_count - int(base.sum())
    fractions = (raw - base).sort_values(ascending=False)
    for cluster in fractions.index[:remainder]:
        base.loc[cluster] += 1

    selected_parts: list[pd.DataFrame] = []
    for cluster, count in base.items():
        cluster_rows = frame[frame["event_cluster"] == cluster].sort_values("clip_id")
        selected_parts.append(cluster_rows.head(count))

    selected = pd.concat(selected_parts, ignore_index=False)
    if len(selected) < target_count:
        remaining = frame.loc[~frame["clip_id"].isin(selected["clip_id"])].sort_values("clip_id")
        needed = target_count - len(selected)
        selected = pd.concat([selected, remaining.head(needed)], ignore_index=False)
    return selected.sort_values(["event_cluster", "clip_id"]).copy()
