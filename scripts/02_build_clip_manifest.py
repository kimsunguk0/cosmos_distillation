#!/usr/bin/env python3
"""WP2 entrypoint: canonical clip manifest build."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


REQUIRED_CAMERA_COLUMNS = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]
REQUIRED_SENSOR_COLUMNS = REQUIRED_CAMERA_COLUMNS + ["egomotion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root for the cosmos_distillation workspace.",
    )
    parser.add_argument(
        "--user-clip-list",
        type=Path,
        default=PROJECT_ROOT / "inputs" / "selected_1400_clip_ids.txt",
        help="Optional user-provided train clip list.",
    )
    parser.add_argument("--train-count", type=int, default=1400)
    parser.add_argument(
        "--include-val",
        action="store_true",
        default=True,
        help="Keep the official validation split in the output manifest.",
    )
    parser.add_argument(
        "--no-include-val",
        dest="include_val",
        action="store_false",
        help="Exclude the validation split from the output manifest.",
    )
    parser.add_argument(
        "--require-human-coc",
        action="store_true",
        default=True,
        help="Require parseable human reasoning events for selection.",
    )
    parser.add_argument(
        "--allow-missing-human-coc",
        dest="require_human_coc",
        action="store_false",
        help="Allow clips with missing events to enter the manifest.",
    )
    parser.add_argument(
        "--allow-short-train-count",
        action="store_true",
        help="Allow fewer than --train-count train clips when strict filtering removes too many.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "clip_manifest.parquet",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "manifest_summary.json",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "data_summary.md",
    )
    return parser.parse_args()


def read_clip_list(path: Path) -> list[str]:
    """Read a newline-separated clip list while ignoring comments and blanks."""
    if not path.exists():
        return []
    items: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(line)
    return items


def parse_event_payload(raw_events: str | float | None) -> tuple[list[dict], str | None, list[int], list[int]]:
    """Parse event payloads into joined human CoC and keyframe fields."""
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


def load_merged_table(project_root: Path) -> pd.DataFrame:
    """Load and merge reasoning, clip index, collection metadata, and sensor presence."""
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
        suffixes=("", "_clip_index"),
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


def chunk_locator(chunk_id: int, prefix: str, stem: str) -> str:
    """Build a human-readable chunk locator pattern."""
    return f"{prefix}/{stem}/{stem}.chunk_{int(chunk_id):04d}.*"


def build_manifest_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Project merged candidate rows into the canonical manifest schema."""
    manifest = source.copy()
    manifest["sample_id"] = manifest["clip_id"].map(lambda clip_id: f"{clip_id}__anchor0")
    manifest["clip_uuid"] = manifest["clip_id"]
    manifest["camera_chunk_locator"] = manifest["chunk"].map(
        lambda chunk_id: chunk_locator(chunk_id, "camera", "camera_front_wide_120fov")
    )
    manifest["timestamp_locator"] = manifest["chunk"].map(
        lambda chunk_id: chunk_locator(chunk_id, "camera", "camera_front_wide_120fov")
    )
    manifest["egomotion_locator"] = manifest["chunk"].map(
        lambda chunk_id: chunk_locator(chunk_id, "labels", "egomotion")
    )
    manifest["keyframes_json"] = manifest["keyframes"].map(json.dumps)
    manifest["keyframe_timestamps_us_json"] = manifest["keyframe_timestamps_us"].map(json.dumps)
    manifest["parsed_events_json"] = manifest["parsed_events"].map(json.dumps)

    keep_columns = [
        "sample_id",
        "clip_uuid",
        "split",
        "event_cluster",
        "t0_us",
        "primary_keyframe",
        "num_events",
        "human_refined_coc",
        "keyframes_json",
        "keyframe_timestamps_us_json",
        "parsed_events_json",
        "reasoning_row_id",
        "clip_index_row_id",
        "feature_presence_row_id",
        "data_collection_row_id",
        "chunk",
        "camera_chunk_locator",
        "timestamp_locator",
        "egomotion_locator",
        "clip_is_valid",
        "has_required_sensors",
        "country",
        "month",
        "hour_of_day",
        "platform_class",
        "radar_config",
    ]
    return manifest[keep_columns].reset_index(drop=True)


def write_summary(
    args: argparse.Namespace,
    *,
    merged: pd.DataFrame,
    train_candidates: pd.DataFrame,
    selected_train: pd.DataFrame,
    selected_val: pd.DataFrame,
    excluded_train: pd.DataFrame,
) -> None:
    """Write machine-readable and markdown summary artifacts."""
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "strict_human_coc": args.require_human_coc,
        "requested_train_count": args.train_count,
        "available_reasoning_rows": int(len(merged)),
        "available_train_rows": int((merged["split"] == "train").sum()),
        "available_val_rows": int((merged["split"] == "val").sum()),
        "available_train_candidates_after_filters": int(len(train_candidates)),
        "selected_train_count": int(len(selected_train)),
        "selected_val_count": int(len(selected_val)),
        "excluded_train_count": int(len(excluded_train)),
        "events_present_total": int(merged["events_present"].sum()),
        "events_present_train": int(((merged["split"] == "train") & merged["events_present"]).sum()),
        "events_present_val": int(((merged["split"] == "val") & merged["events_present"]).sum()),
        "required_sensor_coverage_train": int(
            ((merged["split"] == "train") & merged["has_required_sensors"]).sum()
        ),
        "required_sensor_coverage_val": int(
            ((merged["split"] == "val") & merged["has_required_sensors"]).sum()
        ),
        "selected_train_event_cluster_counts": selected_train["event_cluster"].value_counts().to_dict(),
        "selected_val_event_cluster_counts": selected_val["event_cluster"].value_counts().to_dict(),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Data Summary",
        "",
        f"- Strict human CoC required: `{args.require_human_coc}`",
        f"- Requested train clips: `{args.train_count}`",
        f"- Available reasoning rows: `{len(merged)}`",
        f"- Train candidates after filters: `{len(train_candidates)}`",
        f"- Selected train clips: `{len(selected_train)}`",
        f"- Selected val clips: `{len(selected_val)}`",
        f"- Excluded train clips: `{len(excluded_train)}`",
        f"- Events present total: `{int(merged['events_present'].sum())}`",
        f"- Events present train: `{int(((merged['split'] == 'train') & merged['events_present']).sum())}`",
        f"- Events present val: `{int(((merged['split'] == 'val') & merged['events_present']).sum())}`",
    ]
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    merged = load_merged_table(args.project_root)

    filtered = merged[merged["clip_is_valid"].fillna(False) & merged["has_required_sensors"]].copy()
    if args.require_human_coc:
        filtered = filtered[filtered["events_present"]].copy()

    train_candidates = filtered[filtered["split"] == "train"].copy()
    val_candidates = filtered[filtered["split"] == "val"].copy()

    explicit_train_ids = read_clip_list(args.user_clip_list)
    if explicit_train_ids:
        selected_train = train_candidates[train_candidates["clip_id"].isin(explicit_train_ids)].copy()
        missing_ids = sorted(set(explicit_train_ids) - set(selected_train["clip_id"]))
        if missing_ids:
            raise SystemExit(
                f"User clip list contains {len(missing_ids)} ids not present after filters, "
                f"for example: {missing_ids[:5]}"
            )
    else:
        if len(train_candidates) < args.train_count and not args.allow_short_train_count:
            write_summary(
                args,
                merged=merged,
                train_candidates=train_candidates,
                selected_train=train_candidates,
                selected_val=val_candidates if args.include_val else val_candidates.iloc[0:0],
                excluded_train=train_candidates.iloc[0:0],
            )
            raise SystemExit(
                "Not enough filtered train clips to satisfy the requested train count. "
                f"Requested {args.train_count}, found {len(train_candidates)} after filters. "
                "Re-run with --allow-short-train-count to materialize an audit manifest."
            )

        target_count = min(args.train_count, len(train_candidates))
        selected_train = proportional_cluster_sample(train_candidates, target_count)

    selected_val = val_candidates.copy() if args.include_val else val_candidates.iloc[0:0]
    excluded_train = train_candidates.loc[
        ~train_candidates["clip_id"].isin(selected_train["clip_id"])
    ].copy()

    manifest = build_manifest_frame(pd.concat([selected_train, selected_val], ignore_index=False))
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.manifest_path, index=False)

    (args.project_root / "inputs" / "selected_1400_samples.parquet").parent.mkdir(
        parents=True, exist_ok=True
    )
    build_manifest_frame(selected_train).to_parquet(
        args.project_root / "inputs" / "selected_1400_samples.parquet",
        index=False,
    )
    (args.project_root / "inputs" / "selected_1400_clip_ids.txt").write_text(
        "".join(f"{clip_id}\n" for clip_id in selected_train["clip_id"].tolist()),
        encoding="utf-8",
    )

    write_summary(
        args,
        merged=merged,
        train_candidates=train_candidates,
        selected_train=selected_train,
        selected_val=selected_val,
        excluded_train=excluded_train,
    )
    print(f"[done] manifest_path={args.manifest_path}")
    print(f"[done] train_selected={len(selected_train)} val_selected={len(selected_val)}")


if __name__ == "__main__":
    main()
