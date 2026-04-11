#!/usr/bin/env python3
"""Build a download-oriented 750/200/50 subset manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.selection import load_merged_reasoning_table, proportional_cluster_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--train-count", type=int, default=750)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument(
        "--require-human-coc",
        action="store_true",
        help="Restrict the subset to clips whose reasoning events include human CoC text.",
    )
    parser.add_argument(
        "--test-source",
        choices=("val", "train"),
        default="val",
        help="Where to carve the held-out test split from after train/val selection.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "download_subset_manifest.parquet",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "download_subset_summary.json",
    )
    return parser.parse_args()


def build_manifest_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Project merged rows into a download/materialization manifest."""
    manifest = source.copy()
    manifest["sample_id"] = manifest["clip_id"].map(lambda clip_id: f"{clip_id}__anchor0")
    manifest["clip_uuid"] = manifest["clip_id"]
    manifest["keyframes_json"] = manifest["keyframes"].map(json.dumps)
    manifest["keyframe_timestamps_us_json"] = manifest["keyframe_timestamps_us"].map(json.dumps)
    manifest["parsed_events_json"] = manifest["parsed_events"].map(json.dumps)
    manifest["camera_chunk_locator"] = manifest["chunk"].map(
        lambda chunk: f"camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_{int(chunk):04d}.*"
    )
    manifest["egomotion_locator"] = manifest["chunk"].map(
        lambda chunk: f"labels/egomotion/egomotion.chunk_{int(chunk):04d}.*"
    )
    return manifest[
        [
            "sample_id",
            "clip_uuid",
            "subset_split",
            "split",
            "event_cluster",
            "chunk",
            "t0_us",
            "primary_keyframe",
            "events_present",
            "num_events",
            "human_refined_coc",
            "keyframes_json",
            "keyframe_timestamps_us_json",
            "parsed_events_json",
            "reasoning_row_id",
            "clip_index_row_id",
            "feature_presence_row_id",
            "data_collection_row_id",
            "camera_chunk_locator",
            "egomotion_locator",
        ]
    ].reset_index(drop=True)


def main() -> None:
    args = parse_args()
    merged = load_merged_reasoning_table(args.project_root)
    base = merged[merged["clip_is_valid"].fillna(False) & merged["has_required_sensors"]].copy()
    if args.require_human_coc:
        base = base[base["events_present"]].copy()

    train_pool = base[base["split"] == "train"].copy()
    val_pool = base[base["split"] == "val"].copy()
    if len(train_pool) < args.train_count:
        raise SystemExit(f"Not enough train clips: requested {args.train_count}, found {len(train_pool)}")
    if len(val_pool) < args.val_count + args.test_count:
        raise SystemExit(
            f"Not enough val clips for val+test: requested {args.val_count + args.test_count}, found {len(val_pool)}"
        )

    selected_train = proportional_cluster_sample(train_pool, args.train_count).assign(subset_split="train")
    selected_val = proportional_cluster_sample(val_pool, args.val_count).assign(subset_split="val")

    if args.test_source == "val":
        test_pool = val_pool.loc[~val_pool["clip_id"].isin(selected_val["clip_id"])].copy()
    else:
        test_pool = train_pool.loc[~train_pool["clip_id"].isin(selected_train["clip_id"])].copy()

    if len(test_pool) < args.test_count:
        raise SystemExit(
            f"Not enough clips in test source `{args.test_source}`: "
            f"requested {args.test_count}, found {len(test_pool)}"
        )
    selected_test = proportional_cluster_sample(test_pool, args.test_count).assign(subset_split="test")

    manifest = build_manifest_frame(pd.concat([selected_train, selected_val, selected_test], ignore_index=False))
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.manifest_path, index=False)

    summary = {
        "counts": manifest["subset_split"].value_counts().to_dict(),
        "unique_chunks_total": int(manifest["chunk"].nunique()),
        "unique_chunks_by_split": manifest.groupby("subset_split")["chunk"].nunique().to_dict(),
        "events_present_by_split": manifest.groupby("subset_split")["events_present"].sum().astype(int).to_dict(),
        "event_clusters_by_split": manifest.groupby("subset_split")["event_cluster"].nunique().astype(int).to_dict(),
        "require_human_coc": args.require_human_coc,
        "test_source": args.test_source,
        "note": (
            "This subset is download-oriented. It keeps required sensors and official train/val provenance, "
            "and it can optionally enforce strict human-CoC coverage."
        ),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[done] manifest_path={args.manifest_path}")


if __name__ == "__main__":
    main()
