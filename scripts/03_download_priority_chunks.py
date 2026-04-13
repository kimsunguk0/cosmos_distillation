#!/usr/bin/env python3
"""Download chunks in a teacher-ready priority order."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.download_priority import (
    REQUIRED_FEATURES,
    build_priority_plan,
    local_feature_dir,
    source_chunk_path,
)
from src.data.hf_download import DEFAULT_REPO_ID, existing_cache_root, metadata_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "strict_download_subset_manifest.parquet",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--anchor-feature", default="camera_cross_left_120fov")
    parser.add_argument(
        "--preferred-features",
        nargs="+",
        default=[
            "camera_front_wide_120fov",
            "camera_cross_right_120fov",
            "camera_front_tele_30fov",
            "egomotion",
        ],
    )
    parser.add_argument("--target-chunks", type=int, default=100)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--xet-high-performance", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "priority_download_plan.json",
    )
    return parser.parse_args()


def materialize_from_cache(src: Path, dst: Path, link_mode: str) -> None:
    """Materialize a file from the shared cache into the local output root."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if link_mode == "symlink":
        dst.symlink_to(src)
    elif link_mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {link_mode}")


def estimate_hours(num_bytes: int, mb_per_sec: float) -> float:
    """Convert bytes and throughput into hours."""
    return num_bytes / (mb_per_sec * 1_000_000) / 3600.0


def main() -> None:
    args = parse_args()
    manifest = pd.read_parquet(args.manifest_path)
    output_root = (args.output_dir or metadata_root(PROJECT_ROOT)).resolve()
    cache_root = args.cache_root.resolve() if args.cache_root else existing_cache_root()

    plan = build_priority_plan(
        manifest,
        output_root=output_root,
        cache_root=cache_root,
        anchor_feature=args.anchor_feature,
        preferred_features=args.preferred_features,
        target_chunks=args.target_chunks,
    )

    linked_from_cache = {feature: 0 for feature in args.preferred_features}
    remaining_patterns: list[str] = []
    for chunk in plan.selected_chunks:
        for feature in args.preferred_features:
            dst = local_feature_dir(output_root, feature) / f"{feature}.chunk_{int(chunk):04d}.zip"
            if dst.exists() or dst.is_symlink():
                continue
            src = source_chunk_path(cache_root, feature, chunk) if cache_root else None
            if src is not None:
                dst = local_feature_dir(output_root, feature) / src.name
                materialize_from_cache(src, dst, args.link_mode)
                linked_from_cache[feature] += 1
            else:
                remaining_patterns.append(f"{'camera' if feature.startswith('camera') else 'labels'}/{feature}/{feature}.chunk_{chunk:04d}.*")

    summary = {
        "manifest_path": str(args.manifest_path),
        "output_root": str(output_root),
        "cache_root": str(cache_root) if cache_root else None,
        "anchor_feature": args.anchor_feature,
        "preferred_features": args.preferred_features,
        "target_chunks": int(args.target_chunks),
        "max_workers": int(args.max_workers),
        "xet_high_performance": bool(args.xet_high_performance),
        "ready_chunk_count_now": plan.ready_chunk_count_now,
        "ready_sample_count_now": plan.ready_sample_count_now,
        "selected_chunk_count": len(plan.selected_chunks),
        "selected_sample_count_if_completed": plan.selected_sample_count,
        "selected_chunks": plan.selected_chunks,
        "selected_missing_counts": plan.selected_missing_counts,
        "linked_from_cache": linked_from_cache,
        "remaining_file_patterns": len(remaining_patterns),
        "estimated_remaining_bytes": plan.estimated_selected_missing_bytes,
        "estimated_remaining_gb": round(plan.estimated_selected_missing_bytes / 1_000_000_000, 2),
        "estimated_hours": {
            "50_MBps": round(estimate_hours(plan.estimated_selected_missing_bytes, 50.0), 2),
            "100_MBps": round(estimate_hours(plan.estimated_selected_missing_bytes, 100.0), 2),
            "200_MBps": round(estimate_hours(plan.estimated_selected_missing_bytes, 200.0), 2),
        },
        "top_chunk_table_preview": plan.chunk_table.head(20).to_dict(orient="records"),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.estimate_only or not remaining_patterns:
        return

    if args.xet_high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output_root),
        allow_patterns=remaining_patterns,
        max_workers=args.max_workers,
    )
    print("[done] priority chunk download complete")


if __name__ == "__main__":
    main()
