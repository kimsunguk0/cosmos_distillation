#!/usr/bin/env python3
"""Download or reuse the camera and egomotion chunks needed for a subset manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.hf_download import DEFAULT_REPO_ID, existing_cache_root, metadata_root


REQUIRED_FEATURES = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
    "egomotion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "download_subset_manifest.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dataset root that already contains metadata. Defaults to data/raw/physical_ai_av.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent file-download workers passed to snapshot_download.",
    )
    parser.add_argument(
        "--xet-high-performance",
        action="store_true",
        help="Enable HF_XET_HIGH_PERFORMANCE=1 for higher download parallelism.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "download_plan_summary.json",
    )
    return parser.parse_args()


def feature_prefix(feature: str) -> str:
    return "camera" if feature.startswith("camera") else "labels"


def allow_pattern(feature: str, chunk: int) -> str:
    prefix = feature_prefix(feature)
    return f"{prefix}/{feature}/{feature}.chunk_{int(chunk):04d}.*"


def local_feature_dir(output_root: Path, feature: str) -> Path:
    return output_root / feature_prefix(feature) / feature


def source_chunk_path(cache_root: Path, feature: str, chunk: int) -> Path | None:
    matches = list((cache_root / feature_prefix(feature) / feature).glob(f"{feature}.chunk_{int(chunk):04d}.*"))
    return matches[0] if matches else None


def materialize_from_cache(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if link_mode == "symlink":
        dst.symlink_to(src)
    elif link_mode == "copy":
        import shutil

        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {link_mode}")


def average_feature_sizes(cache_root: Path) -> dict[str, int]:
    """Estimate bytes per chunk by averaging the local cached files for each feature."""
    averages: dict[str, int] = {}
    for feature in REQUIRED_FEATURES:
        files = sorted((cache_root / feature_prefix(feature) / feature).glob(f"{feature}.chunk_*"))
        sample = files[: min(20, len(files))]
        if not sample:
            averages[feature] = 0
            continue
        averages[feature] = int(sum(path.stat().st_size for path in sample) / len(sample))
    return averages


def format_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1_000_000_000:.2f} GB"


def estimate_hours(num_bytes: int, mb_per_sec: float) -> float:
    return num_bytes / (mb_per_sec * 1_000_000) / 3600.0


def main() -> None:
    args = parse_args()
    output_root = (args.output_dir or metadata_root(args.project_root)).resolve()
    manifest = pd.read_parquet(args.manifest_path)
    chunks = sorted({int(chunk) for chunk in manifest["chunk"].unique()})
    cache_root = args.cache_root.resolve() if args.cache_root else existing_cache_root()

    linked_counts = {feature: 0 for feature in REQUIRED_FEATURES}
    missing_counts = {feature: 0 for feature in REQUIRED_FEATURES}
    missing_patterns: list[str] = []

    for feature in REQUIRED_FEATURES:
        for chunk in chunks:
            src = source_chunk_path(cache_root, feature, chunk) if cache_root else None
            if src is not None:
                dst = local_feature_dir(output_root, feature) / src.name
                materialize_from_cache(src, dst, args.link_mode)
                linked_counts[feature] += 1
            else:
                missing_patterns.append(allow_pattern(feature, chunk))
                missing_counts[feature] += 1

    averages = average_feature_sizes(cache_root) if cache_root else {feature: 0 for feature in REQUIRED_FEATURES}
    estimated_missing_bytes = sum(missing_counts[feature] * averages[feature] for feature in REQUIRED_FEATURES)

    summary = {
        "manifest_path": str(args.manifest_path),
        "output_root": str(output_root),
        "cache_root": str(cache_root) if cache_root else None,
        "max_workers": int(args.max_workers),
        "xet_high_performance": bool(args.xet_high_performance),
        "unique_chunks": len(chunks),
        "required_features": REQUIRED_FEATURES,
        "linked_counts": linked_counts,
        "missing_counts": missing_counts,
        "missing_file_patterns": len(missing_patterns),
        "estimated_missing_bytes": estimated_missing_bytes,
        "estimated_missing_gb": round(estimated_missing_bytes / 1_000_000_000, 2),
        "estimated_hours": {
            "50_MBps": round(estimate_hours(estimated_missing_bytes, 50.0), 2),
            "100_MBps": round(estimate_hours(estimated_missing_bytes, 100.0), 2),
            "200_MBps": round(estimate_hours(estimated_missing_bytes, 200.0), 2),
        },
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.estimate_only:
        return

    if not missing_patterns:
        print("[done] all required files already available locally")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required to download chunk files") from exc

    print(
        "[download] estimated_missing="
        f"{format_gb(estimated_missing_bytes)} "
        f"(50MB/s ~ {summary['estimated_hours']['50_MBps']}h, "
        f"100MB/s ~ {summary['estimated_hours']['100_MBps']}h, "
        f"200MB/s ~ {summary['estimated_hours']['200_MBps']}h) "
        f"max_workers={args.max_workers} "
        f"xet_high_performance={args.xet_high_performance}"
    )
    if args.xet_high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output_root),
        allow_patterns=missing_patterns,
        max_workers=args.max_workers,
    )
    print("[done] missing chunk download complete")


if __name__ == "__main__":
    main()
