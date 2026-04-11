#!/usr/bin/env python3
"""WP3 entrypoint: canonical sample materialization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.canonicalize import materialize_sample
from src.data.hf_download import metadata_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "strict_download_subset_manifest.parquet",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root that contains metadata plus any downloaded camera/label chunks.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "canonical_samples",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke runs.")
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Materialize timestamps and ego tensors only, without frame decoding.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "canonical_materialization_summary.json",
    )
    return parser.parse_args()


def required_paths_exist(dataset_root: Path, row: pd.Series) -> bool:
    chunk = int(row["chunk"])
    checks = [
        dataset_root / "labels" / "egomotion" / f"egomotion.chunk_{chunk:04d}.zip",
    ]
    for camera_name in [
        "camera_cross_left_120fov",
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_front_tele_30fov",
    ]:
        checks.append(dataset_root / "camera" / camera_name / f"{camera_name}.chunk_{chunk:04d}.zip")
    return all(path.exists() for path in checks)


def main() -> None:
    args = parse_args()
    dataset_root = (args.dataset_root or metadata_root(args.project_root)).resolve()
    manifest = pd.read_parquet(args.manifest_path)
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    args.output_root.mkdir(parents=True, exist_ok=True)

    materialized = 0
    skipped_missing_chunks = 0
    decoder_failures = 0
    status_examples: list[dict[str, str]] = []

    for _, row in manifest.iterrows():
        if not required_paths_exist(dataset_root, row):
            skipped_missing_chunks += 1
            continue

        meta = materialize_sample(
            row,
            dataset_root=dataset_root,
            sample_root=args.output_root,
            extract_images=not args.skip_images,
        )
        materialized += 1
        failed = {camera: status for camera, status in meta["decoder_status"].items() if status.startswith("failed:")}
        if failed:
            decoder_failures += 1
            if len(status_examples) < 10:
                status_examples.append(failed)

    summary = {
        "manifest_path": str(args.manifest_path),
        "dataset_root": str(dataset_root),
        "output_root": str(args.output_root),
        "requested_rows": int(len(manifest)),
        "materialized_rows": materialized,
        "skipped_missing_chunks": skipped_missing_chunks,
        "decoder_failure_rows": decoder_failures,
        "decoder_failure_examples": status_examples,
        "skip_images": args.skip_images,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
