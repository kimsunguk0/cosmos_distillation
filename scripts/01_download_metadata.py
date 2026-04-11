#!/usr/bin/env python3
"""WP1 entrypoint: metadata-only dataset download."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hf_download import (
    DEFAULT_REPO_ID,
    candidate_cache_roots,
    existing_cache_root,
    is_metadata_complete,
    materialize_from_existing_cache,
    metadata_patterns,
    metadata_root,
    required_metadata_status,
    write_acquisition_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root for the cosmos_distillation workspace.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory that will hold metadata files. Defaults to data/raw/physical_ai_av.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo ID.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        default=True,
        help="Try to reuse an existing local PhysicalAI AV snapshot first.",
    )
    parser.add_argument(
        "--no-reuse-cache",
        dest="reuse_cache",
        action="store_false",
        help="Skip existing-cache reuse and force hub download.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Specific local PhysicalAI AV snapshot root to reuse.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to materialize files from an existing cache.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Always call snapshot_download even if a local cache or local output already exists.",
    )
    return parser.parse_args()


def download_missing_metadata(repo_id: str, output_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed in the active environment."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = metadata_patterns()
    print(f"[download] repo={repo_id}")
    print(f"[download] allow_patterns={list(allow_patterns)}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=list(allow_patterns),
    )


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (args.output_dir or metadata_root(project_root)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_metadata_complete(output_dir) and not args.force_download:
        status = required_metadata_status(output_dir)
        summary_path = write_acquisition_summary(
            output_dir,
            source_cache_root=None,
            source_repo_id=args.repo_id,
            mode="local_already_complete",
            status=status,
        )
        print(f"[skip] metadata already complete at {output_dir}")
        print(f"[summary] {summary_path}")
        return

    cache_root = args.cache_root.resolve() if args.cache_root else None
    if args.reuse_cache and cache_root is None:
        cache_root = existing_cache_root()

    if args.reuse_cache:
        print("[cache] candidates:")
        for candidate in candidate_cache_roots():
            print(f"  - {candidate}")

    if cache_root and cache_root.exists() and not args.force_download:
        print(f"[cache] using existing snapshot root: {cache_root}")
        link_status = materialize_from_existing_cache(
            cache_root=cache_root,
            output_root=output_dir,
            link_mode=args.link_mode,
        )
        if is_metadata_complete(output_dir):
            summary_path = write_acquisition_summary(
                output_dir,
                source_cache_root=cache_root,
                source_repo_id=args.repo_id,
                mode=f"reuse_cache_{args.link_mode}",
                status={**link_status, **required_metadata_status(output_dir)},
            )
            print(f"[done] metadata materialized from local cache into {output_dir}")
            print(f"[summary] {summary_path}")
            return
        print("[cache] existing snapshot did not contain the full required metadata set")

    print("[download] falling back to Hugging Face metadata-only snapshot")
    download_missing_metadata(repo_id=args.repo_id, output_dir=output_dir)
    final_status = required_metadata_status(output_dir)
    summary_path = write_acquisition_summary(
        output_dir,
        source_cache_root=cache_root,
        source_repo_id=args.repo_id,
        mode="snapshot_download",
        status=final_status,
    )
    print(f"[status] {final_status}")
    print(f"[summary] {summary_path}")
    if not all(final_status.values()):
        missing = [name for name, ok in final_status.items() if not ok]
        raise SystemExit(f"Metadata acquisition incomplete, still missing: {missing}")


if __name__ == "__main__":
    main()
