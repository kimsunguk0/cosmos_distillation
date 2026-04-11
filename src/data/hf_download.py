"""Metadata acquisition helpers."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable


DEFAULT_METADATA_PATTERNS = (
    "features.csv",
    "clip_index.parquet",
    "reasoning/ood_reasoning.parquet",
    "metadata/feature_presence.parquet",
    "metadata/sensor_presence.parquet",
    "metadata/data_collection.parquet",
)

DEFAULT_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
DEFAULT_KNOWN_CACHE_CANDIDATES = (
    "/home/pm97/workspace/kjhong/alpamayo_100/.physical_ai_av_cache/"
    "datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots/"
    "2ae73f49ffd2b5db43b404201beb7b92889f7afc",
)


def metadata_patterns(extra_patterns: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return the allowlist for metadata-only downloads."""
    if not extra_patterns:
        return DEFAULT_METADATA_PATTERNS
    return DEFAULT_METADATA_PATTERNS + tuple(extra_patterns)


def metadata_root(project_root: Path) -> Path:
    """Return the local directory used for dataset metadata."""
    return project_root / "data" / "raw" / "physical_ai_av"


def required_metadata_targets(root: Path) -> dict[str, Path]:
    """Return the canonical output paths used by this project."""
    return {
        "features_csv": root / "features.csv",
        "clip_index": root / "clip_index.parquet",
        "reasoning": root / "reasoning" / "ood_reasoning.parquet",
        "feature_presence": root / "metadata" / "feature_presence.parquet",
        "sensor_presence": root / "metadata" / "sensor_presence.parquet",
        "data_collection": root / "metadata" / "data_collection.parquet",
    }


def required_metadata_status(root: Path) -> dict[str, bool]:
    """Report which required artifacts currently exist."""
    targets = required_metadata_targets(root)
    return {
        "features_csv": targets["features_csv"].exists(),
        "clip_index": targets["clip_index"].exists(),
        "reasoning": targets["reasoning"].exists(),
        "data_collection": targets["data_collection"].exists(),
        "presence": targets["feature_presence"].exists() or targets["sensor_presence"].exists(),
    }


def is_metadata_complete(root: Path) -> bool:
    """Return whether the local metadata set is complete enough for WP1/WP2."""
    return all(required_metadata_status(root).values())


def candidate_cache_roots() -> list[Path]:
    """Return possible existing dataset snapshot roots."""
    candidates: list[Path] = []

    for env_name in ("PAI_SNAPSHOT_ROOT", "PAI_LOCAL_DIR"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())

    candidates.extend(Path(path) for path in DEFAULT_KNOWN_CACHE_CANDIDATES)

    # Preserve order while removing duplicates.
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def existing_cache_root() -> Path | None:
    """Return the first known local snapshot root that exists."""
    for candidate in candidate_cache_roots():
        if candidate.exists():
            return candidate.resolve()
    return None


def _copy_or_link(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return

    if link_mode == "symlink":
        dst.symlink_to(src)
        return
    if link_mode == "copy":
        shutil.copy2(src, dst)
        return
    raise ValueError(f"Unsupported link mode: {link_mode}")


def materialize_from_existing_cache(
    cache_root: Path,
    output_root: Path,
    link_mode: str = "symlink",
) -> dict[str, str]:
    """Populate local metadata paths from an existing snapshot root."""
    targets = required_metadata_targets(output_root)
    sources = {
        "features_csv": cache_root / "features.csv",
        "clip_index": cache_root / "clip_index.parquet",
        "reasoning": cache_root / "reasoning" / "ood_reasoning.parquet",
        "feature_presence": cache_root / "metadata" / "feature_presence.parquet",
        "sensor_presence": cache_root / "metadata" / "sensor_presence.parquet",
        "data_collection": cache_root / "metadata" / "data_collection.parquet",
    }

    status: dict[str, str] = {}
    for key, src in sources.items():
        if not src.exists():
            status[key] = "missing"
            continue
        _copy_or_link(src, targets[key], link_mode)
        status[key] = "linked" if link_mode == "symlink" else "copied"
    return status


def write_acquisition_summary(
    output_root: Path,
    *,
    source_cache_root: Path | None,
    source_repo_id: str,
    mode: str,
    status: dict[str, str | bool],
) -> Path:
    """Persist a machine-readable summary of the metadata acquisition step."""
    summary_path = output_root / "metadata_acquisition.json"
    payload = {
        "source_cache_root": str(source_cache_root) if source_cache_root else None,
        "source_repo_id": source_repo_id,
        "mode": mode,
        "status": status,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path
