"""Priority download planning for chunk-completion-first strategies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_FEATURES = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
    "egomotion",
]


def feature_prefix(feature: str) -> str:
    """Return the top-level dataset folder for a feature."""
    return "camera" if feature.startswith("camera") else "labels"


def allow_pattern(feature: str, chunk: int) -> str:
    """Return the HF allow-pattern for a given feature/chunk."""
    return f"{feature_prefix(feature)}/{feature}/{feature}.chunk_{int(chunk):04d}.*"


def source_chunk_path(cache_root: Path, feature: str, chunk: int) -> Path | None:
    """Resolve a chunk file from the local shared cache if available."""
    matches = list((cache_root / feature_prefix(feature) / feature).glob(f"{feature}.chunk_{int(chunk):04d}.*"))
    return matches[0] if matches else None


def local_feature_dir(output_root: Path, feature: str) -> Path:
    """Return the local feature directory."""
    return output_root / feature_prefix(feature) / feature


def average_feature_sizes(cache_root: Path) -> dict[str, int]:
    """Estimate average chunk size per feature from local cache samples."""
    averages: dict[str, int] = {}
    for feature in REQUIRED_FEATURES:
        files = sorted((cache_root / feature_prefix(feature) / feature).glob(f"{feature}.chunk_*"))
        sample = files[: min(20, len(files))]
        averages[feature] = int(sum(path.stat().st_size for path in sample) / len(sample)) if sample else 0
    return averages


def existing_chunk_index(output_root: Path) -> dict[str, set[int]]:
    """Return currently available chunk IDs per feature."""
    existing: dict[str, set[int]] = {}
    for feature in REQUIRED_FEATURES:
        feature_dir = local_feature_dir(output_root, feature)
        existing[feature] = {int(path.stem.split("chunk_")[1]) for path in feature_dir.glob("*.chunk_*.zip")}
    return existing


@dataclass(slots=True)
class PriorityPlan:
    chunk_table: pd.DataFrame
    selected_chunks: list[int]
    selected_patterns: list[str]
    selected_missing_counts: dict[str, int]
    selected_sample_count: int
    ready_sample_count_now: int
    ready_chunk_count_now: int
    estimated_selected_missing_bytes: int


def build_chunk_table(manifest: pd.DataFrame, existing: dict[str, set[int]]) -> pd.DataFrame:
    """Build a per-chunk readiness table."""
    rows: list[dict] = []
    chunk_counts = manifest.groupby("chunk").size().sort_values(ascending=False)
    for chunk, n_samples in chunk_counts.items():
        chunk = int(chunk)
        present = {feature: (chunk in existing[feature]) for feature in REQUIRED_FEATURES}
        ready = all(present.values())
        rows.append(
            {
                "chunk": chunk,
                "samples": int(n_samples),
                **present,
                "ready": ready,
                "missing_count": sum(not value for value in present.values()),
            }
        )
    return pd.DataFrame(rows).sort_values(["samples", "chunk"], ascending=[False, True]).reset_index(drop=True)


def build_priority_plan(
    manifest: pd.DataFrame,
    *,
    output_root: Path,
    cache_root: Path,
    anchor_feature: str,
    preferred_features: list[str],
    target_chunks: int,
) -> PriorityPlan:
    """Choose chunks that are closest to becoming teacher-ready."""
    existing = existing_chunk_index(output_root)
    chunk_table = build_chunk_table(manifest, existing)

    ready_rows = chunk_table[chunk_table["ready"]]
    candidates = chunk_table[(chunk_table[anchor_feature]) & (~chunk_table["ready"])].copy()
    candidates = candidates.sort_values(["samples", "missing_count", "chunk"], ascending=[False, True, True])
    selected = candidates.head(target_chunks).copy()
    selected_chunks = [int(chunk) for chunk in selected["chunk"].tolist()]

    selected_patterns: list[str] = []
    selected_missing_counts = {feature: 0 for feature in preferred_features}
    for chunk in selected_chunks:
        for feature in preferred_features:
            if chunk not in existing[feature]:
                selected_patterns.append(allow_pattern(feature, chunk))
                selected_missing_counts[feature] += 1

    averages = average_feature_sizes(cache_root)
    estimated_selected_missing_bytes = sum(
        selected_missing_counts[feature] * averages.get(feature, 0) for feature in preferred_features
    )

    return PriorityPlan(
        chunk_table=chunk_table,
        selected_chunks=selected_chunks,
        selected_patterns=selected_patterns,
        selected_missing_counts=selected_missing_counts,
        selected_sample_count=int(selected["samples"].sum()),
        ready_sample_count_now=int(ready_rows["samples"].sum()),
        ready_chunk_count_now=int(len(ready_rows)),
        estimated_selected_missing_bytes=int(estimated_selected_missing_bytes),
    )
