"""PhysicalAI AV dataset schema placeholders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MetadataPaths:
    reasoning_parquet: Path
    feature_presence_parquet: Path
    data_collection_parquet: Path


def resolve_metadata_paths(project_root: Path) -> MetadataPaths:
    """Resolve the expected metadata parquet locations."""
    metadata_dir = project_root / "data" / "raw" / "metadata"
    return MetadataPaths(
        reasoning_parquet=metadata_dir / "reasoning" / "ood_reasoning.parquet",
        feature_presence_parquet=metadata_dir / "metadata" / "feature_presence.parquet",
        data_collection_parquet=metadata_dir / "metadata" / "data_collection.parquet",
    )
