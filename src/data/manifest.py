"""Clip manifest data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SampleSource:
    reasoning_parquet_row_id: int | None = None
    feature_presence_row_id: int | None = None


@dataclass(slots=True)
class SamplePaths:
    camera_chunk_locator: str = ""
    timestamp_locator: str = ""
    egomotion_locator: str = ""


@dataclass(slots=True)
class ClipManifestRow:
    sample_id: str
    clip_uuid: str
    split: str
    event_cluster: str
    t0_us: int
    source: SampleSource = field(default_factory=SampleSource)
    paths: SamplePaths = field(default_factory=SamplePaths)
