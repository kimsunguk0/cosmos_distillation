"""Filesystem helpers."""

from __future__ import annotations

from pathlib import Path


def project_root_from_file(file_path: str) -> Path:
    """Resolve the project root from a file path inside this repository."""
    return Path(file_path).resolve().parents[2]
