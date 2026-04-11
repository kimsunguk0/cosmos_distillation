"""Hash helpers for artifact provenance."""

from __future__ import annotations

import hashlib


def sha256_text(value: str) -> str:
    """Hash a string for lightweight provenance tracking."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
