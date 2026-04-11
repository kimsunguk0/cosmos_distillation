"""Schedule helpers."""

from __future__ import annotations


def scheduler_name() -> str:
    """Return the default scheduler family."""
    return "cosine_with_warmup"
