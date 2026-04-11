"""Minimal logging setup helpers."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
