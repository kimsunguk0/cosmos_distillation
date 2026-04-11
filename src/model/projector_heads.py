"""Projection head placeholders for future multimodal adaptation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ProjectorHeadConfig:
    enabled: bool = False
    hidden_dim: int = 0
