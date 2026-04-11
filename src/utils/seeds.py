"""Random seed helpers."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python and NumPy for repeatable local tests."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
