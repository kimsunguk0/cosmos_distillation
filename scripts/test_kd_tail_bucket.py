#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.losses import teacher_logit_kd_loss  # noqa: E402


def main() -> int:
    # One supervised shifted token, vocab size 3. Teacher assigns 0.45/0.45 to
    # top-k support and leaves 0.10 tail mass. The "bad" student preserves the
    # same top-k ratio but puts most mass into the tail token.
    teacher_indices = torch.tensor([[[0, 1]]], dtype=torch.long)
    teacher_logprobs = torch.log(torch.tensor([[[0.45, 0.45]]], dtype=torch.float32))
    teacher_mask = torch.tensor([[True]], dtype=torch.bool)
    cot_mask = torch.tensor([[False, True]], dtype=torch.bool)

    good_logits = torch.full((1, 2, 3), -20.0, dtype=torch.float32)
    good_logits[0, 0] = torch.log(torch.tensor([0.45, 0.45, 0.10]))
    bad_logits = torch.full((1, 2, 3), -20.0, dtype=torch.float32)
    bad_logits[0, 0] = torch.log(torch.tensor([0.05, 0.05, 0.90]))

    restricted_good = teacher_logit_kd_loss(
        good_logits, cot_mask, teacher_indices, teacher_logprobs, teacher_mask, include_tail_bucket=False
    )
    restricted_bad = teacher_logit_kd_loss(
        bad_logits, cot_mask, teacher_indices, teacher_logprobs, teacher_mask, include_tail_bucket=False
    )
    tail_good = teacher_logit_kd_loss(
        good_logits, cot_mask, teacher_indices, teacher_logprobs, teacher_mask, include_tail_bucket=True
    )
    tail_bad = teacher_logit_kd_loss(
        bad_logits, cot_mask, teacher_indices, teacher_logprobs, teacher_mask, include_tail_bucket=True
    )

    assert torch.isfinite(restricted_good)
    assert torch.isfinite(restricted_bad)
    assert torch.isfinite(tail_good)
    assert torch.isfinite(tail_bad)
    assert math.isclose(float(restricted_good), float(restricted_bad), rel_tol=1e-5, abs_tol=1e-5)
    assert float(tail_bad) > float(tail_good) + 0.1
    print(
        {
            "restricted_good": float(restricted_good),
            "restricted_bad": float(restricted_bad),
            "tail_good": float(tail_good),
            "tail_bad": float(tail_bad),
            "ok": True,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
