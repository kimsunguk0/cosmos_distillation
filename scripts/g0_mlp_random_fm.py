#!/usr/bin/env python3
"""G0: Analytical baseline for random FM with a tiny independent MLP.

Replicates V2's "random (x0,t) 1-sample E3" setup but with a small MLP
instead of the AE bundle. If this MLP can learn (cosine -> 1), then the
FM task formula + sampler are themselves learnable; remaining failure
lives in the AE path. If this MLP also collapses (cosine ~ 0 or
negative), then the FM definition / sampler / target formula is wrong.
"""
from __future__ import annotations

import json
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def main() -> None:
    torch.manual_seed(42)
    T = 64
    D = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed target trajectory (matches "1-sample E3" — one sample, learned from
    # random (x0, t) draws). Magnitude similar to what 84 logs (target_act ~ 0.2-0.5).
    x1 = torch.randn(T, D, device=device) * 0.4

    class TFMlp(nn.Module):
        def __init__(self, action_dim=3, num_freqs=20, max_freq=100.0, hidden=512):
            super().__init__()
            freqs = torch.logspace(0, math.log10(max_freq), steps=num_freqs)
            self.register_buffer("freqs", freqs[None, :])
            self.t_dim = num_freqs * 2
            self.net = nn.Sequential(
                nn.Linear(action_dim + self.t_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, action_dim),
            )

        def encode_t(self, t: torch.Tensor) -> torch.Tensor:
            arg = t[..., None] * self.freqs * 2 * math.pi
            return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1) * math.sqrt(2)

        def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            B, T, D = x_t.shape
            t_emb = self.encode_t(t)
            t_emb_tile = t_emb[:, None, :].expand(B, T, self.t_dim)
            h = torch.cat([x_t, t_emb_tile], dim=-1)
            return self.net(h)

    mlp = TFMlp(action_dim=D).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=0.01)

    # Teacher beta sampler (matches alpamayo_base flow_matching.py L54-57).
    beta = torch.distributions.beta.Beta(
        torch.tensor(1.5, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    beta_scale = 0.999
    batch_size = 4
    steps = 1500
    log_every = 100

    print(json.dumps({
        "event": "g0_start",
        "T": T, "D": D, "batch_size": batch_size, "steps": steps,
        "x1_abs_mean": float(x1.abs().mean()),
        "mlp_params": sum(p.numel() for p in mlp.parameters()),
        "device": str(device),
    }), flush=True)

    log = []
    for step in range(1, steps + 1):
        x0 = torch.randn(batch_size, T, D, device=device)
        t = beta.sample((batch_size,)).to(device)
        t = beta_scale - t * beta_scale   # same transform as flow_matching.py
        t_view = t[:, None, None]
        x_t = (1.0 - t_view) * x0 + t_view * x1[None]
        target_v = x1[None] - x0
        pred_v = mlp(x_t, t)
        loss = F.mse_loss(pred_v, target_v)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 1 or step % log_every == 0:
            with torch.no_grad():
                pred_flat = pred_v.reshape(batch_size, -1)
                tgt_flat = target_v.reshape(batch_size, -1)
                cos = F.cosine_similarity(pred_flat, tgt_flat, dim=-1).mean()
                alpha = (pred_v * target_v).sum() / (target_v.pow(2).sum() + 1e-12)
            row = {
                "step": step,
                "loss": float(loss),
                "cosine": float(cos),
                "alpha": float(alpha),
                "pred_abs_mean": float(pred_v.abs().mean()),
                "target_abs_mean": float(target_v.abs().mean()),
                "t_mean": float(t.mean()),
            }
            log.append(row)
            print(json.dumps({"event": "g0_step", **row}), flush=True)

    # Verdict
    final = log[-1]
    passed = final["cosine"] > 0.9
    print(json.dumps({
        "event": "g0_verdict",
        "final_step": final["step"],
        "final_loss": final["loss"],
        "final_cosine": final["cosine"],
        "final_alpha": final["alpha"],
        "passed": passed,
        "interpretation": (
            "PASS: random FM is learnable analytically. Remaining failure is in the AE path."
            if passed else
            "FAIL: even a tiny MLP cannot learn random FM with this formula/sampler. "
            "Target formula / sampler / parameterization itself is suspect."
        ),
    }), flush=True)


if __name__ == "__main__":
    main()
