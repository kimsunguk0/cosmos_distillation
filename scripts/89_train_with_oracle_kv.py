#!/usr/bin/env python3
"""Train AE with oracle KV injection.

Monkey-patches 84.build_batch so that immediately after the student backbone
constructs prompt_cache, we overwrite cache.layers[i].keys/values with a
target-projected, sample-specific oracle K/V.

If the student AE can memorize 32 samples under this setup, then the
hypothesis "student backbone hidden lacks trajectory-relevant info" is
confirmed as the root cause. If it still can't memorize, the issue lies
deeper in the AE training path (FM objective, projections, optimizer).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn


_84_PATH = Path(__file__).resolve().parent / "84_train_student_ae28_official.py"
if not _84_PATH.exists():
    raise FileNotFoundError(f"Cannot locate sibling 84 script at {_84_PATH}")
_spec = importlib.util.spec_from_file_location("script_84", _84_PATH)
script_84 = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec.loader is not None
_spec.loader.exec_module(script_84)


_ORACLE_STATE: dict = {
    "proj": None,
    "num_layers": 28,
    "num_kv_heads": 8,
    "head_dim": 128,
    "mode": "none",
}


def _make_oracle_projection(in_dim: int, out_dim: int, seed: int,
                             dtype: torch.dtype, device) -> nn.Linear:
    """Fixed Xavier-initialized linear projection (no grad)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    proj = nn.Linear(in_dim, out_dim, bias=False)
    with torch.no_grad():
        std = math.sqrt(2.0 / (in_dim + out_dim))
        proj.weight.copy_(torch.randn(proj.weight.shape, generator=g) * std)
    proj = proj.to(device=device, dtype=dtype).eval()
    for p in proj.parameters():
        p.requires_grad_(False)
    return proj


def _inject_oracle_kv(batch: dict) -> dict:
    """Replace prompt KV with target-projected oracle. Operates in-place on batch."""
    state = _ORACLE_STATE
    if state["mode"] != "target_projection" or state["proj"] is None:
        return batch
    target_xyz = batch["target_xyz"]                                    # (B, 64, 3)
    proj = state["proj"]
    target = target_xyz.to(device=proj.weight.device, dtype=proj.weight.dtype)
    B, T, _ = target.shape
    nkv = state["num_kv_heads"]
    hdim = state["head_dim"]
    oracle = proj(target)                                                # (B, T, nkv*hdim)
    oracle = oracle.view(B, T, nkv, hdim)                                # (B, T, nkv, hdim)
    oracle_k = oracle.transpose(1, 2).contiguous()                       # (B, nkv, T, hdim)
    oracle_v = oracle_k.clone()

    cache = batch["cache"]
    layers_attr = getattr(cache, "layers", None)
    if layers_attr is None:
        raise RuntimeError("Cache has no 'layers' attribute; cannot inject oracle KV.")

    for i in range(state["num_layers"]):
        if i >= len(layers_attr):
            break
        layer_cache = layers_attr[i]
        try:
            layer_cache.keys = oracle_k.detach().clone()
            layer_cache.values = oracle_v.detach().clone()
        except AttributeError as exc:
            raise RuntimeError(
                f"Cannot assign cache.layers[{i}].keys/.values directly: {exc}. "
                f"Cache layer class = {type(layer_cache).__name__}"
            ) from exc

    # Update context to reflect the shorter, oracle-only cache.
    context = dict(batch["context"])
    context["kv_cache_seq_len"] = T
    pos_dtype = context["position_ids"].dtype
    device = context["position_ids"].device
    n_diff = int(context["n_diffusion_tokens"])
    new_pos = (
        torch.arange(T, T + n_diff, device=device, dtype=pos_dtype)
        .view(1, 1, -1)
        .repeat(3, B, 1)
    )
    context["position_ids"] = new_pos
    if context.get("attention_mask") is not None:
        context["attention_mask"] = None
    batch["context"] = context
    return batch


_original_build_batch = script_84.build_batch


def _build_batch_with_oracle(*args, **kwargs):
    batch = _original_build_batch(*args, **kwargs)
    return _inject_oracle_kv(batch)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--oracle-kv-mode", choices=("none", "target_projection"),
                     default="target_projection")
    pre.add_argument("--oracle-seed", type=int, default=7777)
    parsed, remaining = pre.parse_known_args(sys.argv[1:])

    if parsed.oracle_kv_mode == "target_projection":
        # Sniff device + ae_dtype from 84 args to build proj on the correct device.
        sniff = argparse.ArgumentParser(add_help=False)
        sniff.add_argument("--device", default="cuda:0")
        sniff.add_argument("--ae-dtype", default="bfloat16")
        sniffed, _ = sniff.parse_known_args(remaining)
        device = torch.device(sniffed.device)
        dtype = script_84.torch_dtype_from_name(sniffed.ae_dtype)

        proj = _make_oracle_projection(
            in_dim=3,
            out_dim=_ORACLE_STATE["num_kv_heads"] * _ORACLE_STATE["head_dim"],
            seed=parsed.oracle_seed,
            dtype=dtype,
            device=device,
        )
        _ORACLE_STATE["proj"] = proj
        _ORACLE_STATE["mode"] = "target_projection"
        script_84.build_batch = _build_batch_with_oracle
        print(json.dumps({
            "event": "oracle_kv_enabled",
            "mode": parsed.oracle_kv_mode,
            "oracle_seed": parsed.oracle_seed,
            "proj_in_dim": 3,
            "proj_out_dim": _ORACLE_STATE["num_kv_heads"] * _ORACLE_STATE["head_dim"],
            "num_kv_heads": _ORACLE_STATE["num_kv_heads"],
            "head_dim": _ORACLE_STATE["head_dim"],
            "device": str(device),
            "dtype": str(dtype),
        }), flush=True)
    else:
        print(json.dumps({"event": "oracle_kv_disabled"}), flush=True)

    # Delegate to 84's main loop (which will now use the patched build_batch).
    sys.argv = [sys.argv[0]] + list(remaining)
    script_84.main()


if __name__ == "__main__":
    main()
