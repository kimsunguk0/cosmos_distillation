#!/usr/bin/env python3
"""Probe AE expert attention weights to locate where conditioning breaks.

Loads a trained AE28 bundle (e.g. the 3.63m checkpoint), forces eager attention
on the expert so we can capture attn_weights via forward hooks, picks
static/dynamic samples from the corpus, and for each sample:

 1. Records prompt KV cache norms per layer BEFORE the expert is called
    (to verify prefill KV is non-zero and structurally present).
 2. Installs a forward hook on each expert layer's self_attn (only the FIRST
    call per layer is captured to keep this comparable across the 10 diffusion
    sampling sub-steps).
 3. Runs sample_paths() once per sample with a fixed probe seed.
 4. From captured attn_weights of shape (B, H, Q=128, K), computes
    attn_to_prefill_frac = sum_K[..., :prefill_seq_len] / sum_K[..., :]
    per layer (and per head).
 5. Reports per-layer attn_to_prefill_frac and KV norms; flags whether the
    key length matches prefill+128 (the structural concat expectation).

Does NOT modify sample_paths, build_batch, or any model internals — only hooks
and config-level force_attention.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


_84_PATH = Path(__file__).resolve().parent / "84_train_student_ae28_official.py"
if not _84_PATH.exists():
    raise FileNotFoundError(f"Cannot locate sibling 84 script at {_84_PATH}")
_spec = importlib.util.spec_from_file_location("script_84", _84_PATH)
script_84 = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec.loader is not None
_spec.loader.exec_module(script_84)


def _extract_probe_args(argv: list[str]) -> tuple[Path, int, list[str]]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--ckpt-path", type=Path, required=True)
    pre.add_argument("--n-extremes", type=int, default=2,
                     help="Number of static and dynamic samples to probe (each).")
    parsed, remaining = pre.parse_known_args(argv)
    return parsed.ckpt_path, parsed.n_extremes, remaining


def _pick_extremes(items: list[dict], n_extremes: int) -> tuple[list, list]:
    mags = []
    for item in items:
        try:
            xyz, _ = script_84.raw_teacher_pred(Path(item["raw_json"]))
            mags.append((float(np.abs(xyz).mean()), item))
        except Exception as exc:
            print(json.dumps({"event": "pick_extremes_skip",
                              "sample_id": item.get("sample_id"),
                              "error": str(exc)}), flush=True)
    mags.sort(key=lambda x: x[0])
    return mags[:n_extremes], mags[-n_extremes:]


def _measure_prompt_kv_norms(cache, num_layers: int) -> list:
    """Per-layer K/V abs mean and norm summary BEFORE expert forward."""
    norms = []
    layers_attr = getattr(cache, "layers", None)
    for i in range(num_layers):
        if layers_attr is None or i >= len(layers_attr):
            norms.append(None)
            continue
        layer_cache = layers_attr[i]
        keys = getattr(layer_cache, "keys", None)
        values = getattr(layer_cache, "values", None)
        if keys is None or values is None:
            norms.append(None)
            continue
        norms.append({
            "shape": list(keys.shape),
            "key_abs_mean": float(keys.float().abs().mean().cpu()),
            "value_abs_mean": float(values.float().abs().mean().cpu()),
            "key_norm_per_pos_mean": float(keys.float().norm(dim=-1).mean().cpu()),
            "value_norm_per_pos_mean": float(values.float().norm(dim=-1).mean().cpu()),
        })
    return norms


def _probe_one(item, bundle, teacher_model, args, student, student_processor,
               student_tokenizer, device, n_diffusion_tokens: int) -> dict:
    batch = script_84.build_batch(
        args=args,
        student=student,
        student_processor=student_processor,
        student_tokenizer=student_tokenizer,
        teacher_model=teacher_model,
        batch_items=[item],
    )
    prefill_seq_len = int(batch["context"]["kv_cache_seq_len"])
    num_layers = len(bundle.expert.layers)
    kv_norms = _measure_prompt_kv_norms(batch["cache"], num_layers)

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, args_in, output):
            if layer_idx in captured:
                return  # Only capture the FIRST sample_paths sub-step.
            attn_w = None
            if isinstance(output, tuple) and len(output) >= 2:
                attn_w = output[1]
            if attn_w is not None:
                try:
                    captured[layer_idx] = attn_w.detach().float().cpu()
                except Exception:
                    pass
        return hook

    handles = []
    for i, layer in enumerate(bundle.expert.layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(i)))

    try:
        pred = script_84.sample_paths(
            bundle=bundle,
            teacher_model=teacher_model,
            batch=batch,
            seed=12345,
            device=device,
        )
    finally:
        for h in handles:
            h.remove()

    per_layer = []
    for i in range(num_layers):
        attn = captured.get(i)
        if attn is None:
            per_layer.append({"layer": i, "captured": False})
            continue
        if attn.ndim != 4:
            per_layer.append({"layer": i, "captured": True,
                              "WARNING": f"unexpected ndim {attn.ndim}",
                              "shape": list(attn.shape)})
            continue
        B, H, Q, K = attn.shape
        expected_K = prefill_seq_len + n_diffusion_tokens
        prefill_weight = attn[..., :prefill_seq_len].sum(dim=-1)  # (B, H, Q)
        total_weight = attn.sum(dim=-1).clamp(min=1e-12)            # (B, H, Q)
        frac = (prefill_weight / total_weight)
        per_layer.append({
            "layer": i,
            "captured": True,
            "attn_shape": list(attn.shape),
            "expected_K": expected_K,
            "key_length_matches": (K == expected_K),
            "attn_to_prefill_frac_mean": float(frac.mean()),
            "attn_to_prefill_frac_min": float(frac.min()),
            "attn_to_prefill_frac_max": float(frac.max()),
            "attn_to_prefill_frac_per_head_mean": [float(x) for x in frac.mean(dim=(0, 2)).tolist()],
            "total_weight_mean": float(total_weight.mean()),
        })

    out = {
        "sample_id": item["sample_id"],
        "prefill_seq_len": prefill_seq_len,
        "n_diffusion_tokens": n_diffusion_tokens,
        "num_layers": num_layers,
        "kv_norms": kv_norms,
        "layers": per_layer,
        "action_abs_mean": float(np.abs(pred["action"]).mean()),
    }
    del batch, pred, captured
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ckpt_path, n_extremes, remaining = _extract_probe_args(sys.argv[1:])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]] + list(remaining)
        args = script_84.parse_args()
    finally:
        sys.argv = saved_argv

    items = script_84.select_items(args)
    student, st_tok, st_proc, base_model = script_84.load_student(args)
    teacher_model, *_ = script_84.load_model_and_processor(
        checkpoint_path=args.teacher_checkpoint_path,
        dtype=script_84.torch_dtype_from_name(args.ae_dtype),
        device=args.teacher_load_device,
        config_json=None, runtime_support=None,
        attn_implementation=args.attn_implementation,
        min_pixels=163840, max_pixels=196608,
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    script_84.force_attention(
        teacher_model.expert,
        "sdpa" if args.attn_implementation != "eager" else "eager",
    )
    bundle, _ = script_84.build_bundle(teacher_model, args, student=student)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bundle.load_state_dict(state["bundle_state_dict"], strict=False)
    bundle = bundle.to(device=args.device, dtype=script_84.torch_dtype_from_name(args.ae_dtype))
    bundle.eval()

    # Force EAGER on bundle.expert so attention returns attn_weights through the hook.
    script_84.force_attention(bundle.expert, "eager")
    print(json.dumps({"event": "expert_attention_forced_eager"}), flush=True)

    if hasattr(teacher_model, "vlm"):
        delattr(teacher_model, "vlm")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device(args.device)
    n_diff = int(teacher_model.action_space.get_action_space_dims()[0])
    static, dynamic = _pick_extremes(items, n_extremes)
    print(json.dumps({
        "event": "probe_start",
        "ckpt_path": str(ckpt_path),
        "n_extremes": int(n_extremes),
        "n_diffusion_tokens": int(n_diff),
        "static_mags": [m for m, _ in static],
        "dynamic_mags": [m for m, _ in dynamic],
    }), flush=True)

    results = []
    for kind, picks in (("static", static), ("dynamic", dynamic)):
        for mag, item in picks:
            r = _probe_one(item, bundle, teacher_model, args, student, st_proc, st_tok, device, n_diff)
            r["kind"] = kind
            r["target_xyz_abs_mean"] = mag
            cap = [l for l in r["layers"] if l.get("captured")]
            r["n_layers_captured"] = len(cap)
            if cap:
                fracs = [l["attn_to_prefill_frac_mean"] for l in cap]
                r["overall_attn_to_prefill_frac_mean"] = float(np.mean(fracs))
                r["overall_attn_to_prefill_frac_min"] = float(np.min(fracs))
                r["overall_attn_to_prefill_frac_max"] = float(np.max(fracs))
                # Heuristic uniform-baseline: under uniform attention,
                # prefill_frac would be prefill / (prefill + 128).
                uniform = r["prefill_seq_len"] / (r["prefill_seq_len"] + n_diff)
                r["uniform_prefill_frac_baseline"] = float(uniform)
            results.append(r)
            print(json.dumps({
                "event": "probe_sample_done",
                "kind": kind,
                "sample_id": r["sample_id"],
                "target_xyz_abs_mean": mag,
                "prefill_seq_len": r["prefill_seq_len"],
                "n_layers_captured": r["n_layers_captured"],
                "overall_attn_to_prefill_frac_mean": r.get("overall_attn_to_prefill_frac_mean"),
                "overall_attn_to_prefill_frac_min": r.get("overall_attn_to_prefill_frac_min"),
                "overall_attn_to_prefill_frac_max": r.get("overall_attn_to_prefill_frac_max"),
                "uniform_prefill_frac_baseline": r.get("uniform_prefill_frac_baseline"),
                "kv_norm_layer0": r["kv_norms"][0] if r["kv_norms"] else None,
            }), flush=True)

    summary = []
    for r in results:
        summary.append({
            "sample_id": r["sample_id"],
            "kind": r["kind"],
            "prefill_seq_len": r["prefill_seq_len"],
            "uniform_prefill_frac_baseline": r.get("uniform_prefill_frac_baseline"),
            "overall_attn_to_prefill_frac_mean": r.get("overall_attn_to_prefill_frac_mean"),
            "per_layer_frac_mean": [l.get("attn_to_prefill_frac_mean") for l in r["layers"]],
            "kv_key_abs_mean_per_layer": [
                (kv["key_abs_mean"] if kv else None) for kv in r["kv_norms"]
            ],
            "kv_value_abs_mean_per_layer": [
                (kv["value_abs_mean"] if kv else None) for kv in r["kv_norms"]
            ],
            "first_layer_key_length_matches": (
                r["layers"][0].get("key_length_matches") if r["layers"][0].get("captured") else None
            ),
        })
    print(json.dumps({"event": "full_summary", "samples": summary}), flush=True)


if __name__ == "__main__":
    main()
