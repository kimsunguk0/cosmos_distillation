#!/usr/bin/env python3
"""Probe RoPE alignment for AE expert.

Verifies whether query position_ids built by build_batch match the canonical
HF transformers convention (cache_position + rope_deltas), and tests whether
forcing position_ids to be a simple arange(prefill, prefill+128) changes the
prefill attention mass.

Inference-only. Does not modify sample_paths, build_batch, or model code.
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
_spec = importlib.util.spec_from_file_location("script_84", _84_PATH)
script_84 = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec.loader is not None
_spec.loader.exec_module(script_84)


def _extract_args(argv: list[str]) -> tuple[Path | None, bool, list[str]]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--ckpt-path", type=Path, default=None)
    pre.add_argument("--skip-ckpt-load", action="store_true",
                     help="Build bundle from scratch (init mode) and skip state_dict load. "
                          "Useful to probe raw init effects without training.")
    parsed, remaining = pre.parse_known_args(argv)
    return parsed.ckpt_path, parsed.skip_ckpt_load, remaining


def _measure_attn(bundle, teacher_model, batch, device, seed: int = 12345) -> dict:
    """Forward-hook capture attn_to_prefill_frac per layer (first sub-step only)."""
    captured: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(module, _args, output):
            if idx in captured:
                return
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                try:
                    captured[idx] = output[1].detach().float().cpu()
                except Exception:
                    pass
        return hook

    handles = [layer.self_attn.register_forward_hook(make_hook(i))
               for i, layer in enumerate(bundle.expert.layers)]
    try:
        pred = script_84.sample_paths(
            bundle=bundle, teacher_model=teacher_model,
            batch=batch, seed=seed, device=device,
        )
    finally:
        for h in handles:
            h.remove()

    prefill = int(batch["context"]["kv_cache_seq_len"])
    n_diff = int(batch["context"]["n_diffusion_tokens"])
    per_layer = []
    for i in range(len(bundle.expert.layers)):
        attn = captured.get(i)
        if attn is None or attn.ndim != 4:
            per_layer.append(None)
            continue
        if attn.shape[-1] != prefill + n_diff:
            per_layer.append(None)
            continue
        prefill_w = attn[..., :prefill].sum(dim=-1)
        total = attn.sum(dim=-1).clamp(min=1e-12)
        per_layer.append(float((prefill_w / total).mean()))
    overall_vals = [f for f in per_layer if f is not None]
    overall = float(np.mean(overall_vals)) if overall_vals else None
    return {
        "per_layer": per_layer,
        "overall": overall,
        "prefill_seq_len": prefill,
        "n_diffusion_tokens": n_diff,
        "action_abs_mean": float(np.abs(pred["action"]).mean()),
    }


def main() -> None:
    ckpt_path, skip_ckpt_load, remaining = _extract_args(sys.argv[1:])
    if not skip_ckpt_load:
        if ckpt_path is None:
            raise ValueError("Either --ckpt-path or --skip-ckpt-load must be provided.")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]] + list(remaining)
        args = script_84.parse_args()
    finally:
        sys.argv = saved_argv

    items = script_84.select_items(args)
    student, st_tok, st_proc, _ = script_84.load_student(args)
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
    if not skip_ckpt_load:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        bundle.load_state_dict(state["bundle_state_dict"], strict=False)
        print(json.dumps({"event": "bundle_state_loaded",
                          "ckpt_payload_step": state.get("payload", {}).get("step")}), flush=True)
    else:
        print(json.dumps({"event": "bundle_state_skip_load",
                          "ae_init_mode": str(args.ae_init_mode)}), flush=True)
    bundle = bundle.to(device=args.device, dtype=script_84.torch_dtype_from_name(args.ae_dtype))
    bundle.eval()
    script_84.force_attention(bundle.expert, "eager")
    print(json.dumps({"event": "expert_attention_forced_eager"}), flush=True)

    if hasattr(teacher_model, "vlm"):
        delattr(teacher_model, "vlm")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device(args.device)

    # Pick one sample (the first in the corpus is fine for this diagnostic).
    item = items[0]
    print(json.dumps({"event": "probe_start", "sample_id": item["sample_id"]}), flush=True)

    batch = script_84.build_batch(
        args=args, student=student, student_processor=st_proc,
        student_tokenizer=st_tok, teacher_model=teacher_model,
        batch_items=[item],
    )
    prefill = int(batch["context"]["kv_cache_seq_len"])
    n_diff = int(batch["context"]["n_diffusion_tokens"])
    pos_ids = batch["context"]["position_ids"]
    first_query_pos = [int(pos_ids[d, 0, 0].cpu()) for d in range(pos_ids.shape[0])]
    last_query_pos = [int(pos_ids[d, 0, -1].cpu()) for d in range(pos_ids.shape[0])]
    rope_deltas_inferred = first_query_pos[0] - prefill
    print(json.dumps({
        "event": "position_diagnostic",
        "prefill_seq_len": prefill,
        "n_diffusion_tokens": n_diff,
        "position_ids_shape": list(pos_ids.shape),
        "first_query_pos_3d": first_query_pos,
        "last_query_pos_3d": last_query_pos,
        "rope_deltas_inferred": rope_deltas_inferred,
        "transformers_formula": "cache_position[0] + rope_deltas == prefill_seq_len + rope_deltas",
        "matches_hf_convention": (first_query_pos[0] == prefill + rope_deltas_inferred),
        "note": "rope_deltas (from get_rope_index) = max(mrope_positions)+1 - seq_len. "
                "For prefill ending on a text token, last text mrope position = "
                "rope_deltas + prefill - 1, so query first position = prefill + rope_deltas "
                "= max(prefill_positions) + 1 (natural continuation).",
    }), flush=True)

    # Baseline attn (HF-canonical position_ids)
    baseline = _measure_attn(bundle, teacher_model, batch, device)
    print(json.dumps({
        "event": "baseline_attn",
        "overall_attn_to_prefill_frac": baseline["overall"],
        "per_layer": baseline["per_layer"],
        "uniform_baseline": prefill / (prefill + n_diff),
    }), flush=True)

    # Override: force position_ids = arange(prefill, prefill+128) on all 3 dims.
    # This puts query "right after" the cached sequence in a simple text-like manner.
    new_pos = (torch.arange(prefill, prefill + n_diff, dtype=pos_ids.dtype, device=pos_ids.device)
               .view(1, 1, -1)
               .repeat(pos_ids.shape[0], pos_ids.shape[1], 1))
    batch_override = dict(batch)
    batch_override["context"] = dict(batch["context"])
    batch_override["context"]["position_ids"] = new_pos
    print(json.dumps({
        "event": "override_position",
        "new_first_pos_3d": [int(new_pos[d, 0, 0].cpu()) for d in range(new_pos.shape[0])],
        "new_last_pos_3d": [int(new_pos[d, 0, -1].cpu()) for d in range(new_pos.shape[0])],
    }), flush=True)
    override = _measure_attn(bundle, teacher_model, batch_override, device)
    print(json.dumps({
        "event": "override_attn",
        "overall_attn_to_prefill_frac": override["overall"],
        "per_layer": override["per_layer"],
    }), flush=True)

    delta = (override["overall"] - baseline["overall"]) if (
        baseline["overall"] is not None and override["overall"] is not None
    ) else None
    verdict = "RoPE_ALIGNMENT_OK"
    if delta is not None and delta > 0.05:
        verdict = "RoPE_ALIGNMENT_PARTIAL_FIX"
    if delta is not None and delta > 0.2:
        verdict = "RoPE_MISALIGNED_BASELINE"
    print(json.dumps({
        "event": "summary",
        "baseline_overall": baseline["overall"],
        "override_overall": override["overall"],
        "delta": delta,
        "verdict": verdict,
        "interpretation": (
            "RoPE_ALIGNMENT_OK: baseline already matches HF convention; "
            "RoPE_MISALIGNED_BASELINE: baseline had wrong position_ids, override significantly raises prefill_frac."
        ),
    }), flush=True)


if __name__ == "__main__":
    main()
