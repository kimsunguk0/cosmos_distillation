#!/usr/bin/env python3
"""Probe whether Alpamayo's action expert can run without generated CoT tokens.

The action expert consumes VLM transformer KV caches, not raw ViT features.  This
script tests the closest useful variants:

1. full_generate: normal Alpamayo path, prompt + generated CoT until traj start
2. prompt_only_kv: prompt/Vision prefill cache only, no generated CoT/traj token
3. no_kv: unconditional expert diffusion with no VLM cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import einops
import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
ALPAMAYO15_SRC = WORKSPACE_ROOT / "alpamayo_repo" / "alpamayo1.5" / "src"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(ALPAMAYO15_SRC) not in sys.path:
    sys.path.insert(0, str(ALPAMAYO15_SRC))

from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import (  # noqa: E402
    build_model_inputs,
    load_materialized_sample,
    load_model_and_processor,
    torch_dtype_from_name,
)


DEFAULT_SAMPLE_DIR = Path(
    "/home/pm97/workspace/dataset/distill_dataset/materialized/"
    "0683baa6-7ba6-47de-bcb7-a45e3064dfb7__sg_00__t0_1600000"
)
DEFAULT_OUTPUT_JSON = (
    Path("/home/pm97/workspace/dataset/distill_dataset/reports/no_nav")
    / "prompt_only_action_expert_probe.json"
)
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "base_weights" / "Alpamayo-1.5-10B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=163840)
    parser.add_argument("--max-pixels", type=int, default=196608)
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--skip-full-generate", action="store_true")
    return parser.parse_args()


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_mem() -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    sync_cuda()
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024**3), 3),
    }


def path_length(xyz: np.ndarray) -> float:
    arr = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    if arr.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=-1).sum())


def summarize_xyz(xyz: torch.Tensor | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(xyz.detach().float().cpu().numpy() if isinstance(xyz, torch.Tensor) else xyz, dtype=np.float32)
    flat = arr.reshape(-1, 3)
    return {
        "shape": list(arr.shape),
        "path_length_m": round(path_length(flat), 4),
        "final_xy_m": [round(float(v), 4) for v in flat[-1, :2].tolist()],
        "mean_abs_xyz": round(float(np.mean(np.abs(flat))), 6),
        "finite": bool(np.isfinite(flat).all()),
    }


def ade_fde(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    aa = np.asarray(a, dtype=np.float32).reshape(-1, 3)
    bb = np.asarray(b, dtype=np.float32).reshape(-1, 3)
    n = min(len(aa), len(bb))
    if n == 0:
        return {"ade_xy_m": float("nan"), "fde_xy_m": float("nan")}
    d = np.linalg.norm(aa[:n, :2] - bb[:n, :2], axis=-1)
    return {"ade_xy_m": round(float(d.mean()), 4), "fde_xy_m": round(float(d[-1]), 4)}


def build_prompt_prefill(
    *,
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    data = build_model_inputs(processor=processor, sample=sample, device=device)
    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]
    tokenized_data = dict(data["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")
    input_ids = model.fuse_traj_tokens(
        input_ids,
        {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        },
    )
    started = time.perf_counter()
    prefill_outputs = model.vlm(
        input_ids=input_ids,
        **tokenized_data,
        use_cache=True,
        logits_to_keep=1,
    )
    sync_cuda()
    rope_deltas = getattr(prefill_outputs, "rope_deltas", None)
    if rope_deltas is None:
        rope_deltas = getattr(model.vlm.model, "rope_deltas", None)
    if rope_deltas is None:
        rope_deltas = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
    prefix_mask = tokenized_data.get("attention_mask")
    return {
        "cache": prefill_outputs.past_key_values,
        "rope_deltas": rope_deltas,
        "prefix_mask": prefix_mask,
        "input_ids": input_ids,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "elapsed_sec": round(time.perf_counter() - started, 6),
    }


def sample_action_expert_from_cache(
    *,
    model: Any,
    prompt_cache: Any | None,
    rope_deltas: torch.Tensor | None,
    prefix_mask: torch.Tensor | None,
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    device: str,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device_obj = torch.device(device)
    dtype = next(model.parameters()).dtype
    n_diffusion_tokens = int(model.action_space.get_action_space_dims()[0])
    batch_size = int(ego_history_xyz.shape[0])

    if prompt_cache is None:
        prefill_seq_len = 0
        offset = torch.zeros((batch_size,), dtype=torch.long, device=device_obj)
        if rope_deltas is None:
            rope_deltas = torch.zeros((batch_size, 1), dtype=torch.long, device=device_obj)
        position_ids = torch.arange(n_diffusion_tokens, device=device_obj)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=batch_size).clone()
        position_ids += rope_deltas.to(position_ids.device)
        attention_mask = torch.zeros(
            (batch_size, 1, n_diffusion_tokens, n_diffusion_tokens),
            dtype=torch.float32,
            device=device_obj,
        )
    else:
        prefill_seq_len = int(prompt_cache.get_seq_length())
        offset = torch.full((batch_size,), prefill_seq_len, dtype=torch.long, device=device_obj)
        position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
            offset=offset,
            rope_deltas=rope_deltas,
            kv_cache_seq_len=prefill_seq_len,
            n_diffusion_tokens=n_diffusion_tokens,
            b_star=batch_size,
            device=device_obj,
            prefix_mask=prefix_mask,
        )

    forward_kwargs: dict[str, Any] = {}
    if bool(getattr(model.config, "expert_non_causal_attention", False)):
        forward_kwargs["is_causal"] = False

    def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        future_token_embeds = model.action_in_proj(x.to(dtype=dtype), t.to(dtype=dtype))
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(x.shape[0], n_diffusion_tokens, -1)
        expert_out = model.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=prompt_cache,
            attention_mask=attention_mask.to(dtype=future_token_embeds.dtype),
            use_cache=True,
            **forward_kwargs,
        )
        if prompt_cache is not None:
            prompt_cache.crop(prefill_seq_len)
        last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
        return model.action_out_proj(last_hidden).view(-1, *model.action_space.get_action_space_dims())

    sync_cuda()
    started = time.perf_counter()
    sampled_action = model.diffusion.sample(
        batch_size=batch_size,
        step_fn=step_fn,
        device=device_obj,
        return_all_steps=False,
    )
    hist_xyz = ego_history_xyz[:, -1].to(device_obj)
    hist_rot = ego_history_rot[:, -1].to(device_obj)
    pred_xyz, pred_rot = model.action_space.action_to_traj(sampled_action, hist_xyz, hist_rot)
    sync_cuda()
    return {
        "status": "ok",
        "elapsed_sec": round(time.perf_counter() - started, 6),
        "prefill_seq_len": int(prefill_seq_len),
        "offset": [int(v) for v in offset.detach().cpu().tolist()],
        "cache_layer_count": len(getattr(prompt_cache, "layers", [])) if prompt_cache is not None else 0,
        "pred_xyz": pred_xyz.detach().float().cpu().numpy(),
        "pred_rot_shape": list(pred_rot.shape),
    }


def run_full_generate(
    *,
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    device: str,
    seed: int,
) -> dict[str, Any]:
    data = build_model_inputs(processor=processor, sample=sample, device=device)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    started = time.perf_counter()
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=data,
        top_p=1.0,
        top_k=None,
        temperature=1.0,
        num_traj_samples=1,
        max_generation_length=256,
        return_extra=True,
    )
    sync_cuda()
    return {
        "status": "ok",
        "elapsed_sec": round(time.perf_counter() - started, 6),
        "pred_xyz": pred_xyz.detach().float().cpu().numpy().reshape(-1, 64, 3)[0],
        "pred_rot_shape": list(pred_rot.shape),
        "cot_preview": str(np.asarray(extra.get("cot")).reshape(-1)[0])[:240] if isinstance(extra, dict) else None,
        "meta_action_preview": str(np.asarray(extra.get("meta_action")).reshape(-1)[0])[:240]
        if isinstance(extra, dict) and "meta_action" in extra
        else None,
    }


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    dtype = torch_dtype_from_name(args.dtype)
    model, processor, config, config_path, runtime_support_path = load_model_and_processor(
        Path(args.checkpoint_path),
        dtype=dtype,
        device=args.device,
        config_json=None,
        runtime_support=None,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    sample_dir = Path(args.sample_dir)
    sample = load_materialized_sample(sample_dir)

    autocast_context = (
        torch.autocast("cuda", dtype=dtype)
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else nullcontext()
    )
    results: dict[str, Any] = {
        "sample_dir": str(sample_dir),
        "checkpoint_path": str(args.checkpoint_path),
        "config_path": str(config_path),
        "runtime_support_path": str(runtime_support_path) if runtime_support_path is not None else None,
        "dtype": str(dtype).replace("torch.", ""),
        "attn_implementation": args.attn_implementation,
        "expert_layers": int(model.expert.config.num_hidden_layers),
        "expert_hidden_size": int(model.expert.config.hidden_size),
        "note": "ViT features alone are not layerwise KV; prompt_only_kv is the no-CoT KV test.",
    }

    with torch.inference_mode(), autocast_context:
        if not args.skip_full_generate:
            try:
                full = run_full_generate(
                    model=model,
                    processor=processor,
                    sample=sample,
                    device=args.device,
                    seed=args.seed,
                )
                full["summary"] = summarize_xyz(full["pred_xyz"])
                results["full_generate"] = {k: v for k, v in full.items() if k != "pred_xyz"}
                full_xyz = np.asarray(full["pred_xyz"], dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                results["full_generate"] = {"status": "failed", "error": repr(exc)}
                full_xyz = None
        else:
            full_xyz = None

        try:
            prefill = build_prompt_prefill(
                model=model,
                processor=processor,
                sample=sample,
                device=args.device,
            )
            prompt_only = sample_action_expert_from_cache(
                model=model,
                prompt_cache=prefill["cache"],
                rope_deltas=prefill["rope_deltas"],
                prefix_mask=prefill["prefix_mask"],
                ego_history_xyz=prefill["ego_history_xyz"],
                ego_history_rot=prefill["ego_history_rot"],
                device=args.device,
                seed=args.seed,
            )
            prompt_only_xyz = np.asarray(prompt_only["pred_xyz"], dtype=np.float32).reshape(-1, 64, 3)[0]
            prompt_only["prefill_elapsed_sec"] = prefill["elapsed_sec"]
            prompt_only["prompt_token_count"] = int(prefill["input_ids"].shape[1])
            prompt_only["summary"] = summarize_xyz(prompt_only_xyz)
            if full_xyz is not None:
                prompt_only["vs_full_generate"] = ade_fde(prompt_only_xyz, full_xyz)
            results["prompt_only_kv"] = {k: v for k, v in prompt_only.items() if k != "pred_xyz"}
        except Exception as exc:  # noqa: BLE001
            results["prompt_only_kv"] = {"status": "failed", "error": repr(exc)}

        try:
            # Reuse prompt-loaded ego tensors but intentionally provide no VLM cache.
            no_kv_source = prefill if "prefill" in locals() else build_prompt_prefill(
                model=model,
                processor=processor,
                sample=sample,
                device=args.device,
            )
            no_kv = sample_action_expert_from_cache(
                model=model,
                prompt_cache=None,
                rope_deltas=None,
                prefix_mask=None,
                ego_history_xyz=no_kv_source["ego_history_xyz"],
                ego_history_rot=no_kv_source["ego_history_rot"],
                device=args.device,
                seed=args.seed,
            )
            no_kv_xyz = np.asarray(no_kv["pred_xyz"], dtype=np.float32).reshape(-1, 64, 3)[0]
            no_kv["summary"] = summarize_xyz(no_kv_xyz)
            if full_xyz is not None:
                no_kv["vs_full_generate"] = ade_fde(no_kv_xyz, full_xyz)
            results["no_kv"] = {k: v for k, v in no_kv.items() if k != "pred_xyz"}
        except Exception as exc:  # noqa: BLE001
            results["no_kv"] = {"status": "failed", "error": repr(exc)}

    results["cuda_mem"] = cuda_mem()
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
