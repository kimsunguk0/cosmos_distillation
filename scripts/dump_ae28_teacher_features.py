#!/usr/bin/env python3
"""Dump AE28 teacher 10-step features for AE14 consistency distillation.

For each sample:
  1. Run student backbone forward → KV cache
  2. Run AE28 10-step flow matching, capturing step-wise v and hidden states
  3. Compute reflow targets (straight-line velocity, teacher residuals)
  4. Save compact .pt dump per sample (~2.6 MB each)

Output per sample (.pt):
  - x_all_steps: [11, 64, 2]     (positions at all 10+1 steps, float32)
  - v_steps: [10, 64, 2]         (velocity at each step, float32)
  - last_hidden_steps: [10, 64, 2048]  (expert hidden at each step, bfloat16)
  - time_steps: [11]             (time values 0.0 .. 1.0)
  - sampled_action: [64, 2]      (final teacher action, float32)
  - target_action: [64, 2]       (GT action, float32)
  - target_xyz: [1, ...]         (GT trajectory xyz)
  - ego_history_xyz: [4, 3]
  - ego_history_rot: [4, 3, 3]
  - position_ids: [3, 1, 64]     (RoPE positions for expert)
  - kv_cache_seq_len: int
  - n_diffusion_tokens: int
  - stage2_attention_mode: str

Usage:
    python scripts/dump_ae28_teacher_features.py \
        --student-checkpoint-dir outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final \
        --ae28-checkpoint outputs/action_expert/flex_k512_6ep_ae_18k_s10k_b16/best.pt \
        --corpus-jsonl data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl \
        --output-dir outputs/ae28_teacher_dumps/flex_k512_fp16 \
        --batch-size 1
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))


def _import_ae_script():
    """Import 84_train_student_ae28_official.py (numeric filename workaround)."""
    spec = importlib.util.spec_from_file_location(
        "ae_train",
        str(PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def capturing_inference(
    *,
    bundle: nn.Module,
    teacher_model: Any,
    prompt_cache: Any,
    context: dict[str, Any],
    device: torch.device,
    num_steps: int = 10,
    temperature: float = 1.0,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Run AE28 10-step flow matching inference, capturing all intermediate states.

    Returns dict with x_all_steps, v_steps, last_hidden_steps, time_steps, sampled_action.
    """
    dtype = next(bundle.parameters()).dtype
    batch_size = int(context["position_ids"].shape[1])
    prefill_seq_len = int(context["kv_cache_seq_len"])
    n_diffusion_tokens = int(context["n_diffusion_tokens"])
    action_dims = teacher_model.action_space.get_action_space_dims()
    kwargs: dict[str, Any] = {}
    if bool(getattr(teacher_model.config, "expert_non_causal_attention", False)):
        kwargs["is_causal"] = False

    # Storage for captured features
    captured_v: list[torch.Tensor] = []
    captured_hidden: list[torch.Tensor] = []

    def step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        future_token_embeds = bundle.action_in_proj(x.to(dtype=dtype), t.to(dtype=dtype))
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(x.shape[0], n_diffusion_tokens, -1)
        expert_attention_mask = context.get("attention_mask")
        if expert_attention_mask is not None:
            expert_attention_mask = expert_attention_mask.to(dtype=future_token_embeds.dtype)
        out = bundle.expert(
            inputs_embeds=future_token_embeds,
            position_ids=context["position_ids"],
            past_key_values=prompt_cache,
            attention_mask=expert_attention_mask,
            use_cache=True,
            **kwargs,
        )
        prompt_cache.crop(prefill_seq_len)
        last_hidden = out.last_hidden_state[:, -n_diffusion_tokens:]
        v = bundle.action_out_proj(last_hidden).view(-1, *action_dims)
        # Capture
        captured_v.append(v.detach().cpu().float())
        captured_hidden.append(last_hidden.detach().cpu().to(torch.bfloat16))
        return v

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        result = teacher_model.diffusion.sample(
            batch_size=batch_size,
            step_fn=step_fn,
            device=device,
            inference_step=num_steps,
            temperature=temperature,
            return_all_steps=True,
        )
        x_all_steps, time_steps = result  # [B, 11, 64, 2], [11]

    return {
        "x_all_steps": x_all_steps.detach().cpu().float(),        # [B, 11, 64, 2]
        "v_steps": torch.stack(captured_v, dim=1),                 # [B, 10, 64, 2]
        "last_hidden_steps": torch.stack(captured_hidden, dim=1),  # [B, 10, 64, 2048]
        "time_steps": time_steps.detach().cpu().float(),           # [11]
        "sampled_action": x_all_steps[:, -1].detach().cpu().float(),  # [B, 64, 2]
    }


def build_reflow_targets(
    x_all_steps: torch.Tensor,  # [B, 11, 64, 2]
    v_steps: torch.Tensor,      # [B, 10, 64, 2]
    sampled_action: torch.Tensor,  # [B, 64, 2]
) -> dict[str, torch.Tensor]:
    """Compute reflow straight-line targets and teacher residuals."""
    x_start = x_all_steps[:, 0]     # [B, 64, 2] noise
    x_target = sampled_action        # [B, 64, 2] final action
    straight_delta = x_target - x_start  # [B, 64, 2]

    # Straight-line velocity (constant across all steps)
    reflow_target_v = straight_delta.unsqueeze(1).expand_as(v_steps)  # [B, 10, 64, 2]

    # Teacher residual: how teacher deviates from straight line
    teacher_residual = v_steps - reflow_target_v  # [B, 10, 64, 2]

    return {
        "reflow_target_v_steps": reflow_target_v.float(),
        "teacher_residual_steps": teacher_residual.float(),
    }


def main():
    parser = argparse.ArgumentParser(description="Dump AE28 teacher features")
    parser.add_argument("--student-checkpoint-dir", type=str, required=True)
    parser.add_argument("--ae28-checkpoint", type=str, required=True)
    parser.add_argument("--corpus-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=18000)
    parser.add_argument("--val-samples", type=int, default=1900)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--student-dtype", type=str, default="bfloat16")
    parser.add_argument("--prefix-mode", type=str, default="teacher_forced",
                        choices=["teacher_forced", "student_free"])
    parser.add_argument("--preserve-flex-positions", action="store_true", default=True)
    parser.add_argument("--flex-selection-strategy", type=str, default="uniform")
    parser.add_argument("--flex-scene-deepstack", action="store_true", default=True)
    parser.add_argument("--qat-quantization", type=str, default="")
    parser.add_argument("--qat-calib-samples", type=int, default=256)
    parser.add_argument("--target-source", type=str, default="teacher")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--split-scan-all", action="store_true", default=True)
    parser.add_argument("--stage2-attention-mode", type=str, default="official_none")
    parser.add_argument("--student-model", type=str, default="")
    parser.add_argument("--teacher-checkpoint-path", type=str,
                        default=str(PROJECT_ROOT.parent.parent / "base_weights" / "Alpamayo-1.5-10B"))
    parser.add_argument("--ae-init-mode", type=str, default="student_backbone_init")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--disable-student-deepstack", action="store_true", default=False)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--split-cache-json", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--compressed-layers", type=int, default=28)
    parser.add_argument("--mapping", type=str, default="linspace_round")
    parser.add_argument("--ae-dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Convert to Path objects
    args.student_checkpoint_dir = Path(args.student_checkpoint_dir)
    args.corpus_jsonl = Path(args.corpus_jsonl)
    args.teacher_checkpoint_path = Path(args.teacher_checkpoint_path)
    ae28_path = Path(args.ae28_checkpoint)

    print(json.dumps({
        "event": "dump_start",
        "ae28_checkpoint": str(ae28_path),
        "student_checkpoint": str(args.student_checkpoint_dir),
        "output_dir": str(output_dir),
        "num_inference_steps": args.num_inference_steps,
    }), flush=True)

    # Import AE script
    ae = _import_ae_script()

    # Load student backbone
    student, student_tokenizer, student_processor, base_model = ae.load_student(args)

    # Load teacher model (for action space, diffusion, target computation)
    def _torch_dtype(name: str) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(name, torch.bfloat16)

    args.teacher_load_device = "cpu"
    print(json.dumps({"event": "load_teacher_start"}), flush=True)
    _load_fn = getattr(ae, "load_model_and_processor", None)
    if _load_fn is None:
        from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import load_model_and_processor as _load_fn
    teacher_model, _, _, _, _ = _load_fn(
        checkpoint_path=args.teacher_checkpoint_path,
        dtype=_torch_dtype(args.student_dtype),
        device=args.teacher_load_device,
        config_json=None,
        runtime_support=None,
        attn_implementation=args.attn_implementation,
        min_pixels=163840,
        max_pixels=196608,
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)
    teacher_model.to(device)
    print(json.dumps({"event": "load_teacher_done"}), flush=True)

    # Build AE28 bundle from teacher, then load trained weights
    bundle, selected_layers = ae.build_bundle(teacher_model, args, student=student)
    payload = ae.load_bundle_checkpoint(ae28_path, bundle=bundle)
    bundle.eval()
    for param in bundle.parameters():
        param.requires_grad_(False)
    bundle.to(device)
    print(json.dumps({
        "event": "ae28_loaded",
        "checkpoint": str(ae28_path),
        "step": payload.get("step", "unknown"),
        "selected_layers": selected_layers,
    }), flush=True)

    # Get train+val items
    train_items, val_items, split_summary = ae.select_train_val_items(args)
    all_items = train_items + val_items

    print(json.dumps({
        "event": "data_ready",
        "train": len(train_items),
        "val": len(val_items),
        "total": len(all_items),
    }), flush=True)

    batches = list(ae.iter_batches(all_items, args.batch_size))
    total_samples = 0
    total_bytes = 0
    skipped = 0
    started = time.time()

    for batch_idx, batch_items in enumerate(batches):
        # Build batch: backbone forward + KV cache + targets
        try:
            batch = ae.build_batch(
                args=args,
                student=student,
                student_processor=student_processor,
                student_tokenizer=student_tokenizer,
                teacher_model=teacher_model,
                batch_items=batch_items,
            )
        except Exception as e:
            print(json.dumps({"event": "batch_error", "batch": batch_idx, "error": str(e)}), flush=True)
            skipped += len(batch_items)
            continue

        # Run AE28 10-step inference with capturing
        prompt_cache = batch["cache"]
        context = batch["context"]
        features = capturing_inference(
            bundle=bundle,
            teacher_model=teacher_model,
            prompt_cache=prompt_cache,
            context=context,
            device=device,
            num_steps=args.num_inference_steps,
            temperature=args.temperature,
            seed=args.seed + batch_idx,
        )

        # Compute reflow targets
        reflow = build_reflow_targets(
            features["x_all_steps"],
            features["v_steps"],
            features["sampled_action"],
        )

        sample_ids = batch["sample_ids"]
        for i, sample_id in enumerate(sample_ids):
            safe_id = sample_id.replace("/", "_").replace("\\", "_")
            sample_path = output_dir / f"{safe_id}.pt"

            dump = {
                "sample_id": sample_id,
                # Teacher features
                "x_all_steps": features["x_all_steps"][i].cpu(),               # [11, 64, 2]
                "v_steps": features["v_steps"][i].cpu(),                       # [10, 64, 2]
                "last_hidden_steps": features["last_hidden_steps"][i].cpu(),   # [10, 64, 2048] bf16
                "time_steps": features["time_steps"].cpu(),                    # [11]
                "sampled_action": features["sampled_action"][i].cpu(),         # [64, 2]
                # Reflow targets
                "reflow_target_v_steps": reflow["reflow_target_v_steps"][i].cpu(),   # [10, 64, 2]
                "teacher_residual_steps": reflow["teacher_residual_steps"][i].cpu(), # [10, 64, 2]
                # GT targets
                "target_action": batch["target_action"][i].detach().cpu().float(),   # [64, 2]
                "target_xyz": batch["target_xyz"][i:i+1].detach().cpu().float(),
                "ego_history_xyz": batch["ego_history_xyz"][i].detach().cpu().float(),
                "ego_history_rot": batch["ego_history_rot"][i].detach().cpu().float(),
                # Context (for online KV regeneration during training)
                "kv_cache_seq_len": int(context["kv_cache_seq_len"]),
                "n_diffusion_tokens": int(context["n_diffusion_tokens"]),
                "position_ids": context["position_ids"][:, i:i+1, :].detach().cpu(),
                "stage2_attention_mode": str(context.get("stage2_attention_mode", "official_none")),
            }
            torch.save(dump, sample_path)
            total_bytes += sample_path.stat().st_size
            total_samples += 1

        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            elapsed = time.time() - started
            sps = total_samples / max(elapsed, 0.01)
            remaining = (len(all_items) - total_samples - skipped) / max(sps, 0.01)
            print(json.dumps({
                "event": "progress",
                "batch": batch_idx + 1,
                "total_batches": len(batches),
                "samples": total_samples,
                "skipped": skipped,
                "total": len(all_items),
                "elapsed_min": round(elapsed / 60, 1),
                "eta_min": round(remaining / 60, 1),
                "mb_per_sample": round(total_bytes / max(total_samples, 1) / 1e6, 1),
                "total_gb": round(total_bytes / 1e9, 2),
                "samples_per_sec": round(sps, 2),
            }), flush=True)

        del batch, features, reflow
        if (batch_idx + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Save manifest
    manifest = {
        "total_samples": total_samples,
        "skipped": skipped,
        "total_gb": round(total_bytes / 1e9, 2),
        "mb_per_sample": round(total_bytes / max(total_samples, 1) / 1e6, 1),
        "num_inference_steps": args.num_inference_steps,
        "temperature": args.temperature,
        "ae28_checkpoint": str(ae28_path),
        "student_checkpoint": str(args.student_checkpoint_dir),
        "prefix_mode": args.prefix_mode,
        "elapsed_min": round((time.time() - started) / 60, 1),
        "sample_ids": [item["sample_id"] for item in all_items if item["sample_id"]],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps({
        "event": "dump_done",
        "samples": total_samples,
        "skipped": skipped,
        "total_gb": round(total_bytes / 1e9, 2),
        "elapsed_min": round((time.time() - started) / 60, 1),
    }), flush=True)


if __name__ == "__main__":
    main()
