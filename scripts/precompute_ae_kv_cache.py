#!/usr/bin/env python3
"""Pre-compute student KV cache for offline AE training.

Runs student forward (ViT → FLEX → LLM) on all samples once, saves KV cache
+ targets to disk. AE training then loads from disk with --kv-cache-dir.

Usage:
    python scripts/precompute_ae_kv_cache.py \
        --student-checkpoint-dir outputs/checkpoints/qat_mlflex_k512_int4awq_20k_e3/final \
        --corpus-jsonl data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl \
        --output-dir outputs/kv_cache/qat_flex_k512_18k \
        --prefix-mode teacher_forced \
        --batch-size 4
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))


def _import_ae_script():
    """Import the AE training script despite its numeric filename."""
    spec = importlib.util.spec_from_file_location(
        "ae_train",
        str(PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="Pre-compute AE KV cache")
    parser.add_argument("--student-checkpoint-dir", type=str, required=True)
    parser.add_argument("--corpus-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=18000)
    parser.add_argument("--val-samples", type=int, default=1900)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefix-mode", type=str, default="teacher_forced",
                        choices=["teacher_forced", "student_free"])
    parser.add_argument("--preserve-flex-positions", action="store_true", default=True)
    parser.add_argument("--flex-selection-strategy", type=str, default="uniform")
    parser.add_argument("--flex-scene-deepstack", action="store_true", default=True)
    parser.add_argument("--qat-quantization", type=str, default="")
    parser.add_argument("--qat-calib-samples", type=int, default=256)
    parser.add_argument("--target-source", type=str, default="teacher")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--student-dtype", type=str, default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--split-scan-all", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(json.dumps({
        "event": "precompute_start",
        "checkpoint": args.student_checkpoint_dir,
        "num_samples": args.num_samples,
        "prefix_mode": args.prefix_mode,
        "output_dir": str(output_dir),
    }), flush=True)

    # Import AE script functions
    ae = _import_ae_script()

    # Convert string paths to Path objects (AE script expects Path)
    args.student_checkpoint_dir = Path(args.student_checkpoint_dir)
    args.corpus_jsonl = Path(args.corpus_jsonl)
    args.teacher_checkpoint_path = Path(args.teacher_checkpoint_path)

    # Load student + teacher
    student, student_tokenizer, student_processor, base_model = ae.load_student(args)

    # Load teacher model (for target action computation)
    def _torch_dtype(name: str) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(name, torch.bfloat16)

    args.teacher_load_device = "cpu"
    print(json.dumps({"event": "load_teacher_start"}), flush=True)
    # Use the same loader as AE script
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

    # Get train/val items
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
    started = time.time()

    for batch_idx, batch_items in enumerate(batches):
        batch = ae.build_batch(
            args=args,
            student=student,
            student_processor=student_processor,
            student_tokenizer=student_tokenizer,
            teacher_model=teacher_model,
            batch_items=batch_items,
        )

        cache = batch["cache"]
        sample_ids = batch["sample_ids"]

        for i, sample_id in enumerate(sample_ids):
            safe_id = sample_id.replace("/", "_").replace("\\", "_")
            sample_path = output_dir / f"{safe_id}.pt"

            kv_list = []
            for layer_idx in range(len(cache)):
                k = cache[layer_idx][0][i:i+1].detach().cpu()
                v = cache[layer_idx][1][i:i+1].detach().cpu()
                kv_list.append((k, v))

            ctx = batch["context"]
            position_ids_i = ctx["position_ids"][:, i:i+1, :].detach().cpu()

            sample_data = {
                "sample_id": sample_id,
                "kv_cache": kv_list,
                "kv_cache_seq_len": ctx["kv_cache_seq_len"],
                "n_diffusion_tokens": ctx["n_diffusion_tokens"],
                "position_ids": position_ids_i,
                "attention_mask": ctx.get("attention_mask"),
                "stage2_attention_mode": ctx["stage2_attention_mode"],
                "target_action": batch["target_action"][i:i+1].detach().cpu(),
                "target_xyz": batch["target_xyz"][i:i+1].detach().cpu(),
                "ego_history_xyz": batch["ego_history_xyz"][i:i+1].detach().cpu(),
                "ego_history_rot": batch["ego_history_rot"][i:i+1].detach().cpu(),
            }

            torch.save(sample_data, sample_path)
            total_bytes += sample_path.stat().st_size
            total_samples += 1

        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            elapsed = time.time() - started
            sps = total_samples / max(elapsed, 0.01)
            remaining = (len(all_items) - total_samples) / max(sps, 0.01)
            print(json.dumps({
                "event": "progress",
                "batch": batch_idx + 1,
                "samples": total_samples,
                "total": len(all_items),
                "elapsed_min": round(elapsed / 60, 1),
                "eta_min": round(remaining / 60, 1),
                "mb_per_sample": round(total_bytes / max(total_samples, 1) / 1e6, 1),
                "total_gb": round(total_bytes / 1e9, 1),
            }), flush=True)

        del batch
        if (batch_idx + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    manifest = {
        "total_samples": total_samples,
        "total_gb": round(total_bytes / 1e9, 1),
        "mb_per_sample": round(total_bytes / max(total_samples, 1) / 1e6, 1),
        "prefix_mode": args.prefix_mode,
        "student_checkpoint": args.student_checkpoint_dir,
        "qat_quantization": args.qat_quantization or "none",
        "elapsed_min": round((time.time() - started) / 60, 1),
        "sample_ids": [item["sample_id"] for item in all_items],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps({
        "event": "precompute_done",
        "samples": total_samples,
        "total_gb": round(total_bytes / 1e9, 1),
        "elapsed_min": round((time.time() - started) / 60, 1),
    }), flush=True)


if __name__ == "__main__":
    main()
