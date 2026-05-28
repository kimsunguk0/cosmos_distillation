#!/usr/bin/env python3
"""KV-cache distillation training pipeline.

Teacher  : Alpamayo-1.5-10B VLM (Qwen3-VL, 36 layers, KV-dim=1024), fully frozen.
Student  : Cosmos-Reason2-2B (28 layers, KV-dim=1024), trainable backbone + new
           vit_projection Linear(4096→2048).

Loss     : CE (student auto-regressive) + weighted KV-distillation loss
           (per-layer RMSNorm + Huber on K and V, optional K-gram term,
           linear layer weight schedule with maximum weight at the last layer).

Layer mapping (student_idx → teacher_idx, 0-indexed):
  0→0, 1→1, 2→3, 3→4, 4→5, 5→6, 6→8, 7→9, 8→10, 9→12, 10→13, 11→14,
  12→16, 13→17, 14→18, 15→19, 16→21, 17→22, 18→23, 19→25, 20→26, 21→27,
  22→29, 23→30, 24→31, 25→32, 26→34, 27→35
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer


# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUKIM_ROOT = PROJECT_ROOT.parents[1]
for path in (PROJECT_ROOT, SUKIM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.model.student_wrapper import (  # noqa: E402
    StudentWrapperConfig,
    build_student_model,
    load_student_processor,
    load_student_tokenizer,
)
from src.training.losses import kv_cache_distillation_loss  # noqa: E402
from src.utils.runtime_paths import remap_external_path  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "kv_distill_7k_balanced.jsonl"
DEFAULT_STUDENT_MODEL = str(SUKIM_ROOT / "base_weights" / "cosmos-reason-2b")
DEFAULT_TEACHER_VLM = str(SUKIM_ROOT / "base_weights" / "alpamayo15_vlm_weights")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "kv_distill_pipeline"

# Layer mapping: (student_layer_idx, teacher_layer_idx), 0-indexed
LAYER_MAPPING: list[tuple[int, int]] = [
    (0, 0), (1, 1), (2, 3), (3, 4), (4, 5), (5, 6),
    (6, 8), (7, 9), (8, 10), (9, 12), (10, 13), (11, 14),
    (12, 16), (13, 17), (14, 18), (15, 19), (16, 21), (17, 22),
    (18, 23), (19, 25), (20, 26), (21, 27), (22, 29), (23, 30),
    (24, 31), (25, 32), (26, 34), (27, 35),
]

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--student-model", default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--teacher-vlm", default=DEFAULT_TEACHER_VLM)
    parser.add_argument("--student-checkpoint", type=Path, default=None,
                        help="Optional path to an existing student checkpoint (.pt).")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of training samples (for debugging).")
    parser.add_argument("--kv-loss-weight", type=float, default=0.1)
    parser.add_argument("--kv-huber-delta", type=float, default=1.0)
    parser.add_argument("--kv-gram-weight", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--teacher-device", default="cuda:0",
                        help="Device for teacher (cpu recommended when VRAM is tight).")
    parser.add_argument("--attn-implementation",
                        choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--seed", type=int, default=97)
    return parser.parse_args()


def _dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def _resolve_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    remapped = remap_external_path(raw)
    if remapped is None:
        return None
    p = Path(remapped)
    return p if p.exists() else None


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _unwrap_text(value: Any) -> str:
    while isinstance(value, list) and value:
        value = value[0]
    return str(value or "").strip()


def teacher_cot_text(row: dict) -> str:
    for key in ("teacher_target", "hard_target"):
        text = _unwrap_text((row.get(key) or {}).get("cot_text"))
        if text:
            return text
    return ""


def select_items(corpus: list[dict], split: str, max_samples: int | None) -> list[dict]:
    items = []
    for row in corpus:
        if split and row.get("split") != split:
            continue
        sample_dir = _resolve_path((row.get("input") or {}).get("materialized_sample_path"))
        if sample_dir is None:
            continue
        items.append({"row": row, "sample_dir": sample_dir})
        if max_samples is not None and len(items) >= max_samples:
            break
    return items


def load_images_for_row(row: dict) -> list[Any]:
    """Load PIL images from the materialized sample directory."""
    from PIL import Image

    sample_dir = _resolve_path((row.get("input") or {}).get("materialized_sample_path"))
    if sample_dir is None:
        return []
    image_names = sorted(p.name for p in sample_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    images = []
    for name in image_names:
        try:
            img = Image.open(sample_dir / name).convert("RGB")
            images.append(img)
        except Exception:  # noqa: BLE001
            pass
    return images


def build_prompt_text(row: dict) -> str:
    """Return a simple driving assistant user prompt from the row metadata."""
    metadata = row.get("metadata") or {}
    location = metadata.get("location_description") or "unknown location"
    return f"You are in a vehicle at {location}. Describe what you observe and decide the appropriate driving action."


def encode_batch(
    processor: Any,
    tokenizer: Any,
    items: list[dict],
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build a padded batch from a list of corpus items."""
    from transformers import BatchFeature

    batch_input_ids: list[torch.Tensor] = []
    batch_labels: list[torch.Tensor] = []
    batch_pixel_values: list[torch.Tensor] = []
    batch_image_grid_thw: list[torch.Tensor] = []

    for item in items:
        row = item["row"]
        images = load_images_for_row(row)
        cot = teacher_cot_text(row)

        # Build a minimal chat message
        user_content: list[dict] = []
        for _ in images:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": build_prompt_text(row)})

        assistant_text = cot if cot else "The vehicle should proceed safely."
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_text},
        ]
        try:
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            if images:
                inputs = processor(
                    text=[text_prompt],
                    images=images,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )
            else:
                inputs = tokenizer(
                    text_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
        except Exception as exc:  # noqa: BLE001
            # Fall back to text-only on image processing failure
            fallback_text = f"User: {build_prompt_text(row)}\nAssistant: {assistant_text}"
            inputs = tokenizer(
                fallback_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            print(f"WARNING: image encode failed ({exc}), falling back to text-only.", flush=True)

        ids = inputs["input_ids"][0]  # [seq]
        # Labels: mask everything before the assistant turn with IGNORE_INDEX
        # Simple heuristic: find the last occurrence of the assistant prefix tokens
        labels = ids.clone()
        # mask padding/system tokens (we train on full sequence for simplicity)
        batch_input_ids.append(ids)
        batch_labels.append(labels)
        if "pixel_values" in inputs:
            batch_pixel_values.append(inputs["pixel_values"])
        if "image_grid_thw" in inputs:
            batch_image_grid_thw.append(inputs["image_grid_thw"])

    # Pad to max seq length in batch
    max_seq = max(t.shape[0] for t in batch_input_ids)
    pad_id = tokenizer.pad_token_id or 0

    padded_ids = torch.full((len(items), max_seq), pad_id, dtype=torch.long)
    padded_labels = torch.full((len(items), max_seq), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(items), max_seq), dtype=torch.long)

    for i, (ids, labs) in enumerate(zip(batch_input_ids, batch_labels)):
        seq_len = ids.shape[0]
        padded_ids[i, :seq_len] = ids
        padded_labels[i, :seq_len] = labs
        attention_mask[i, :seq_len] = 1

    result = {
        "input_ids": padded_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": padded_labels.to(device),
    }
    if batch_pixel_values:
        result["pixel_values"] = torch.cat(batch_pixel_values, dim=0).to(device)
    if batch_image_grid_thw:
        result["image_grid_thw"] = torch.cat(batch_image_grid_thw, dim=0).to(device)
    return result


# ---------------------------------------------------------------------------
# Teacher forward + KV / ViT feature extraction
# ---------------------------------------------------------------------------


def extract_teacher_outputs(
    teacher_vlm: nn.Module,
    batch: dict[str, torch.Tensor],
    teacher_device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor | None]:
    """Run teacher forward pass and return (teacher_kvs, teacher_image_embeds).

    teacher_kvs:          list of (K, V) per teacher layer.
    teacher_image_embeds: [N_image_tokens, 4096] or None if no images.
    """
    # Move batch to teacher device
    t_input_ids = batch["input_ids"].to(teacher_device)
    t_attn_mask = batch["attention_mask"].to(teacher_device)
    t_pixel_values = batch.get("pixel_values")
    t_grid_thw = batch.get("image_grid_thw")

    teacher_visual_feats: dict[str, torch.Tensor] = {}
    hook_handle = None

    if t_pixel_values is not None:
        t_pixel_values = t_pixel_values.to(device=teacher_device, dtype=dtype)

        # Register hook on the visual merger to capture 4096-dim image embeddings
        visual_module = None
        for attr_path in ("visual", "model.visual"):
            obj = teacher_vlm
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                visual_module = obj
                break
            except AttributeError:
                pass

        if visual_module is not None:
            merger = getattr(visual_module, "merger", None)
            if merger is not None:
                def _hook(module: nn.Module, inp: Any, out: Any) -> None:
                    teacher_visual_feats["embeds"] = out.detach()

                hook_handle = merger.register_forward_hook(_hook)

    with torch.no_grad():
        fwd_kwargs: dict[str, Any] = {
            "input_ids": t_input_ids,
            "attention_mask": t_attn_mask,
            "use_cache": True,
            "return_dict": True,
            "output_hidden_states": False,
        }
        if t_pixel_values is not None and t_grid_thw is not None:
            fwd_kwargs["pixel_values"] = t_pixel_values
            fwd_kwargs["image_grid_thw"] = t_grid_thw.to(teacher_device)

        try:
            teacher_out = teacher_vlm(**fwd_kwargs)
        except Exception as exc:  # noqa: BLE001
            # Some models wrap their LM under .model – try that
            try:
                teacher_out = teacher_vlm.model(**fwd_kwargs)
            except Exception:
                raise exc
        finally:
            if hook_handle is not None:
                hook_handle.remove()

    # Extract past_key_values
    pkv = getattr(teacher_out, "past_key_values", None)
    teacher_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
    if pkv is not None:
        # Handle both tuple-of-tuples and Cache objects
        if hasattr(pkv, "key_cache"):
            # transformers Cache object
            for k, v in zip(pkv.key_cache, pkv.value_cache):
                teacher_kvs.append((k.detach().cpu(), v.detach().cpu()))
        else:
            for layer_kv in pkv:
                if isinstance(layer_kv, (tuple, list)) and len(layer_kv) >= 2:
                    teacher_kvs.append((layer_kv[0].detach().cpu(), layer_kv[1].detach().cpu()))

    teacher_image_embeds = teacher_visual_feats.get("embeds")
    return teacher_kvs, teacher_image_embeds


# ---------------------------------------------------------------------------
# Student forward + KV extraction
# ---------------------------------------------------------------------------


def extract_student_kvs(
    student_backbone: nn.Module,  # unused param (kept for call-site compat)
    batch: dict[str, torch.Tensor],
    student_model: Any,
    device: torch.device,
    dtype: torch.dtype,
    teacher_image_embeds: torch.Tensor | None,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Run student backbone forward and return (logits, student_kvs).

    If teacher_image_embeds is provided and student has vit_projection,
    inject projected teacher features instead of student ViT features.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch.get("pixel_values")
    image_grid_thw = batch.get("image_grid_thw")
    image_token_id = getattr(student_model, "image_token_id", None)

    fwd_kwargs: dict[str, Any] = {
        "attention_mask": attention_mask,
        "use_cache": True,
        "return_dict": True,
        "output_hidden_states": False,
    }

    # Decide whether to use projected teacher ViT features
    use_teacher_vit = (
        teacher_image_embeds is not None
        and student_model.vit_projection is not None
        and image_token_id is not None
    )

    if use_teacher_vit:
        assert teacher_image_embeds is not None
        t_embeds = teacher_image_embeds.to(device=device, dtype=dtype)
        try:
            inputs_embeds = student_model.embed_with_teacher_vit_features(
                input_ids, t_embeds, image_token_id
            )
            fwd_kwargs["inputs_embeds"] = inputs_embeds
            # Do NOT pass input_ids when inputs_embeds is given
            student_out = student_model.backbone(
                input_ids=None,
                **fwd_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: embed_with_teacher_vit_features failed ({exc}); falling back to standard forward.", flush=True)
            use_teacher_vit = False

    if not use_teacher_vit:
        std_kwargs: dict[str, Any] = dict(fwd_kwargs)
        if pixel_values is not None:
            std_kwargs["pixel_values"] = pixel_values.to(dtype=dtype)
        if image_grid_thw is not None:
            std_kwargs["image_grid_thw"] = image_grid_thw
        student_out = student_model.backbone(
            input_ids=input_ids,
            **std_kwargs,
        )

    logits = getattr(student_out, "logits", None)
    if logits is None:
        raise ValueError("Student backbone did not return logits.")

    pkv = getattr(student_out, "past_key_values", None)
    student_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
    if pkv is not None:
        if hasattr(pkv, "key_cache"):
            for k, v in zip(pkv.key_cache, pkv.value_cache):
                student_kvs.append((k, v))
        else:
            for layer_kv in pkv:
                if isinstance(layer_kv, (tuple, list)) and len(layer_kv) >= 2:
                    student_kvs.append((layer_kv[0], layer_kv[1]))

    return logits, student_kvs


# ---------------------------------------------------------------------------
# CE loss
# ---------------------------------------------------------------------------


def causal_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard causal LM cross-entropy, shifted by 1."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def save_checkpoint(path: Path, student: Any, optimizer: Any, step: int, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **extra,
        },
        path,
    )
    print(json.dumps({"event": "checkpoint_saved", "path": str(path), "step": step}), flush=True)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    teacher_device = torch.device(args.teacher_device)
    dtype = _dtype_from_name(args.dtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    log_handle = log_path.open("a", encoding="utf-8")

    summary: dict[str, Any] = {"args": {k: str(v) if hasattr(v, "__fspath__") else v for k, v in vars(args).items()}, "status": "running"}
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # 1. Load teacher VLM (frozen)
    # ------------------------------------------------------------------
    print(json.dumps({"event": "loading_teacher", "path": args.teacher_vlm}), flush=True)
    teacher_vlm = AutoModelForVision2Seq.from_pretrained(
        args.teacher_vlm,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(teacher_device)
    teacher_vlm.eval()
    teacher_vlm.requires_grad_(False)
    print(json.dumps({"event": "teacher_loaded"}), flush=True)

    # ------------------------------------------------------------------
    # 2. Load student model (trainable)
    # ------------------------------------------------------------------
    print(json.dumps({"event": "loading_student", "path": args.student_model}), flush=True)
    _student_local = Path(args.student_model).exists()
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student_model, trust_remote_code=True, local_files_only=_student_local
    )
    student_processor = AutoProcessor.from_pretrained(
        args.student_model,
        trust_remote_code=True,
        local_files_only=_student_local,
    )
    student_processor.tokenizer = student_tokenizer

    wrapper_config = StudentWrapperConfig(
        student_model_name=args.student_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=Path(args.student_model).exists(),
        attn_implementation=args.attn_implementation,
        vit_in_dim=4096,
        use_vit_projection=True,
    )
    student = build_student_model(wrapper_config, student_tokenizer)
    student = student.to(device=device, dtype=dtype)
    student.train()

    if args.student_checkpoint is not None:
        ckpt = torch.load(args.student_checkpoint, map_location=device)
        state = ckpt.get("student_state_dict", ckpt)
        missing, unexpected = student.load_state_dict(state, strict=False)
        print(json.dumps({
            "event": "student_checkpoint_loaded",
            "missing": len(missing),
            "unexpected": len(unexpected),
        }), flush=True)

    print(json.dumps({"event": "student_loaded"}), flush=True)

    # ------------------------------------------------------------------
    # 3. Optimizer (only student parameters)
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ------------------------------------------------------------------
    # 4. Data
    # ------------------------------------------------------------------
    corpus = read_jsonl(args.corpus_jsonl)
    items = select_items(corpus, args.split, args.max_samples)
    if not items:
        raise RuntimeError(f"No usable items found in {args.corpus_jsonl} with split={args.split!r}.")
    print(json.dumps({"event": "data_loaded", "n_items": len(items)}), flush=True)

    import random
    rng = random.Random(args.seed)

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(items)
        batches = [items[i: i + args.batch_size] for i in range(0, len(items), args.batch_size)]

        for batch_items in batches:
            global_step += 1
            optimizer.zero_grad(set_to_none=True)

            # Encode batch
            batch = encode_batch(
                student_processor,
                student_tokenizer,
                batch_items,
                args.max_length,
                device,
            )
            labels = batch.pop("labels")

            # ------------------------------------------------------------------
            # Teacher forward (no_grad, teacher device)
            # ------------------------------------------------------------------
            with torch.no_grad():
                teacher_batch_cpu = {k: v.cpu() for k, v in batch.items()}
                teacher_batch_td: dict[str, torch.Tensor] = {}
                for k, v in batch.items():
                    teacher_batch_td[k] = v.to(teacher_device)

                teacher_kvs, teacher_image_embeds = extract_teacher_outputs(
                    teacher_vlm, teacher_batch_td, teacher_device, dtype
                )

            # Move teacher KVs to student device (lazy: done inside loss)
            teacher_kvs_dev = [
                (k.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype))
                for k, v in teacher_kvs
            ]
            if teacher_image_embeds is not None:
                teacher_image_embeds = teacher_image_embeds.to(device=device, dtype=dtype)

            # ------------------------------------------------------------------
            # Student forward
            # ------------------------------------------------------------------
            logits, student_kvs = extract_student_kvs(
                student_backbone=student.backbone,
                batch=batch,
                student_model=student,
                device=device,
                dtype=dtype,
                teacher_image_embeds=teacher_image_embeds,
            )

            # ------------------------------------------------------------------
            # Losses
            # ------------------------------------------------------------------
            ce_loss = causal_ce_loss(logits, labels)

            stats: dict[str, float] = {"ce_loss": float(ce_loss.detach().cpu())}

            if student_kvs and teacher_kvs_dev and len(teacher_kvs_dev) > max(t for _, t in LAYER_MAPPING):
                kv_loss, kv_stats = kv_cache_distillation_loss(
                    student_kvs=student_kvs,
                    teacher_kvs=teacher_kvs_dev,
                    layer_mapping=LAYER_MAPPING,
                    huber_delta=args.kv_huber_delta,
                    gram_weight=args.kv_gram_weight,
                )
                stats.update(kv_stats)
                total_loss = ce_loss + args.kv_loss_weight * kv_loss
                stats["kv_loss_weighted"] = float((args.kv_loss_weight * kv_loss).detach().cpu())
            else:
                total_loss = ce_loss
                if not teacher_kvs_dev:
                    print("WARNING: No teacher KVs extracted; skipping KV loss this step.", flush=True)

            stats["total_loss"] = float(total_loss.detach().cpu())

            # ------------------------------------------------------------------
            # Backward
            # ------------------------------------------------------------------
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, student.parameters()),
                args.grad_clip_norm,
            )
            optimizer.step()

            stats["grad_norm"] = float(grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm)

            if global_step % args.log_every == 0:
                row = {
                    "event": "train_step",
                    "epoch": epoch,
                    "step": global_step,
                    "elapsed_sec": round(time.perf_counter() - started, 3),
                    **stats,
                }
                print(json.dumps(row), flush=True)
                log_handle.write(json.dumps(row) + "\n")
                log_handle.flush()

            if args.save_every and global_step % args.save_every == 0:
                save_checkpoint(
                    args.output_dir / f"step_{global_step:06d}.pt",
                    student=student,
                    optimizer=optimizer,
                    step=global_step,
                    extra={"epoch": epoch},
                )

            # Cleanup to avoid OOM
            del batch, labels, logits, student_kvs, teacher_kvs, teacher_kvs_dev
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(json.dumps({"event": "epoch_done", "epoch": epoch, "step": global_step}), flush=True)

    # Final checkpoint
    save_checkpoint(
        args.output_dir / "final.pt",
        student=student,
        optimizer=optimizer,
        step=global_step,
        extra={"epochs": args.epochs},
    )

    summary.update({"status": "ok", "total_steps": global_step, "elapsed_sec": round(time.perf_counter() - started, 3)})
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log_handle.close()
    print(json.dumps({"event": "done", "summary_json": str(summary_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
