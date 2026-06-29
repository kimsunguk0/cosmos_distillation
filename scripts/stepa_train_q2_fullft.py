#!/usr/bin/env python3
"""Full fine-tune Cosmos-Reason2-2B on Step A Q2 VQA distillation."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vqa.q2_stepa import (
    IGNORE_INDEX,
    create_vqa_messages,
    encode_messages,
    labels_from_prompt_and_full,
    load_row_frame_tensors,
    read_jsonl,
)


DEFAULT_MODEL = Path("/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b")
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "vqa_q2_stepa" / "teacher_topk32" / "records_with_topk.jsonl"
DEFAULT_VAL = PROJECT_ROOT / "data" / "vqa_q2_stepa" / "val_q2_stepa.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "checkpoints" / "stepa_q2_vqa_fullft_smoke"
VQA_CONTROL_TOKENS = ["<|question_start|>", "<|question_end|>", "<|answer_start|>", "<|answer_end|>"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


class StepAQ2Dataset(Dataset):
    def __init__(self, path: Path, *, max_samples: int | None = None, require_topk: bool = True) -> None:
        rows = read_jsonl(path)
        if require_topk:
            rows = [row for row in rows if row.get("teacher_topk_ready") and row.get("teacher_topk_path")]
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        if not rows:
            raise ValueError(f"No rows loaded from {path}")
        self.path = path
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        frames, camera_indices = load_row_frame_tensors(row)
        item = {
            "row": row,
            "frames": frames,
            "camera_indices": camera_indices,
        }
        topk_path = row.get("teacher_topk_path")
        if topk_path:
            topk = np.load(str(topk_path))
            item["teacher_topk_indices"] = topk["topk_indices"].astype(np.int64)
            item["teacher_topk_logprobs"] = topk["topk_logprobs"].astype(np.float32)
            item["teacher_target_token_ids"] = topk["target_token_ids"].astype(np.int64)
        return item


@dataclass(slots=True)
class BatchCollator:
    processor: Any
    max_length: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_messages = []
        full_messages = []
        sample_ids = []
        raw_rows = []
        for item in features:
            row = item["row"]
            sample_ids.append(str(row["sample_id"]))
            raw_rows.append(row)
            prompt_messages.append(
                create_vqa_messages(
                    frames=item["frames"],
                    camera_indices=item["camera_indices"],
                    question=str(row["question"]),
                    answer_text=None,
                )
            )
            full_messages.append(
                create_vqa_messages(
                    frames=item["frames"],
                    camera_indices=item["camera_indices"],
                    question=str(row["question"]),
                    answer_text=str(row["teacher_answer_short"]),
                )
            )
        prompt_batch = encode_messages(
            self.processor,
            prompt_messages,
            continue_final_message=True,
            max_length=self.max_length,
        )
        full_batch = encode_messages(
            self.processor,
            full_messages,
            continue_final_message=False,
            max_length=self.max_length,
        )
        labels = labels_from_prompt_and_full(prompt_batch, full_batch)
        batch: dict[str, Any] = {
            "input_ids": full_batch["input_ids"],
            "attention_mask": full_batch["attention_mask"],
            "pixel_values": full_batch.get("pixel_values"),
            "image_grid_thw": full_batch.get("image_grid_thw"),
            "labels": labels,
            "sample_ids": sample_ids,
            "rows": raw_rows,
        }

        topk_items = [item for item in features if "teacher_topk_indices" in item]
        if topk_items:
            max_tokens = max(int(item["teacher_topk_indices"].shape[0]) for item in features)
            k = int(topk_items[0]["teacher_topk_indices"].shape[-1])
            indices = torch.zeros((len(features), max_tokens, k), dtype=torch.long)
            logprobs = torch.zeros((len(features), max_tokens, k), dtype=torch.float32)
            mask = torch.zeros((len(features), max_tokens), dtype=torch.bool)
            target_ids = torch.full((len(features), max_tokens), -1, dtype=torch.long)
            for row_idx, item in enumerate(features):
                if "teacher_topk_indices" not in item:
                    continue
                token_count = int(item["teacher_topk_indices"].shape[0])
                indices[row_idx, :token_count] = torch.from_numpy(item["teacher_topk_indices"]).long()
                logprobs[row_idx, :token_count] = torch.from_numpy(item["teacher_topk_logprobs"]).float()
                target_ids[row_idx, :token_count] = torch.from_numpy(item["teacher_target_token_ids"]).long()
                mask[row_idx, :token_count] = True
            batch["teacher_topk_indices"] = indices
            batch["teacher_topk_logprobs"] = logprobs
            batch["teacher_topk_mask"] = mask
            batch["teacher_target_token_ids"] = target_ids
        return batch


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def answer_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != IGNORE_INDEX
    token_count = int(valid.sum().item())
    if token_count == 0:
        return torch.zeros((), device=logits.device), 0
    loss = F.cross_entropy(shift_logits[valid], shift_labels[valid], reduction="mean")
    return loss, token_count


def answer_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != IGNORE_INDEX
    if not valid.any():
        return torch.zeros((), device=logits.device)
    pred = shift_logits[valid].argmax(dim=-1)
    return (pred == shift_labels[valid]).float().mean()


def sparse_topk_kd_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_topk_indices: torch.Tensor | None,
    teacher_topk_logprobs: torch.Tensor | None,
    teacher_topk_mask: torch.Tensor | None,
    *,
    temperature: float,
) -> torch.Tensor:
    """Standard sparse teacher-to-student KL on answer token positions.

    Reverse KL is ill-defined with teacher-only top-k support, so this uses the
    standard KD direction: KL(teacher_topk || student_on_teacher_topk).
    """
    if teacher_topk_indices is None or teacher_topk_logprobs is None or teacher_topk_mask is None:
        return torch.zeros((), device=logits.device)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab_size = int(shift_logits.shape[-1])
    losses: list[torch.Tensor] = []
    for row_idx in range(shift_logits.shape[0]):
        active_shift_positions = torch.nonzero(shift_labels[row_idx] != IGNORE_INDEX, as_tuple=False).flatten()
        available = torch.nonzero(teacher_topk_mask[row_idx], as_tuple=False).flatten()
        n = min(int(active_shift_positions.numel()), int(available.numel()))
        if n <= 0:
            continue
        active_shift_positions = active_shift_positions[:n]
        topk_rows = available[:n]
        student_rows = shift_logits[row_idx].index_select(0, active_shift_positions)
        row_losses: list[torch.Tensor] = []
        for token_idx in range(n):
            ids = teacher_topk_indices[row_idx, topk_rows[token_idx]].to(logits.device)
            vals = teacher_topk_logprobs[row_idx, topk_rows[token_idx]].to(logits.device)
            keep = (ids >= 0) & (ids < vocab_size)
            if not keep.any():
                continue
            ids = ids[keep]
            vals = vals[keep]
            selected_student = student_rows[token_idx].gather(0, ids)
            student_log_probs = F.log_softmax(selected_student / float(temperature), dim=-1)
            teacher_probs = F.softmax(vals / float(temperature), dim=-1)
            row_losses.append(F.kl_div(student_log_probs, teacher_probs, reduction="sum") * (float(temperature) ** 2))
        if row_losses:
            losses.append(torch.stack(row_losses).mean())
    if not losses:
        return torch.zeros((), device=logits.device)
    return torch.stack(losses).mean()


def load_student(args: argparse.Namespace, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.student_model),
        trust_remote_code=True,
        local_files_only=Path(args.student_model).exists(),
    )
    if args.add_vqa_control_tokens:
        missing = [token for token in VQA_CONTROL_TOKENS if token not in tokenizer.get_vocab()]
        if missing:
            tokenizer.add_special_tokens({"additional_special_tokens": missing})
    tokenizer.padding_side = "right"
    processor = AutoProcessor.from_pretrained(
        str(args.student_model),
        trust_remote_code=True,
        local_files_only=Path(args.student_model).exists(),
        min_pixels=int(args.min_pixels),
        max_pixels=int(args.max_pixels),
    )
    processor.tokenizer = tokenizer
    processor.tokenizer.padding_side = "right"
    model = AutoModelForVision2Seq.from_pretrained(
        str(args.student_model),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=Path(args.student_model).exists(),
    )
    embedding_rows = int(model.get_input_embeddings().num_embeddings)
    if len(tokenizer) > embedding_rows:
        model.resize_token_embeddings(len(tokenizer))
    if bool(args.gradient_checkpointing):
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.to(device)
    model.train()
    return model, processor, tokenizer


def make_optimizer(args: argparse.Namespace, model) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    if args.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            if not args.allow_adamw_fallback:
                raise RuntimeError(
                    "bitsandbytes is not installed, but --optimizer adamw8bit was requested. "
                    "Install bitsandbytes or pass --optimizer adamw --allow-adamw-fallback for a smoke run."
                ) from exc
            return torch.optim.AdamW(params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
        return bnb.optim.AdamW8bit(params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    if args.optimizer == "adamw":
        if not args.allow_adamw_fallback:
            raise RuntimeError("Use --allow-adamw-fallback when choosing full AdamW for a smoke run.")
        return torch.optim.AdamW(params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def make_scheduler(optimizer: torch.optim.Optimizer, *, max_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(round(int(max_steps) * float(warmup_ratio))))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, int(max_steps) - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(output_dir: Path, model, processor, tokenizer, *, step: int, summary: dict[str, Any]) -> None:
    ckpt = output_dir / f"step_{int(step):06d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt), safe_serialization=True)
    tokenizer.save_pretrained(str(ckpt / "tokenizer"))
    processor.save_pretrained(str(ckpt / "processor"))
    save_json(ckpt / "train_summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--student-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument("--kd-temperature", type=float, default=1.5)
    parser.add_argument("--optimizer", choices=["adamw8bit", "adamw"], default="adamw8bit")
    parser.add_argument("--allow-adamw-fallback", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-vqa-control-tokens", action="store_true")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-pixels", type=int, default=163840)
    parser.add_argument("--max-pixels", type=int, default=196608)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this command outside the sandbox on the H200.")
    device = torch.device(args.device if str(args.device).startswith("cuda") else "cpu")
    started = time.time()
    model, processor, tokenizer = load_student(args, device)
    collator = BatchCollator(processor=processor, max_length=args.max_length)
    train_ds = StepAQ2Dataset(args.train_jsonl, max_samples=args.max_train_samples, require_topk=float(args.lambda_kl) > 0)
    loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collator,
    )

    first_batch = next(iter(loader))
    labeled_counts = (first_batch["labels"] != IGNORE_INDEX).sum(dim=1).tolist()
    sanity = {
        "event": "mask_sanity",
        "sample_ids": first_batch["sample_ids"],
        "input_shape": list(first_batch["input_ids"].shape),
        "labeled_token_counts": [int(v) for v in labeled_counts],
        "has_topk": "teacher_topk_indices" in first_batch,
        "tokenizer_len": int(len(tokenizer)),
        "model_embedding_rows": int(model.get_input_embeddings().num_embeddings),
        "add_vqa_control_tokens": bool(args.add_vqa_control_tokens),
    }
    print(json.dumps(sanity, ensure_ascii=True), flush=True)
    if args.sanity_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(args.output_dir / "sanity_summary.json", sanity)
        return

    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(optimizer, max_steps=int(args.max_steps), warmup_ratio=float(args.warmup_ratio))
    scaler = None
    global_step = 0
    running: dict[str, float] = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "acc": 0.0}
    optimizer.zero_grad(set_to_none=True)
    autocast_context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    while global_step < int(args.max_steps):
        for batch in loader:
            batch = move_to_device(batch, device)
            with autocast_context:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch.get("pixel_values"),
                    image_grid_thw=batch.get("image_grid_thw"),
                    use_cache=False,
                    return_dict=True,
                )
                ce, token_count = answer_ce_loss(outputs.logits, batch["labels"])
                kl = sparse_topk_kd_loss(
                    outputs.logits,
                    batch["labels"],
                    batch.get("teacher_topk_indices"),
                    batch.get("teacher_topk_logprobs"),
                    batch.get("teacher_topk_mask"),
                    temperature=float(args.kd_temperature),
                )
                loss = ce + float(args.lambda_kl) * kl
                loss = loss / int(args.grad_accum_steps)
            loss.backward()
            if (global_step + 1) % int(args.grad_accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            acc = answer_token_accuracy(outputs.logits.detach(), batch["labels"])
            running["loss"] += float((loss * int(args.grad_accum_steps)).detach().cpu())
            running["ce"] += float(ce.detach().cpu())
            running["kl"] += float(kl.detach().cpu())
            running["acc"] += float(acc.detach().cpu())
            global_step += 1
            if int(args.log_every) > 0 and global_step % int(args.log_every) == 0:
                denom = float(int(args.log_every))
                log = {
                    "event": "train_step",
                    "step": global_step,
                    "max_steps": int(args.max_steps),
                    "loss": running["loss"] / denom,
                    "ce": running["ce"] / denom,
                    "kl": running["kl"] / denom,
                    "answer_token_acc": running["acc"] / denom,
                    "answer_tokens": int(token_count),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                print(json.dumps(log, ensure_ascii=True), flush=True)
                running = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "acc": 0.0}
            if global_step >= int(args.max_steps):
                break

    summary = {
        "created_at": utc_now(),
        "train_jsonl": str(args.train_jsonl),
        "student_model": str(args.student_model),
        "output_dir": str(args.output_dir),
        "max_steps": int(args.max_steps),
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "learning_rate": float(args.learning_rate),
        "lambda_kl": float(args.lambda_kl),
        "kd_temperature": float(args.kd_temperature),
        "optimizer": str(args.optimizer),
        "elapsed_sec": round(time.time() - started, 3),
        "sanity": sanity,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "train_summary.json", summary)
    if args.save_final:
        save_checkpoint(args.output_dir, model, processor, tokenizer, step=global_step, summary=summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

