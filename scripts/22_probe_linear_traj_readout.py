#!/usr/bin/env python3
"""Train frozen-feature linear probes for trajectory-token readout."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.checkpoint_io import load_student_checkpoint
from src.model.student_wrapper import (
    StudentWrapperConfig,
    build_student_model,
    load_student_processor,
    load_student_tokenizer,
)
from src.training.collator import DistillationCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "distill_v3_2_959.jsonl",
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--prefix-tokens", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "linear_traj_readout_probe.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _traj_vocab_size(tokenizer) -> tuple[int, int]:
    start_idx = int(getattr(tokenizer, "traj_token_start_idx", 0))
    end_idx = int(getattr(tokenizer, "traj_token_end_idx", start_idx + 4000))
    size = max(end_idx - start_idx + 1, 0)
    if size <= 0:
        raise ValueError("Tokenizer is missing valid trajectory token range.")
    return start_idx, size


def _get_output_head(model):
    backbone = model.backbone
    if hasattr(backbone, "get_output_embeddings"):
        head = backbone.get_output_embeddings()
        if head is not None:
            return head
    head = getattr(backbone, "lm_head", None)
    if head is None:
        raise ValueError("Student backbone is missing output embedding head.")
    return head


@torch.inference_mode()
def extract_features(
    *,
    model,
    collator,
    records: list[dict[str, Any]],
    device: torch.device,
    prefix_tokens: int,
    traj_start_id: int,
    traj_vocab_size: int,
) -> dict[str, Any]:
    output_head = _get_output_head(model)
    raw_features: list[np.ndarray] = []
    proj_features: list[np.ndarray] = []
    labels_local: list[np.ndarray] = []
    lm_logits_local: list[np.ndarray] = []
    sample_rows: list[dict[str, Any]] = []

    for record in records:
        batch = collator([record])
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.backbone.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
                logits_to_keep=1,
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
            )

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None and hasattr(outputs, "language_model_outputs"):
            hidden_states = getattr(outputs.language_model_outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Backbone did not return hidden states.")
        final_hidden = hidden_states[-1]
        traj_positions = torch.nonzero(
            batch["traj_token_mask"][0] & (batch["labels"][0] != -100),
            as_tuple=False,
        ).flatten()[:prefix_tokens]
        if traj_positions.numel() == 0:
            continue

        selected_hidden = final_hidden[0, traj_positions, :].float()
        selected_labels = batch["labels"][0, traj_positions].long()
        local_labels = selected_labels - int(traj_start_id)
        local_labels = local_labels.clamp(min=0, max=traj_vocab_size - 1)

        if model.traj_hidden_projector is not None:
            projected_hidden = model.traj_hidden_projector(final_hidden)[0, traj_positions, :].float()
        else:
            projected_hidden = selected_hidden

        lm_logits = output_head(selected_hidden)[:, traj_start_id : traj_start_id + traj_vocab_size].float()

        raw_features.append(selected_hidden.cpu().numpy())
        proj_features.append(projected_hidden.cpu().numpy())
        labels_local.append(local_labels.cpu().numpy())
        lm_logits_local.append(lm_logits.cpu().numpy())
        sample_rows.append(
            {
                "sample_id": str(record.get("sample_id")),
                "token_count": int(traj_positions.numel()),
            }
        )

    if not raw_features:
        raise RuntimeError("No trajectory features were extracted for the requested split.")

    return {
        "raw_features": np.concatenate(raw_features, axis=0).astype(np.float32),
        "proj_features": np.concatenate(proj_features, axis=0).astype(np.float32),
        "labels_local": np.concatenate(labels_local, axis=0).astype(np.int64),
        "lm_logits_local": np.concatenate(lm_logits_local, axis=0).astype(np.float32),
        "sample_rows": sample_rows,
    }


def _standardize(
    train_x: np.ndarray,
    val_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.clip(std, 1e-6, None)
    return (train_x - mean) / std, (val_x - mean) / std, mean, std


def _evaluate_logits(logits: torch.Tensor, labels: torch.Tensor, traj_1499_index: int) -> dict[str, float]:
    target_logits = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    token1499 = torch.full_like(labels, int(traj_1499_index))
    token1499_logits = logits.gather(dim=-1, index=token1499.unsqueeze(-1)).squeeze(-1)
    target_ranks = (logits > target_logits.unsqueeze(-1)).sum(dim=-1) + 1
    top1 = logits.argmax(dim=-1)
    top1_hist = Counter(int(value) for value in top1.tolist())
    return {
        "target_logit_mean": float(target_logits.mean().item()),
        "token1499_logit_mean": float(token1499_logits.mean().item()),
        "target_vs_1499_margin_mean": float((target_logits - token1499_logits).mean().item()),
        "target_rank_mean": float(target_ranks.float().mean().item()),
        "top1_1499_ratio": float((top1 == traj_1499_index).float().mean().item()),
        "top1_accuracy": float((top1 == labels).float().mean().item()),
        "top1_histogram_top20": dict(top1_hist.most_common(20)),
    }


def train_linear_probe(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    num_classes: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    traj_1499_index: int,
    device: torch.device,
) -> dict[str, Any]:
    train_x_std, val_x_std, mean, std = _standardize(train_x, val_x)
    train_inputs = torch.from_numpy(train_x_std).to(device=device, dtype=torch.float32)
    train_targets = torch.from_numpy(train_y).to(device=device, dtype=torch.long)
    val_inputs = torch.from_numpy(val_x_std).to(device=device, dtype=torch.float32)
    val_targets = torch.from_numpy(val_y).to(device=device, dtype=torch.long)

    head = torch.nn.Linear(int(train_inputs.shape[-1]), int(num_classes), bias=True).to(device=device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = None
    best_val_rank = math.inf
    history: list[dict[str, float]] = []

    num_train = int(train_inputs.shape[0])
    for epoch in range(int(epochs)):
        order = torch.randperm(num_train, device=device)
        head.train()
        train_loss = 0.0
        seen = 0
        for start in range(0, num_train, batch_size):
            indices = order[start : start + batch_size]
            batch_x = train_inputs[indices]
            batch_y = train_targets[indices]
            logits = head(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().item()) * int(batch_y.shape[0])
            seen += int(batch_y.shape[0])

        head.eval()
        with torch.no_grad():
            val_logits = head(val_inputs)
            metrics = _evaluate_logits(val_logits, val_targets, traj_1499_index)
            val_loss = float(F.cross_entropy(val_logits, val_targets).item())
        epoch_row = {
            "epoch": float(epoch + 1),
            "train_loss": float(train_loss / max(seen, 1)),
            "val_loss": val_loss,
            "val_target_rank_mean": float(metrics["target_rank_mean"]),
            "val_target_vs_1499_margin_mean": float(metrics["target_vs_1499_margin_mean"]),
            "val_top1_accuracy": float(metrics["top1_accuracy"]),
        }
        history.append(epoch_row)
        if float(metrics["target_rank_mean"]) < best_val_rank:
            best_val_rank = float(metrics["target_rank_mean"])
            best_state = {
                "weight": head.weight.detach().cpu().clone(),
                "bias": head.bias.detach().cpu().clone() if head.bias is not None else None,
            }

    if best_state is None:
        raise RuntimeError("Linear probe did not record a best checkpoint.")

    with torch.no_grad():
        head.weight.copy_(best_state["weight"].to(device=device))
        if head.bias is not None and best_state["bias"] is not None:
            head.bias.copy_(best_state["bias"].to(device=device))
        train_metrics = _evaluate_logits(head(train_inputs), train_targets, traj_1499_index)
        val_metrics = _evaluate_logits(head(val_inputs), val_targets, traj_1499_index)

    return {
        "input_dim": int(train_inputs.shape[-1]),
        "num_classes": int(num_classes),
        "train_examples": int(train_inputs.shape[0]),
        "val_examples": int(val_inputs.shape[0]),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "weight_decay": float(weight_decay),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "history_tail": history[-5:],
        "standardization_mean_abs": float(np.abs(mean).mean()),
        "standardization_std_mean": float(std.mean()),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    rows = load_jsonl(args.corpus_jsonl)
    train_records = [row for row in rows if row.get("split") == args.train_split]
    val_records = [row for row in rows if row.get("split") == args.val_split]
    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_records = val_records[: args.max_val_samples]

    config = StudentWrapperConfig(student_model_name="/workspace/base_models_weights/Cosmos-Reason2-2B")
    tokenizer = load_student_tokenizer(config)
    processor = load_student_processor(config)
    collator = DistillationCollator(
        tokenizer=tokenizer,
        processor=processor,
        project_root=PROJECT_ROOT,
        teacher_pair_target=False,
        enable_teacher_view=False,
        enable_action_aux=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_student_model(config, tokenizer).to(device).eval()
    load_student_checkpoint(args.checkpoint_dir, model, use_lora=True)
    model = model.to(device).eval()

    traj_start_id, traj_vocab_size = _traj_vocab_size(tokenizer)
    traj_1499_index = 1499

    train_bundle = extract_features(
        model=model,
        collator=collator,
        records=train_records,
        device=device,
        prefix_tokens=int(args.prefix_tokens),
        traj_start_id=traj_start_id,
        traj_vocab_size=traj_vocab_size,
    )
    val_bundle = extract_features(
        model=model,
        collator=collator,
        records=val_records,
        device=device,
        prefix_tokens=int(args.prefix_tokens),
        traj_start_id=traj_start_id,
        traj_vocab_size=traj_vocab_size,
    )

    lm_val_logits = torch.from_numpy(val_bundle["lm_logits_local"]).to(device=device, dtype=torch.float32)
    lm_val_labels = torch.from_numpy(val_bundle["labels_local"]).to(device=device, dtype=torch.long)
    lm_train_logits = torch.from_numpy(train_bundle["lm_logits_local"]).to(device=device, dtype=torch.float32)
    lm_train_labels = torch.from_numpy(train_bundle["labels_local"]).to(device=device, dtype=torch.long)

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "corpus_jsonl": str(args.corpus_jsonl),
        "train_split": str(args.train_split),
        "val_split": str(args.val_split),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "prefix_tokens": int(args.prefix_tokens),
        "traj_vocab_size": int(traj_vocab_size),
        "lm_head_traj_vocab": {
            "train_metrics": _evaluate_logits(lm_train_logits, lm_train_labels, traj_1499_index),
            "val_metrics": _evaluate_logits(lm_val_logits, lm_val_labels, traj_1499_index),
        },
        "linear_probe_raw": train_linear_probe(
            train_x=train_bundle["raw_features"],
            train_y=train_bundle["labels_local"],
            val_x=val_bundle["raw_features"],
            val_y=val_bundle["labels_local"],
            num_classes=traj_vocab_size,
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            traj_1499_index=traj_1499_index,
            device=device,
        ),
        "linear_probe_projected": train_linear_probe(
            train_x=train_bundle["proj_features"],
            train_y=train_bundle["labels_local"],
            val_x=val_bundle["proj_features"],
            val_y=val_bundle["labels_local"],
            num_classes=traj_vocab_size,
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            traj_1499_index=traj_1499_index,
            device=device,
        ),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
