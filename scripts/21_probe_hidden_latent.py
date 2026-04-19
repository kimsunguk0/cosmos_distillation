#!/usr/bin/env python3
"""Probe student/teacher hidden geometry on traj-body latent targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

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
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--prefix-tokens", type=int, default=16)
    parser.add_argument(
        "--teacher-traj-cache-dir",
        type=Path,
        default=Path("/data/teacher_cache/traj15"),
    )
    parser.add_argument("--teacher-traj-hidden-source", type=str, default="latent")
    parser.add_argument("--teacher-traj-latent-suffix", type=str, default="lat32")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "hiddenprobe.json",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def effective_rank(x: np.ndarray) -> float:
    centered = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    weights = singular_values**2
    denom = weights.sum()
    if denom <= 0:
        return 0.0
    probs = weights / denom
    probs = probs[probs > 1e-12]
    return float(np.exp(-(probs * np.log(probs)).sum()))


def offdiag_cosine_mean(x: np.ndarray) -> float:
    norms = np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    x_unit = x / norms
    gram = x_unit @ x_unit.T
    n = gram.shape[0]
    if n <= 1:
        return 1.0
    return float((gram.sum() - np.trace(gram)) / (n * (n - 1)))


def token_cosine_mean(student: np.ndarray, teacher: np.ndarray) -> float:
    s_norm = student / np.clip(np.linalg.norm(student, axis=1, keepdims=True), 1e-8, None)
    t_norm = teacher / np.clip(np.linalg.norm(teacher, axis=1, keepdims=True), 1e-8, None)
    return float(np.sum(s_norm * t_norm, axis=1).mean())


def main() -> None:
    args = parse_args()
    records = [row for row in load_jsonl(args.corpus_jsonl) if row.get("split") == args.split][: args.num_samples]
    config = StudentWrapperConfig(
        student_model_name="/workspace/base_models_weights/Cosmos-Reason2-2B",
        traj_teacher_hidden_size=32 if args.teacher_traj_hidden_source == "latent" else None,
    )
    tokenizer = load_student_tokenizer(config)
    processor = load_student_processor(config)
    collator = DistillationCollator(
        tokenizer=tokenizer,
        processor=processor,
        project_root=PROJECT_ROOT,
        teacher_pair_target=True,
        enable_teacher_view=False,
        enable_action_aux=False,
        teacher_traj_cache_dir=args.teacher_traj_cache_dir,
        teacher_traj_hidden_source=args.teacher_traj_hidden_source,
        teacher_traj_latent_suffix=args.teacher_traj_latent_suffix,
    )
    model = build_student_model(config, tokenizer).cuda().eval()
    load_student_checkpoint(args.checkpoint_dir, model, use_lora=True)

    sample_rows: list[dict[str, float | str]] = []
    for record in records:
        batch = collator([record])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            backbone_outputs = model.backbone.model(
                input_ids=batch["input_ids"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
                output_hidden_states=True,
                return_dict=True,
                logits_to_keep=1,
                pixel_values=batch["pixel_values"].cuda(),
                image_grid_thw=batch["image_grid_thw"].cuda(),
            )
        hidden_states = getattr(backbone_outputs, "hidden_states", None)
        if hidden_states is None and hasattr(backbone_outputs, "language_model_outputs"):
            hidden_states = getattr(backbone_outputs.language_model_outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("Student backbone did not return hidden states for probing.")
        final_hidden = hidden_states[-1]
        mask = batch["traj_token_mask"].bool().cpu().numpy()[0]
        teacher = batch["teacher_traj_hidden"].cpu().numpy()[0][: args.prefix_tokens]
        if model.traj_hidden_projector is not None:
            student_hidden = model.traj_hidden_projector(final_hidden)
        else:
            student_hidden = final_hidden
        student = student_hidden.detach().cpu().float().numpy()[0][mask][: args.prefix_tokens]
        sample_rows.append(
            {
                "sample_id": str(record.get("sample_id")),
                "student_effective_rank": effective_rank(student),
                "teacher_effective_rank": effective_rank(teacher),
                "student_offdiag_cosine_mean": offdiag_cosine_mean(student),
                "teacher_offdiag_cosine_mean": offdiag_cosine_mean(teacher),
                "token_cosine_mean": token_cosine_mean(student, teacher),
            }
        )

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "corpus_jsonl": str(args.corpus_jsonl),
        "split": args.split,
        "records": len(sample_rows),
        "prefix_tokens": args.prefix_tokens,
        "teacher_traj_hidden_source": args.teacher_traj_hidden_source,
        "teacher_traj_latent_suffix": args.teacher_traj_latent_suffix,
        "student_effective_rank_mean": float(np.mean([row["student_effective_rank"] for row in sample_rows])),
        "teacher_effective_rank_mean": float(np.mean([row["teacher_effective_rank"] for row in sample_rows])),
        "student_offdiag_cosine_mean": float(np.mean([row["student_offdiag_cosine_mean"] for row in sample_rows])),
        "teacher_offdiag_cosine_mean": float(np.mean([row["teacher_offdiag_cosine_mean"] for row in sample_rows])),
        "token_cosine_mean": float(np.mean([row["token_cosine_mean"] for row in sample_rows])),
        "samples": sample_rows,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
