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
    x = x.astype(np.float32, copy=False)
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
    x = x.astype(np.float32, copy=False)
    norms = np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    x_unit = x / norms
    gram = x_unit @ x_unit.T
    n = gram.shape[0]
    if n <= 1:
        return 1.0
    return float((gram.sum() - np.trace(gram)) / (n * (n - 1)))


def centered_offdiag_cosine_mean(x: np.ndarray) -> float:
    x = x.astype(np.float32, copy=False)
    centered = x - x.mean(axis=0, keepdims=True)
    return offdiag_cosine_mean(centered)


def token_cosine_mean(student: np.ndarray, teacher: np.ndarray) -> float:
    student = student.astype(np.float32, copy=False)
    teacher = teacher.astype(np.float32, copy=False)
    s_norm = student / np.clip(np.linalg.norm(student, axis=1, keepdims=True), 1e-8, None)
    t_norm = teacher / np.clip(np.linalg.norm(teacher, axis=1, keepdims=True), 1e-8, None)
    return float(np.sum(s_norm * t_norm, axis=1).mean())


def gram_corr(student: np.ndarray, teacher: np.ndarray) -> float:
    student = student.astype(np.float32, copy=False)
    teacher = teacher.astype(np.float32, copy=False)
    s_norm = student / np.clip(np.linalg.norm(student, axis=1, keepdims=True), 1e-8, None)
    t_norm = teacher / np.clip(np.linalg.norm(teacher, axis=1, keepdims=True), 1e-8, None)
    s_gram = s_norm @ s_norm.T
    t_gram = t_norm @ t_norm.T
    s_flat = s_gram.reshape(-1) - s_gram.mean()
    t_flat = t_gram.reshape(-1) - t_gram.mean()
    denom = np.linalg.norm(s_flat) * np.linalg.norm(t_flat)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(s_flat, t_flat) / denom)


def centered_gram_corr(student: np.ndarray, teacher: np.ndarray) -> float:
    student = student.astype(np.float32, copy=False) - student.astype(np.float32, copy=False).mean(axis=0, keepdims=True)
    teacher = teacher.astype(np.float32, copy=False) - teacher.astype(np.float32, copy=False).mean(axis=0, keepdims=True)
    return gram_corr(student, teacher)


def common_ratio(x: np.ndarray) -> float:
    x = x.astype(np.float32, copy=False)
    token_mean = x.mean(axis=0)
    mean_sq = float(np.dot(token_mean, token_mean))
    token_energy = float(np.mean(np.sum(x * x, axis=1)))
    if token_energy <= 1e-12:
        return 0.0
    return mean_sq / token_energy


def spectral_entropy(x: np.ndarray) -> float:
    x = x.astype(np.float32, copy=False)
    centered = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    weights = singular_values**2
    denom = weights.sum()
    if denom <= 0:
        return 0.0
    probs = weights / denom
    probs = probs[probs > 1e-12]
    return float(-(probs * np.log(probs)).sum())


def mean_metric(rows: list[dict[str, float | str]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _get_output_head(model):
    backbone = model.backbone
    if hasattr(backbone, "get_output_embeddings"):
        head = backbone.get_output_embeddings()
        if head is not None:
            return head
    head = getattr(backbone, "lm_head", None)
    if head is None:
        raise ValueError("Student backbone is missing output embedding head for logit probing.")
    return head


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
    output_head = _get_output_head(model)
    traj_1499_id = int(getattr(tokenizer, "traj_token_start_idx", 0)) + 1499

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
        sample_id = str(record.get("sample_id"))
        teacher_latent = batch["teacher_traj_hidden"].cpu().numpy()[0][: args.prefix_tokens]
        teacher_raw_path = args.teacher_traj_cache_dir / "hidden" / f"{sample_id}.teacher_traj15.hidden.npy"
        teacher_raw = np.load(teacher_raw_path)[: args.prefix_tokens]
        raw_student = final_hidden.detach().cpu().float().numpy()[0][mask][: args.prefix_tokens]
        active_positions = torch.nonzero(batch["traj_token_mask"][0] & (batch["labels"][0] != -100), as_tuple=False).flatten()
        active_positions = active_positions[: args.prefix_tokens]
        target_token_ids = batch["labels"][0, active_positions].to(device=final_hidden.device, dtype=torch.long)
        selected_hidden = final_hidden[0, active_positions, :].float()
        selected_logits = output_head(selected_hidden)
        if selected_logits.ndim != 2:
            raise ValueError("Expected 2D logits for selected trajectory positions.")
        top1_ids = selected_logits.argmax(dim=-1)
        top1_1499_ratio = float((top1_ids == traj_1499_id).float().mean().item()) if top1_ids.numel() > 0 else 0.0
        target_logits = torch.gather(selected_logits, dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)
        token1499_ids = torch.full_like(target_token_ids, traj_1499_id)
        token1499_logits = torch.gather(selected_logits, dim=-1, index=token1499_ids.unsqueeze(-1)).squeeze(-1)
        target_vs_1499_margin = target_logits - token1499_logits
        target_ranks = (selected_logits > target_logits.unsqueeze(-1)).sum(dim=-1) + 1
        if model.traj_hidden_projector is not None:
            projected_student_hidden = model.traj_hidden_projector(final_hidden)
        else:
            projected_student_hidden = final_hidden
        projected_student = projected_student_hidden.detach().cpu().float().numpy()[0][mask][: args.prefix_tokens]
        sample_rows.append(
            {
                "sample_id": sample_id,
                "raw_student_effective_rank": effective_rank(raw_student),
                "raw_teacher_effective_rank": effective_rank(teacher_raw),
                "raw_student_offdiag_cosine_mean": offdiag_cosine_mean(raw_student),
                "raw_student_centered_offdiag_cosine_mean": centered_offdiag_cosine_mean(raw_student),
                "raw_teacher_offdiag_cosine_mean": offdiag_cosine_mean(teacher_raw),
                "raw_gram_corr": gram_corr(raw_student, teacher_raw),
                "raw_centered_gram_corr": centered_gram_corr(raw_student, teacher_raw),
                "raw_common_ratio": common_ratio(raw_student),
                "raw_spectral_entropy": spectral_entropy(raw_student),
                "projected_student_effective_rank": effective_rank(projected_student),
                "projected_teacher_effective_rank": effective_rank(teacher_latent),
                "projected_student_offdiag_cosine_mean": offdiag_cosine_mean(projected_student),
                "projected_teacher_offdiag_cosine_mean": offdiag_cosine_mean(teacher_latent),
                "projected_token_cosine_mean": token_cosine_mean(projected_student, teacher_latent),
                "projected_gram_corr": gram_corr(projected_student, teacher_latent),
                "top1_1499_ratio": top1_1499_ratio,
                "target_logit_mean": float(target_logits.mean().item()),
                "token1499_logit_mean": float(token1499_logits.mean().item()),
                "target_vs_1499_margin_mean": float(target_vs_1499_margin.mean().item()),
                "target_rank_mean": float(target_ranks.float().mean().item()),
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
        "raw_student_effective_rank_mean": mean_metric(sample_rows, "raw_student_effective_rank"),
        "raw_teacher_effective_rank_mean": mean_metric(sample_rows, "raw_teacher_effective_rank"),
        "raw_student_offdiag_cosine_mean": mean_metric(sample_rows, "raw_student_offdiag_cosine_mean"),
        "raw_student_centered_offdiag_cosine_mean": mean_metric(sample_rows, "raw_student_centered_offdiag_cosine_mean"),
        "raw_teacher_offdiag_cosine_mean": mean_metric(sample_rows, "raw_teacher_offdiag_cosine_mean"),
        "raw_gram_corr_mean": mean_metric(sample_rows, "raw_gram_corr"),
        "raw_centered_gram_corr_mean": mean_metric(sample_rows, "raw_centered_gram_corr"),
        "raw_common_ratio_mean": mean_metric(sample_rows, "raw_common_ratio"),
        "raw_spectral_entropy_mean": mean_metric(sample_rows, "raw_spectral_entropy"),
        "projected_student_effective_rank_mean": mean_metric(sample_rows, "projected_student_effective_rank"),
        "projected_teacher_effective_rank_mean": mean_metric(sample_rows, "projected_teacher_effective_rank"),
        "projected_student_offdiag_cosine_mean": mean_metric(sample_rows, "projected_student_offdiag_cosine_mean"),
        "projected_teacher_offdiag_cosine_mean": mean_metric(sample_rows, "projected_teacher_offdiag_cosine_mean"),
        "projected_token_cosine_mean": mean_metric(sample_rows, "projected_token_cosine_mean"),
        "projected_gram_corr_mean": mean_metric(sample_rows, "projected_gram_corr"),
        "top1_1499_ratio_mean": mean_metric(sample_rows, "top1_1499_ratio"),
        "target_logit_mean": mean_metric(sample_rows, "target_logit_mean"),
        "token1499_logit_mean": mean_metric(sample_rows, "token1499_logit_mean"),
        "target_vs_1499_margin_mean": mean_metric(sample_rows, "target_vs_1499_margin_mean"),
        "target_rank_mean": mean_metric(sample_rows, "target_rank_mean"),
        "samples": sample_rows,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
