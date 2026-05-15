#!/usr/bin/env python3
"""Compare no-nav teacher cached hidden states with base and distilled student states.

The goal is intentionally modest: before training trajectory probes, verify that
student hidden distributions are healthy and whether a traj-only checkpoint moves
the student representation closer to the cached Alpamayo teacher interface.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.checkpoint_io import load_student_checkpoint  # noqa: E402
from src.model.student_wrapper import (  # noqa: E402
    StudentWrapperConfig,
    build_student_model,
    load_student_processor,
    load_student_tokenizer,
)
from src.training.collator import BOUNDARY_HIDDEN_NAMES, DistillationCollator  # noqa: E402


POSITION_NAMES = ("prompt_last", "cot_end", "traj_start", "action_pre", "traj_body_mean", "traj_body_first16")
TEACHER_POSITION_NAMES = ("cot_end", "traj_start", "action_pre", "traj_body_mean", "traj_body_first16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument(
        "--student-model",
        type=Path,
        default=WORKSPACE_ROOT / "base_weights" / "cosmos-reason-2b",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "checkpoints"
            / "no_nav_bp5_vlm_interface_hidden_kd"
            / "no_nav_bp5_trajreadout_200k_b8_20260507_084536"
            / "step_012500"
        ),
    )
    parser.add_argument("--checkpoint-name", default="traj_only_2b")
    parser.add_argument("--attn-implementation", default="flash_attention_2", choices=("flash_attention_2", "sdpa", "eager"))
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "no_nav_hidden_qc",
    )
    parser.add_argument("--report-name", default="base_vs_trajonly_hidden_qc.json")
    parser.add_argument("--markdown-name", default="base_vs_trajonly_hidden_qc.md")
    parser.add_argument("--save-embeddings", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_records(path: Path, *, split: str, count: int, seed: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if split and row.get("split") != split:
                continue
            if not has_teacher_hidden(row):
                continue
            candidates.append(row)
    if len(candidates) < count:
        raise RuntimeError(f"Only found {len(candidates)} usable rows for split={split!r}; requested {count}.")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:count]


def has_teacher_hidden(row: dict[str, Any]) -> bool:
    traj_path = (((row.get("teacher_traj_target") or {}).get("hidden_path")) or "").strip()
    boundary = ((row.get("teacher_cache") or {}).get("boundary_hidden_paths") or {})
    return bool(traj_path and Path(traj_path).exists() and all(Path(str(boundary.get(name, ""))).exists() for name in BOUNDARY_HIDDEN_NAMES))


def batches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def tensor_to_device(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    return value


def get_hidden_states(outputs: Any) -> tuple[torch.Tensor, ...]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and hasattr(outputs, "language_model_outputs"):
        hidden_states = getattr(outputs.language_model_outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Student backbone did not return hidden_states.")
    return tuple(hidden_states)


def forward_student(model: torch.nn.Module, batch: dict[str, Any], *, device: str, dtype: torch.dtype) -> torch.Tensor:
    kwargs = {
        "input_ids": tensor_to_device(batch["input_ids"], device),
        "attention_mask": tensor_to_device(batch["attention_mask"], device),
        "output_hidden_states": True,
        "return_dict": True,
    }
    if batch.get("pixel_values") is not None:
        kwargs["pixel_values"] = tensor_to_device(batch["pixel_values"], device)
    if batch.get("image_grid_thw") is not None:
        kwargs["image_grid_thw"] = tensor_to_device(batch["image_grid_thw"], device)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=(device == "cuda" and dtype != torch.float32)):
            outputs = model.backbone(**kwargs, logits_to_keep=1)
    except TypeError:
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=(device == "cuda" and dtype != torch.float32)):
            outputs = model.backbone(**kwargs)
    return get_hidden_states(outputs)[-1].detach().float().cpu()


def append_vector(store: dict[str, list[np.ndarray]], name: str, tensor: torch.Tensor | np.ndarray) -> None:
    if isinstance(tensor, torch.Tensor):
        value = tensor.detach().cpu().float().numpy()
    else:
        value = np.asarray(tensor, dtype=np.float32)
    store[name].append(value.reshape(-1).astype(np.float32, copy=False))


def collect_model_vectors(
    *,
    model_name: str,
    checkpoint_dir: Path | None,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch_dtype(args.dtype)
    config = StudentWrapperConfig(
        student_model_name=str(args.student_model),
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
    )
    tokenizer = load_student_tokenizer(config)
    processor = load_student_processor(config, tokenizer)
    try:
        processor.tokenizer.padding_side = "right"
    except Exception:  # noqa: BLE001
        pass
    collator = DistillationCollator(
        tokenizer=tokenizer,
        processor=processor,
        project_root=PROJECT_ROOT,
        max_length=args.max_length,
        prompt_mode="joint",
        target_mode="joint",
        teacher_pair_target=True,
        enable_teacher_view=False,
        enable_action_aux=False,
        teacher_traj_hidden_source="hidden",
    )
    model = build_student_model(config, tokenizer).to(device).eval()
    checkpoint_load: dict[str, Any] | None = None
    if checkpoint_dir is not None:
        checkpoint_load = load_student_checkpoint(checkpoint_dir, model, use_lora=True)
        model.to(device).eval()

    student_store: dict[str, list[np.ndarray]] = defaultdict(list)
    teacher_store: dict[str, list[np.ndarray]] = defaultdict(list)
    counts = {
        "batches": 0,
        "samples": 0,
        "invalid_boundary_positions": 0,
        "invalid_traj_positions": 0,
    }

    for batch_records in batches(records, args.batch_size):
        batch = collator(batch_records)
        final_hidden = forward_student(model, batch, device=device, dtype=dtype)
        labels = batch["labels"]
        boundary_positions = batch.get("teacher_text_boundary_hidden_positions")
        boundary_mask = batch.get("teacher_text_boundary_hidden_mask")
        teacher_boundary = batch.get("teacher_text_boundary_hidden")
        teacher_traj_hidden = batch.get("teacher_traj_hidden")
        teacher_traj_hidden_mask = batch.get("teacher_traj_hidden_mask")
        traj_token_mask = batch["traj_token_mask"] & (labels != -100)

        for row_index, sample_id in enumerate(batch["sample_ids"]):
            valid_label_positions = torch.nonzero(labels[row_index] != -100, as_tuple=False).flatten()
            if valid_label_positions.numel() > 0:
                prompt_last = int(valid_label_positions[0].item()) - 1
                if prompt_last >= 0:
                    append_vector(student_store, "prompt_last", final_hidden[row_index, prompt_last])

            if boundary_positions is not None and boundary_mask is not None and teacher_boundary is not None:
                for boundary_index, name in enumerate(BOUNDARY_HIDDEN_NAMES):
                    position = int(boundary_positions[row_index, boundary_index].item())
                    is_valid = bool(boundary_mask[row_index, boundary_index].item()) and position >= 0
                    if not is_valid:
                        counts["invalid_boundary_positions"] += 1
                        continue
                    append_vector(student_store, name, final_hidden[row_index, position])
                    append_vector(teacher_store, name, teacher_boundary[row_index, boundary_index])

            traj_positions = torch.nonzero(traj_token_mask[row_index], as_tuple=False).flatten()
            if traj_positions.numel() <= 0:
                counts["invalid_traj_positions"] += 1
                continue
            student_traj = final_hidden[row_index, traj_positions]
            append_vector(student_store, "traj_body_mean", student_traj.mean(dim=0))
            append_vector(student_store, "traj_body_first16", student_traj[: min(16, student_traj.shape[0])].mean(dim=0))
            if teacher_traj_hidden is not None and teacher_traj_hidden_mask is not None:
                teacher_mask = teacher_traj_hidden_mask[row_index].bool()
                teacher_traj = teacher_traj_hidden[row_index, teacher_mask]
                if teacher_traj.numel() > 0:
                    append_vector(teacher_store, "traj_body_mean", teacher_traj.mean(dim=0))
                    append_vector(teacher_store, "traj_body_first16", teacher_traj[: min(16, teacher_traj.shape[0])].mean(dim=0))
            counts["samples"] += 1
        counts["batches"] += 1
        if device == "cuda":
            torch.cuda.empty_cache()

    student_arrays = {name: np.stack(items, axis=0) for name, items in student_store.items() if items}
    teacher_arrays = {name: np.stack(items, axis=0) for name, items in teacher_store.items() if items}
    metadata = {
        "model_name": model_name,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "checkpoint_load": checkpoint_load,
        "counts": counts,
        "hidden_size": int(getattr(model, "hidden_size", -1)),
    }
    del model, processor, tokenizer, collator
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return student_arrays, teacher_arrays, metadata


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def offdiag_cosine_stats(x: np.ndarray) -> dict[str, float | None]:
    if x.shape[0] <= 1:
        return {"offdiag_cos_mean": None, "offdiag_cos_p95": None, "offdiag_cos_p99": None}
    x = x.astype(np.float32, copy=False)
    norms = np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    unit = x / norms
    gram = unit @ unit.T
    mask = ~np.eye(gram.shape[0], dtype=bool)
    values = gram[mask]
    return {
        "offdiag_cos_mean": as_float(np.mean(values)),
        "offdiag_cos_p95": as_float(np.percentile(values, 95)),
        "offdiag_cos_p99": as_float(np.percentile(values, 99)),
    }


def spectral_stats(x: np.ndarray) -> dict[str, float | None]:
    if x.shape[0] <= 1:
        return {"effective_rank": None, "top_pc_var_ratio": None, "spectral_entropy": None}
    centered = x.astype(np.float32, copy=False) - x.astype(np.float32, copy=False).mean(axis=0, keepdims=True)
    try:
        singular_values = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return {"effective_rank": None, "top_pc_var_ratio": None, "spectral_entropy": None}
    weights = singular_values**2
    total = float(weights.sum())
    if total <= 1e-12:
        return {"effective_rank": 0.0, "top_pc_var_ratio": 1.0, "spectral_entropy": 0.0}
    probs = weights / total
    nz = probs[probs > 1e-12]
    entropy = float(-(nz * np.log(nz)).sum())
    return {
        "effective_rank": as_float(np.exp(entropy)),
        "top_pc_var_ratio": as_float(probs[0]),
        "spectral_entropy": as_float(entropy),
    }


def vector_summary(x: np.ndarray) -> dict[str, Any]:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1)
    dim_mean = x.mean(axis=0)
    dim_std = x.std(axis=0)
    summary: dict[str, Any] = {
        "shape": list(x.shape),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
        "mean": as_float(x.mean()),
        "std": as_float(x.std()),
        "abs_mean": as_float(np.abs(x).mean()),
        "max_abs": as_float(np.abs(x).max()),
        "norm_mean": as_float(norms.mean()),
        "norm_std": as_float(norms.std()),
        "norm_p50": as_float(np.percentile(norms, 50)),
        "norm_p95": as_float(np.percentile(norms, 95)),
        "norm_max": as_float(norms.max()),
        "dim_mean_abs_mean": as_float(np.abs(dim_mean).mean()),
        "dim_std_mean": as_float(dim_std.mean()),
        "dim_std_p05": as_float(np.percentile(dim_std, 5)),
        "dim_std_p95": as_float(np.percentile(dim_std, 95)),
        "common_ratio": common_ratio(x),
    }
    summary.update(offdiag_cosine_stats(x))
    summary.update(spectral_stats(x))
    return summary


def common_ratio(x: np.ndarray) -> float | None:
    x = x.astype(np.float32, copy=False)
    token_mean = x.mean(axis=0)
    mean_sq = float(np.dot(token_mean, token_mean))
    token_energy = float(np.mean(np.sum(x * x, axis=1)))
    if token_energy <= 1e-12:
        return None
    return as_float(mean_sq / token_energy)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.shape[0] != y.shape[0] or x.shape[0] <= 1:
        return None
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    numerator = float(np.sum(xty * xty))
    denominator = float(np.sqrt(np.sum(xtx * xtx) * np.sum(yty * yty)))
    if denominator <= 1e-12:
        return None
    return as_float(numerator / denominator)


def summarize_all(
    *,
    teacher_arrays: dict[str, np.ndarray],
    model_arrays: dict[str, dict[str, np.ndarray]],
    model_metadata: dict[str, dict[str, Any]],
    sample_ids: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    teacher_summary = {name: vector_summary(array) for name, array in teacher_arrays.items()}
    student_summary: dict[str, Any] = {}
    cka_summary: dict[str, Any] = {}
    for model_name, arrays in model_arrays.items():
        student_summary[model_name] = {name: vector_summary(array) for name, array in arrays.items()}
        cka_summary[model_name] = {}
        for position_name in TEACHER_POSITION_NAMES:
            if position_name in arrays and position_name in teacher_arrays:
                cka_summary[model_name][position_name] = {
                    "linear_cka_to_teacher": linear_cka(arrays[position_name], teacher_arrays[position_name]),
                    "student_shape": list(arrays[position_name].shape),
                    "teacher_shape": list(teacher_arrays[position_name].shape),
                }

    deltas: dict[str, Any] = {}
    checkpoint_name = str(args.checkpoint_name)
    if "base_2b" in cka_summary and checkpoint_name in cka_summary:
        for position_name in TEACHER_POSITION_NAMES:
            base_value = (cka_summary["base_2b"].get(position_name) or {}).get("linear_cka_to_teacher")
            traj_value = (cka_summary[checkpoint_name].get(position_name) or {}).get("linear_cka_to_teacher")
            if base_value is not None and traj_value is not None:
                deltas[position_name] = {
                    "checkpoint_minus_base_linear_cka": as_float(float(traj_value) - float(base_value)),
                    "base_linear_cka": base_value,
                    "checkpoint_linear_cka": traj_value,
                }

    return {
        "schema_version": "no_nav_hidden_qc_v1",
        "corpus_jsonl": str(args.corpus_jsonl),
        "split": args.split,
        "num_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "student_model": str(args.student_model),
        "student_checkpoint": str(args.student_checkpoint),
        "checkpoint_name": checkpoint_name,
        "positions": {
            "prompt_last": "last token before assistant target span; causal-equivalent to prompt/prefill final state for the student sequence",
            "cot_end": "teacher-forced <|cot_end|> hidden position",
            "traj_start": "teacher-forced <|traj_future_start|> hidden position",
            "action_pre": "same cutpoint as traj_start in the current collator contract",
            "traj_body_mean": "mean hidden over 128 teacher trajectory token positions",
            "traj_body_first16": "mean hidden over first 16 teacher trajectory token positions",
        },
        "notes": [
            "Teacher prompt_last hidden is not cached in the current no-nav teacher cache, so prompt_last is student-only.",
            "Cross-model comparison uses linear CKA because teacher/student hidden dimensions differ.",
            "This is a distribution/interface QC, not a trajectory ADE/FDE probe.",
        ],
        "model_metadata": model_metadata,
        "teacher": teacher_summary,
        "students": student_summary,
        "cka_to_teacher": cka_summary,
        "checkpoint_minus_base": deltas,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    checkpoint_name = str(report.get("checkpoint_name") or "student_checkpoint")
    lines.append(f"# No-Nav Hidden QC: Base 2B vs {checkpoint_name}")
    lines.append("")
    lines.append(f"- samples: {report['num_samples']} `{report['split']}` rows")
    lines.append(f"- student base: `{report['student_model']}`")
    lines.append(f"- checkpoint `{checkpoint_name}`: `{report['student_checkpoint']}`")
    lines.append("")
    lines.append("## Linear CKA To Teacher")
    lines.append("")
    lines.append(f"| position | base_2b | {checkpoint_name} | delta |")
    lines.append("|---|---:|---:|---:|")
    for position_name in TEACHER_POSITION_NAMES:
        base = (((report["cka_to_teacher"].get("base_2b") or {}).get(position_name) or {}).get("linear_cka_to_teacher"))
        traj = (((report["cka_to_teacher"].get(checkpoint_name) or {}).get(position_name) or {}).get("linear_cka_to_teacher"))
        delta = None if base is None or traj is None else float(traj) - float(base)
        lines.append(
            f"| {position_name} | {fmt(base)} | {fmt(traj)} | {fmt(delta)} |"
        )
    lines.append("")
    lines.append("## Collapse / Anisotropy Checks")
    lines.append("")
    lines.append("| model | position | norm mean | offdiag cos mean | top PC var | effective rank | common ratio |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for model_name, model_summary in report["students"].items():
        for position_name in POSITION_NAMES:
            if position_name not in model_summary:
                continue
            row = model_summary[position_name]
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_name,
                        position_name,
                        fmt(row.get("norm_mean")),
                        fmt(row.get("offdiag_cos_mean")),
                        fmt(row.get("top_pc_var_ratio")),
                        fmt(row.get("effective_rank")),
                        fmt(row.get("common_ratio")),
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("## Teacher Reference")
    lines.append("")
    lines.append("| position | norm mean | offdiag cos mean | top PC var | effective rank | common ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for position_name in TEACHER_POSITION_NAMES:
        row = report["teacher"].get(position_name)
        if not row:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    position_name,
                    fmt(row.get("norm_mean")),
                    fmt(row.get("offdiag_cos_mean")),
                    fmt(row.get("top_pc_var_ratio")),
                    fmt(row.get("effective_rank")),
                    fmt(row.get("common_ratio")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Read")
    lines.append("")
    deltas = report.get("checkpoint_minus_base") or {}
    positive = [name for name, value in deltas.items() if (value.get("checkpoint_minus_base_linear_cka") or 0.0) > 0]
    negative = [name for name, value in deltas.items() if (value.get("checkpoint_minus_base_linear_cka") or 0.0) < 0]
    lines.append(
        f"- CKA improved at: {', '.join(positive) if positive else 'none'}."
    )
    lines.append(
        f"- CKA regressed at: {', '.join(negative) if negative else 'none'}."
    )
    lines.append("- `prompt_last` has no teacher reference in this cache; use it only to catch student distribution collapse/outliers.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.corpus_jsonl, split=args.split, count=args.num_samples, seed=args.seed)
    sample_ids = [str(row.get("sample_id")) for row in records]

    model_arrays: dict[str, dict[str, np.ndarray]] = {}
    teacher_arrays: dict[str, np.ndarray] | None = None
    model_metadata: dict[str, dict[str, Any]] = {}

    for model_name, checkpoint_dir in (
        ("base_2b", None),
        (str(args.checkpoint_name), args.student_checkpoint),
    ):
        print(json.dumps({"event": "model_start", "model": model_name, "checkpoint": str(checkpoint_dir) if checkpoint_dir else None}), flush=True)
        student, teacher, metadata = collect_model_vectors(
            model_name=model_name,
            checkpoint_dir=checkpoint_dir,
            records=records,
            args=args,
        )
        model_arrays[model_name] = student
        model_metadata[model_name] = metadata
        if teacher_arrays is None:
            teacher_arrays = teacher
        print(json.dumps({"event": "model_done", "model": model_name, "samples": metadata["counts"]["samples"]}), flush=True)

    if teacher_arrays is None:
        raise RuntimeError("No teacher arrays were collected.")
    report = summarize_all(
        teacher_arrays=teacher_arrays,
        model_arrays=model_arrays,
        model_metadata=model_metadata,
        sample_ids=sample_ids,
        args=args,
    )
    report_path = args.output_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = args.output_dir / args.markdown_name
    write_markdown(report, markdown_path)

    if args.save_embeddings:
        for model_name, arrays in model_arrays.items():
            np.savez_compressed(args.output_dir / f"{model_name}_hidden_vectors.npz", **arrays)
        np.savez_compressed(args.output_dir / "teacher_hidden_vectors.npz", **teacher_arrays)

    print(json.dumps({"event": "done", "report_json": str(report_path), "report_md": str(markdown_path)}), flush=True)


if __name__ == "__main__":
    main()
