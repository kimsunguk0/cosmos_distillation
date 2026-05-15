#!/usr/bin/env python3
"""Compare true prompt-prefill hidden states between Alpamayo teacher and students.

This probe is deliberately prompt-only.  It extracts the final hidden state after
the image stack, ego history, user prompt, and assistant prefix have been
prefilled, before any CoT or future trajectory token is decoded.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
ALPAMAYO15_SRC = WORKSPACE_ROOT / "alpamayo_repo" / "alpamayo1.5" / "src"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ALPAMAYO15_SRC) not in sys.path:
    sys.path.insert(0, str(ALPAMAYO15_SRC))

from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import (  # noqa: E402
    build_model_inputs,
    load_materialized_sample,
    load_model_and_processor,
    torch_dtype_from_name,
)
from src.model.checkpoint_io import load_student_checkpoint  # noqa: E402
from src.model.student_wrapper import (  # noqa: E402
    StudentWrapperConfig,
    build_student_model,
    load_student_processor,
    load_student_tokenizer,
)
from src.training.collator import (  # noqa: E402
    _encode_messages,
    build_messages,
    build_user_prompt,
    load_sample_images,
)


DEFAULT_TEACHER = WORKSPACE_ROOT / "base_weights" / "Alpamayo-1.5-10B"
DEFAULT_STUDENT = WORKSPACE_ROOT / "base_weights" / "cosmos-reason-2b"
DEFAULT_BP3 = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "no_nav_bp3_h200fast_b4"
    / "no_nav_bp3_h200fast_b4_from_step2288_20260504_053208"
    / "final"
)
DEFAULT_TRAJ_ONLY = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "no_nav_bp5_vlm_interface_hidden_kd"
    / "no_nav_bp5_trajreadout_200k_b8_20260507_084536"
    / "step_012500"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--student-model", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--student-attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="flash_attention_2")
    parser.add_argument(
        "--student-checkpoint",
        action="append",
        default=[],
        help="Student checkpoint as name=/abs/path. Defaults to BP3 cot+traj and traj-only step12500.",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--student-batch-size", type=int, default=4)
    parser.add_argument("--min-pixels", type=int, default=163840)
    parser.add_argument("--max-pixels", type=int, default=196608)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "no_nav_prefill_qc",
    )
    parser.add_argument("--report-name", default="prefill_base_bp3_trajonly_qc.json")
    parser.add_argument("--markdown-name", default="prefill_base_bp3_trajonly_qc.md")
    parser.add_argument("--save-embeddings", action="store_true")
    return parser.parse_args()


def parse_checkpoint_specs(raw_specs: list[str]) -> list[tuple[str, Path]]:
    if not raw_specs:
        return [
            ("bp3_cot_traj_2b", DEFAULT_BP3),
            ("traj_only_step12500_2b", DEFAULT_TRAJ_ONLY),
        ]
    specs: list[tuple[str, Path]] = []
    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(f"--student-checkpoint must be name=/path, got {spec!r}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip())
        if not name:
            raise ValueError(f"Empty checkpoint name in {spec!r}")
        specs.append((name, path))
    return specs


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
            sample_path = (((row.get("input") or {}).get("materialized_sample_path")) or "").strip()
            if not sample_path or not Path(sample_path).exists():
                continue
            candidates.append(row)
    if len(candidates) < count:
        raise RuntimeError(f"Only found {len(candidates)} usable rows for split={split!r}; requested {count}.")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:count]


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def append_vector(store: dict[str, list[np.ndarray]], name: str, value: torch.Tensor | np.ndarray) -> None:
    if isinstance(value, torch.Tensor):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    store.setdefault(name, []).append(array.reshape(-1).astype(np.float32, copy=False))


def get_hidden_states(outputs: Any) -> tuple[torch.Tensor, ...]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and hasattr(outputs, "language_model_outputs"):
        hidden_states = getattr(outputs.language_model_outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model forward did not return hidden_states.")
    return tuple(hidden_states)


def last_nonpad_positions(attention_mask: torch.Tensor | None, seq_len: int, batch_size: int) -> torch.Tensor:
    if attention_mask is None:
        return torch.full((batch_size,), seq_len - 1, dtype=torch.long)
    mask = attention_mask.detach().long().cpu()
    positions: list[int] = []
    for row in mask:
        valid = torch.nonzero(row > 0, as_tuple=False).flatten()
        positions.append(int(valid[-1].item()) if valid.numel() else seq_len - 1)
    return torch.tensor(positions, dtype=torch.long)


def sync_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def collect_teacher_prefill(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    dtype = torch_dtype_from_name(args.dtype)
    model, processor, config, config_path, runtime_support_path = load_model_and_processor(
        args.teacher_checkpoint,
        dtype=dtype,
        device=device,
        config_json=None,
        runtime_support=None,
        attn_implementation=args.teacher_attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    store: dict[str, list[np.ndarray]] = {}
    prompt_lengths: list[int] = []
    elapsed: list[float] = []
    route_used = 0
    errors: list[dict[str, str]] = []

    for index, row in enumerate(records, start=1):
        sample_id = str(row.get("sample_id"))
        started = time.perf_counter()
        try:
            sample_path = Path(str((row.get("input") or {}).get("materialized_sample_path")))
            sample = load_materialized_sample(sample_path)
            sample_input = row.get("input") or {}
            nav_text = sample_input.get("nav_text") if bool(sample_input.get("nav_available")) else None
            if nav_text is not None and str(nav_text).strip():
                sample["nav_text"] = str(nav_text)
                route_used += 1
            else:
                sample["nav_text"] = None
            data = build_model_inputs(processor=processor, sample=sample, device=device)
            tokenized_data = dict(data["tokenized_data"])
            input_ids = tokenized_data.pop("input_ids")
            input_ids = model.fuse_traj_tokens(
                input_ids,
                {
                    "ego_history_xyz": data["ego_history_xyz"],
                    "ego_history_rot": data["ego_history_rot"],
                },
            )
            attention_mask = tokenized_data.get("attention_mask")
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=(device.startswith("cuda") and dtype != torch.float32)):
                outputs = model.vlm(
                    input_ids=input_ids,
                    **tokenized_data,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                    logits_to_keep=1,
                )
            final_hidden = get_hidden_states(outputs)[-1].detach()
            positions = last_nonpad_positions(attention_mask, final_hidden.shape[1], final_hidden.shape[0])
            append_vector(store, "prefill_last", final_hidden[0, int(positions[0].item())])
            prompt_lengths.append(int((attention_mask.sum().item() if attention_mask is not None else input_ids.shape[1])))
            sync_cuda(device)
            elapsed.append(time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001
            errors.append({"sample_id": sample_id, "error": repr(exc)})
        if index % 8 == 0:
            print(json.dumps({"event": "teacher_prefill_progress", "done": index, "total": len(records)}), flush=True)

    arrays = {name: np.stack(values, axis=0) for name, values in store.items() if values}
    metadata = {
        "model_name": "alpamayo15_8b_teacher",
        "checkpoint": str(args.teacher_checkpoint),
        "config_path": str(config_path),
        "runtime_support_path": str(runtime_support_path) if runtime_support_path is not None else None,
        "attn_implementation": str(config.attn_implementation),
        "dtype": args.dtype,
        "samples_requested": len(records),
        "samples_collected": int(arrays.get("prefill_last", np.empty((0,))).shape[0]),
        "route_used_count": int(route_used),
        "prompt_length_mean": as_float(np.mean(prompt_lengths)) if prompt_lengths else None,
        "prompt_length_p50": as_float(np.percentile(prompt_lengths, 50)) if prompt_lengths else None,
        "prompt_length_p95": as_float(np.percentile(prompt_lengths, 95)) if prompt_lengths else None,
        "elapsed_sec_mean": as_float(np.mean(elapsed)) if elapsed else None,
        "elapsed_sec_p50": as_float(np.percentile(elapsed, 50)) if elapsed else None,
        "elapsed_sec_p95": as_float(np.percentile(elapsed, 95)) if elapsed else None,
        "errors": errors[:20],
        "error_count": len(errors),
    }
    del model, processor
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return arrays, metadata


def batches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def tensor_to_device(value: Any, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    return value


def encode_student_prefill_batch(
    *,
    processor: Any,
    records: list[dict[str, Any]],
    max_length: int,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    messages_batch: list[list[dict[str, Any]]] = []
    image_batch = []
    for sample in records:
        prompt_text = build_user_prompt(sample, PROJECT_ROOT)
        images = load_sample_images(sample, PROJECT_ROOT)
        messages_batch.append(build_messages(prompt_text, len(images), assistant_prefix="<|cot_start|>"))
        image_batch.append(images)
    encoded = _encode_messages(
        processor,
        messages_batch,
        image_batch,
        max_length=max_length,
        continue_final_message=True,
    )
    lengths = encoded["attention_mask"].sum(dim=1).detach().cpu().tolist()
    return encoded, [int(value) for value in lengths]


def forward_student_prefill(model: torch.nn.Module, batch: dict[str, Any], *, device: str, dtype: torch.dtype) -> torch.Tensor:
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
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=(device.startswith("cuda") and dtype != torch.float32)):
            outputs = model.backbone(**kwargs, logits_to_keep=1)
    except TypeError:
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=(device.startswith("cuda") and dtype != torch.float32)):
            outputs = model.backbone(**kwargs)
    final_hidden = get_hidden_states(outputs)[-1].detach()
    positions = last_nonpad_positions(kwargs.get("attention_mask"), final_hidden.shape[1], final_hidden.shape[0]).to(final_hidden.device)
    return final_hidden[torch.arange(final_hidden.shape[0], device=final_hidden.device), positions].float().cpu()


def collect_student_prefill(
    *,
    model_name: str,
    checkpoint_dir: Path | None,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    dtype = torch_dtype(args.dtype)
    config = StudentWrapperConfig(
        student_model_name=str(args.student_model),
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation=args.student_attn_implementation,
    )
    tokenizer = load_student_tokenizer(config)
    processor = load_student_processor(config, tokenizer)
    try:
        processor.tokenizer.padding_side = "right"
        tokenizer.padding_side = "right"
    except Exception:  # noqa: BLE001
        pass
    model = build_student_model(config, tokenizer).to(device).eval()
    checkpoint_load: dict[str, Any] | None = None
    if checkpoint_dir is not None:
        checkpoint_load = load_student_checkpoint(checkpoint_dir, model, use_lora=True)
        model.to(device).eval()

    store: dict[str, list[np.ndarray]] = {}
    prompt_lengths: list[int] = []
    elapsed: list[float] = []
    errors: list[dict[str, str]] = []
    done = 0
    for group in batches(records, args.student_batch_size):
        started = time.perf_counter()
        try:
            batch, lengths = encode_student_prefill_batch(
                processor=processor,
                records=group,
                max_length=args.max_length,
            )
            vectors = forward_student_prefill(model, batch, device=device, dtype=dtype)
            for vector in vectors:
                append_vector(store, "prefill_last", vector)
            prompt_lengths.extend(lengths)
            sync_cuda(device)
            elapsed.append(time.perf_counter() - started)
            done += len(group)
        except Exception as exc:  # noqa: BLE001
            for row in group:
                errors.append({"sample_id": str(row.get("sample_id")), "error": repr(exc)})
        print(json.dumps({"event": "student_prefill_progress", "model": model_name, "done": done, "total": len(records)}), flush=True)
    arrays = {name: np.stack(values, axis=0) for name, values in store.items() if values}
    metadata = {
        "model_name": model_name,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "checkpoint_load": checkpoint_load,
        "attn_implementation": args.student_attn_implementation,
        "dtype": args.dtype,
        "samples_requested": len(records),
        "samples_collected": int(arrays.get("prefill_last", np.empty((0,))).shape[0]),
        "prompt_length_mean": as_float(np.mean(prompt_lengths)) if prompt_lengths else None,
        "prompt_length_p50": as_float(np.percentile(prompt_lengths, 50)) if prompt_lengths else None,
        "prompt_length_p95": as_float(np.percentile(prompt_lengths, 95)) if prompt_lengths else None,
        "elapsed_batch_sec_mean": as_float(np.mean(elapsed)) if elapsed else None,
        "elapsed_batch_sec_p50": as_float(np.percentile(elapsed, 50)) if elapsed else None,
        "elapsed_batch_sec_p95": as_float(np.percentile(elapsed, 95)) if elapsed else None,
        "errors": errors[:20],
        "error_count": len(errors),
    }
    del model, processor, tokenizer
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return arrays, metadata


def common_ratio(x: np.ndarray) -> float | None:
    x = x.astype(np.float32, copy=False)
    token_mean = x.mean(axis=0)
    mean_sq = float(np.dot(token_mean, token_mean))
    token_energy = float(np.mean(np.sum(x * x, axis=1)))
    if token_energy <= 1e-12:
        return None
    return as_float(mean_sq / token_energy)


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


def summarize_report(
    *,
    teacher_arrays: dict[str, np.ndarray],
    student_arrays: dict[str, dict[str, np.ndarray]],
    metadata: dict[str, Any],
    sample_ids: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    teacher = {name: vector_summary(array) for name, array in teacher_arrays.items()}
    students = {
        model_name: {name: vector_summary(array) for name, array in arrays.items()}
        for model_name, arrays in student_arrays.items()
    }
    cka: dict[str, Any] = {}
    teacher_prefill = teacher_arrays.get("prefill_last")
    for model_name, arrays in student_arrays.items():
        value = None
        if teacher_prefill is not None and "prefill_last" in arrays:
            n = min(teacher_prefill.shape[0], arrays["prefill_last"].shape[0])
            value = linear_cka(arrays["prefill_last"][:n], teacher_prefill[:n])
        cka[model_name] = {
            "prefill_last_linear_cka_to_teacher": value,
            "student_shape": list(arrays.get("prefill_last", np.empty((0, 0))).shape),
            "teacher_shape": list(teacher_prefill.shape) if teacher_prefill is not None else None,
        }

    deltas: dict[str, Any] = {}
    base_value = (cka.get("base_2b") or {}).get("prefill_last_linear_cka_to_teacher")
    for model_name, item in cka.items():
        if model_name == "base_2b":
            continue
        value = item.get("prefill_last_linear_cka_to_teacher")
        if base_value is not None and value is not None:
            deltas[model_name] = {
                "checkpoint_minus_base_prefill_cka": as_float(float(value) - float(base_value)),
                "base_prefill_cka": base_value,
                "checkpoint_prefill_cka": value,
            }

    return {
        "schema_version": "no_nav_true_prefill_hidden_qc_v1",
        "corpus_jsonl": str(args.corpus_jsonl),
        "split": args.split,
        "num_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "teacher_model": str(args.teacher_checkpoint),
        "student_model": str(args.student_model),
        "comparison_target": "final hidden at prompt-only prefill end, before decoding CoT/traj tokens",
        "important_caveat": (
            "Teacher uses the official Alpamayo prompt with fused <|traj_history|> placeholders; "
            "student uses the current distillation collator prompt with numeric ego history text. "
            "This matches the actual student training setup, but is not token-identical."
        ),
        "metadata": metadata,
        "teacher": teacher,
        "students": students,
        "cka_to_teacher": cka,
        "checkpoint_minus_base": deltas,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# No-Nav True Prefill Hidden QC")
    lines.append("")
    lines.append(f"- samples: {report['num_samples']} `{report['split']}` rows")
    lines.append("- target: final hidden after prompt-only prefill, before CoT/traj decode")
    lines.append(f"- caveat: {report['important_caveat']}")
    lines.append("")
    lines.append("## Prefill CKA To Teacher")
    lines.append("")
    lines.append("| model | CKA | delta vs base | hidden shape |")
    lines.append("|---|---:|---:|---|")
    base_value = (report["cka_to_teacher"].get("base_2b") or {}).get("prefill_last_linear_cka_to_teacher")
    for model_name, item in report["cka_to_teacher"].items():
        value = item.get("prefill_last_linear_cka_to_teacher")
        delta = None if model_name == "base_2b" or value is None or base_value is None else float(value) - float(base_value)
        lines.append(f"| {model_name} | {fmt(value)} | {fmt(delta)} | `{item.get('student_shape')}` |")
    lines.append("")
    lines.append("## Distribution / Collapse Checks")
    lines.append("")
    lines.append("| model | norm mean | offdiag cos mean | top PC var | effective rank | common ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    teacher_row = (report["teacher"].get("prefill_last") or {})
    lines.append(
        "| teacher_8b | "
        + " | ".join(
            [
                fmt(teacher_row.get("norm_mean")),
                fmt(teacher_row.get("offdiag_cos_mean")),
                fmt(teacher_row.get("top_pc_var_ratio")),
                fmt(teacher_row.get("effective_rank")),
                fmt(teacher_row.get("common_ratio")),
            ]
        )
        + " |"
    )
    for model_name, model_summary in report["students"].items():
        row = model_summary.get("prefill_last") or {}
        lines.append(
            f"| {model_name} | "
            + " | ".join(
                [
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
    lines.append("## Prompt Length / Timing")
    lines.append("")
    lines.append("| model | collected | prompt p50 | prompt p95 | elapsed p50 | elapsed p95 | errors |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    teacher_meta = report["metadata"].get("teacher", {})
    lines.append(
        f"| teacher_8b | {teacher_meta.get('samples_collected')} | "
        f"{fmt(teacher_meta.get('prompt_length_p50'))} | {fmt(teacher_meta.get('prompt_length_p95'))} | "
        f"{fmt(teacher_meta.get('elapsed_sec_p50'))} | {fmt(teacher_meta.get('elapsed_sec_p95'))} | "
        f"{teacher_meta.get('error_count')} |"
    )
    for model_name, model_meta in (report["metadata"].get("students") or {}).items():
        lines.append(
            f"| {model_name} | {model_meta.get('samples_collected')} | "
            f"{fmt(model_meta.get('prompt_length_p50'))} | {fmt(model_meta.get('prompt_length_p95'))} | "
            f"{fmt(model_meta.get('elapsed_batch_sec_p50'))} | {fmt(model_meta.get('elapsed_batch_sec_p95'))} | "
            f"{model_meta.get('error_count')} |"
        )
    lines.append("")
    lines.append("## Read")
    lines.append("")
    deltas = report.get("checkpoint_minus_base") or {}
    if not deltas:
        lines.append("- No checkpoint delta was computed.")
    else:
        best_name = max(deltas, key=lambda name: deltas[name].get("checkpoint_prefill_cka") or -1.0)
        best = deltas[best_name]
        lines.append(
            f"- Best checkpoint by true prefill CKA here: `{best_name}` "
            f"({fmt(best.get('checkpoint_prefill_cka'))}, delta {fmt(best.get('checkpoint_minus_base_prefill_cka'))})."
        )
    lines.append("- Treat this as an interface/distribution QC, not ADE/FDE. The important signal is whether distillation moves prompt-only planning state toward teacher without representation collapse.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.corpus_jsonl, split=args.split, count=args.num_samples, seed=args.seed)
    sample_ids = [str(row.get("sample_id")) for row in records]
    checkpoint_specs = parse_checkpoint_specs(args.student_checkpoint)

    print(json.dumps({"event": "teacher_prefill_start", "samples": len(records)}), flush=True)
    teacher_arrays, teacher_metadata = collect_teacher_prefill(records, args)
    if not teacher_arrays:
        raise RuntimeError("No teacher prefill vectors were collected.")
    print(json.dumps({"event": "teacher_prefill_done", "metadata": teacher_metadata}), flush=True)

    student_arrays: dict[str, dict[str, np.ndarray]] = {}
    student_metadata: dict[str, Any] = {}
    for model_name, checkpoint_dir in [("base_2b", None), *checkpoint_specs]:
        if checkpoint_dir is not None and not checkpoint_dir.exists():
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {checkpoint_dir}")
        print(json.dumps({"event": "student_prefill_start", "model": model_name, "checkpoint": str(checkpoint_dir) if checkpoint_dir else None}), flush=True)
        arrays, metadata = collect_student_prefill(
            model_name=model_name,
            checkpoint_dir=checkpoint_dir,
            records=records,
            args=args,
        )
        student_arrays[model_name] = arrays
        student_metadata[model_name] = metadata
        print(json.dumps({"event": "student_prefill_done", "model": model_name, "metadata": metadata}), flush=True)

    report = summarize_report(
        teacher_arrays=teacher_arrays,
        student_arrays=student_arrays,
        metadata={"teacher": teacher_metadata, "students": student_metadata},
        sample_ids=sample_ids,
        args=args,
    )
    report_json = args.output_dir / args.report_name
    report_md = args.output_dir / args.markdown_name
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, report_md)
    if args.save_embeddings:
        np.savez_compressed(args.output_dir / "teacher_prefill_vectors.npz", **teacher_arrays)
        for model_name, arrays in student_arrays.items():
            np.savez_compressed(args.output_dir / f"{model_name}_prefill_vectors.npz", **arrays)
    print(json.dumps({"event": "done", "report_json": str(report_json), "report_md": str(report_md)}), flush=True)


if __name__ == "__main__":
    main()
