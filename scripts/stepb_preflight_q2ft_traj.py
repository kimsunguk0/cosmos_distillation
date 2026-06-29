#!/usr/bin/env python3
"""Preflight the Step B Q2-FT trajectory-only distillation contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.student_wrapper import StudentWrapperConfig, load_student_processor, load_student_tokenizer
from src.model.tokenizer_ext import missing_special_tokens
from src.training.collator import DistillationCollator, load_traj_future_token_ids
from src.training.losses import export_loss_weights
from src.utils.runtime_paths import remap_external_path, resolve_student_model_path
from src.utils.traj_tokens import DEFAULT_TRAJ_VOCAB_SIZE, discrete_traj_token


def _load_train_module():
    path = PROJECT_ROOT / "scripts" / "09_train_distill.py"
    spec = importlib.util.spec_from_file_location("train_distill_09", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _path_exists(raw_path: str | Path | None) -> bool:
    if raw_path in (None, ""):
        return False
    resolved = remap_external_path(raw_path)
    return resolved is not None and Path(resolved).exists()


def _first_topk_summary(records: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    shapes: list[list[int]] = []
    topk_values: list[int] = []
    raw_min = None
    raw_max = None
    checked = 0
    missing = 0
    for record in records[: max(int(limit), 0)]:
        checked += 1
        target = record.get("teacher_traj_target") or {}
        raw_path = (
            target.get("topk_logits_path")
            or target.get("topk_path")
            or target.get("teacher_traj_topk_ids_path")
        )
        resolved = remap_external_path(raw_path)
        if resolved is None or not Path(resolved).exists():
            missing += 1
            continue
        with np.load(Path(resolved)) as payload:
            index_key = "topk_indices" if "topk_indices" in payload.files else "topk_ids"
            arr = np.asarray(payload[index_key])
        shapes.append([int(v) for v in arr.shape])
        if arr.ndim >= 2:
            topk_values.append(int(arr.shape[-1]))
        if arr.size:
            current_min = int(arr.min())
            current_max = int(arr.max())
            raw_min = current_min if raw_min is None else min(raw_min, current_min)
            raw_max = current_max if raw_max is None else max(raw_max, current_max)
    return {
        "checked": checked,
        "missing": missing,
        "unique_topk": sorted(set(topk_values)),
        "example_shapes": shapes[:5],
        "raw_index_min": raw_min,
        "raw_index_max": raw_max,
    }


def _split_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [record for record in records if record.get("split") == "train"]
    val = [record for record in records if record.get("split") == "val"]
    return train, val


def _mask_counts(batch: dict[str, Any]) -> list[dict[str, int]]:
    rows = int(batch["input_ids"].shape[0])
    out = []
    for row in range(rows):
        out.append(
            {
                "labels": int(torch.count_nonzero(batch["labels"][row] != -100).item()),
                "traj_token_mask": int(torch.count_nonzero(batch["traj_token_mask"][row]).item()),
                "traj_span_mask": int(torch.count_nonzero(batch["traj_span_mask"][row]).item()),
                "cot_span_mask": int(torch.count_nonzero(batch["cot_span_mask"][row]).item()),
                "cot_content_mask": int(torch.count_nonzero(batch["cot_content_mask"][row]).item()),
                "format_token_mask": int(torch.count_nonzero(batch["format_token_mask"][row]).item()),
            }
        )
    return out


def _shape(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.shape]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl",
    )
    parser.add_argument(
        "--stage-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train" / "stage_stepb_q2ft_trajonly_fullft.yaml",
    )
    parser.add_argument(
        "--student-model",
        default=str(PROJECT_ROOT / "outputs" / "checkpoints" / "stepa_q2_vqa_fullft_repaired_v1_bs8_e1" / "step_003488"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=16)
    parser.add_argument("--max-val-samples", type=int, default=8)
    parser.add_argument("--topk-check-samples", type=int, default=64)
    parser.add_argument("--expected-topk", type=int, default=64)
    parser.add_argument("--future-bins", type=int, default=3000)
    parser.add_argument("--skip-asset-check", action="store_true")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "outputs" / "reports" / "stepb_preflight_q2ft_traj.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_mod = _load_train_module()
    trainer_cfg, loss_weights, stage_options = train_mod.stage_weights_from_yaml(args.stage_config)
    data_view = dict(stage_options.get("data_view") or {})
    student_model = str(resolve_student_model_path(args.student_model))

    records = train_mod.load_jsonl(args.corpus_jsonl)
    all_train, all_val = _split_records(records)
    if args.skip_asset_check:
        train_records = list(all_train)
        val_records = list(all_val)
    else:
        train_records = [record for record in all_train if train_mod.has_required_materialized_assets(record)]
        val_records = [record for record in all_val if train_mod.has_required_materialized_assets(record)]
    train_records = train_records[: args.max_train_samples]
    val_records = val_records[: args.max_val_samples]
    if not train_records:
        raise RuntimeError("No train records are available after filtering.")

    raw_tokenizer_source = Path(student_model) / "tokenizer" if (Path(student_model) / "tokenizer").exists() else Path(student_model)
    raw_tokenizer = AutoTokenizer.from_pretrained(
        str(raw_tokenizer_source),
        trust_remote_code=True,
        local_files_only=Path(student_model).exists(),
    )
    raw_vocab = set(raw_tokenizer.get_vocab().keys())
    missing_before = missing_special_tokens(raw_vocab, traj_vocab_size=DEFAULT_TRAJ_VOCAB_SIZE)

    wrapper_cfg = StudentWrapperConfig(
        student_model_name=student_model,
        max_length=int(trainer_cfg.max_length),
        torch_dtype=train_mod.preferred_model_dtype(bf16=bool(trainer_cfg.bf16)),
        local_files_only=Path(student_model).exists(),
        attn_implementation=stage_options.get("attn_implementation"),
    )
    tokenizer = load_student_tokenizer(wrapper_cfg)
    processor = load_student_processor(wrapper_cfg, tokenizer=tokenizer)
    traj_token_start = int(tokenizer.convert_tokens_to_ids("<i0>"))
    traj_token_end_future = int(tokenizer.convert_tokens_to_ids("<i2999>"))
    traj_token_end_vocab = int(tokenizer.convert_tokens_to_ids(f"<i{DEFAULT_TRAJ_VOCAB_SIZE - 1}>"))

    collator = DistillationCollator(
        tokenizer=tokenizer,
        processor=processor,
        project_root=PROJECT_ROOT,
        max_length=int(trainer_cfg.max_length),
        prompt_mode=str(data_view.get("prompt_mode", "joint")),
        target_mode=str(data_view.get("target_mode", "joint")),
        teacher_pair_target=bool(data_view.get("teacher_pair_target", False)),
        enable_teacher_view=bool(data_view.get("enable_teacher_view", False)),
        enable_action_aux=bool(data_view.get("enable_action_aux", False)),
        teacher_traj_cache_dir=None,
        teacher_traj_hidden_source=str(data_view.get("teacher_traj_hidden_source", "hidden")),
        teacher_traj_latent_suffix=str(data_view.get("teacher_traj_latent_suffix", "lat32")),
        hard_view_uses_teacher_cot=bool(data_view.get("hard_view_uses_teacher_cot", False)),
        teacher_view_force_enable=bool(data_view.get("teacher_view_force_enable", False)),
        teacher_view_uses_teacher_traj=bool(data_view.get("teacher_view_uses_teacher_traj", False)),
        teacher_view_default_traj_weight=float(data_view.get("teacher_view_default_traj_weight", 0.0) or 0.0),
        teacher_traj_topk_on_teacher_view=bool(data_view.get("teacher_traj_topk_on_teacher_view", False)),
        image_prompt_style=str(data_view.get("image_prompt_style", "compact")),
        prompt_text_style=str(data_view.get("prompt_text_style", "numeric_history_question")),
        fuse_history_tokens=bool(data_view.get("fuse_history_tokens", False)),
    )
    batch = collator(train_records[: max(int(args.batch_size), 1)])

    sample = train_records[0]
    sample_input = sample.get("input") or {}
    sample_target = sample.get("hard_target") or {}
    token_ids = load_traj_future_token_ids(sample_target, PROJECT_ROOT)
    image_names = list(sample_input.get("image_names") or [])
    topk_summary = _first_topk_summary(train_records, limit=args.topk_check_samples)
    batch_topk = batch.get("teacher_traj_topk_indices")
    warnings: list[str] = []
    observed_topk = topk_summary.get("unique_topk") or []
    if observed_topk and observed_topk != [int(args.expected_topk)]:
        warnings.append(f"teacher trajectory top-k cache is {observed_topk}, expected top-k{args.expected_topk}")
    if int(args.future_bins) > int(args.expected_topk):
        warnings.append(f"future bin count {args.future_bins} > top-k {args.expected_topk}; this is sparse KL, not full KL")
    if not bool((stage_options.get("flex") or {}).get("enabled", False)):
        warnings.append("relative_timestamps are batched, but standard non-FLEX Qwen forward ignores them as tensor conditioning")

    masks = _mask_counts(batch)
    for row, counts in enumerate(masks):
        if counts["traj_token_mask"] != 128:
            warnings.append(f"row {row} has traj_token_mask={counts['traj_token_mask']}, expected 128")
        if counts["cot_span_mask"] != 0 or counts["cot_content_mask"] != 0:
            warnings.append(f"row {row} has nonzero CoT mask in traj-only mode")

    summary = {
        "mode": "stepb_q2ft_traj_preflight",
        "corpus_jsonl": str(args.corpus_jsonl),
        "stage_config": str(args.stage_config),
        "student_model": student_model,
        "records": {
            "total": len(records),
            "all_train": len(all_train),
            "all_val": len(all_val),
            "selected_train": len(train_records),
            "selected_val": len(val_records),
            "skip_asset_check": bool(args.skip_asset_check),
        },
        "stage": {
            "name": trainer_cfg.stage_name,
            "bf16": bool(trainer_cfg.bf16),
            "gradient_checkpointing": bool(trainer_cfg.gradient_checkpointing),
            "learning_rate": float(trainer_cfg.learning_rate),
            "attn_implementation": stage_options.get("attn_implementation"),
            "data_view": data_view,
            "loss_weights": export_loss_weights(loss_weights),
        },
        "tokenizer": {
            "raw_length": len(raw_tokenizer),
            "extended_length": len(tokenizer),
            "missing_custom_tokens_before_extension": len(missing_before),
            "traj_token_start_id": traj_token_start,
            "future_token_end_id_i2999": traj_token_end_future,
            "traj_vocab_end_id_i3999": traj_token_end_vocab,
            "answer_start_id": int(tokenizer.convert_tokens_to_ids("<|answer_start|>")),
            "traj_future_start_id": int(tokenizer.convert_tokens_to_ids("<|traj_future_start|>")),
            "traj_future_end_id": int(tokenizer.convert_tokens_to_ids("<|traj_future_end|>")),
            "traj_vocab_size": int(DEFAULT_TRAJ_VOCAB_SIZE),
            "future_bins": int(args.future_bins),
            "future_token_example": discrete_traj_token(0),
        },
        "sample0": {
            "sample_id": sample.get("sample_id"),
            "image_layout": sample_input.get("image_layout"),
            "image_count": int(sample_input.get("image_count") or len(image_names) or len(sample_input.get("image_paths") or [])),
            "camera_count": sample_input.get("camera_count"),
            "num_frames_per_camera": sample_input.get("num_frames_per_camera"),
            "image_names_count": len(image_names),
            "materialized_sample_path_exists": _path_exists(sample_input.get("materialized_sample_path")),
            "metadata_path_exists": _path_exists(sample_input.get("metadata_path")),
            "ego_history_path_exists": _path_exists(sample_input.get("ego_history_path")),
            "target_token_path_exists": _path_exists(sample_target.get("traj_future_token_ids_path")),
            "target_token_count": len(token_ids),
            "target_token_min": min(token_ids) if token_ids else None,
            "target_token_max": max(token_ids) if token_ids else None,
        },
        "teacher_traj_topk": topk_summary,
        "batch": {
            "shapes": {
                "input_ids": _shape(batch.get("input_ids")),
                "attention_mask": _shape(batch.get("attention_mask")),
                "labels": _shape(batch.get("labels")),
                "pixel_values": _shape(batch.get("pixel_values")),
                "image_grid_thw": _shape(batch.get("image_grid_thw")),
                "camera_indices": _shape(batch.get("camera_indices")),
                "relative_timestamps": _shape(batch.get("relative_timestamps")),
                "frames_per_camera": _shape(batch.get("frames_per_camera")),
                "teacher_traj_topk_indices": _shape(batch.get("teacher_traj_topk_indices")),
                "teacher_traj_topk_logprobs": _shape(batch.get("teacher_traj_topk_logprobs")),
                "teacher_traj_topk_mask": _shape(batch.get("teacher_traj_topk_mask")),
            },
            "mask_counts": masks,
            "teacher_traj_topk_token_id_min": int(batch_topk.min().item()) if isinstance(batch_topk, torch.Tensor) and batch_topk.numel() else None,
            "teacher_traj_topk_token_id_max": int(batch_topk.max().item()) if isinstance(batch_topk, torch.Tensor) and batch_topk.numel() else None,
        },
        "warnings": warnings,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
