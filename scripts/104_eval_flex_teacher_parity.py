#!/usr/bin/env python3
"""Evaluate FLEX compression parity against a frozen no-FLEX teacher model.

Teacher: B0 no-FLEX checkpoint.
Student: same base checkpoint plus FLEX, usually F0 or F1.

The key metric is not Alpamayo teacher accuracy.  It is whether the FLEX
student preserves the frozen no-FLEX model's hidden states, trajectory logits,
and teacher-forced trajectory geometry under the compressed visual prefix.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint  # noqa: E402
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora  # noqa: E402
from src.model.student_wrapper import StudentWrapperConfig, build_student_model  # noqa: E402
from src.model.tokenizer_ext import distill_trainable_token_ids, ensure_special_tokens  # noqa: E402
from src.training.collator import DistillationCollator, load_ego_history_xyz, load_traj_future_token_ids  # noqa: E402
from src.training.flex_batch import attach_qwen_mrope_position_ids, compress_batch_for_flex  # noqa: E402
from src.utils.runtime_paths import resolve_student_model_path  # noqa: E402


def _load_decode_module():
    path = PROJECT_ROOT / "scripts" / "25_decode_checkpoint_overlays.py"
    spec = importlib.util.spec_from_file_location("decode_checkpoint_overlays_25", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import decode helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


decode_mod = _load_decode_module()
TrajectoryTokenDecoder = decode_mod.TrajectoryTokenDecoder
load_ego_history_rot = decode_mod.load_ego_history_rot
resolve_traj_tokenizer_config_path = decode_mod.resolve_traj_tokenizer_config_path

BOUNDARY_NAMES = ("cot_end", "traj_start", "action_pre")
IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--student-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--student-model", default=resolve_student_model_path())
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--samples-jsonl", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preserve-flex-positions", action="store_true")
    parser.add_argument("--flex-selection-strategy", choices=("first", "uniform"), default="first")
    parser.add_argument("--flex-dummy-image-slots", action="store_true")
    parser.add_argument("--flex-residual-image-slots", action="store_true")
    parser.add_argument("--flex-residual-scale", type=float, default=1.0)
    parser.add_argument("--flex-passthrough-image-slots", action="store_true")
    parser.add_argument("--flex-scene-deepstack", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batched(rows: list[dict[str, Any]], batch_size: int):
    width = max(int(batch_size), 1)
    for index in range(0, len(rows), width):
        yield rows[index : index + width]


def mean(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def percentile(values: list[float | int | None], q: float) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.percentile(clean, q)) if clean else None


def ratio(num: int, den: int) -> float:
    return float(num / max(int(den), 1))


def ade_fde(pred: np.ndarray | None, target: np.ndarray | None) -> tuple[float, float]:
    if pred is None or target is None:
        return float("nan"), float("nan")
    n = min(int(pred.shape[0]), int(target.shape[0]))
    if n <= 0:
        return float("nan"), float("nan")
    dist = np.linalg.norm(pred[:n, :2] - target[:n, :2], axis=-1)
    return float(dist.mean()), float(dist[-1])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_model(checkpoint_dir: Path, *, student_model: str, device: torch.device):
    train_config = _load_json(checkpoint_dir / "train_config.json")
    checkpoint_manifest = _load_json(checkpoint_dir / "checkpoint_manifest.json")
    base_model = str((train_config.get("args") or {}).get("student_model") or student_model)
    use_lora = not bool((train_config.get("args") or {}).get("disable_lora", False))
    data_view = train_config.get("data_view") or {}

    from transformers import AutoProcessor, AutoTokenizer

    tokenizer_dir = checkpoint_dir / "tokenizer"
    processor_dir = checkpoint_dir / "processor"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir if tokenizer_dir.exists() else base_model,
        local_files_only=True,
    )
    ensure_special_tokens(tokenizer)
    processor = AutoProcessor.from_pretrained(
        processor_dir if processor_dir.exists() else base_model,
        local_files_only=True,
    )
    processor.tokenizer = tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"

    wrapper_cfg = StudentWrapperConfig(
        student_model_name=base_model,
        max_length=int((train_config.get("trainer_config") or {}).get("max_length", 4096)),
        torch_dtype=torch.bfloat16 if device.type == "cuda" else None,
        local_files_only=Path(base_model).expanduser().exists(),
        traj_teacher_hidden_size=(
            int(data_view.get("teacher_traj_hidden_size"))
            if data_view.get("teacher_traj_hidden_size") not in (None, "", 0)
            else None
        ),
        traj_hidden_bridge_size=(
            int(checkpoint_manifest.get("traj_hidden_bridge_size"))
            if checkpoint_manifest.get("traj_hidden_bridge_size") not in (None, "", 0)
            else None
        ),
    )
    model = build_student_model(wrapper_cfg, tokenizer)
    if detect_checkpoint_format(checkpoint_dir) == "full_state_dict" and use_lora:
        model.backbone = maybe_apply_lora(
            model.backbone,
            LoraConfigSpec(trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer))),
            enabled=True,
        )
    load_student_checkpoint(checkpoint_dir, model, use_lora=use_lora, adapter_trainable=False)
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)
    return model.eval(), tokenizer, processor, base_model, train_config


def move_batch(batch: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def forward_model(model, moved: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        "input_ids": moved["input_ids"],
        "attention_mask": moved["attention_mask"],
        "return_hidden_states": True,
        "compute_meta_action": False,
        "compute_traj_aux": False,
    }
    for optional_key in (
        "pixel_values",
        "image_grid_thw",
        "camera_indices",
        "relative_timestamps",
        "camera_counts",
        "frames_per_camera",
        "position_ids",
        "flex_allow_dummy_image_slots",
        "flex_residual_image_slots",
        "flex_residual_scale",
        "flex_passthrough_image_slots",
        "flex_scene_deepstack",
    ):
        if optional_key in moved and moved[optional_key] is not None:
            kwargs[optional_key] = moved[optional_key]
    return model(**kwargs)


def _label_positions(batch: dict[str, Any], row_index: int, mask_key: str) -> torch.Tensor:
    labels = batch["labels"]
    mask = batch[mask_key].bool() & (labels != IGNORE_INDEX)
    positions = torch.nonzero(mask[row_index], as_tuple=False).flatten()
    return positions[positions > 0]


def _traj_logits(
    logits: torch.Tensor,
    positions: torch.Tensor,
    *,
    row_index: int,
    traj_start: int,
    num_bins: int,
) -> torch.Tensor:
    return logits[row_index, positions - 1, traj_start : traj_start + num_bins].float()


def _full_logits(logits: torch.Tensor, positions: torch.Tensor, *, row_index: int) -> torch.Tensor:
    return logits[row_index, positions - 1, :].float()


def _kl_teacher_student(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    return (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)


def _hidden_cosines(
    teacher_hidden: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_batch: dict[str, Any],
    student_batch: dict[str, Any],
    row_index: int,
) -> dict[str, float | None]:
    teacher_positions = teacher_batch.get("teacher_text_boundary_hidden_positions")
    student_positions = student_batch.get("teacher_text_boundary_hidden_positions")
    out: dict[str, float | None] = {}
    if not isinstance(teacher_positions, torch.Tensor) or not isinstance(student_positions, torch.Tensor):
        return {name: None for name in BOUNDARY_NAMES}
    for boundary_index, name in enumerate(BOUNDARY_NAMES):
        t_pos = int(teacher_positions[row_index, boundary_index].item())
        s_pos = int(student_positions[row_index, boundary_index].item())
        if t_pos < 0 or s_pos < 0 or t_pos >= teacher_hidden.shape[1] or s_pos >= student_hidden.shape[1]:
            out[name] = None
            continue
        t_vec = teacher_hidden[row_index, t_pos].float()
        s_vec = student_hidden[row_index, s_pos].float()
        out[name] = float(F.cosine_similarity(t_vec, s_vec, dim=0).detach().cpu())
        out[f"{name}_norm_ratio"] = float(
            (s_vec.norm() / t_vec.norm().clamp(min=1e-6)).detach().cpu()
        )
    return out


def main() -> int:
    args = parse_args()
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    if args.samples_jsonl is None:
        args.samples_jsonl = args.summary_json.with_suffix(".samples.jsonl")

    rows = [row for row in load_jsonl(args.corpus_jsonl) if row.get("split") == args.split]
    if args.num_samples > 0:
        rows = rows[: args.num_samples]
    if not rows:
        raise SystemExit(f"No rows selected for split={args.split!r}")

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    teacher, tokenizer, processor, base_model, teacher_train_config = load_model(
        args.teacher_checkpoint_dir,
        student_model=args.student_model,
        device=device,
    )
    student, _, _, _, student_train_config = load_model(
        args.student_checkpoint_dir,
        student_model=args.student_model,
        device=device,
    )
    if not (hasattr(student, "flex_enabled") and student.flex_enabled()):
        raise SystemExit("Student checkpoint does not have FLEX enabled; parity eval expects F0/F1/F2 student.")

    data_view = teacher_train_config.get("data_view") or {}
    collator = DistillationCollator(
        tokenizer=tokenizer,
        processor=processor,
        project_root=PROJECT_ROOT,
        teacher_pair_target=bool(data_view.get("teacher_pair_target", False)),
        enable_teacher_view=False,
        enable_action_aux=False,
        hard_view_uses_teacher_cot=bool(data_view.get("hard_view_uses_teacher_cot", True)),
        prompt_mode=str(data_view.get("prompt_mode") or "joint"),
        target_mode=str(data_view.get("target_mode") or "joint"),
        image_prompt_style=str(data_view.get("image_prompt_style") or "camera_labeled"),
        prompt_text_style=str(data_view.get("prompt_text_style") or "official_alpamayo"),
        fuse_history_tokens=bool(data_view.get("fuse_history_tokens", True)),
        max_length=int((teacher_train_config.get("trainer_config") or {}).get("max_length", 4096)),
    )

    decoder_path = resolve_traj_tokenizer_config_path(base_model)
    if decoder_path is None:
        raise SystemExit("Could not find Alpamayo trajectory tokenizer config.")
    decoder = TrajectoryTokenDecoder(config_path=decoder_path)
    traj_start = int(getattr(tokenizer, "traj_token_start_idx", tokenizer.convert_tokens_to_ids("<i0>")))
    num_bins = int(decoder.num_bins)
    dtype = next(teacher.backbone.parameters()).dtype

    metrics: dict[str, list[float]] = defaultdict(list)
    sample_rows: list[dict[str, Any]] = []
    student_flex_config = getattr(student, "flex_scene_config")
    teacher_flex_stats: dict[str, Any] = {
        "student_flex_enabled": bool(student.flex_enabled()),
        "student_flex_config": (
            asdict(student_flex_config) if is_dataclass(student_flex_config) else dict(student_flex_config or {})
        ),
    }

    for batch_rows in batched(rows, args.batch_size):
        batch = collator(batch_rows)
        student_batch = batch
        flex_cfg = getattr(student, "flex_scene_config")
        if bool(args.flex_dummy_image_slots) or bool(args.flex_residual_image_slots):
            student_batch = dict(student_batch)
            if bool(args.flex_dummy_image_slots):
                student_batch["flex_allow_dummy_image_slots"] = True
            if bool(args.flex_residual_image_slots):
                student_batch["flex_residual_image_slots"] = True
                student_batch["flex_residual_scale"] = float(args.flex_residual_scale)
            if str(args.flex_selection_strategy) != "first":
                student_batch["flex_selection_strategy"] = str(args.flex_selection_strategy)
        else:
            if bool(args.preserve_flex_positions):
                student_batch = attach_qwen_mrope_position_ids(student_batch, student)
            student_batch = compress_batch_for_flex(
                student_batch,
                image_token_id=int(getattr(student, "image_token_id")),
                tokens_per_image=int(getattr(flex_cfg, "tokens_per_image")),
                pad_token_id=int(getattr(tokenizer, "pad_token_id", 0) or 0),
                preserve_original_position_ids=bool(args.preserve_flex_positions),
                selection_strategy=str(args.flex_selection_strategy),
            )
            if str(args.flex_selection_strategy) != "first":
                student_batch["flex_selection_strategy"] = str(args.flex_selection_strategy)
            if bool(args.flex_scene_deepstack):
                student_batch["flex_scene_deepstack"] = True
            if bool(args.flex_passthrough_image_slots):
                student_batch["flex_passthrough_image_slots"] = True
        teacher_moved = move_batch(batch, device=device, dtype=dtype)
        student_moved = move_batch(student_batch, device=device, dtype=dtype)
        with torch.inference_mode():
            teacher_out = forward_model(teacher, teacher_moved)
            student_out = forward_model(student, student_moved)
        teacher_logits = teacher_out["logits"].float()
        student_logits = student_out["logits"].float()
        teacher_hidden = teacher_out["hidden_states"].float()
        student_hidden = student_out["hidden_states"].float()

        for row_index, sample in enumerate(batch_rows):
            sid = str(sample.get("sample_id"))
            teacher_traj_pos = _label_positions(teacher_moved, row_index, "traj_token_mask")
            student_traj_pos = _label_positions(student_moved, row_index, "traj_token_mask")
            usable = min(int(teacher_traj_pos.numel()), int(student_traj_pos.numel()), 128)
            row: dict[str, Any] = {
                "sample_id": sid,
                "teacher_seq_len": int(teacher_moved["input_ids"].shape[1]),
                "student_seq_len": int(student_moved["input_ids"].shape[1]),
                "usable_traj_tokens": int(usable),
            }
            if usable > 0:
                t_logits = _traj_logits(
                    teacher_logits,
                    teacher_traj_pos[:usable],
                    row_index=row_index,
                    traj_start=traj_start,
                    num_bins=num_bins,
                )
                s_logits = _traj_logits(
                    student_logits,
                    student_traj_pos[:usable],
                    row_index=row_index,
                    traj_start=traj_start,
                    num_bins=num_bins,
                )
                kl = _kl_teacher_student(t_logits, s_logits)
                t_pred = t_logits.argmax(dim=-1)
                s_pred = s_logits.argmax(dim=-1)
                top5 = torch.topk(s_logits, k=5, dim=-1).indices
                agree = t_pred == s_pred
                top5_agree = (top5 == t_pred[:, None]).any(dim=-1)
                row["traj_teacher_student_kl"] = float(kl.mean().detach().cpu())
                row["traj_top1_agreement"] = float(agree.float().mean().detach().cpu())
                row["traj_teacher_top1_in_student_top5"] = float(top5_agree.float().mean().detach().cpu())
                row["student_unique_tf_argmax"] = int(torch.unique(s_pred).numel())
                row["teacher_unique_tf_argmax"] = int(torch.unique(t_pred).numel())
                metrics["traj_teacher_student_kl"].append(row["traj_teacher_student_kl"])
                metrics["traj_top1_agreement"].append(row["traj_top1_agreement"])
                metrics["traj_teacher_top1_in_student_top5"].append(row["traj_teacher_top1_in_student_top5"])
                metrics["student_unique_tf_argmax"].append(row["student_unique_tf_argmax"])
                metrics["teacher_unique_tf_argmax"].append(row["teacher_unique_tf_argmax"])

                target_ids = load_traj_future_token_ids(sample.get("hard_target") or {}, PROJECT_ROOT)
                history_xyz = load_ego_history_xyz(sample, PROJECT_ROOT)
                history_rot = load_ego_history_rot(sample, PROJECT_ROOT)
                target_xyz = decoder.decode(history_xyz, history_rot, target_ids[:128])
                teacher_xyz = decoder.decode(history_xyz, history_rot, [int(x) for x in t_pred[:128].detach().cpu().tolist()])
                student_xyz = decoder.decode(history_xyz, history_rot, [int(x) for x in s_pred[:128].detach().cpu().tolist()])
                teacher_ade, teacher_fde = ade_fde(teacher_xyz, target_xyz)
                student_ade, student_fde = ade_fde(student_xyz, target_xyz)
                row["teacher_tf_argmax_ade"] = teacher_ade if math.isfinite(teacher_ade) else None
                row["teacher_tf_argmax_fde"] = teacher_fde if math.isfinite(teacher_fde) else None
                row["student_tf_argmax_ade"] = student_ade if math.isfinite(student_ade) else None
                row["student_tf_argmax_fde"] = student_fde if math.isfinite(student_fde) else None
                row["student_minus_teacher_tf_argmax_ade"] = (
                    student_ade - teacher_ade if math.isfinite(student_ade) and math.isfinite(teacher_ade) else None
                )
                for key in (
                    "teacher_tf_argmax_ade",
                    "teacher_tf_argmax_fde",
                    "student_tf_argmax_ade",
                    "student_tf_argmax_fde",
                    "student_minus_teacher_tf_argmax_ade",
                ):
                    metrics[key].append(row[key])

            for mask_key, metric_prefix in (("cot_span_mask", "text"), ("format_token_mask", "format")):
                teacher_pos = _label_positions(teacher_moved, row_index, mask_key)
                student_pos = _label_positions(student_moved, row_index, mask_key)
                text_usable = min(int(teacher_pos.numel()), int(student_pos.numel()))
                if text_usable <= 0:
                    continue
                t_full = _full_logits(teacher_logits, teacher_pos[:text_usable], row_index=row_index)
                s_full = _full_logits(student_logits, student_pos[:text_usable], row_index=row_index)
                text_kl = _kl_teacher_student(t_full, s_full)
                text_agree = t_full.argmax(dim=-1) == s_full.argmax(dim=-1)
                row[f"{metric_prefix}_teacher_student_kl"] = float(text_kl.mean().detach().cpu())
                row[f"{metric_prefix}_top1_agreement"] = float(text_agree.float().mean().detach().cpu())
                metrics[f"{metric_prefix}_teacher_student_kl"].append(row[f"{metric_prefix}_teacher_student_kl"])
                metrics[f"{metric_prefix}_top1_agreement"].append(row[f"{metric_prefix}_top1_agreement"])

            hidden_metrics = _hidden_cosines(
                teacher_hidden,
                student_hidden,
                teacher_moved,
                student_moved,
                row_index,
            )
            for key, value in hidden_metrics.items():
                row[key] = value
                metrics[key].append(value)
            sample_rows.append(row)
        print(json.dumps({"event": "parity_batch_done", "done": len(sample_rows), "total": len(rows)}), flush=True)

    summary = {
        "teacher_checkpoint_dir": str(args.teacher_checkpoint_dir),
        "student_checkpoint_dir": str(args.student_checkpoint_dir),
        "split": args.split,
        "num_samples": len(sample_rows),
        "batch_size": int(args.batch_size),
        "input_contract": {
            "prompt_mode": str((teacher_train_config.get("data_view") or {}).get("prompt_mode") or "joint"),
            "target_mode": str((teacher_train_config.get("data_view") or {}).get("target_mode") or "joint"),
            "image_prompt_style": str((teacher_train_config.get("data_view") or {}).get("image_prompt_style") or "camera_labeled"),
            "prompt_text_style": str((teacher_train_config.get("data_view") or {}).get("prompt_text_style") or "official_alpamayo"),
            "fuse_history_tokens": bool((teacher_train_config.get("data_view") or {}).get("fuse_history_tokens", True)),
            "hard_view_uses_teacher_cot": bool((teacher_train_config.get("data_view") or {}).get("hard_view_uses_teacher_cot", True)),
            "preserve_flex_positions": bool(args.preserve_flex_positions),
            "flex_selection_strategy": str(args.flex_selection_strategy),
            "flex_dummy_image_slots": bool(args.flex_dummy_image_slots),
            "flex_residual_image_slots": bool(args.flex_residual_image_slots),
            "flex_residual_scale": float(args.flex_residual_scale),
            "flex_passthrough_image_slots": bool(args.flex_passthrough_image_slots),
            "flex_scene_deepstack": bool(args.flex_scene_deepstack),
        },
        "flex": teacher_flex_stats,
        "metrics": {
            key: {
                "mean": mean(values),
                "p50": percentile(values, 50),
                "p90": percentile(values, 90),
            }
            for key, values in sorted(metrics.items())
        },
        "samples": sample_rows,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    with args.samples_jsonl.open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
