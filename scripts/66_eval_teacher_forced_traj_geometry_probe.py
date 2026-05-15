#!/usr/bin/env python3
"""Teacher-forced trajectory geometry probe for no-nav checkpoints.

This is intentionally different from free-run decode.  It feeds the gold CoT
and gold trajectory prefix through the student, takes the LM-head top-1 at the
128 trajectory-token label positions, decodes those tokens, and compares the
resulting path to the cached teacher trajectory tokens.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint  # noqa: E402
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora  # noqa: E402
from src.model.student_wrapper import StudentWrapperConfig, build_student_model  # noqa: E402
from src.model.tokenizer_ext import distill_trainable_token_ids, ensure_special_tokens  # noqa: E402
from src.training.collator import (  # noqa: E402
    DistillationCollator,
    load_ego_history_xyz,
    load_traj_future_token_ids,
)
from src.utils.runtime_paths import resolve_student_model_path  # noqa: E402

def load_decode_module():
    path = PROJECT_ROOT / "scripts" / "25_decode_checkpoint_overlays.py"
    spec = importlib.util.spec_from_file_location("decode_checkpoint_overlays_25", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import decode helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


decode_mod = load_decode_module()
TrajectoryTokenDecoder = decode_mod.TrajectoryTokenDecoder
load_ego_history_rot = decode_mod.load_ego_history_rot
resolve_traj_tokenizer_config_path = decode_mod.resolve_traj_tokenizer_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--student-model", default=resolve_student_model_path())
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
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


def mean(values: list[float | int]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def ade_fde(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    n = min(int(pred.shape[0]), int(target.shape[0]))
    if n <= 0:
        return float("nan"), float("nan")
    dist = np.linalg.norm(pred[:n, :2] - target[:n, :2], axis=-1)
    return float(dist.mean()), float(dist[-1])


def load_model(args: argparse.Namespace):
    train_config_path = args.checkpoint_dir / "train_config.json"
    train_config = json.loads(train_config_path.read_text(encoding="utf-8")) if train_config_path.exists() else {}
    checkpoint_manifest_path = args.checkpoint_dir / "checkpoint_manifest.json"
    checkpoint_manifest = (
        json.loads(checkpoint_manifest_path.read_text(encoding="utf-8")) if checkpoint_manifest_path.exists() else {}
    )
    base_model = str((train_config.get("args") or {}).get("student_model") or args.student_model)
    use_lora = not bool((train_config.get("args") or {}).get("disable_lora", False))
    data_view = train_config.get("data_view") or {}

    from transformers import AutoProcessor, AutoTokenizer

    tokenizer_dir = args.checkpoint_dir / "tokenizer"
    processor_dir = args.checkpoint_dir / "processor"
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
    # The training collator compares prompt/full prefixes before masking labels;
    # keep right padding here so both encodings share the same token origin.
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
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
    if detect_checkpoint_format(args.checkpoint_dir) == "full_state_dict" and use_lora:
        model.backbone = maybe_apply_lora(
            model.backbone,
            LoraConfigSpec(trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer))),
            enabled=True,
        )
    load_student_checkpoint(args.checkpoint_dir, model, use_lora=use_lora)
    return model.to(device).eval(), tokenizer, processor, device, base_model, train_config


def main() -> int:
    args = parse_args()
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in load_jsonl(args.corpus_jsonl) if row.get("split") == args.split]
    if args.num_samples > 0:
        rows = rows[: args.num_samples]
    if not rows:
        raise SystemExit(f"No rows selected for split={args.split!r}")

    model, tokenizer, processor, device, base_model, train_config = load_model(args)
    collator = DistillationCollator(
        tokenizer=tokenizer,
        processor=processor,
        project_root=PROJECT_ROOT,
        teacher_pair_target=False,
        enable_teacher_view=False,
        enable_action_aux=False,
        prompt_mode=str((train_config.get("data_view") or {}).get("prompt_mode") or "joint"),
        target_mode=str((train_config.get("data_view") or {}).get("target_mode") or "joint"),
        max_length=int((train_config.get("trainer_config") or {}).get("max_length", 4096)),
    )

    decoder_path = resolve_traj_tokenizer_config_path(base_model)
    if decoder_path is None:
        raise SystemExit("Could not find Alpamayo traj tokenizer config.")
    decoder = TrajectoryTokenDecoder(config_path=decoder_path)

    ade_values: list[float] = []
    fde_values: list[float] = []
    token_acc_values: list[float] = []
    cot_correct = 0
    cot_total = 0
    invalid_counts: list[int] = []
    count_malformed = 0
    top_tokens: Counter[int] = Counter()
    samples: list[dict[str, Any]] = []

    for batch_rows in batched(rows, args.batch_size):
        batch = collator(batch_rows)
        moved = {}
        model_dtype = next(model.backbone.parameters()).dtype
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if torch.is_floating_point(value):
                    moved[key] = value.to(device=device, dtype=model_dtype)
                else:
                    moved[key] = value.to(device=device)
            else:
                moved[key] = value

        forward_kwargs = {
            "input_ids": moved["input_ids"],
            "attention_mask": moved["attention_mask"],
            "return_hidden_states": False,
            "compute_meta_action": False,
            "compute_traj_aux": False,
        }
        for optional_key in ("pixel_values", "image_grid_thw"):
            if optional_key in moved:
                forward_kwargs[optional_key] = moved[optional_key]
        with torch.inference_mode():
            outputs = model(**forward_kwargs)
        logits = outputs["logits"].float()
        labels = moved["labels"]
        traj_mask = moved["traj_token_mask"].bool() & (labels != -100)
        cot_mask = moved["cot_span_mask"].bool() & (labels != -100)

        for row_index, sample in enumerate(batch_rows):
            cot_positions = torch.nonzero(cot_mask[row_index], as_tuple=False).flatten()
            cot_positions = cot_positions[cot_positions > 0]
            if cot_positions.numel() > 0:
                cot_preds = logits[row_index, cot_positions - 1, :].argmax(dim=-1)
                cot_labels = labels[row_index, cot_positions]
                cot_correct += int((cot_preds == cot_labels).sum().item())
                cot_total += int(cot_labels.numel())

            positions = torch.nonzero(traj_mask[row_index], as_tuple=False).flatten()
            positions = positions[positions > 0]
            pred_token_ids = logits[row_index, positions - 1, :].argmax(dim=-1).detach().cpu().tolist()
            target_token_ids = load_traj_future_token_ids(sample.get("hard_target") or {}, PROJECT_ROOT)
            pred_token_ids = [int(token_id) - int(getattr(tokenizer, "traj_token_start_idx", 0)) for token_id in pred_token_ids]
            invalid_count = sum(1 for token_id in pred_token_ids if token_id < 0 or token_id >= 3000)
            invalid_counts.append(invalid_count)
            top_tokens.update(pred_token_ids)
            if len(pred_token_ids) != 128 or invalid_count:
                count_malformed += 1

            n = min(len(pred_token_ids), len(target_token_ids))
            token_acc = (
                sum(1 for left, right in zip(pred_token_ids[:n], target_token_ids[:n]) if int(left) == int(right))
                / max(n, 1)
            )
            token_acc_values.append(float(token_acc))
            student_xyz = None
            teacher_xyz = None
            if len(pred_token_ids) == 128 and len(target_token_ids) == 128 and invalid_count == 0:
                history_xyz = load_ego_history_xyz(sample, PROJECT_ROOT)
                history_rot = load_ego_history_rot(sample, PROJECT_ROOT)
                student_xyz = decoder.decode(history_xyz, history_rot, pred_token_ids)
                teacher_xyz = decoder.decode(history_xyz, history_rot, target_token_ids)
            if student_xyz is not None and teacher_xyz is not None:
                ade, fde = ade_fde(student_xyz, teacher_xyz)
                ade_values.append(ade)
                fde_values.append(fde)
            else:
                ade = fde = float("nan")

            samples.append(
                {
                    "sample_id": str(sample.get("sample_id")),
                    "pred_token_count": len(pred_token_ids),
                    "target_token_count": len(target_token_ids),
                    "invalid_count": int(invalid_count),
                    "token_acc": float(token_acc),
                    "ade_m": ade if math.isfinite(ade) else None,
                    "fde_m": fde if math.isfinite(fde) else None,
                }
            )
        print(
            json.dumps(
                {
                    "event": "probe_batch_done",
                    "done": len(samples),
                    "total": len(rows),
                    "avg_ade_m": mean(ade_values),
                    "avg_fde_m": mean(fde_values),
                }
            ),
            flush=True,
        )

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "split": args.split,
        "num_samples": len(samples),
        "batch_size": int(args.batch_size),
        "probe_type": "teacher_forced_lm_head_top1_traj_geometry",
        "avg_ade_m": mean(ade_values),
        "avg_fde_m": mean(fde_values),
        "avg_token_acc": mean(token_acc_values),
        "cot_token_acc": float(cot_correct / max(cot_total, 1)),
        "cot_token_count": int(cot_total),
        "malformed_count": int(count_malformed),
        "malformed_rate": float(count_malformed / max(len(samples), 1)),
        "invalid_future_token_rate_i3000_plus": float(sum(1 for count in invalid_counts if count > 0) / max(len(invalid_counts), 1)),
        "avg_invalid_future_tokens_i3000_plus": mean(invalid_counts),
        "top_token_histogram": [
            {"token": int(token), "count": int(count), "mass": float(count / max(sum(top_tokens.values()), 1))}
            for token, count in top_tokens.most_common(30)
        ],
        "samples": samples,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
