#!/usr/bin/env python3
"""Train FLEX-only compression parity against a frozen no-FLEX checkpoint.

Teacher: frozen B0 no-FLEX checkpoint.
Student: same checkpoint with FLEX attached, usually F0.  Only
``flex_scene_encoder`` is trainable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import shutil
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

from src.model.checkpoint_io import save_student_checkpoint  # noqa: E402
from src.training.collator import DistillationCollator  # noqa: E402
from src.training.flex_batch import attach_qwen_mrope_position_ids, compress_batch_for_flex  # noqa: E402
from src.utils.runtime_paths import resolve_student_model_path  # noqa: E402

IGNORE_INDEX = -100
BOUNDARY_NAMES = ("cot_end", "traj_start", "action_pre")


def _load_eval104():
    path = PROJECT_ROOT / "scripts" / "104_eval_flex_teacher_parity.py"
    spec = importlib.util.spec_from_file_location("flex_teacher_parity_eval104", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval104 = _load_eval104()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--student-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-model", default=resolve_student_model_path())
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-train-samples", type=int, default=16)
    parser.add_argument(
        "--shuffle-train-samples",
        action="store_true",
        help="Shuffle selected split rows with --seed before applying --max-train-samples.",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--flex-lr",
        type=float,
        default=None,
        help="Optional LR override for flex_scene_encoder parameters.",
    )
    parser.add_argument(
        "--lora-lr",
        type=float,
        default=None,
        help="Optional LR override for LoRA parameters.",
    )
    parser.add_argument(
        "--multimodal-projector-lr",
        type=float,
        default=None,
        help="Optional LR override for multimodal projector / visual merger parameters.",
    )
    parser.add_argument(
        "--deepstack-projector-lr",
        type=float,
        default=None,
        help="Optional LR override for flex_deepstack_projector parameters.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--traj-kl-weight", type=float, default=1.0)
    parser.add_argument(
        "--traj-token-ce-weight",
        type=float,
        default=0.0,
        help="Cross-entropy weight for supervised trajectory labels at traj_token_mask positions.",
    )
    parser.add_argument("--text-kl-weight", type=float, default=0.2)
    parser.add_argument("--format-kl-weight", type=float, default=0.05)
    parser.add_argument("--boundary-cos-weight", type=float, default=0.05)
    parser.add_argument("--boundary-norm-weight", type=float, default=0.10)
    parser.add_argument("--boundary-mse-weight", type=float, default=0.0)
    parser.add_argument(
        "--traj-state-cos-weight",
        type=float,
        default=0.0,
        help="Cosine loss on hidden states used to predict the 128 trajectory tokens.",
    )
    parser.add_argument(
        "--traj-state-norm-weight",
        type=float,
        default=0.0,
        help="Log-norm matching loss on hidden states used to predict the 128 trajectory tokens.",
    )
    parser.add_argument(
        "--traj-state-mse-weight",
        type=float,
        default=0.0,
        help="Smooth-L1 loss on hidden states used to predict the 128 trajectory tokens.",
    )
    parser.add_argument("--cache-teacher-targets", action="store_true")
    parser.add_argument("--cache-collated-batches", action="store_true")
    parser.add_argument("--preserve-flex-positions", action="store_true")
    parser.add_argument("--flex-selection-strategy", choices=("config", "first", "uniform"), default="config")
    parser.add_argument(
        "--flex-dummy-image-slots",
        action="store_true",
        help=(
            "Diagnostic mode: keep original image-token length and insert FLEX scene tokens "
            "into the first K slots of each image block."
        ),
    )
    parser.add_argument(
        "--flex-residual-image-slots",
        action="store_true",
        help=(
            "Diagnostic mode: keep original visual embeddings and add FLEX scene tokens as "
            "a residual to the first K slots of each image block."
        ),
    )
    parser.add_argument(
        "--flex-residual-scale",
        type=float,
        default=1.0,
        help="Residual multiplier used with --flex-residual-image-slots.",
    )
    parser.add_argument(
        "--flex-passthrough-image-slots",
        action="store_true",
        help=(
            "For compressed FLEX batches, bypass the FLEX scene encoder and fill retained "
            "image slots with original Qwen visual features. Diagnostic only."
        ),
    )
    parser.add_argument(
        "--flex-scene-deepstack",
        action="store_true",
        help="For compressed FLEX batches, inject scene tokens through Qwen3-VL DeepStack visual hooks.",
    )
    parser.add_argument(
        "--flex-deepstack-projector-rank",
        type=int,
        default=0,
        help=(
            "If >0, attach a layer-specific low-rank residual projector for compressed "
            "FLEX DeepStack scene tokens."
        ),
    )
    parser.add_argument(
        "--flex-deepstack-projector-dropout",
        type=float,
        default=0.0,
        help="Dropout used inside --flex-deepstack-projector-rank adapters.",
    )
    parser.add_argument(
        "--train-flex-deepstack-projector",
        action="store_true",
        help="Unfreeze the FLEX DeepStack projector parameters.",
    )
    parser.add_argument(
        "--deepstack-feature-tokens-per-image",
        type=int,
        default=0,
        help=(
            "When DeepStack feature parity is enabled, select the first K teacher "
            "DeepStack features from each original image block. Use the FLEX "
            "tokens_per_image value."
        ),
    )
    parser.add_argument(
        "--image-feature-tokens-per-image",
        type=int,
        default=0,
        help=(
            "When image feature parity is enabled, select K no-FLEX Qwen image "
            "features from each original image block. Use the FLEX tokens_per_image value."
        ),
    )
    parser.add_argument(
        "--image-feature-cos-weight",
        type=float,
        default=0.0,
        help="Cosine loss weight for compressed FLEX scene embeddings vs selected no-FLEX image features.",
    )
    parser.add_argument(
        "--image-feature-norm-weight",
        type=float,
        default=0.0,
        help="Log-norm matching loss weight for compressed FLEX scene embeddings.",
    )
    parser.add_argument(
        "--image-feature-mse-weight",
        type=float,
        default=0.0,
        help="Smooth-L1 loss weight for compressed FLEX scene embeddings.",
    )
    parser.add_argument(
        "--deepstack-feature-cos-weight",
        type=float,
        default=0.0,
        help="Cosine loss weight for compressed DeepStack projector outputs vs teacher DeepStack features.",
    )
    parser.add_argument(
        "--deepstack-feature-norm-weight",
        type=float,
        default=0.0,
        help="Log-norm matching loss weight for compressed DeepStack projector outputs.",
    )
    parser.add_argument(
        "--deepstack-feature-mse-weight",
        type=float,
        default=0.0,
        help="Smooth-L1 loss weight for compressed DeepStack projector outputs.",
    )
    parser.add_argument(
        "--prompt-mode-override",
        choices=("", "joint", "traj_only"),
        default="",
        help="Override teacher checkpoint data_view.prompt_mode for FLEX diagnostics.",
    )
    parser.add_argument(
        "--target-mode-override",
        choices=("", "joint", "traj_only"),
        default="",
        help="Override teacher checkpoint data_view.target_mode for FLEX diagnostics.",
    )
    parser.add_argument(
        "--image-ablations",
        default="normal",
        help="Comma-separated image ablations to train on, e.g. normal,camera_shuffle,black.",
    )
    parser.add_argument(
        "--paired-ablation",
        choices=("none", "camera_shuffle", "black"),
        default="none",
        help="Train each step on [normal, paired_ablation] for the same sample.",
    )
    parser.add_argument(
        "--pairwise-boundary-delta-cos-weight",
        type=float,
        default=0.0,
        help="Match teacher/student normal-vs-ablation action_pre delta direction.",
    )
    parser.add_argument(
        "--pairwise-boundary-delta-norm-weight",
        type=float,
        default=0.0,
        help="Match teacher/student normal-vs-ablation action_pre delta norm.",
    )
    parser.add_argument(
        "--pairwise-traj-logprob-delta-weight",
        type=float,
        default=0.0,
        help="Match teacher/student normal-vs-ablation trajectory log-prob deltas.",
    )
    parser.add_argument(
        "--pairwise-free-run-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Rank B0 free-run token targets for paired normal-vs-ablation rows: "
            "normal input should prefer normal targets and ablated input should prefer ablated targets."
        ),
    )
    parser.add_argument(
        "--pairwise-free-run-margin",
        type=float,
        default=0.10,
        help="Per-token average log-prob margin used by --pairwise-free-run-margin-weight.",
    )
    parser.add_argument(
        "--free-run-token-targets",
        default="",
        help=(
            "Comma-separated mode=decode_summary.json mappings. "
            "Uses summary samples[*].generated_traj_tokens as trajectory CE targets."
        ),
    )
    parser.add_argument(
        "--free-run-token-ce-weight",
        type=float,
        default=0.0,
        help="Cross-entropy weight for external B0 free-run trajectory token targets.",
    )
    parser.add_argument(
        "--free-run-token-ce-modes",
        default="*",
        help=(
            "Comma-separated image modes to apply --free-run-token-ce-weight to. "
            "Use '*' for every loaded mode, or e.g. 'normal' to anchor only normal free-run output."
        ),
    )
    parser.add_argument(
        "--free-run-end-token-ce-weight",
        type=float,
        default=0.0,
        help=(
            "Cross-entropy weight for the first supervised format/end token after the 128 trajectory tokens, "
            "under the selected free-run token context."
        ),
    )
    parser.add_argument(
        "--prefix-token-ce-weight",
        type=float,
        default=0.0,
        help="Cross-entropy weight for supervised CoT/format prefix tokens before the trajectory body.",
    )
    parser.add_argument(
        "--free-run-token-force-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace trajectory input/label tokens with external free-run targets before forward.",
    )
    parser.add_argument(
        "--free-run-token-context-source",
        choices=("target", "student_greedy"),
        default="target",
        help=(
            "Trajectory context used for external free-run token CE. "
            "'target' is teacher forcing; 'student_greedy' trains on the current student's rollout prefix."
        ),
    )
    parser.add_argument(
        "--student-greedy-context-refresh-steps",
        type=int,
        default=250,
        help="Refresh interval for student_greedy context cache. Use 1 for strict per-step rollout.",
    )
    parser.add_argument(
        "--student-greedy-invalid-context",
        choices=("raw", "target", "skip"),
        default="raw",
        help=(
            "Fallback for malformed student_greedy rollouts. "
            "'raw' keeps previous behavior, 'target' uses the external/teacher target tokens, "
            "and 'skip' leaves the original batch context unchanged."
        ),
    )
    parser.add_argument("--train-flex", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unfreeze-lora-last-n-layers", type=int, default=0)
    parser.add_argument("--unfreeze-all-lora", action="store_true")
    parser.add_argument("--unfreeze-multimodal-projector", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--no-save-final", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cycle_batches(rows: list[dict[str, Any]], batch_size: int):
    if not rows:
        raise ValueError("Cannot train on an empty row list.")
    width = max(int(batch_size), 1)
    index = 0
    while True:
        batch_rows = []
        for _ in range(width):
            batch_rows.append(rows[index % len(rows)])
            index += 1
        yield batch_rows


def cycle_pair_batches(base_rows: list[dict[str, Any]], paired_ablation: str):
    if not base_rows:
        raise ValueError("Cannot train on an empty base row list.")
    index = 0
    while True:
        row = base_rows[index % len(base_rows)]
        index += 1
        yield [
            dict(row, _image_ablation="normal"),
            dict(row, _image_ablation=paired_ablation),
        ]


def parse_image_ablations(raw: str) -> list[str]:
    allowed = {"normal", "black", "gray", "noise", "camera_shuffle"}
    modes = [item.strip().lower() for item in str(raw or "normal").split(",") if item.strip()]
    if not modes:
        modes = ["normal"]
    deduped: list[str] = []
    for mode in modes:
        if mode not in allowed:
            raise ValueError(f"Unsupported image ablation {mode!r}; allowed={sorted(allowed)}")
        if mode not in deduped:
            deduped.append(mode)
    return deduped


def expand_rows_for_image_ablations(rows: list[dict[str, Any]], modes: list[str]) -> list[dict[str, Any]]:
    if modes == ["normal"]:
        return [dict(row, _image_ablation="normal") for row in rows]
    expanded: list[dict[str, Any]] = []
    for row in rows:
        for mode in modes:
            expanded.append(dict(row, _image_ablation=mode))
    return expanded


def expand_rows_for_pairwise_cache(rows: list[dict[str, Any]], paired_ablation: str) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        expanded.append(dict(row, _image_ablation="normal"))
        expanded.append(dict(row, _image_ablation=paired_ablation))
    return expanded


def free_run_ce_enabled_for_mode(raw_modes: str, mode: str) -> bool:
    raw = str(raw_modes or "*").strip()
    if raw in {"*", "all"}:
        return True
    allowed = {part.strip() for part in raw.split(",") if part.strip()}
    return str(mode or "normal") in allowed


def resolve_flex_selection_strategy(flex_cfg, args: argparse.Namespace) -> str:
    raw = str(getattr(args, "flex_selection_strategy", "config") or "config")
    if raw == "config":
        return str(getattr(flex_cfg, "selection_strategy", "first") or "first")
    return raw


def cache_key(row: dict[str, Any]) -> str:
    return f"{row.get('sample_id')}::{row.get('_image_ablation') or 'normal'}"


def load_free_run_token_targets(raw: str) -> dict[str, list[int]]:
    targets: dict[str, list[int]] = {}
    raw = str(raw or "").strip()
    if not raw:
        return targets
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "--free-run-token-targets entries must look like mode=/path/summary.json; "
                f"got {item!r}"
            )
        mode, path_raw = item.split("=", 1)
        mode = mode.strip().lower()
        path = Path(path_raw.strip())
        if not path.exists():
            raise FileNotFoundError(f"free-run token target summary not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = payload.get("samples") or []
        loaded = 0
        for sample in samples:
            sample_id = sample.get("sample_id")
            tokens = sample.get("generated_traj_tokens")
            if not sample_id or not isinstance(tokens, list):
                continue
            targets[f"{sample_id}::{mode}"] = [int(token) for token in tokens]
            loaded += 1
        print(
            json.dumps(
                {
                    "event": "free_run_token_targets_loaded",
                    "mode": mode,
                    "path": str(path),
                    "samples": loaded,
                }
            ),
            flush=True,
        )
    return targets


def apply_free_run_token_targets_to_batch(
    batch: dict[str, Any],
    batch_rows: list[dict[str, Any]],
    targets: dict[str, list[int]],
    *,
    traj_start: int,
    num_bins: int,
) -> None:
    if not targets:
        return
    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
        return
    for row_index, row in enumerate(batch_rows):
        tokens = targets.get(cache_key(row))
        if not tokens:
            continue
        positions = _label_positions(batch, row_index, "traj_token_mask")[:128]
        usable = min(int(positions.numel()), len(tokens), 128)
        if usable <= 0:
            continue
        token_tensor = torch.as_tensor(tokens[:usable], device=input_ids.device, dtype=torch.long)
        valid = (token_tensor >= 0) & (token_tensor < int(num_bins))
        if not bool(valid.any().item()):
            continue
        pos = positions[:usable].to(device=input_ids.device)[valid]
        token_ids = token_tensor[valid] + int(traj_start)
        input_ids[row_index, pos] = token_ids
        labels[row_index, pos] = token_ids


def clone_tensor_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        cloned[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return cloned


def flex_student_batch(
    batch: dict[str, Any],
    *,
    student,
    tokenizer,
    flex_cfg,
    args: argparse.Namespace,
) -> dict[str, Any]:
    selection_strategy = resolve_flex_selection_strategy(flex_cfg, args)
    if bool(args.flex_dummy_image_slots) or bool(args.flex_residual_image_slots):
        student_batch = clone_tensor_batch(batch)
        if bool(args.flex_dummy_image_slots):
            student_batch["flex_allow_dummy_image_slots"] = True
        if bool(args.flex_residual_image_slots):
            student_batch["flex_residual_image_slots"] = True
            student_batch["flex_residual_scale"] = float(args.flex_residual_scale)
        input_ids = student_batch["input_ids"]
        original_image_tokens = int((input_ids == int(getattr(student, "image_token_id"))).sum().detach().cpu())
        expected_scene_tokens = int(getattr(flex_cfg, "tokens_per_image")) * int(
            getattr(flex_cfg, "expected_images_per_sample")
        )
        student_batch["flex_stats"] = {
            "flex_original_seq_len": float(input_ids.shape[1]),
            "flex_compressed_seq_len": float(input_ids.shape[1]),
            "flex_original_image_tokens": float(original_image_tokens),
            "flex_compressed_image_tokens": float(expected_scene_tokens * int(input_ids.shape[0])),
            "flex_image_token_compression": 1.0,
            "flex_dummy_image_slots": float(bool(args.flex_dummy_image_slots)),
            "flex_residual_image_slots": float(bool(args.flex_residual_image_slots)),
            "flex_residual_scale": float(args.flex_residual_scale) if bool(args.flex_residual_image_slots) else 0.0,
            "flex_selection_uniform": float(selection_strategy == "uniform"),
        }
        if selection_strategy != "first":
            student_batch["flex_selection_strategy"] = selection_strategy
        return student_batch
    source_batch = attach_qwen_mrope_position_ids(batch, student) if bool(args.preserve_flex_positions) else batch
    student_batch = compress_batch_for_flex(
        source_batch,
        image_token_id=int(getattr(student, "image_token_id")),
        tokens_per_image=int(getattr(flex_cfg, "tokens_per_image")),
        pad_token_id=int(getattr(tokenizer, "pad_token_id", 0) or 0),
        preserve_original_position_ids=bool(args.preserve_flex_positions),
        selection_strategy=selection_strategy,
    )
    if selection_strategy != "first":
        student_batch["flex_selection_strategy"] = selection_strategy
        stats = dict(student_batch.get("flex_stats") or {})
        stats["flex_selection_uniform"] = float(selection_strategy == "uniform")
        student_batch["flex_stats"] = stats
    if bool(args.flex_scene_deepstack):
        student_batch["flex_scene_deepstack"] = True
        stats = dict(student_batch.get("flex_stats") or {})
        stats["flex_scene_deepstack"] = 1.0
        student_batch["flex_stats"] = stats
    if bool(args.flex_passthrough_image_slots):
        student_batch["flex_passthrough_image_slots"] = True
        stats = dict(student_batch.get("flex_stats") or {})
        stats["flex_passthrough_image_slots"] = 1.0
        student_batch["flex_stats"] = stats
    return student_batch


def apply_free_run_token_context_to_batch(
    batch: dict[str, Any],
    batch_rows: list[dict[str, Any]],
    *,
    context_tokens: dict[str, list[int]],
    label_targets: dict[str, list[int]],
    traj_start: int,
    num_bins: int,
) -> None:
    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
        return
    for row_index, row in enumerate(batch_rows):
        key = cache_key(row)
        context = context_tokens.get(key)
        target = label_targets.get(key)
        if not context:
            continue
        positions = _label_positions(batch, row_index, "traj_token_mask")[:128]
        usable = min(int(positions.numel()), len(context), 128)
        if usable <= 0:
            continue
        context_tensor = torch.as_tensor(context[:usable], device=input_ids.device, dtype=torch.long)
        valid_context = (context_tensor >= 0) & (context_tensor < int(num_bins))
        if bool(valid_context.any().item()):
            pos = positions[:usable].to(device=input_ids.device)[valid_context]
            input_ids[row_index, pos] = context_tensor[valid_context] + int(traj_start)
        if target:
            target_usable = min(int(positions.numel()), len(target), 128)
            if target_usable > 0:
                target_tensor = torch.as_tensor(target[:target_usable], device=labels.device, dtype=torch.long)
                valid_target = (target_tensor >= 0) & (target_tensor < int(num_bins))
                if bool(valid_target.any().item()):
                    pos = positions[:target_usable].to(device=labels.device)[valid_target]
                    labels[row_index, pos] = target_tensor[valid_target] + int(traj_start)


def generate_student_greedy_contexts(
    *,
    student,
    rows: list[dict[str, Any]],
    tokenizer,
    processor,
    data_view: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    traj_token_targets: dict[str, list[int]],
    flex_cfg,
    args: argparse.Namespace,
) -> dict[str, list[int]]:
    decode_mod = eval104.decode_mod
    contexts: dict[str, list[int]] = {}
    match_values: list[float] = []
    token_count_values: list[int] = []
    invalid_values: list[int] = []
    exact_128_values: list[float] = []
    over_128_values: list[float] = []
    fallback_target_values: list[float] = []
    skipped_values: list[float] = []
    debug_overlong_examples = 0

    student.eval()
    for index, row in enumerate(rows, start=1):
        sample = row
        sample_id = str(sample.get("sample_id") or "")
        mode = str(row.get("_image_ablation") or "normal")
        history_xyz = decode_mod.load_ego_history_xyz(sample, PROJECT_ROOT)
        prompt_mode = str(data_view.get("prompt_mode") or "joint")
        target_mode = str(data_view.get("target_mode") or "joint")
        prompt_text_style = str(data_view.get("prompt_text_style") or "official_alpamayo")
        image_prompt_style = str(data_view.get("image_prompt_style") or "camera_labeled")
        fuse_history_tokens = bool(data_view.get("fuse_history_tokens", True))
        prompt_text = (
            decode_mod.build_traj_only_prompt(sample, PROJECT_ROOT, ego_history_xyz=history_xyz)
            if prompt_mode == "traj_only"
            else decode_mod.build_user_prompt(
                sample,
                PROJECT_ROOT,
                ego_history_xyz=history_xyz,
                prompt_text_style=prompt_text_style,
            )
        )
        assistant_prefix = "<|traj_future_start|>" if target_mode == "traj_only" else "<|cot_start|>"
        images = decode_mod._apply_image_ablation(
            decode_mod.load_sample_images(sample, PROJECT_ROOT),
            mode,
            sample_id=sample_id,
        )
        camera_indices = decode_mod.resolve_camera_indices(sample, PROJECT_ROOT, image_count=len(images))
        frames_per_camera = max(len(images) // max(len(camera_indices), 1), 1)
        messages = decode_mod.build_messages(
            prompt_text,
            len(images),
            assistant_prefix=assistant_prefix,
            image_prompt_style=image_prompt_style,
            camera_indices=camera_indices,
            num_frames_per_camera=frames_per_camera,
        )
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        batch = processor(text=[text], images=[images], return_tensors="pt", padding=True, truncation=True)
        if fuse_history_tokens:
            batch["input_ids"] = decode_mod.fuse_history_tokens_in_input_ids(
                batch["input_ids"],
                tokenizer,
                [history_xyz],
            )
        relative_timestamps = decode_mod.resolve_image_relative_timestamps(
            sample,
            PROJECT_ROOT,
            camera_count=len(camera_indices),
            frames_per_camera=frames_per_camera,
        )
        batch["camera_indices"] = torch.tensor([camera_indices], dtype=torch.long)
        batch["relative_timestamps"] = torch.tensor([relative_timestamps], dtype=torch.float32)
        batch["camera_counts"] = torch.tensor([len(camera_indices)], dtype=torch.long)
        batch["frames_per_camera"] = torch.tensor([frames_per_camera], dtype=torch.long)
        batch = flex_student_batch(
            batch,
            student=student,
            tokenizer=tokenizer,
            flex_cfg=flex_cfg,
            args=args,
        )
        batch = {
            key: (
                value.to(device=device, dtype=dtype)
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
                else value.to(device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in batch.items()
        }
        target = traj_token_targets.get(cache_key(row))
        if not target:
            target = decode_mod.load_traj_future_token_ids(sample.get("hard_target") or {}, PROJECT_ROOT)
        prompt_lengths = [int(batch["input_ids"].shape[1])]
        if target_mode == "traj_only":
            contract = decode_mod.TrajOnlyDecodingContract.from_tokenizer(
                tokenizer,
                prompt_lengths=prompt_lengths,
                traj_token_count=len(target),
            )
            logits_processor = decode_mod.LogitsProcessorList([decode_mod.TrajOnlyLogitsProcessor(contract)])
            stopping_criteria = decode_mod.StoppingCriteriaList([decode_mod.StopOnTrajOnlyEndCriteria(contract)])
        else:
            contract = decode_mod.TrajDecodingContract.from_tokenizer(
                tokenizer,
                prompt_lengths=prompt_lengths,
                traj_token_count=len(target),
            )
            logits_processor = decode_mod.LogitsProcessorList([decode_mod.TrajSpanLogitsProcessor(contract)])
            stopping_criteria = decode_mod.StoppingCriteriaList([decode_mod.StopOnTrajEndCriteria(contract)])
        with torch.inference_mode():
            generated = decode_mod._manual_flex_generate(
                student,
                batch,
                max_new_tokens=192,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
            )
        generated_text = decode_mod._extract_generated_text(tokenizer, batch["input_ids"], generated, row_index=0)
        raw_tokens = [int(token) for token in decode_mod._extract_generated_traj_tokens(generated_text)]
        traj_start_marker = "<|traj_future_start|>"
        traj_end_marker = "<|traj_future_end|>"
        span_text = ""
        traj_start_pos = generated_text.find(traj_start_marker)
        traj_end_pos = -1
        if traj_start_pos >= 0:
            span_start = traj_start_pos + len(traj_start_marker)
            traj_end_pos = generated_text.find(traj_end_marker, span_start)
            span_text = generated_text[span_start : traj_end_pos if traj_end_pos >= 0 else len(generated_text)]
        span_tokens = [int(match.group(1)) for match in re.finditer(r"<i(\d+)>", span_text)]
        tokens_source = "raw"
        tokens = raw_tokens
        if len(span_tokens) == 128:
            tokens = span_tokens
            tokens_source = "span"
        elif len(raw_tokens) == 128:
            tokens = raw_tokens
            tokens_source = "raw_exact128"
        elif str(args.student_greedy_invalid_context) == "target" and target:
            tokens = [int(token) for token in target[:128]]
            tokens_source = "target_fallback"
        elif str(args.student_greedy_invalid_context) == "skip":
            tokens = []
            tokens_source = "skip"
        contexts[cache_key(row)] = tokens
        fallback_target_values.append(float(tokens_source == "target_fallback"))
        skipped_values.append(float(tokens_source == "skip"))
        token_count_values.append(len(tokens))
        exact_128_values.append(float(len(tokens) == 128))
        over_128_values.append(float(len(tokens) > 128))
        invalid_values.append(sum(1 for token in tokens if token < 0 or token >= 3000))
        if target:
            usable = min(len(tokens), len(target), 128)
            if usable > 0:
                match_values.append(
                    float(sum(1 for left, right in zip(tokens[:usable], target[:usable]) if int(left) == int(right)))
                    / float(usable)
                )
        print(
            json.dumps(
                {
                    "event": "student_greedy_context_done",
                    "done": index,
                    "total": len(rows),
                    "sample_id": sample_id,
                    "mode": mode,
                    "tokens": len(tokens),
                    "raw_tokens": len(raw_tokens),
                    "span_tokens": len(span_tokens),
                    "tokens_source": tokens_source,
                }
            ),
            flush=True,
        )
        if len(tokens) != 128 and debug_overlong_examples < 8:
            debug_overlong_examples += 1
            pre_span_text = generated_text[:traj_start_pos] if traj_start_pos >= 0 else generated_text
            post_end_text = generated_text[traj_end_pos + len(traj_end_marker) :] if traj_end_pos >= 0 else ""
            print(
                json.dumps(
                    {
                        "event": "student_greedy_context_debug",
                        "sample_id": sample_id,
                        "mode": mode,
                        "all_i_count": len(tokens),
                        "span_i_count": len(span_tokens),
                        "pre_span_i_count": len(re.findall(r"<i\\d+>", pre_span_text)),
                        "post_end_i_count": len(re.findall(r"<i\\d+>", post_end_text)),
                        "cot_end_count": generated_text.count("<|cot_end|>"),
                        "traj_start_count": generated_text.count(traj_start_marker),
                        "traj_end_count": generated_text.count(traj_end_marker),
                        "text_prefix": generated_text[:900],
                        "text_suffix": generated_text[-900:],
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "event": "student_greedy_context_refresh_done",
                "rows": len(rows),
                "mean_token_count": mean(token_count_values),
                "exact_128_rate": mean(exact_128_values),
                "over_128_rate": mean(over_128_values),
                "mean_invalid_count": mean(invalid_values),
                "mean_target_match": mean(match_values),
                "fallback_target_rate": mean(fallback_target_values),
                "skip_rate": mean(skipped_values),
            }
        ),
        flush=True,
    )
    return contexts


def mean(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


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


def _boundary_vectors(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    row_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_positions = positions[row_index].long()
    valid = (row_positions >= 0) & (row_positions < hidden.shape[1])
    safe = row_positions.clamp(min=0, max=max(int(hidden.shape[1]) - 1, 0))
    vectors = hidden[row_index].index_select(0, safe)
    vectors = vectors * valid.to(dtype=vectors.dtype, device=vectors.device).unsqueeze(-1)
    return vectors.float(), valid


def _boundary_loss(
    teacher_hidden: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_batch: dict[str, Any],
    student_batch: dict[str, Any],
    *,
    cos_weight: float,
    norm_weight: float,
    mse_weight: float,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    device = student_hidden.device
    loss = torch.zeros((), device=device, dtype=torch.float32)
    metrics: dict[str, list[float]] = defaultdict(list)
    teacher_positions = teacher_batch.get("teacher_text_boundary_hidden_positions")
    student_positions = student_batch.get("teacher_text_boundary_hidden_positions")
    if not isinstance(teacher_positions, torch.Tensor) or not isinstance(student_positions, torch.Tensor):
        return loss, {f"{name}_cos": None for name in BOUNDARY_NAMES}

    for row_index in range(int(student_hidden.shape[0])):
        t_vecs, t_valid = _boundary_vectors(teacher_hidden, teacher_positions, row_index)
        s_vecs, s_valid = _boundary_vectors(student_hidden, student_positions, row_index)
        valid = t_valid & s_valid
        if not bool(valid.any().item()):
            continue
        t_vecs = t_vecs[valid].detach()
        s_vecs = s_vecs[valid]
        cos = F.cosine_similarity(s_vecs, t_vecs, dim=-1)
        t_norm = t_vecs.norm(dim=-1).clamp(min=1e-6)
        s_norm = s_vecs.norm(dim=-1).clamp(min=1e-6)
        if cos_weight:
            loss = loss + float(cos_weight) * (1.0 - cos).mean()
        if norm_weight:
            loss = loss + float(norm_weight) * F.smooth_l1_loss(torch.log(s_norm), torch.log(t_norm))
        if mse_weight:
            loss = loss + float(mse_weight) * F.smooth_l1_loss(s_vecs, t_vecs)
        for local_index, boundary_index in enumerate(torch.nonzero(valid, as_tuple=False).flatten().tolist()):
            name = BOUNDARY_NAMES[int(boundary_index)]
            metrics[f"{name}_cos"].append(float(cos[local_index].detach().cpu()))
            metrics[f"{name}_norm_ratio"].append(float((s_norm[local_index] / t_norm[local_index]).detach().cpu()))
    return loss, {key: mean(values) for key, values in metrics.items()}


def _cached_boundary_loss(
    student_hidden: torch.Tensor,
    student_batch: dict[str, Any],
    teacher_entries: list[dict[str, torch.Tensor]],
    *,
    cos_weight: float,
    norm_weight: float,
    mse_weight: float,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    device = student_hidden.device
    loss = torch.zeros((), device=device, dtype=torch.float32)
    metrics: dict[str, list[float]] = defaultdict(list)
    student_positions = student_batch.get("teacher_text_boundary_hidden_positions")
    if not isinstance(student_positions, torch.Tensor):
        return loss, {f"{name}_cos": None for name in BOUNDARY_NAMES}

    for row_index, entry in enumerate(teacher_entries):
        t_vecs = entry["boundary_vecs"].to(device=device).float().detach()
        t_valid = entry["boundary_valid"].to(device=device).bool()
        s_vecs, s_valid = _boundary_vectors(student_hidden, student_positions, row_index)
        valid = t_valid & s_valid
        if not bool(valid.any().item()):
            continue
        t_vecs = t_vecs[valid]
        s_vecs = s_vecs[valid]
        cos = F.cosine_similarity(s_vecs, t_vecs, dim=-1)
        t_norm = t_vecs.norm(dim=-1).clamp(min=1e-6)
        s_norm = s_vecs.norm(dim=-1).clamp(min=1e-6)
        if cos_weight:
            loss = loss + float(cos_weight) * (1.0 - cos).mean()
        if norm_weight:
            loss = loss + float(norm_weight) * F.smooth_l1_loss(torch.log(s_norm), torch.log(t_norm))
        if mse_weight:
            loss = loss + float(mse_weight) * F.smooth_l1_loss(s_vecs, t_vecs)
        for local_index, boundary_index in enumerate(torch.nonzero(valid, as_tuple=False).flatten().tolist()):
            name = BOUNDARY_NAMES[int(boundary_index)]
            metrics[f"{name}_cos"].append(float(cos[local_index].detach().cpu()))
            metrics[f"{name}_norm_ratio"].append(float((s_norm[local_index] / t_norm[local_index]).detach().cpu()))
    return loss, {key: mean(values) for key, values in metrics.items()}


def _trajectory_state_vectors(
    hidden: torch.Tensor,
    batch: dict[str, Any],
    row_index: int,
) -> torch.Tensor:
    """Hidden states that produce the next-token logits for trajectory labels."""
    traj_positions = _label_positions(batch, row_index, "traj_token_mask")[:128]
    if int(traj_positions.numel()) <= 0:
        return hidden.new_zeros((0, int(hidden.shape[-1])), dtype=hidden.dtype)
    state_positions = (traj_positions - 1).clamp(min=0, max=max(int(hidden.shape[1]) - 1, 0)).long()
    return hidden[row_index].index_select(0, state_positions).float()


def _trajectory_state_loss_from_vectors(
    *,
    teacher_vecs: torch.Tensor,
    student_vecs: torch.Tensor,
    cos_weight: float,
    norm_weight: float,
    mse_weight: float,
) -> tuple[torch.Tensor, dict[str, list[float]]]:
    device = student_vecs.device
    loss = torch.zeros((), device=device, dtype=torch.float32)
    metrics: dict[str, list[float]] = defaultdict(list)
    usable = min(int(teacher_vecs.shape[0]), int(student_vecs.shape[0]), 128)
    if usable <= 0:
        return loss, metrics
    t_vecs = teacher_vecs[:usable].to(device=device).float().detach()
    s_vecs = student_vecs[:usable].float()
    cos = F.cosine_similarity(s_vecs, t_vecs, dim=-1)
    t_norm = t_vecs.norm(dim=-1).clamp(min=1e-6)
    s_norm = s_vecs.norm(dim=-1).clamp(min=1e-6)
    if cos_weight:
        loss = loss + float(cos_weight) * (1.0 - cos).mean()
    if norm_weight:
        loss = loss + float(norm_weight) * F.smooth_l1_loss(torch.log(s_norm), torch.log(t_norm))
    if mse_weight:
        loss = loss + float(mse_weight) * F.smooth_l1_loss(s_vecs, t_vecs)
    metrics["traj_state_cos"].append(float(cos.mean().detach().cpu()))
    metrics["traj_state_norm_ratio"].append(float((s_norm / t_norm).mean().detach().cpu()))
    metrics["traj_state_mse"].append(float(F.smooth_l1_loss(s_vecs.detach(), t_vecs).detach().cpu()))
    metrics["traj_state_usable"].append(float(usable))
    return loss, metrics


def _trajectory_state_loss(
    teacher_hidden: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_batch: dict[str, Any],
    student_batch: dict[str, Any],
    *,
    cos_weight: float,
    norm_weight: float,
    mse_weight: float,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    device = student_hidden.device
    loss = torch.zeros((), device=device, dtype=torch.float32)
    metrics: dict[str, list[float]] = defaultdict(list)
    for row_index in range(int(student_hidden.shape[0])):
        t_vecs = _trajectory_state_vectors(teacher_hidden, teacher_batch, row_index)
        s_vecs = _trajectory_state_vectors(student_hidden, student_batch, row_index)
        row_loss, row_metrics = _trajectory_state_loss_from_vectors(
            teacher_vecs=t_vecs,
            student_vecs=s_vecs,
            cos_weight=cos_weight,
            norm_weight=norm_weight,
            mse_weight=mse_weight,
        )
        loss = loss + row_loss
        for key, values in row_metrics.items():
            metrics[key].extend(values)
    return loss, {key: mean(values) for key, values in metrics.items()}


def _cached_trajectory_state_loss(
    student_hidden: torch.Tensor,
    student_batch: dict[str, Any],
    teacher_entries: list[dict[str, torch.Tensor]],
    *,
    cos_weight: float,
    norm_weight: float,
    mse_weight: float,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    device = student_hidden.device
    loss = torch.zeros((), device=device, dtype=torch.float32)
    metrics: dict[str, list[float]] = defaultdict(list)
    for row_index, entry in enumerate(teacher_entries):
        t_vecs = entry.get("traj_state_vecs")
        if not isinstance(t_vecs, torch.Tensor):
            continue
        s_vecs = _trajectory_state_vectors(student_hidden, student_batch, row_index)
        row_loss, row_metrics = _trajectory_state_loss_from_vectors(
            teacher_vecs=t_vecs,
            student_vecs=s_vecs,
            cos_weight=cos_weight,
            norm_weight=norm_weight,
            mse_weight=mse_weight,
        )
        loss = loss + row_loss
        for key, values in row_metrics.items():
            metrics[key].extend(values)
    return loss, {key: mean(values) for key, values in metrics.items()}


def _student_compressed_deepstack_outputs(
    student,
    student_moved: dict[str, Any],
) -> list[torch.Tensor] | None:
    if not bool(student_moved.get("flex_scene_deepstack", False)):
        return None
    if not hasattr(student, "_flex_inputs_embeds"):
        return None
    _, _, deepstack = student._flex_inputs_embeds(
        student_moved["input_ids"],
        student_moved["pixel_values"],
        student_moved["image_grid_thw"],
        camera_indices=student_moved.get("camera_indices"),
        relative_timestamps=student_moved.get("relative_timestamps"),
        camera_counts=student_moved.get("camera_counts"),
        frames_per_camera=student_moved.get("frames_per_camera"),
        allow_dummy_image_slots=bool(student_moved.get("flex_allow_dummy_image_slots", False)),
        residual_image_slots=bool(student_moved.get("flex_residual_image_slots", False)),
        residual_scale=float(student_moved.get("flex_residual_scale", 1.0)),
        passthrough_image_slots=bool(student_moved.get("flex_passthrough_image_slots", False)),
        selection_strategy=str(student_moved.get("flex_selection_strategy", "first") or "first"),
        scene_deepstack=True,
    )
    return deepstack


def _student_compressed_scene_outputs(student, student_moved: dict[str, Any]) -> torch.Tensor:
    if not hasattr(student, "_flex_inputs_embeds"):
        raise RuntimeError("Student model does not expose _flex_inputs_embeds for image feature parity.")
    inputs_embeds, _, _ = student._flex_inputs_embeds(
        student_moved["input_ids"],
        student_moved["pixel_values"],
        student_moved["image_grid_thw"],
        camera_indices=student_moved.get("camera_indices"),
        relative_timestamps=student_moved.get("relative_timestamps"),
        camera_counts=student_moved.get("camera_counts"),
        frames_per_camera=student_moved.get("frames_per_camera"),
        allow_dummy_image_slots=bool(student_moved.get("flex_allow_dummy_image_slots", False)),
        residual_image_slots=bool(student_moved.get("flex_residual_image_slots", False)),
        residual_scale=float(student_moved.get("flex_residual_scale", 1.0)),
        passthrough_image_slots=bool(student_moved.get("flex_passthrough_image_slots", False)),
        selection_strategy=str(student_moved.get("flex_selection_strategy", "first") or "first"),
        scene_deepstack=False,
    )
    image_token_id = int(getattr(student, "image_token_id"))
    image_mask = student_moved["input_ids"] == image_token_id
    batch_size = int(student_moved["input_ids"].shape[0])
    scene_tokens = int(getattr(getattr(student, "flex_scene_config", None), "scene_tokens", 0) or 0)
    if scene_tokens <= 0:
        raise RuntimeError("Could not infer FLEX scene token count for image feature parity.")
    counts = image_mask.sum(dim=1)
    if not bool(torch.all(counts == scene_tokens).item()):
        raise ValueError(
            "Image feature parity expects compressed image-token count to match scene tokens; "
            f"counts={counts.detach().cpu().tolist()}, scene_tokens={scene_tokens}."
        )
    return inputs_embeds[image_mask].reshape(batch_size, scene_tokens, -1)


def _cached_image_feature_loss(
    *,
    student,
    student_moved: dict[str, Any],
    teacher_entries: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    device = student_moved["input_ids"].device
    loss = torch.zeros((), device=device, dtype=torch.float32)
    metrics: dict[str, list[float]] = defaultdict(list)
    if not image_feature_loss_enabled(args):
        return loss, {}
    scene_outputs = _student_compressed_scene_outputs(student, student_moved).float()
    for row_index, entry in enumerate(teacher_entries):
        targets = entry.get("image_feature_targets")
        if not isinstance(targets, torch.Tensor):
            continue
        t_vecs = targets.to(device=device).float().detach()
        s_vecs = scene_outputs[row_index]
        usable = min(int(t_vecs.shape[0]), int(s_vecs.shape[0]))
        if usable <= 0:
            continue
        t_vecs = t_vecs[:usable]
        s_vecs = s_vecs[:usable]
        cos = F.cosine_similarity(s_vecs, t_vecs, dim=-1)
        t_norm = t_vecs.norm(dim=-1).clamp(min=1e-6)
        s_norm = s_vecs.norm(dim=-1).clamp(min=1e-6)
        if float(args.image_feature_cos_weight):
            loss = loss + float(args.image_feature_cos_weight) * (1.0 - cos).mean()
        if float(args.image_feature_norm_weight):
            loss = loss + float(args.image_feature_norm_weight) * F.smooth_l1_loss(
                torch.log(s_norm),
                torch.log(t_norm),
            )
        if float(args.image_feature_mse_weight):
            loss = loss + float(args.image_feature_mse_weight) * F.smooth_l1_loss(s_vecs, t_vecs)
        metrics["image_feature_cos"].append(float(cos.mean().detach().cpu()))
        metrics["image_feature_norm_ratio"].append(float((s_norm / t_norm).mean().detach().cpu()))
        metrics["image_feature_mse"].append(float(F.smooth_l1_loss(s_vecs.detach(), t_vecs).detach().cpu()))
        metrics["image_feature_usable"].append(float(usable))
    return loss, {key: mean(values) for key, values in metrics.items()}


def _cached_deepstack_feature_loss(
    *,
    student,
    student_moved: dict[str, Any],
    teacher_entries: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    device = student_moved["input_ids"].device
    loss = torch.zeros((), device=device, dtype=torch.float32)
    metrics: dict[str, list[float]] = defaultdict(list)
    if not deepstack_feature_loss_enabled(args):
        return loss, {}
    deepstack_outputs = _student_compressed_deepstack_outputs(student, student_moved)
    if not deepstack_outputs:
        raise RuntimeError("Student did not produce compressed DeepStack outputs.")
    batch_size = int(student_moved["input_ids"].shape[0])
    scene_tokens = int(getattr(getattr(student, "flex_scene_config", None), "scene_tokens", 0) or 0)
    if scene_tokens <= 0:
        raise RuntimeError("Could not infer FLEX scene token count for DeepStack feature parity.")
    for layer_index, layer_output in enumerate(deepstack_outputs):
        if layer_output.ndim != 2:
            raise ValueError(f"DeepStack output must be [batch*tokens, hidden], got {tuple(layer_output.shape)}")
        if int(layer_output.shape[0]) < batch_size * scene_tokens:
            raise ValueError(
                "DeepStack output has fewer rows than batch*scene_tokens; "
                f"got {tuple(layer_output.shape)}, batch={batch_size}, scene_tokens={scene_tokens}."
            )
        row_outputs = layer_output[: batch_size * scene_tokens].reshape(batch_size, scene_tokens, -1).float()
        for row_index, entry in enumerate(teacher_entries):
            targets = entry.get("deepstack_feature_targets")
            if not isinstance(targets, torch.Tensor):
                continue
            if layer_index >= int(targets.shape[0]):
                continue
            t_vecs = targets[layer_index].to(device=device).float().detach()
            s_vecs = row_outputs[row_index]
            usable = min(int(t_vecs.shape[0]), int(s_vecs.shape[0]))
            if usable <= 0:
                continue
            t_vecs = t_vecs[:usable]
            s_vecs = s_vecs[:usable]
            cos = F.cosine_similarity(s_vecs, t_vecs, dim=-1)
            t_norm = t_vecs.norm(dim=-1).clamp(min=1e-6)
            s_norm = s_vecs.norm(dim=-1).clamp(min=1e-6)
            if float(args.deepstack_feature_cos_weight):
                loss = loss + float(args.deepstack_feature_cos_weight) * (1.0 - cos).mean()
            if float(args.deepstack_feature_norm_weight):
                loss = loss + float(args.deepstack_feature_norm_weight) * F.smooth_l1_loss(
                    torch.log(s_norm),
                    torch.log(t_norm),
                )
            if float(args.deepstack_feature_mse_weight):
                loss = loss + float(args.deepstack_feature_mse_weight) * F.smooth_l1_loss(s_vecs, t_vecs)
            metrics["deepstack_feature_cos"].append(float(cos.mean().detach().cpu()))
            metrics["deepstack_feature_norm_ratio"].append(float((s_norm / t_norm).mean().detach().cpu()))
            metrics["deepstack_feature_mse"].append(float(F.smooth_l1_loss(s_vecs.detach(), t_vecs).detach().cpu()))
            metrics[f"deepstack_feature_l{layer_index}_cos"].append(float(cos.mean().detach().cpu()))
            metrics["deepstack_feature_usable"].append(float(usable))
    return loss, {key: mean(values) for key, values in metrics.items()}


def _find_paired_indices(batch_rows: list[dict[str, Any]], paired_ablation: str) -> list[tuple[int, int]]:
    by_sample: dict[str, dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(batch_rows):
        sid = str(row.get("sample_id"))
        mode = str(row.get("_image_ablation") or "normal")
        by_sample[sid][mode] = index
    pairs: list[tuple[int, int]] = []
    for modes in by_sample.values():
        if "normal" in modes and paired_ablation in modes:
            pairs.append((modes["normal"], modes[paired_ablation]))
    return pairs


def _pairwise_delta_loss(
    *,
    student_logits: torch.Tensor,
    student_hidden: torch.Tensor,
    student_batch: dict[str, Any],
    batch_rows: list[dict[str, Any]],
    teacher_entries: list[dict[str, torch.Tensor]] | None,
    teacher_logits: torch.Tensor | None,
    teacher_hidden: torch.Tensor | None,
    teacher_moved: dict[str, Any] | None,
    free_run_token_targets: dict[str, list[int]],
    device: torch.device,
    traj_start: int,
    num_bins: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    paired_ablation = str(args.paired_ablation)
    if paired_ablation == "none":
        return torch.zeros((), device=device, dtype=torch.float32), {}
    pairs = _find_paired_indices(batch_rows, paired_ablation)
    if not pairs:
        return torch.zeros((), device=device, dtype=torch.float32), {"pair_count": 0.0}

    loss = torch.zeros((), device=device, dtype=torch.float32)
    metrics: dict[str, list[float]] = defaultdict(list)
    student_positions = student_batch.get("teacher_text_boundary_hidden_positions")
    teacher_positions = teacher_moved.get("teacher_text_boundary_hidden_positions") if teacher_moved is not None else None

    for normal_index, ablated_index in pairs:
        if (
            (float(args.pairwise_boundary_delta_cos_weight) or float(args.pairwise_boundary_delta_norm_weight))
            and isinstance(student_positions, torch.Tensor)
        ):
            s_normal, s_normal_valid = _boundary_vectors(student_hidden.float(), student_positions, normal_index)
            s_ablated, s_ablated_valid = _boundary_vectors(student_hidden.float(), student_positions, ablated_index)
            if teacher_entries is not None:
                t_normal = teacher_entries[normal_index]["boundary_vecs"].to(device=device).float().detach()
                t_ablated = teacher_entries[ablated_index]["boundary_vecs"].to(device=device).float().detach()
                t_valid = (
                    teacher_entries[normal_index]["boundary_valid"].to(device=device).bool()
                    & teacher_entries[ablated_index]["boundary_valid"].to(device=device).bool()
                )
            else:
                assert teacher_hidden is not None and isinstance(teacher_positions, torch.Tensor)
                t_normal, t_normal_valid = _boundary_vectors(teacher_hidden.float(), teacher_positions, normal_index)
                t_ablated, t_ablated_valid = _boundary_vectors(teacher_hidden.float(), teacher_positions, ablated_index)
                t_valid = t_normal_valid & t_ablated_valid
            valid = s_normal_valid & s_ablated_valid & t_valid
            action_index = BOUNDARY_NAMES.index("action_pre")
            if bool(valid[action_index].item()):
                s_delta = s_normal[action_index] - s_ablated[action_index]
                t_delta = t_normal[action_index] - t_ablated[action_index]
                s_norm = s_delta.norm().clamp(min=1e-6)
                t_norm = t_delta.norm().clamp(min=1e-6)
                cos = F.cosine_similarity(s_delta[None, :], t_delta[None, :], dim=-1).mean()
                if float(args.pairwise_boundary_delta_cos_weight):
                    loss = loss + float(args.pairwise_boundary_delta_cos_weight) * (1.0 - cos)
                if float(args.pairwise_boundary_delta_norm_weight):
                    loss = loss + float(args.pairwise_boundary_delta_norm_weight) * F.smooth_l1_loss(
                        torch.log(s_norm),
                        torch.log(t_norm),
                    )
                metrics["pair_action_pre_delta_cos"].append(float(cos.detach().cpu()))
                metrics["pair_action_pre_student_delta_norm"].append(float(s_norm.detach().cpu()))
                metrics["pair_action_pre_teacher_delta_norm"].append(float(t_norm.detach().cpu()))
                metrics["pair_action_pre_delta_norm_ratio"].append(float((s_norm / t_norm).detach().cpu()))

        if float(args.pairwise_traj_logprob_delta_weight):
            s_pos_normal = _label_positions(student_batch, normal_index, "traj_token_mask")[:128]
            s_pos_ablated = _label_positions(student_batch, ablated_index, "traj_token_mask")[:128]
            if teacher_entries is not None:
                t_logits_normal = teacher_entries[normal_index]["traj_logits"].to(device=device)
                t_logits_ablated = teacher_entries[ablated_index]["traj_logits"].to(device=device)
            else:
                assert teacher_moved is not None and teacher_logits is not None
                t_pos_normal = _label_positions(teacher_moved, normal_index, "traj_token_mask")[:128]
                t_pos_ablated = _label_positions(teacher_moved, ablated_index, "traj_token_mask")[:128]
                t_logits_normal = _traj_logits(
                    teacher_logits,
                    t_pos_normal,
                    row_index=normal_index,
                    traj_start=traj_start,
                    num_bins=num_bins,
                )
                t_logits_ablated = _traj_logits(
                    teacher_logits,
                    t_pos_ablated,
                    row_index=ablated_index,
                    traj_start=traj_start,
                    num_bins=num_bins,
                )
            usable = min(
                int(s_pos_normal.numel()),
                int(s_pos_ablated.numel()),
                int(t_logits_normal.shape[0]),
                int(t_logits_ablated.shape[0]),
                128,
            )
            if usable > 0:
                s_logits_normal = _traj_logits(
                    student_logits,
                    s_pos_normal[:usable],
                    row_index=normal_index,
                    traj_start=traj_start,
                    num_bins=num_bins,
                )
                s_logits_ablated = _traj_logits(
                    student_logits,
                    s_pos_ablated[:usable],
                    row_index=ablated_index,
                    traj_start=traj_start,
                    num_bins=num_bins,
                )
                s_delta = F.log_softmax(s_logits_normal, dim=-1) - F.log_softmax(s_logits_ablated, dim=-1)
                t_delta = F.log_softmax(t_logits_normal[:usable], dim=-1) - F.log_softmax(
                    t_logits_ablated[:usable],
                    dim=-1,
                )
                traj_delta = F.smooth_l1_loss(s_delta, t_delta.detach())
                loss = loss + float(args.pairwise_traj_logprob_delta_weight) * traj_delta
                metrics["pair_traj_logprob_delta_loss"].append(float(traj_delta.detach().cpu()))
                metrics["pair_traj_logprob_teacher_delta_l1"].append(float(t_delta.detach().abs().mean().cpu()))
                metrics["pair_traj_logprob_student_delta_l1"].append(float(s_delta.detach().abs().mean().cpu()))

        if float(args.pairwise_free_run_margin_weight):
            s_pos_normal = _label_positions(student_batch, normal_index, "traj_token_mask")[:128]
            s_pos_ablated = _label_positions(student_batch, ablated_index, "traj_token_mask")[:128]
            normal_targets = free_run_token_targets.get(cache_key(batch_rows[normal_index]))
            ablated_targets = free_run_token_targets.get(cache_key(batch_rows[ablated_index]))
            usable = min(
                int(s_pos_normal.numel()),
                int(s_pos_ablated.numel()),
                len(normal_targets or []),
                len(ablated_targets or []),
                128,
            )
            if usable > 0 and normal_targets and ablated_targets:
                normal_target = torch.as_tensor(normal_targets[:usable], device=device, dtype=torch.long)
                ablated_target = torch.as_tensor(ablated_targets[:usable], device=device, dtype=torch.long)
                valid = (
                    (normal_target >= 0)
                    & (normal_target < int(num_bins))
                    & (ablated_target >= 0)
                    & (ablated_target < int(num_bins))
                )
                target_mismatch = (normal_target != ablated_target) & valid
                if bool(target_mismatch.any().item()):
                    normal_target = normal_target[valid]
                    ablated_target = ablated_target[valid]
                    pos_normal = s_pos_normal[:usable].to(device=device)[valid]
                    pos_ablated = s_pos_ablated[:usable].to(device=device)[valid]
                    logits_normal = _traj_logits(
                        student_logits,
                        pos_normal,
                        row_index=normal_index,
                        traj_start=traj_start,
                        num_bins=num_bins,
                    )
                    logits_ablated = _traj_logits(
                        student_logits,
                        pos_ablated,
                        row_index=ablated_index,
                        traj_start=traj_start,
                        num_bins=num_bins,
                    )
                    logp_normal = F.log_softmax(logits_normal.float(), dim=-1)
                    logp_ablated = F.log_softmax(logits_ablated.float(), dim=-1)
                    lp_normal_on_normal = logp_normal.gather(1, normal_target[:, None]).squeeze(1).mean()
                    lp_ablated_on_normal = logp_normal.gather(1, ablated_target[:, None]).squeeze(1).mean()
                    lp_ablated_on_ablated = logp_ablated.gather(1, ablated_target[:, None]).squeeze(1).mean()
                    lp_normal_on_ablated = logp_ablated.gather(1, normal_target[:, None]).squeeze(1).mean()
                    normal_margin = lp_normal_on_normal - lp_ablated_on_normal
                    ablated_margin = lp_ablated_on_ablated - lp_normal_on_ablated
                    margin = torch.as_tensor(float(args.pairwise_free_run_margin), device=device, dtype=torch.float32)
                    margin_loss = 0.5 * (
                        F.relu(margin - normal_margin) + F.relu(margin - ablated_margin)
                    )
                    loss = loss + float(args.pairwise_free_run_margin_weight) * margin_loss
                    metrics["pair_free_run_margin_loss"].append(float(margin_loss.detach().cpu()))
                    metrics["pair_free_run_normal_margin"].append(float(normal_margin.detach().cpu()))
                    metrics["pair_free_run_ablated_margin"].append(float(ablated_margin.detach().cpu()))
                    metrics["pair_free_run_target_mismatch"].append(float(target_mismatch.float().mean().detach().cpu()))
                    metrics["pair_free_run_valid_tokens"].append(float(valid.sum().detach().cpu()))
    metrics["pair_count"].append(float(len(pairs)))
    return loss, {key: mean(values) for key, values in metrics.items()}


def _layer_index_from_name(name: str) -> int | None:
    matches = re.findall(r"(?:layers|h)\.(\d+)\.", name)
    if not matches:
        return None
    return int(matches[-1])


def configure_trainable_parameters(student, args: argparse.Namespace) -> tuple[int, int, dict[str, int]]:
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    groups: dict[str, int] = defaultdict(int)

    if bool(args.train_flex):
        flex = getattr(student, "flex_scene_encoder", None)
        if flex is None:
            raise RuntimeError("Student checkpoint does not have flex_scene_encoder.")
        flex.to(dtype=torch.float32)
        for parameter in flex.parameters():
            parameter.requires_grad_(True)
            groups["flex_scene_encoder"] += int(parameter.numel())

    if bool(args.train_flex_deepstack_projector):
        projector = getattr(student, "flex_deepstack_projector", None)
        if projector is None:
            raise RuntimeError("Student checkpoint does not have flex_deepstack_projector.")
        projector.to(dtype=torch.float32)
        for parameter in projector.parameters():
            parameter.requires_grad_(True)
            groups["flex_deepstack_projector"] += int(parameter.numel())

    lora_params = [(name, parameter) for name, parameter in student.named_parameters() if "lora_" in name.lower()]
    if bool(args.unfreeze_all_lora) or int(args.unfreeze_lora_last_n_layers) > 0:
        layer_indices = [index for name, _ in lora_params if (index := _layer_index_from_name(name)) is not None]
        max_layer = max(layer_indices) if layer_indices else None
        threshold = None
        if max_layer is not None and int(args.unfreeze_lora_last_n_layers) > 0:
            threshold = max_layer - int(args.unfreeze_lora_last_n_layers) + 1
        for name, parameter in lora_params:
            layer_index = _layer_index_from_name(name)
            train = bool(args.unfreeze_all_lora)
            if threshold is not None and layer_index is not None and layer_index >= threshold:
                train = True
            if train:
                parameter.requires_grad_(True)
                groups["language_lora"] += int(parameter.numel())

    if bool(args.unfreeze_multimodal_projector):
        projector_markers = (
            "multi_modal_projector",
            "multimodal_projector",
            "mm_projector",
            "visual.merger",
            "visual_merger",
        )
        for name, parameter in student.named_parameters():
            lowered = name.lower()
            if any(marker in lowered for marker in projector_markers):
                parameter.requires_grad_(True)
                groups["multimodal_projector"] += int(parameter.numel())

    trainable = sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in student.parameters())
    if trainable <= 0:
        raise RuntimeError("No trainable parameters selected.")
    return trainable, total, dict(groups)


def _param_group_name(name: str) -> str:
    lowered = name.lower()
    if "flex_deepstack_projector" in lowered:
        return "flex_deepstack_projector"
    if "flex_scene_encoder" in lowered:
        return "flex_scene_encoder"
    if "lora_" in lowered:
        return "language_lora"
    projector_markers = (
        "multi_modal_projector",
        "multimodal_projector",
        "mm_projector",
        "visual.merger",
        "visual_merger",
    )
    if any(marker in lowered for marker in projector_markers):
        return "multimodal_projector"
    return "other"


def build_optimizer_param_groups(student, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | int]]]:
    base_lr = float(args.learning_rate)
    lr_by_group = {
        "flex_scene_encoder": float(args.flex_lr) if args.flex_lr is not None else base_lr,
        "language_lora": float(args.lora_lr) if args.lora_lr is not None else base_lr,
        "multimodal_projector": (
            float(args.multimodal_projector_lr) if args.multimodal_projector_lr is not None else base_lr
        ),
        "flex_deepstack_projector": (
            float(args.deepstack_projector_lr) if args.deepstack_projector_lr is not None else base_lr
        ),
        "other": base_lr,
    }
    grouped_params: dict[str, list[torch.nn.Parameter]] = defaultdict(list)
    grouped_counts: dict[str, int] = defaultdict(int)
    for name, parameter in student.named_parameters():
        if not parameter.requires_grad:
            continue
        group_name = _param_group_name(name)
        grouped_params[group_name].append(parameter)
        grouped_counts[group_name] += int(parameter.numel())
    if not grouped_params:
        raise RuntimeError("No trainable parameters selected for optimizer.")
    optimizer_groups: list[dict[str, Any]] = []
    summary: dict[str, dict[str, float | int]] = {}
    for group_name in sorted(grouped_params):
        params = grouped_params[group_name]
        lr = float(lr_by_group[group_name])
        optimizer_groups.append({"params": params, "lr": lr})
        summary[group_name] = {"lr": lr, "params": int(grouped_counts[group_name])}
    return optimizer_groups, summary


def infer_flex_deepstack_layer_count(student) -> int:
    conditional = student._conditional_backbone() if hasattr(student, "_conditional_backbone") else None
    visual_model = getattr(conditional, "visual", None)
    layer_count = len(getattr(visual_model, "deepstack_visual_indexes", []) or [])
    if layer_count <= 0:
        layer_count = len(getattr(visual_model, "deepstack_merger_list", []) or [])
    language_model = getattr(getattr(conditional, "model", None), "language_model", None)
    if layer_count <= 0:
        layer_count = len(getattr(language_model, "deepstack_visual_indexes", []) or [])
    if layer_count <= 0:
        layer_count = len(getattr(language_model, "deepstack_merger_list", []) or [])
    if layer_count <= 0:
        raise RuntimeError("Could not infer Qwen3-VL DeepStack layer count from the student backbone.")
    return int(layer_count)


def deepstack_feature_loss_enabled(args: argparse.Namespace) -> bool:
    return bool(
        float(args.deepstack_feature_cos_weight)
        or float(args.deepstack_feature_norm_weight)
        or float(args.deepstack_feature_mse_weight)
    )


def image_feature_loss_enabled(args: argparse.Namespace) -> bool:
    return bool(
        float(args.image_feature_cos_weight)
        or float(args.image_feature_norm_weight)
        or float(args.image_feature_mse_weight)
    )


def _select_k_offsets(*, length: int, tokens_per_image: int, strategy: str) -> torch.Tensor:
    length = max(int(length), 0)
    keep_count = min(max(int(tokens_per_image), 0), length)
    if keep_count <= 0:
        return torch.empty((0,), dtype=torch.long)
    strategy = str(strategy or "first").lower()
    if strategy == "first":
        return torch.arange(keep_count, dtype=torch.long)
    if strategy == "uniform":
        if keep_count == length:
            return torch.arange(length, dtype=torch.long)
        offsets = torch.div(
            (torch.arange(keep_count, dtype=torch.long) * 2 + 1) * length,
            2 * keep_count,
            rounding_mode="floor",
        )
        return offsets.clamp_(0, length - 1)
    raise ValueError(f"Unsupported FLEX image-token selection strategy: {strategy!r}.")


def _select_k_image_targets(
    *,
    image_embeds: Any,
    tokens_per_image: int,
    selection_strategy: str,
) -> torch.Tensor | None:
    image_parts = list(image_embeds)
    if not image_parts:
        return None
    k = int(tokens_per_image)
    if k <= 0:
        raise ValueError("--image-feature-tokens-per-image must be >0 when image feature loss is enabled.")
    selected: list[torch.Tensor] = []
    for image_tensor in image_parts:
        length = int(image_tensor.shape[0])
        offsets = _select_k_offsets(
            length=length,
            tokens_per_image=k,
            strategy=selection_strategy,
        ).to(device=image_tensor.device)
        if int(offsets.numel()) > 0:
            selected.append(image_tensor[:length].index_select(0, offsets))
    if not selected:
        return None
    return torch.cat(selected, dim=0).detach().cpu()


def _select_k_deepstack_targets(
    *,
    image_embeds: Any,
    deepstack_image_embeds: Any,
    tokens_per_image: int,
    selection_strategy: str,
) -> torch.Tensor | None:
    if deepstack_image_embeds is None:
        return None
    image_parts = list(image_embeds)
    layer_parts = list(deepstack_image_embeds)
    if not image_parts or not layer_parts:
        return None
    k = int(tokens_per_image)
    if k <= 0:
        raise ValueError("--deepstack-feature-tokens-per-image must be >0 when DeepStack feature loss is enabled.")
    lengths = [int(part.shape[0]) for part in image_parts]
    layer_targets: list[torch.Tensor] = []
    for layer_tensor in layer_parts:
        offset = 0
        selected: list[torch.Tensor] = []
        for length in lengths:
            offsets = _select_k_offsets(
                length=int(length),
                tokens_per_image=k,
                strategy=selection_strategy,
            ).to(device=layer_tensor.device)
            if int(offsets.numel()) > 0:
                selected.append(layer_tensor[offset : offset + int(length)].index_select(0, offsets))
            offset += int(length)
        if not selected:
            continue
        layer_targets.append(torch.cat(selected, dim=0).detach().cpu())
    if not layer_targets:
        return None
    return torch.stack(layer_targets, dim=0)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def forward_model(model, moved: dict[str, Any]) -> dict[str, Any]:
    return eval104.forward_model(model, moved)


def build_teacher_cache(
    *,
    teacher,
    rows: list[dict[str, Any]],
    collator: DistillationCollator,
    device: torch.device,
    dtype: torch.dtype,
    traj_start: int,
    num_bins: int,
    args: argparse.Namespace,
) -> dict[str, dict[str, torch.Tensor]]:
    cache: dict[str, dict[str, torch.Tensor]] = {}
    for index, row in enumerate(rows, start=1):
        sid = cache_key(row)
        batch = collator([row])
        moved = eval104.move_batch(batch, device=device, dtype=dtype)
        with torch.inference_mode():
            out = forward_model(teacher, moved)
        logits = out["logits"].detach()
        hidden = out["hidden_states"].detach()
        entry: dict[str, torch.Tensor] = {}
        if deepstack_feature_loss_enabled(args) or image_feature_loss_enabled(args):
            conditional = teacher._conditional_backbone() if hasattr(teacher, "_conditional_backbone") else None
            if conditional is None or not hasattr(conditional, "get_image_features"):
                raise RuntimeError("Teacher model does not expose get_image_features for visual feature parity.")
            with torch.inference_mode():
                image_embeds, deepstack_image_embeds = conditional.get_image_features(
                    moved["pixel_values"],
                    moved["image_grid_thw"],
                )
            if image_feature_loss_enabled(args):
                image_targets = _select_k_image_targets(
                    image_embeds=image_embeds,
                    tokens_per_image=int(args.image_feature_tokens_per_image),
                    selection_strategy=str(args.flex_selection_strategy),
                )
                if image_targets is None:
                    raise RuntimeError("Teacher did not produce image feature targets.")
                entry["image_feature_targets"] = image_targets
            if deepstack_feature_loss_enabled(args):
                deepstack_targets = _select_k_deepstack_targets(
                    image_embeds=image_embeds,
                    deepstack_image_embeds=deepstack_image_embeds,
                    tokens_per_image=int(args.deepstack_feature_tokens_per_image),
                    selection_strategy=str(args.flex_selection_strategy),
                )
                if deepstack_targets is None:
                    raise RuntimeError("Teacher did not produce DeepStack feature targets.")
                entry["deepstack_feature_targets"] = deepstack_targets
        traj_pos = _label_positions(moved, 0, "traj_token_mask")[:128]
        entry["traj_logits"] = _traj_logits(
            logits,
            traj_pos,
            row_index=0,
            traj_start=traj_start,
            num_bins=num_bins,
        ).detach().cpu()
        if (
            float(args.traj_state_cos_weight)
            or float(args.traj_state_norm_weight)
            or float(args.traj_state_mse_weight)
        ):
            entry["traj_state_vecs"] = _trajectory_state_vectors(hidden.float(), moved, 0).detach().cpu()
        for mask_key, prefix, weight in (
            ("cot_span_mask", "text", float(args.text_kl_weight)),
            ("format_token_mask", "format", float(args.format_kl_weight)),
        ):
            if not weight:
                continue
            positions = _label_positions(moved, 0, mask_key)
            entry[f"{prefix}_logits"] = _full_logits(logits, positions, row_index=0).detach().cpu()
        teacher_positions = moved.get("teacher_text_boundary_hidden_positions")
        if isinstance(teacher_positions, torch.Tensor):
            vectors, valid = _boundary_vectors(hidden.float(), teacher_positions, 0)
            entry["boundary_vecs"] = vectors.detach().cpu()
            entry["boundary_valid"] = valid.detach().cpu()
        else:
            entry["boundary_vecs"] = torch.zeros((len(BOUNDARY_NAMES), hidden.shape[-1]))
            entry["boundary_valid"] = torch.zeros((len(BOUNDARY_NAMES),), dtype=torch.bool)
        cache[sid] = entry
        print(json.dumps({"event": "teacher_cache_done", "done": index, "total": len(rows)}), flush=True)
    return cache


def build_collated_cache(
    *,
    rows: list[dict[str, Any]],
    collator: DistillationCollator,
    student,
    tokenizer,
    flex_cfg,
    free_run_token_targets: dict[str, list[int]],
    traj_start: int,
    num_bins: int,
    args: argparse.Namespace,
) -> list[tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]]:
    if int(args.batch_size) != 1:
        raise ValueError("--cache-collated-batches currently expects --batch-size 1.")
    cache: list[tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]] = []
    for index, row in enumerate(rows, start=1):
        batch_rows = [row]
        batch = collator(batch_rows)
        if bool(args.free_run_token_force_context) and str(args.free_run_token_context_source) == "target":
            apply_free_run_token_targets_to_batch(
                batch,
                batch_rows,
                free_run_token_targets,
                traj_start=traj_start,
                num_bins=num_bins,
            )
        student_batch = flex_student_batch(
            batch,
            student=student,
            tokenizer=tokenizer,
            flex_cfg=flex_cfg,
            args=args,
        )
        cache.append((batch_rows, batch, student_batch))
        print(json.dumps({"event": "collated_cache_done", "done": index, "total": len(rows)}), flush=True)
    return cache


def build_paired_collated_cache(
    *,
    base_rows: list[dict[str, Any]],
    paired_ablation: str,
    collator: DistillationCollator,
    student,
    tokenizer,
    flex_cfg,
    free_run_token_targets: dict[str, list[int]],
    traj_start: int,
    num_bins: int,
    args: argparse.Namespace,
) -> list[tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]]:
    cache: list[tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]] = []
    for index, row in enumerate(base_rows, start=1):
        batch_rows = [
            dict(row, _image_ablation="normal"),
            dict(row, _image_ablation=paired_ablation),
        ]
        batch = collator(batch_rows)
        if bool(args.free_run_token_force_context) and str(args.free_run_token_context_source) == "target":
            apply_free_run_token_targets_to_batch(
                batch,
                batch_rows,
                free_run_token_targets,
                traj_start=traj_start,
                num_bins=num_bins,
            )
        student_batch = flex_student_batch(
            batch,
            student=student,
            tokenizer=tokenizer,
            flex_cfg=flex_cfg,
            args=args,
        )
        cache.append((batch_rows, batch, student_batch))
        print(
            json.dumps(
                {
                    "event": "paired_collated_cache_done",
                    "done": index,
                    "total": len(base_rows),
                    "paired_ablation": paired_ablation,
                }
            ),
            flush=True,
        )
    return cache


def compute_batch_loss(
    *,
    teacher,
    student,
    batch: dict[str, Any],
    student_batch: dict[str, Any],
    batch_rows: list[dict[str, Any]],
    teacher_cache: dict[str, dict[str, torch.Tensor]] | None,
    free_run_token_targets: dict[str, list[int]],
    device: torch.device,
    dtype: torch.dtype,
    traj_start: int,
    num_bins: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    student_moved = eval104.move_batch(student_batch, device=device, dtype=dtype)

    if teacher_cache is None:
        teacher_moved = eval104.move_batch(batch, device=device, dtype=dtype)
        with torch.inference_mode():
            teacher_out = forward_model(teacher, teacher_moved)
        teacher_logits = teacher_out["logits"].detach()
        teacher_hidden = teacher_out["hidden_states"].detach()
        teacher_entries = None
    else:
        teacher_moved = None
        teacher_logits = None
        teacher_hidden = None
        teacher_entries = [teacher_cache[cache_key(row)] for row in batch_rows]
    student_out = forward_model(student, student_moved)

    student_logits = student_out["logits"]
    student_hidden = student_out["hidden_states"]

    losses: dict[str, list[torch.Tensor]] = defaultdict(list)
    stats: dict[str, list[float]] = defaultdict(list)
    batch_size = int(student_logits.shape[0])

    for row_index in range(batch_size):
        student_traj_pos = _label_positions(student_moved, row_index, "traj_token_mask")
        if teacher_entries is None:
            assert teacher_moved is not None and teacher_logits is not None
            teacher_traj_pos = _label_positions(teacher_moved, row_index, "traj_token_mask")
            teacher_traj = _traj_logits(
                teacher_logits,
                teacher_traj_pos[:128],
                row_index=row_index,
                traj_start=traj_start,
                num_bins=num_bins,
            )
        else:
            teacher_traj = teacher_entries[row_index]["traj_logits"].to(device=device)
        usable = min(int(teacher_traj.shape[0]), int(student_traj_pos.numel()), 128)
        stats["usable_traj_tokens"].append(float(usable))
        image_mode = str(batch_rows[row_index].get("_image_ablation") or "normal")
        if free_run_ce_enabled_for_mode(str(args.free_run_token_ce_modes), image_mode):
            target_tokens = free_run_token_targets.get(cache_key(batch_rows[row_index]))
        else:
            target_tokens = None
        if usable > 0 and float(args.traj_token_ce_weight):
            labels = student_moved.get("labels")
            if isinstance(labels, torch.Tensor):
                target = labels[row_index, student_traj_pos[:usable]].long() - int(traj_start)
                valid = (target >= 0) & (target < int(num_bins))
                if bool(valid.any().item()):
                    s_logits_target = _traj_logits(
                        student_logits,
                        student_traj_pos[:usable],
                        row_index=row_index,
                        traj_start=traj_start,
                        num_bins=num_bins,
                    )[valid]
                    target = target[valid]
                    traj_token_ce = F.cross_entropy(s_logits_target.float(), target)
                    losses["traj_token_ce"].append(traj_token_ce)
                    pred = s_logits_target.detach().argmax(dim=-1)
                    stats["traj_token_ce_acc"].append(float((pred == target).float().mean().detach().cpu()))
                    stats["traj_token_ce_usable"].append(float(valid.sum().detach().cpu()))
        if usable > 0 and target_tokens and float(args.free_run_token_ce_weight):
            target_usable = min(int(student_traj_pos.numel()), len(target_tokens), 128)
            if target_usable > 0:
                target = torch.as_tensor(
                    target_tokens[:target_usable],
                    device=device,
                    dtype=torch.long,
                )
                valid = (target >= 0) & (target < int(num_bins))
                if bool(valid.any().item()):
                    s_logits_target = _traj_logits(
                        student_logits,
                        student_traj_pos[:target_usable],
                        row_index=row_index,
                        traj_start=traj_start,
                        num_bins=num_bins,
                    )[valid]
                    target = target[valid]
                    free_run_ce = F.cross_entropy(s_logits_target.float(), target)
                    losses["free_run_token_ce"].append(free_run_ce)
                    pred = s_logits_target.detach().argmax(dim=-1)
                    stats["free_run_token_acc"].append(float((pred == target).float().mean().detach().cpu()))
                    stats["free_run_token_usable"].append(float(valid.sum().detach().cpu()))
        if usable > 0 and float(args.free_run_end_token_ce_weight):
            labels = student_moved.get("labels")
            if isinstance(labels, torch.Tensor) and int(student_traj_pos.numel()) >= 128:
                last_traj_pos = int(student_traj_pos[:128][-1].detach().cpu().item())
                row_labels = labels[row_index]
                valid_after = torch.nonzero(
                    (row_labels != IGNORE_INDEX)
                    & (
                        torch.arange(
                            int(row_labels.shape[0]),
                            device=row_labels.device,
                            dtype=torch.long,
                        )
                        > last_traj_pos
                    ),
                    as_tuple=False,
                ).flatten()
                if int(valid_after.numel()) > 0:
                    end_pos = int(valid_after[0].detach().cpu().item())
                    if end_pos > 0 and end_pos <= int(student_logits.shape[1]):
                        end_label = row_labels[end_pos].long()
                        if int(end_label.detach().cpu().item()) >= 0:
                            end_logits = student_logits[row_index, end_pos - 1, :].float().unsqueeze(0)
                            end_target = end_label.reshape(1)
                            end_ce = F.cross_entropy(end_logits, end_target)
                            losses["free_run_end_token_ce"].append(end_ce)
                            end_pred = end_logits.detach().argmax(dim=-1)
                            stats["free_run_end_token_acc"].append(
                                float((end_pred == end_target).float().mean().detach().cpu())
                            )
        if float(args.prefix_token_ce_weight):
            labels = student_moved.get("labels")
            if isinstance(labels, torch.Tensor):
                prefix_positions = []
                for mask_key in ("cot_span_mask", "format_token_mask"):
                    pos = _label_positions(student_moved, row_index, mask_key)
                    if int(pos.numel()) > 0:
                        prefix_positions.append(pos)
                if prefix_positions:
                    positions = torch.cat(prefix_positions).unique(sorted=True)
                    targets = labels[row_index, positions].long()
                    valid_prefix = targets != IGNORE_INDEX
                    if bool(valid_prefix.any().item()):
                        positions = positions[valid_prefix]
                        targets = targets[valid_prefix]
                        prefix_logits = _full_logits(student_logits, positions, row_index=row_index)
                        prefix_ce = F.cross_entropy(prefix_logits.float(), targets)
                        losses["prefix_token_ce"].append(prefix_ce)
                        prefix_pred = prefix_logits.detach().argmax(dim=-1)
                        stats["prefix_token_acc"].append(
                            float((prefix_pred == targets).float().mean().detach().cpu())
                        )
                        stats["prefix_token_usable"].append(float(targets.numel()))
        if usable > 0 and args.traj_kl_weight:
            t_logits = teacher_traj[:usable].float()
            s_logits = _traj_logits(
                student_logits,
                student_traj_pos[:usable],
                row_index=row_index,
                traj_start=traj_start,
                num_bins=num_bins,
            )
            traj_kl = _kl_teacher_student(t_logits, s_logits).mean()
            losses["traj_kl"].append(traj_kl)
            t_pred = t_logits.argmax(dim=-1)
            s_pred = s_logits.argmax(dim=-1)
            top5 = torch.topk(s_logits.detach(), k=5, dim=-1).indices
            stats["traj_top1_agreement"].append(float((t_pred == s_pred).float().mean().detach().cpu()))
            stats["traj_teacher_top1_in_student_top5"].append(
                float((top5 == t_pred[:, None]).any(dim=-1).float().mean().detach().cpu())
            )

        for mask_key, loss_name, weight in (
            ("cot_span_mask", "text_kl", args.text_kl_weight),
            ("format_token_mask", "format_kl", args.format_kl_weight),
        ):
            if not weight:
                continue
            student_pos = _label_positions(student_moved, row_index, mask_key)
            if teacher_entries is None:
                assert teacher_moved is not None and teacher_logits is not None
                teacher_pos = _label_positions(teacher_moved, row_index, mask_key)
                teacher_text = _full_logits(teacher_logits, teacher_pos, row_index=row_index)
            else:
                teacher_text = teacher_entries[row_index][f"{loss_name.removesuffix('_kl')}_logits"].to(device=device)
            usable_text = min(int(teacher_text.shape[0]), int(student_pos.numel()))
            if usable_text <= 0:
                continue
            t_full = teacher_text[:usable_text].float()
            s_full = _full_logits(student_logits, student_pos[:usable_text], row_index=row_index)
            kl = _kl_teacher_student(t_full, s_full).mean()
            losses[loss_name].append(kl)
            stats[f"{loss_name}_top1_agreement"].append(
                float((t_full.argmax(dim=-1) == s_full.argmax(dim=-1)).float().mean().detach().cpu())
            )

    total = torch.zeros((), device=device, dtype=torch.float32)
    out_stats: dict[str, float | None] = {}
    if losses["traj_kl"]:
        traj_kl = torch.stack(losses["traj_kl"]).mean()
        total = total + float(args.traj_kl_weight) * traj_kl
        out_stats["traj_kl"] = float(traj_kl.detach().cpu())
    if losses["traj_token_ce"]:
        traj_token_ce = torch.stack(losses["traj_token_ce"]).mean()
        total = total + float(args.traj_token_ce_weight) * traj_token_ce
        out_stats["traj_token_ce"] = float(traj_token_ce.detach().cpu())
    if losses["free_run_token_ce"]:
        free_run_token_ce = torch.stack(losses["free_run_token_ce"]).mean()
        total = total + float(args.free_run_token_ce_weight) * free_run_token_ce
        out_stats["free_run_token_ce"] = float(free_run_token_ce.detach().cpu())
    if losses["free_run_end_token_ce"]:
        free_run_end_token_ce = torch.stack(losses["free_run_end_token_ce"]).mean()
        total = total + float(args.free_run_end_token_ce_weight) * free_run_end_token_ce
        out_stats["free_run_end_token_ce"] = float(free_run_end_token_ce.detach().cpu())
    if losses["prefix_token_ce"]:
        prefix_token_ce = torch.stack(losses["prefix_token_ce"]).mean()
        total = total + float(args.prefix_token_ce_weight) * prefix_token_ce
        out_stats["prefix_token_ce"] = float(prefix_token_ce.detach().cpu())
    if losses["text_kl"]:
        text_kl = torch.stack(losses["text_kl"]).mean()
        total = total + float(args.text_kl_weight) * text_kl
        out_stats["text_kl"] = float(text_kl.detach().cpu())
    if losses["format_kl"]:
        format_kl = torch.stack(losses["format_kl"]).mean()
        total = total + float(args.format_kl_weight) * format_kl
        out_stats["format_kl"] = float(format_kl.detach().cpu())

    if teacher_entries is None:
        assert teacher_moved is not None and teacher_hidden is not None
        boundary, boundary_stats = _boundary_loss(
            teacher_hidden.float(),
            student_hidden.float(),
            teacher_moved,
            student_moved,
            cos_weight=float(args.boundary_cos_weight),
            norm_weight=float(args.boundary_norm_weight),
            mse_weight=float(args.boundary_mse_weight),
        )
    else:
        boundary, boundary_stats = _cached_boundary_loss(
            student_hidden.float(),
            student_moved,
            teacher_entries,
            cos_weight=float(args.boundary_cos_weight),
            norm_weight=float(args.boundary_norm_weight),
            mse_weight=float(args.boundary_mse_weight),
        )
    total = total + boundary
    out_stats["boundary_loss"] = float(boundary.detach().cpu())
    out_stats.update(boundary_stats)
    if (
        float(args.traj_state_cos_weight)
        or float(args.traj_state_norm_weight)
        or float(args.traj_state_mse_weight)
    ):
        if teacher_entries is None:
            assert teacher_moved is not None and teacher_hidden is not None
            traj_state_loss, traj_state_stats = _trajectory_state_loss(
                teacher_hidden.float(),
                student_hidden.float(),
                teacher_moved,
                student_moved,
                cos_weight=float(args.traj_state_cos_weight),
                norm_weight=float(args.traj_state_norm_weight),
                mse_weight=float(args.traj_state_mse_weight),
            )
        else:
            traj_state_loss, traj_state_stats = _cached_trajectory_state_loss(
                student_hidden.float(),
                student_moved,
                teacher_entries,
                cos_weight=float(args.traj_state_cos_weight),
                norm_weight=float(args.traj_state_norm_weight),
                mse_weight=float(args.traj_state_mse_weight),
            )
        total = total + traj_state_loss
        out_stats["traj_state_loss"] = float(traj_state_loss.detach().cpu())
        out_stats.update(traj_state_stats)
    if image_feature_loss_enabled(args):
        if teacher_entries is None:
            raise RuntimeError("Image feature parity requires --cache-teacher-targets.")
        image_feature_loss, image_feature_stats = _cached_image_feature_loss(
            student=student,
            student_moved=student_moved,
            teacher_entries=teacher_entries,
            args=args,
        )
        total = total + image_feature_loss
        out_stats["image_feature_loss"] = float(image_feature_loss.detach().cpu())
        out_stats.update(image_feature_stats)
    if deepstack_feature_loss_enabled(args):
        if teacher_entries is None:
            raise RuntimeError("DeepStack feature parity requires --cache-teacher-targets.")
        deepstack_feature_loss, deepstack_feature_stats = _cached_deepstack_feature_loss(
            student=student,
            student_moved=student_moved,
            teacher_entries=teacher_entries,
            args=args,
        )
        total = total + deepstack_feature_loss
        out_stats["deepstack_feature_loss"] = float(deepstack_feature_loss.detach().cpu())
        out_stats.update(deepstack_feature_stats)
    pairwise, pairwise_stats = _pairwise_delta_loss(
        student_logits=student_logits,
        student_hidden=student_hidden.float(),
        student_batch=student_moved,
        batch_rows=batch_rows,
        teacher_entries=teacher_entries,
        teacher_logits=teacher_logits,
        teacher_hidden=teacher_hidden.float() if teacher_hidden is not None else None,
        teacher_moved=teacher_moved,
        free_run_token_targets=free_run_token_targets,
        device=device,
        traj_start=traj_start,
        num_bins=num_bins,
        args=args,
    )
    total = total + pairwise
    if pairwise_stats:
        out_stats["pairwise_loss"] = float(pairwise.detach().cpu())
        out_stats.update(pairwise_stats)
    for key, values in stats.items():
        out_stats[key] = mean(values)
    out_stats["loss"] = float(total.detach().cpu())
    return total, out_stats


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_json is None:
        args.summary_json = args.output_dir / "train_summary.json"

    rows = [row for row in load_jsonl(args.corpus_jsonl) if row.get("split") == args.split]
    if bool(args.shuffle_train_samples):
        random.Random(int(args.seed)).shuffle(rows)
    if args.max_train_samples > 0:
        rows = rows[: args.max_train_samples]
    if not rows:
        raise SystemExit(f"No rows selected for split={args.split!r}")
    base_sample_count = len(rows)
    base_rows = list(rows)
    image_ablations = parse_image_ablations(args.image_ablations)
    paired_ablation = str(args.paired_ablation)
    if paired_ablation != "none":
        image_ablations = ["normal", paired_ablation]
        rows = expand_rows_for_pairwise_cache(base_rows, paired_ablation)
        if int(args.batch_size) != 2:
            raise ValueError("--paired-ablation expects --batch-size 2.")
    else:
        rows = expand_rows_for_image_ablations(rows, image_ablations)

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    teacher, tokenizer, processor, base_model, teacher_train_config = eval104.load_model(
        args.teacher_checkpoint_dir,
        student_model=args.student_model,
        device=device,
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    data_view = dict(teacher_train_config.get("data_view") or {})
    if str(args.prompt_mode_override):
        data_view["prompt_mode"] = str(args.prompt_mode_override)
    if str(args.target_mode_override):
        data_view["target_mode"] = str(args.target_mode_override)
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

    decoder_path = eval104.resolve_traj_tokenizer_config_path(base_model)
    if decoder_path is None:
        raise SystemExit("Could not find Alpamayo trajectory tokenizer config.")
    decoder = eval104.TrajectoryTokenDecoder(config_path=decoder_path)
    traj_start = int(getattr(tokenizer, "traj_token_start_idx", tokenizer.convert_tokens_to_ids("<i0>")))
    dtype = next(teacher.backbone.parameters()).dtype
    num_bins = int(decoder.num_bins)
    free_run_token_targets = load_free_run_token_targets(str(args.free_run_token_targets))
    if str(args.free_run_token_context_source) == "student_greedy" and not free_run_token_targets:
        raise ValueError("--free-run-token-context-source student_greedy requires --free-run-token-targets.")
    if deepstack_feature_loss_enabled(args):
        if not bool(args.cache_teacher_targets):
            raise ValueError("DeepStack feature parity requires --cache-teacher-targets.")
        if not bool(args.flex_scene_deepstack):
            raise ValueError("DeepStack feature parity requires --flex-scene-deepstack.")
        if int(args.deepstack_feature_tokens_per_image) <= 0:
            raise ValueError(
                "DeepStack feature parity requires --deepstack-feature-tokens-per-image "
                "to match FLEX tokens_per_image."
            )
    if image_feature_loss_enabled(args):
        if not bool(args.cache_teacher_targets):
            raise ValueError("Image feature parity requires --cache-teacher-targets.")
        if int(args.image_feature_tokens_per_image) <= 0:
            raise ValueError(
                "Image feature parity requires --image-feature-tokens-per-image "
                "to match FLEX tokens_per_image."
            )

    teacher_cache: dict[str, dict[str, torch.Tensor]] | None = None
    if args.cache_teacher_targets:
        teacher_cache = build_teacher_cache(
            teacher=teacher,
            rows=rows,
            collator=collator,
            device=device,
            dtype=dtype,
            traj_start=traj_start,
            num_bins=num_bins,
            args=args,
        )
        teacher = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

    student, _, _, _, student_train_config = eval104.load_model(
        args.student_checkpoint_dir,
        student_model=args.student_model,
        device=device,
    )
    if not (hasattr(student, "flex_enabled") and student.flex_enabled()):
        raise SystemExit("Student checkpoint does not have FLEX enabled.")
    if int(args.flex_deepstack_projector_rank) > 0:
        if not bool(args.flex_scene_deepstack):
            raise ValueError("--flex-deepstack-projector-rank requires --flex-scene-deepstack.")
        student.configure_flex_deepstack_projector(
            num_layers=infer_flex_deepstack_layer_count(student),
            rank=int(args.flex_deepstack_projector_rank),
            dropout=float(args.flex_deepstack_projector_dropout),
        )

    student.eval()
    trainable, total_params, trainable_groups = configure_trainable_parameters(student, args)
    if getattr(student, "flex_scene_encoder", None) is not None:
        student.flex_scene_encoder.train()
    if getattr(student, "flex_deepstack_projector", None) is not None:
        student.flex_deepstack_projector.train()
    flex_cfg = getattr(student, "flex_scene_config")
    flex_config_dict = asdict(flex_cfg) if is_dataclass(flex_cfg) else dict(flex_cfg or {})

    if bool(args.cache_collated_batches) and paired_ablation != "none":
        collated_cache = build_paired_collated_cache(
            base_rows=base_rows,
            paired_ablation=paired_ablation,
            collator=collator,
            student=student,
            tokenizer=tokenizer,
            flex_cfg=flex_cfg,
            free_run_token_targets=free_run_token_targets,
            traj_start=traj_start,
            num_bins=num_bins,
            args=args,
        )
    elif bool(args.cache_collated_batches):
        collated_cache = build_collated_cache(
            rows=rows,
            collator=collator,
            student=student,
            tokenizer=tokenizer,
            flex_cfg=flex_cfg,
            free_run_token_targets=free_run_token_targets,
            traj_start=traj_start,
            num_bins=num_bins,
            args=args,
        )
    else:
        collated_cache = None

    optimizer_param_groups, optimizer_group_summary = build_optimizer_param_groups(student, args)
    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    print(
        json.dumps(
            {
                "event": "flex_parity_train_start",
                "teacher_checkpoint_dir": str(args.teacher_checkpoint_dir),
                "student_checkpoint_dir": str(args.student_checkpoint_dir),
                "output_dir": str(args.output_dir),
                "samples": len(rows),
                "base_samples": int(base_sample_count),
                "image_ablations": image_ablations,
                "paired_ablation": paired_ablation,
                "max_steps": int(args.max_steps),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "optimizer_param_groups": optimizer_group_summary,
                "pairwise_boundary_delta_cos_weight": float(args.pairwise_boundary_delta_cos_weight),
                "pairwise_boundary_delta_norm_weight": float(args.pairwise_boundary_delta_norm_weight),
                "pairwise_traj_logprob_delta_weight": float(args.pairwise_traj_logprob_delta_weight),
                "pairwise_free_run_margin_weight": float(args.pairwise_free_run_margin_weight),
                "pairwise_free_run_margin": float(args.pairwise_free_run_margin),
                "free_run_token_ce_weight": float(args.free_run_token_ce_weight),
                "free_run_token_ce_modes": str(args.free_run_token_ce_modes),
                "free_run_end_token_ce_weight": float(args.free_run_end_token_ce_weight),
                "prefix_token_ce_weight": float(args.prefix_token_ce_weight),
                "traj_state_cos_weight": float(args.traj_state_cos_weight),
                "traj_state_norm_weight": float(args.traj_state_norm_weight),
                "traj_state_mse_weight": float(args.traj_state_mse_weight),
                "traj_token_ce_weight": float(args.traj_token_ce_weight),
                "flex_residual_image_slots": bool(args.flex_residual_image_slots),
                "flex_residual_scale": float(args.flex_residual_scale),
                "flex_passthrough_image_slots": bool(args.flex_passthrough_image_slots),
                "flex_selection_strategy": str(args.flex_selection_strategy),
                "flex_scene_deepstack": bool(args.flex_scene_deepstack),
                "flex_deepstack_projector": getattr(student, "flex_deepstack_projector_config", None),
                "train_flex_deepstack_projector": bool(args.train_flex_deepstack_projector),
                "image_feature_tokens_per_image": int(args.image_feature_tokens_per_image),
                "image_feature_cos_weight": float(args.image_feature_cos_weight),
                "image_feature_norm_weight": float(args.image_feature_norm_weight),
                "image_feature_mse_weight": float(args.image_feature_mse_weight),
                "deepstack_feature_tokens_per_image": int(args.deepstack_feature_tokens_per_image),
                "deepstack_feature_cos_weight": float(args.deepstack_feature_cos_weight),
                "deepstack_feature_norm_weight": float(args.deepstack_feature_norm_weight),
                "deepstack_feature_mse_weight": float(args.deepstack_feature_mse_weight),
                "free_run_token_targets": len(free_run_token_targets),
                "free_run_token_force_context": bool(args.free_run_token_force_context),
                "free_run_token_context_source": str(args.free_run_token_context_source),
                "student_greedy_context_refresh_steps": int(args.student_greedy_context_refresh_steps),
                "student_greedy_invalid_context": str(args.student_greedy_invalid_context),
                "cache_teacher_targets": bool(args.cache_teacher_targets),
                "cache_collated_batches": bool(args.cache_collated_batches),
                "trainable_params": int(trainable),
                "total_params": int(total_params),
                "trainable_groups": trainable_groups,
                "flex_config": flex_config_dict,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    history: list[dict[str, Any]] = []
    running: dict[str, list[float]] = defaultdict(list)
    batch_iter = (
        cycle_pair_batches(base_rows, paired_ablation)
        if paired_ablation != "none"
        else cycle_batches(rows, args.batch_size)
    )
    student_greedy_contexts: dict[str, list[int]] = {}
    student_greedy_refresh_steps = max(int(args.student_greedy_context_refresh_steps), 1)
    for step in range(1, int(args.max_steps) + 1):
        if str(args.free_run_token_context_source) == "student_greedy" and (
            step == 1 or ((step - 1) % student_greedy_refresh_steps == 0)
        ):
            print(
                json.dumps(
                    {
                        "event": "student_greedy_context_refresh_start",
                        "step": step,
                        "rows": len(rows),
                    }
                ),
                flush=True,
            )
            student_greedy_contexts = generate_student_greedy_contexts(
                student=student,
                rows=rows,
                tokenizer=tokenizer,
                processor=processor,
                data_view=data_view,
                device=device,
                dtype=dtype,
                traj_token_targets=free_run_token_targets,
                flex_cfg=flex_cfg,
                args=args,
            )
        if collated_cache is not None:
            batch_rows, batch, student_batch = collated_cache[(step - 1) % len(collated_cache)]
            if str(args.free_run_token_context_source) == "student_greedy":
                batch = clone_tensor_batch(batch)
                apply_free_run_token_context_to_batch(
                    batch,
                    batch_rows,
                    context_tokens=student_greedy_contexts,
                    label_targets=free_run_token_targets,
                    traj_start=traj_start,
                    num_bins=num_bins,
                )
                student_batch = flex_student_batch(
                    batch,
                    student=student,
                    tokenizer=tokenizer,
                    flex_cfg=flex_cfg,
                    args=args,
                )
        else:
            batch_rows = next(batch_iter)
            batch = collator(batch_rows)
            if bool(args.free_run_token_force_context) and str(args.free_run_token_context_source) == "target":
                apply_free_run_token_targets_to_batch(
                    batch,
                    batch_rows,
                    free_run_token_targets,
                    traj_start=traj_start,
                    num_bins=num_bins,
                )
            elif str(args.free_run_token_context_source) == "student_greedy":
                apply_free_run_token_context_to_batch(
                    batch,
                    batch_rows,
                    context_tokens=student_greedy_contexts,
                    label_targets=free_run_token_targets,
                    traj_start=traj_start,
                    num_bins=num_bins,
                )
            student_batch = flex_student_batch(
                batch,
                student=student,
                tokenizer=tokenizer,
                flex_cfg=flex_cfg,
                args=args,
            )
        optimizer.zero_grad(set_to_none=True)
        loss, stats = compute_batch_loss(
            teacher=teacher,
            student=student,
            batch=batch,
            student_batch=student_batch,
            batch_rows=batch_rows,
            teacher_cache=teacher_cache,
            free_run_token_targets=free_run_token_targets,
            device=device,
            dtype=dtype,
            traj_start=traj_start,
            num_bins=num_bins,
            args=args,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {float(loss.detach().cpu())}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in student.parameters() if parameter.requires_grad],
            max_norm=float(args.grad_clip_norm),
        )
        optimizer.step()

        stats["grad_norm"] = float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        for key, value in stats.items():
            if value is not None and math.isfinite(float(value)):
                running[key].append(float(value))

        if step == 1 or step % max(int(args.log_every), 1) == 0 or step == int(args.max_steps):
            row = {
                "event": "flex_parity_train_step",
                "step": step,
                "metrics": {key: mean(values[-max(int(args.log_every), 1) :]) for key, values in sorted(running.items())},
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)

        if args.save_every and step % int(args.save_every) == 0:
            save_dir = args.output_dir / f"step_{step:06d}"
            save_student_checkpoint(save_dir, student, tokenizer, processor, use_lora=True)

    final_dir = args.output_dir / "final"
    if not args.no_save_final:
        manifest = save_student_checkpoint(final_dir, student, tokenizer, processor, use_lora=True)
    else:
        manifest = {}

    train_config = {
        "args": _jsonable(vars(args) | {"student_model": args.student_model}),
        "teacher_checkpoint_dir": str(args.teacher_checkpoint_dir),
        "student_init_checkpoint_dir": str(args.student_checkpoint_dir),
        "trainer_config": {
            "stage_name": "stage_flex_f1_parity",
            "max_steps": int(args.max_steps),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "bf16": bool(dtype == torch.bfloat16),
            "base_samples": int(base_sample_count),
            "image_ablations": image_ablations,
            "paired_ablation": paired_ablation,
            "free_run_token_targets": int(len(free_run_token_targets)),
            "free_run_token_ce_modes": str(args.free_run_token_ce_modes),
            "free_run_token_force_context": bool(args.free_run_token_force_context),
            "free_run_token_context_source": str(args.free_run_token_context_source),
            "student_greedy_context_refresh_steps": int(args.student_greedy_context_refresh_steps),
            "student_greedy_invalid_context": str(args.student_greedy_invalid_context),
            "preserve_flex_positions": bool(args.preserve_flex_positions),
            "flex_dummy_image_slots": bool(args.flex_dummy_image_slots),
            "flex_residual_image_slots": bool(args.flex_residual_image_slots),
            "flex_residual_scale": float(args.flex_residual_scale),
            "flex_passthrough_image_slots": bool(args.flex_passthrough_image_slots),
            "flex_scene_deepstack": bool(args.flex_scene_deepstack),
            "flex_deepstack_projector_rank": int(args.flex_deepstack_projector_rank),
            "flex_deepstack_projector_dropout": float(args.flex_deepstack_projector_dropout),
        },
        "data_view": data_view,
        "flex": flex_config_dict,
        "flex_deepstack_projector": getattr(student, "flex_deepstack_projector_config", None),
        "optimization": {
            "freeze_all_parameters": True,
            "unfreeze_flex_scene_encoder": bool(args.train_flex),
            "unfreeze_flex_deepstack_projector": bool(args.train_flex_deepstack_projector),
            "unfreeze_lora_last_n_layers": int(args.unfreeze_lora_last_n_layers),
            "unfreeze_all_lora": bool(args.unfreeze_all_lora),
            "unfreeze_multimodal_projector": bool(args.unfreeze_multimodal_projector),
            "learning_rate": float(args.learning_rate),
            "flex_lr": float(args.flex_lr) if args.flex_lr is not None else None,
            "lora_lr": float(args.lora_lr) if args.lora_lr is not None else None,
            "multimodal_projector_lr": (
                float(args.multimodal_projector_lr) if args.multimodal_projector_lr is not None else None
            ),
            "deepstack_projector_lr": (
                float(args.deepstack_projector_lr) if args.deepstack_projector_lr is not None else None
            ),
            "optimizer_param_groups": optimizer_group_summary,
            "trainable_groups": trainable_groups,
            "trainable_params": int(trainable),
            "total_params": int(total_params),
        },
        "loss_weights": {
            "traj_kl": float(args.traj_kl_weight),
            "traj_token_ce": float(args.traj_token_ce_weight),
            "text_kl": float(args.text_kl_weight),
            "format_kl": float(args.format_kl_weight),
            "boundary_cos": float(args.boundary_cos_weight),
            "boundary_norm": float(args.boundary_norm_weight),
            "boundary_mse": float(args.boundary_mse_weight),
            "pairwise_boundary_delta_cos": float(args.pairwise_boundary_delta_cos_weight),
            "pairwise_boundary_delta_norm": float(args.pairwise_boundary_delta_norm_weight),
            "pairwise_traj_logprob_delta": float(args.pairwise_traj_logprob_delta_weight),
            "pairwise_free_run_margin": float(args.pairwise_free_run_margin_weight),
            "pairwise_free_run_margin_value": float(args.pairwise_free_run_margin),
            "free_run_token_ce": float(args.free_run_token_ce_weight),
            "free_run_token_ce_modes": str(args.free_run_token_ce_modes),
            "free_run_end_token_ce": float(args.free_run_end_token_ce_weight),
            "prefix_token_ce": float(args.prefix_token_ce_weight),
            "traj_state_cos": float(args.traj_state_cos_weight),
            "traj_state_norm": float(args.traj_state_norm_weight),
            "traj_state_mse": float(args.traj_state_mse_weight),
            "image_feature_tokens_per_image": int(args.image_feature_tokens_per_image),
            "image_feature_cos": float(args.image_feature_cos_weight),
            "image_feature_norm": float(args.image_feature_norm_weight),
            "image_feature_mse": float(args.image_feature_mse_weight),
            "deepstack_feature_tokens_per_image": int(args.deepstack_feature_tokens_per_image),
            "deepstack_feature_cos": float(args.deepstack_feature_cos_weight),
            "deepstack_feature_norm": float(args.deepstack_feature_norm_weight),
            "deepstack_feature_mse": float(args.deepstack_feature_mse_weight),
        },
        "checkpoint": manifest,
        "history": history,
    }
    args.summary_json.write_text(json.dumps(train_config, indent=2, ensure_ascii=True), encoding="utf-8")
    if not args.no_save_final:
        (final_dir / "train_config.json").write_text(json.dumps(train_config, indent=2, ensure_ascii=True), encoding="utf-8")
        source_config = Path(args.student_checkpoint_dir) / "train_config.json"
        if source_config.exists() and not (args.output_dir / "student_init_train_config.json").exists():
            shutil.copy2(source_config, args.output_dir / "student_init_train_config.json")

    print(
        json.dumps(
            {
                "event": "flex_parity_train_done",
                "output_dir": str(args.output_dir),
                "final_checkpoint_dir": str(final_dir) if not args.no_save_final else None,
                "summary_json": str(args.summary_json),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
