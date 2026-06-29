#!/usr/bin/env python3
"""Train a student-compatible Alpamayo-style AE28 action expert.

This is the formal-compatible action expert path:

  frozen student VLM 2B -> 28-layer student KV cache
  AE28 expert decoder + action_in_proj/action_out_proj
  FlowMatching target in Alpamayo action space [64, 2]

The teacher Alpamayo model is used to provide action_space / flow-matching
utilities, and optionally to initialize the AE modules. Teacher VLM weights may
be loaded on CPU; the student backbone and AE28 trainable modules live on GPU.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import random
import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUKIM_ROOT = PROJECT_ROOT.parents[1]
ALPAMAYO_SRC = SUKIM_ROOT / "alpamayo_repo" / "alpamayo1.5" / "src"
VIS_ROOT = SUKIM_ROOT / "visualization"
for path in (PROJECT_ROOT, SUKIM_ROOT, ALPAMAYO_SRC, VIS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import load_model_and_processor  # noqa: E402
from probe_teacher_kv_28layer_expert_compression import (  # noqa: E402
    ade_fde,
    build_28layer_expert,
    force_attention,
    layer_mapping,
    path_len,
    torch_dtype_from_name,
)
from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint  # noqa: E402
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora  # noqa: E402
from src.model.student_wrapper import StudentWrapperConfig, build_student_model  # noqa: E402
from src.model.tokenizer_ext import distill_trainable_token_ids  # noqa: E402
from src.training.collator import (  # noqa: E402
    _encode_messages,
    build_messages,
    build_user_prompt,
    fuse_history_tokens_in_input_ids,
    load_ego_future_xyz,
    load_ego_history_xyz,
    load_sample_images,
    resolve_camera_indices,
)
from src.training.flex_batch import attach_qwen_mrope_position_ids, compress_batch_for_flex  # noqa: E402
from src.inference.checkpoint_eval import load_ego_future_rot, load_ego_history_rot  # noqa: E402
from src.utils.runtime_paths import (  # noqa: E402
    DEFAULT_MATERIALIZED_ROOT,
    DEFAULT_STATE_ROOT,
    DEFAULT_TEACHER_CACHE_ROOT,
    remap_external_path,
    resolve_student_model_path,
)


DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl"
DEFAULT_TEACHER = SUKIM_ROOT / "base_weights" / "Alpamayo-1.5-10B"
DEFAULT_STUDENT_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "no_nav_camera_labeled_official_200k"
    / "no_nav_official12500_topk_sched16_ar_ramp_p20_rowscale_evalfix_20260517"
    / "best_decode"
)
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "action_expert" / "student_ae28_official"


class StopAfterToken(StoppingCriteria):
    """Stop one decode step after every row has generated a target token.

    The action expert consumes the VLM KV cache at the trajectory-start boundary.
    In HF generation, the sampled token is not guaranteed to be represented in
    the returned cache until the next decode step consumes it. This mirrors the
    official Alpamayo StopAfterEOS behavior: once all rows have emitted the
    boundary token, allow one more generation step so the boundary token is in
    the cache; later tokens are masked out by the expert attention mask.
    """

    def __init__(self, token_id: int, prompt_lengths: list[int]) -> None:
        self.token_id = int(token_id)
        self.prompt_lengths = [int(x) for x in prompt_lengths]
        self.token_found: torch.Tensor | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        batch_size = int(input_ids.shape[0])
        if self.token_found is None or int(self.token_found.numel()) != batch_size:
            self.token_found = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        if bool(self.token_found.all()):
            return True

        last_tokens = input_ids[:, -1]
        just_found = last_tokens == self.token_id
        for row in range(batch_size):
            prompt_len = self.prompt_lengths[min(row, len(self.prompt_lengths) - 1)]
            if int(input_ids.shape[1]) <= prompt_len:
                just_found[row] = False
        self.token_found = self.token_found | just_found
        return False


def select_kv_cache_layers(prompt_cache, kv_layer_indices: list[int]):
    """Create a new DynamicCache with only the selected layers.

    Used when the expert has fewer layers than the backbone KV cache
    (e.g. AE14 expert with 14 layers vs 28-layer backbone cache).
    """
    from transformers.cache_utils import DynamicCache
    new_cache = DynamicCache()
    for new_idx, old_idx in enumerate(kv_layer_indices):
        layer = prompt_cache.layers[old_idx]
        new_cache.update(layer.keys, layer.values, layer_idx=new_idx)
    return new_cache


class AE28Bundle(nn.Module):
    def __init__(self, *, expert: nn.Module, action_in_proj: nn.Module, action_out_proj: nn.Module) -> None:
        super().__init__()
        self.expert = expert
        self.action_in_proj = action_in_proj
        self.action_out_proj = action_out_proj


class ManualGenerateOutput:
    def __init__(self, *, sequences: torch.Tensor, past_key_values: Any) -> None:
        self.sequences = sequences
        self.past_key_values = past_key_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument(
        "--eval-num-paths",
        type=int,
        default=1,
        help=(
            "Number of trajectory samples per eval input (best-of-N diagnostic). "
            "N=1 preserves the existing single-sample eval behavior exactly. "
            "N>1: re-runs sample_paths() with different seeds and adds "
            "ade_best_of_n_*, ade_mean_over_paths_*, ade_std_over_paths_* keys."
        ),
    )
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=1.0,
        help=(
            "FlowMatching inference temperature passed to diffusion.sample(). "
            "Default 1.0 preserves the official sampler behavior."
        ),
    )
    parser.add_argument(
        "--eval-selection-method",
        choices=("single", "oracle_best", "medoid", "mean_traj"),
        default="single",
        help=(
            "Which sampled trajectory is reported as eval ade_m/fde_m. "
            "single preserves legacy path-0 eval; oracle_best is diagnostic only; "
            "medoid and mean_traj are deployable N-path selection/aggregation methods."
        ),
    )
    parser.add_argument(
        "--eval-vectorize-paths",
        action="store_true",
        help=(
            "Evaluate multiple diffusion paths by repeating the eval batch and sampling path chunks "
            "in one call. Faster than the legacy Python loop over paths, but uses more VRAM."
        ),
    )
    parser.add_argument(
        "--eval-path-batch-size",
        type=int,
        default=0,
        help=(
            "When --eval-vectorize-paths is set, number of paths to sample per vectorized chunk. "
            "0 means all --eval-num-paths at once."
        ),
    )
    parser.add_argument(
        "--eval-log-rows",
        type=int,
        default=-1,
        help=(
            "Number of per-sample eval rows to include in JSON logs. -1 keeps all rows "
            "(legacy behavior); 0 logs only aggregates."
        ),
    )
    parser.add_argument(
        "--train-ade-every",
        type=int,
        default=0,
        help=(
            "If > 0, every N training steps, additionally run sample_paths() on the "
            "CURRENT training batch and log in-distribution train_inb_ade_m. "
            "Diagnostic only — does not affect gradients (bundle.eval() + torch.no_grad, "
            "RNG state saved/restored). Default 0 disables (no behavior change)."
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help=(
            "Deterministic group-level held-out fraction within --split. "
            "Groups are split by stable hash of the scene id, so eval samples are never in train."
        ),
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=0,
        help="Held-out validation sample cap. 0 uses --eval-samples.",
    )
    parser.add_argument(
        "--eval-train-samples",
        type=int,
        default=0,
        help="If >0, also run eval on this many train samples at each eval point to track generalization gap.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for deterministic train/val split. Defaults to --seed.",
    )
    parser.add_argument(
        "--split-scan-all",
        action="store_true",
        help="Scan the full corpus before selecting train/val caps. Slower but gives full candidate counts.",
    )
    parser.add_argument(
        "--split-cache-json",
        type=Path,
        default=None,
        help="Optional precomputed train/val split cache. If it exists, load it instead of scanning corpus.",
    )
    parser.add_argument("--student-model", default=resolve_student_model_path())
    parser.add_argument("--student-checkpoint-dir", type=Path, default=DEFAULT_STUDENT_CKPT)
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--disable-student-deepstack",
        action="store_true",
        help="Runtime ablation: clear Qwen3-VL visual DeepStack indexes in the student backbone.",
    )
    parser.add_argument(
        "--preserve-flex-positions",
        action="store_true",
        help="For FLEX student checkpoints, preserve official Qwen MRoPE position ids before compression.",
    )
    parser.add_argument(
        "--flex-selection-strategy",
        choices=("first", "uniform"),
        default="first",
        help="For FLEX student checkpoints, choose which image placeholders survive compression.",
    )
    parser.add_argument(
        "--flex-scene-deepstack",
        action="store_true",
        help="For ML-FLEX student checkpoints, inject compressed scene tokens through Qwen3-VL DeepStack hooks.",
    )
    parser.add_argument(
        "--qat-quantization",
        type=str,
        default="",
        choices=["", "int4_awq", "int4_blockwise", "int4_ffn_only"],
        help="Re-apply ModelOpt fake-quantization to student backbone language_model before AE training.",
    )
    parser.add_argument("--qat-calib-samples", type=int, default=256)
    parser.add_argument(
        "--kv-cache-dir",
        type=str,
        default="",
        help="Directory with pre-computed KV cache .pt files (from precompute_ae_kv_cache.py). "
             "When set, skips student forward entirely and loads KV cache from disk.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--ae-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa")
    parser.add_argument("--teacher-load-device", default="cpu")
    parser.add_argument("--mapping", choices=("linspace_round", "first_n"), default="linspace_round")
    parser.add_argument("--compressed-layers", type=int, default=28)
    parser.add_argument(
        "--prefix-mode",
        choices=("student_free", "teacher_forced"),
        default="student_free",
        help=(
            "student_free generates CoT until <|traj_future_start|> before caching KV. "
            "teacher_forced caches KV from teacher CoT + <|traj_future_start|> directly."
        ),
    )
    parser.add_argument(
        "--ae-init-mode",
        choices=("teacher_compressed", "scratch", "student_backbone_init", "student_backbone_init_teacher_q", "ae_checkpoint_compressed"),
        default="teacher_compressed",
        help=(
            "teacher_compressed copies selected teacher expert layers and action projections. "
            "scratch keeps the Alpamayo-compatible AE structure but randomly initializes "
            "expert/action projection weights for student-KV training. "
            "ae_checkpoint_compressed loads a trained AE checkpoint (--init-ae-source-checkpoint) "
            "and selects --compressed-layers from its expert layers (e.g. AE28->AE14)."
        ),
    )
    parser.add_argument(
        "--init-ae-source-checkpoint",
        type=str,
        default="",
        help="Path to a trained AE checkpoint (.pt) to compress layers from. Used with ae_checkpoint_compressed.",
    )
    parser.add_argument(
        "--target-source",
        choices=("teacher", "gt"),
        default="teacher",
        help=(
            "Flow-matching regression target source. 'teacher' uses raw_json pred_xyz/pred_rot "
            "(default, preserves existing behavior). 'gt' uses canonicalized ego_future_xyz/rot "
            "from sample directory (ego-local at t0)."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--num-time-samples", type=int, default=16)
    parser.add_argument(
        "--train-timestep-sampler",
        choices=("uniform", "beta"),
        default="beta",
        help=(
            "Flow-matching training timestep sampler. Alpamayo base Stage-2 uses "
            "beta with t = 0.999 - Beta(1.5, 1.0) * 0.999."
        ),
    )
    parser.add_argument(
        "--stage2-attention-mode",
        choices=("official_none", "masked"),
        default="official_none",
        help=(
            "official_none matches alpamayo_base Stage-2 TrainableAlpamayoR1, "
            "which calls the expert with attention_mask=None. masked keeps the "
            "older local inference-style expert attention mask."
        ),
    )
    parser.add_argument("--expert-lr", type=float, default=1e-4)
    parser.add_argument("--proj-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--lr-warmup-steps", type=int, default=0,
        help="Linear warmup steps for cosine schedule. 0 disables the schedule (constant LR).")
    parser.add_argument("--min-lr", type=float, default=1e-6,
        help="Minimum learning rate at end of cosine decay.")
    parser.add_argument("--no-norm-bias-decay", action="store_true",
        help="Skip weight decay for biases, LayerNorm/RMSNorm scales (matches alpamayo SFT).")
    parser.add_argument(
        "--allow-train-cache-mutation",
        action="store_true",
        help=(
            "Skip deepcopy() before repeating the prompt KV cache in train_step. "
            "Faster, but incompatible with train_inb_ade diagnostics that reuse the same batch cache."
        ),
    )
    parser.add_argument(
        "--fused-adamw",
        action="store_true",
        help="Use torch.optim.AdamW(..., fused=True). Requires CUDA parameters and compatible PyTorch.",
    )
    parser.add_argument("--train-backbone-lora", action="store_true",
        help="Joint-train student backbone LoRA params. Only valid in teacher_forced mode.")
    parser.add_argument("--backbone-lora-lr", type=float, default=5e-6,
        help="Learning rate for student backbone LoRA params when joint-trained.")
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument(
        "--eval-seed-mode",
        choices=("fixed", "step"),
        default="step",
        help=(
            "Use a constant diffusion sampling seed at every eval, or include "
            "the training step in the eval seed. `fixed` is the right setting "
            "for overfit/reconstruction sanity checks."
        ),
    )
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument(
        "--resume-ae-checkpoint",
        type=Path,
        default=None,
        help=(
            "Load an action-expert checkpoint saved by this script before training. "
            "Only bundle weights are restored; optimizer state is intentionally not restored."
        ),
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=None,
        help=(
            "Absolute step to resume counting from. Defaults to the checkpoint payload step "
            "when --resume-ae-checkpoint is provided, otherwise 0."
        ),
    )
    parser.add_argument(
        "--cleanup-every",
        type=int,
        default=1,
        help="Run gc.collect()/torch.cuda.empty_cache() every N train steps. 0 disables step cleanup.",
    )
    parser.add_argument(
        "--eval-cleanup-every",
        type=int,
        default=1,
        help="Run gc.collect()/torch.cuda.empty_cache() every N eval path chunks. 0 disables eval cleanup.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load/build the bundle, run val/train eval once, write logs, and exit without training.",
    )
    parser.add_argument(
        "--eval-sweep-json",
        default=None,
        help=(
            "Optional JSON list of eval-only overrides. Each item may set label, "
            "eval_temperature, eval_num_paths, and eval_selection_method. This reuses "
            "one loaded model for cheap inference sweeps."
        ),
    )
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    return parser.parse_args()


def _resolve_path(raw: str | Path | None) -> Path | None:
    remapped = remap_external_path(raw)
    if remapped in (None, ""):
        return None
    path = Path(remapped)
    return path if path.exists() else None


def _resolve_path_no_stat(raw: str | Path | None) -> Path | None:
    if raw in (None, ""):
        return None
    path_str = str(raw)
    replacements = (
        ("/data/materialized", str(DEFAULT_MATERIALIZED_ROOT)),
        ("/data/teacher_cache", str(DEFAULT_TEACHER_CACHE_ROOT)),
        ("/data/state", str(DEFAULT_STATE_ROOT)),
        ("/workspace/sukim/alpamayo_teacher_prep/materialized", str(DEFAULT_MATERIALIZED_ROOT)),
        ("/workspace/sukim/alpamayo_teacher_prep/teacher_cache", str(DEFAULT_TEACHER_CACHE_ROOT)),
    )
    for old_prefix, new_prefix in replacements:
        if path_str.startswith(old_prefix):
            path_str = path_str.replace(old_prefix, new_prefix, 1)
            break
    return Path(path_str).expanduser()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_raw_json(record: dict[str, Any]) -> Path | None:
    raw = ((record.get("teacher_cache") or {}).get("text_raw_json_path"))
    return _resolve_path(raw)


def select_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    scanned = 0
    for row in iter_jsonl(args.corpus_jsonl):
        scanned += 1
        if args.split and row.get("split") != args.split:
            continue
        raw_path = _resolve_path_no_stat((row.get("teacher_cache") or {}).get("text_raw_json_path"))
        sample_dir = _resolve_path_no_stat((row.get("input") or {}).get("materialized_sample_path"))
        if raw_path is None or sample_dir is None:
            continue
        items.append(
            {
                "sample_id": str(row["sample_id"]),
                "row": row,
                "sample_dir": str(sample_dir),
                "raw_json": str(raw_path),
            }
        )
        if len(items) >= int(args.num_samples):
            break
    if not items:
        raise RuntimeError("No usable AE28 samples found.")
    print(
        json.dumps(
            {
                "event": "select_items_done",
                "selected_count": len(items),
                "scanned_count": scanned,
                "corpus_jsonl": str(args.corpus_jsonl),
            }
        ),
        flush=True,
    )
    return items


def split_group_id(sample_id: str) -> str:
    return str(sample_id).split("__sg_", 1)[0]


def stable_unit_hash(*, seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{int(seed)}:{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(1 << 64)


def make_item(row: dict[str, Any], *, raw_path: Path, sample_dir: Path) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    return {
        "sample_id": sample_id,
        "split_group_id": split_group_id(sample_id),
        "row": row,
        "sample_dir": str(sample_dir),
        "raw_json": str(raw_path),
    }


def select_train_val_items(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    split_cache_json = getattr(args, "split_cache_json", None)
    if split_cache_json is not None:
        split_cache_path = Path(split_cache_json)
        if split_cache_path.exists():
            payload = json.loads(split_cache_path.read_text(encoding="utf-8"))
            train_items = list(payload["train_items"])
            val_items = list(payload["val_items"])
            summary = dict(payload["summary"])
            summary.update(
                {
                    "event": "train_val_split_done",
                    "loaded_from_cache": True,
                    "split_cache_json": str(split_cache_path),
                    "train_selected_count": len(train_items),
                    "val_selected_count": len(val_items),
                }
            )
            train_ids = {item["sample_id"] for item in train_items}
            val_ids = {item["sample_id"] for item in val_items}
            train_groups = {item["split_group_id"] for item in train_items}
            val_groups = {item["split_group_id"] for item in val_items}
            summary["sample_id_overlap_count"] = len(train_ids & val_ids)
            summary["split_group_overlap_count"] = len(train_groups & val_groups)
            if summary["sample_id_overlap_count"] or summary["split_group_overlap_count"]:
                raise RuntimeError(f"Cached train/val split overlap detected: {summary}")
            print(json.dumps(summary), flush=True)
            return train_items, val_items, summary

    train_items: list[dict[str, Any]] = []
    val_items: list[dict[str, Any]] = []
    scanned = 0
    eligible = 0
    train_candidates = 0
    val_candidates = 0
    train_target = int(args.num_samples)
    val_target = int(args.val_samples) if int(args.val_samples) > 0 else int(args.eval_samples)
    split_seed = int(args.split_seed if args.split_seed is not None else args.seed)
    val_fraction = float(args.val_fraction)
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"--val-fraction must be in (0, 1), got {val_fraction}")

    for row in iter_jsonl(args.corpus_jsonl):
        scanned += 1
        if args.split and row.get("split") != args.split:
            continue
        raw_path = resolve_raw_json(row)
        sample_dir = _resolve_path((row.get("input") or {}).get("materialized_sample_path"))
        if raw_path is None or sample_dir is None:
            continue
        eligible += 1
        item = make_item(row, raw_path=raw_path, sample_dir=sample_dir)
        group_id = str(item["split_group_id"])
        is_val = stable_unit_hash(seed=split_seed, key=group_id) < val_fraction
        if is_val:
            val_candidates += 1
            if len(val_items) < val_target:
                val_items.append(item)
        else:
            train_candidates += 1
            if len(train_items) < train_target:
                train_items.append(item)
        if (
            not bool(args.split_scan_all)
            and len(train_items) >= train_target
            and len(val_items) >= val_target
        ):
            break

    if len(train_items) < train_target:
        raise RuntimeError(f"Not enough train items: requested={train_target} got={len(train_items)}")
    if len(val_items) < val_target:
        raise RuntimeError(f"Not enough val items: requested={val_target} got={len(val_items)}")

    train_ids = {item["sample_id"] for item in train_items}
    val_ids = {item["sample_id"] for item in val_items}
    train_groups = {item["split_group_id"] for item in train_items}
    val_groups = {item["split_group_id"] for item in val_items}
    sample_overlap = sorted(train_ids & val_ids)
    group_overlap = sorted(train_groups & val_groups)
    summary = {
        "event": "train_val_split_done",
        "corpus_jsonl": str(args.corpus_jsonl),
        "source_split": str(args.split),
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "split_group_key": "sample_id before __sg_",
        "split_scan_all": bool(args.split_scan_all),
        "path_exists_checked": False,
        "scanned_count": scanned,
        "eligible_count": eligible,
        "train_candidate_count": train_candidates,
        "val_candidate_count": val_candidates,
        "train_selected_count": len(train_items),
        "val_selected_count": len(val_items),
        "train_group_count": len(train_groups),
        "val_group_count": len(val_groups),
        "sample_id_overlap_count": len(sample_overlap),
        "split_group_overlap_count": len(group_overlap),
        "sample_id_overlap_head": sample_overlap[:16],
        "split_group_overlap_head": group_overlap[:16],
        "train_sample_ids_head": [item["sample_id"] for item in train_items[:8]],
        "val_sample_ids_head": [item["sample_id"] for item in val_items[:8]],
    }
    if sample_overlap or group_overlap:
        raise RuntimeError(f"Train/val split overlap detected: {summary}")
    if split_cache_json is not None:
        split_cache_path = Path(split_cache_json)
        split_cache_path.parent.mkdir(parents=True, exist_ok=True)
        split_cache_path.write_text(
            json.dumps(
                {"train_items": train_items, "val_items": val_items, "summary": summary},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        summary["split_cache_json"] = str(split_cache_path)
    print(json.dumps(summary), flush=True)
    return train_items, val_items, summary


def raw_teacher_pred(raw_json: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(raw_json.read_text(encoding="utf-8"))
    result = (payload.get("results") or [None])[0]
    if not isinstance(result, dict):
        raise ValueError(f"Missing results[0] in {raw_json}")
    xyz = np.asarray(result.get("pred_xyz"), dtype=np.float32).reshape(-1, 64, 3)[0]
    rot = np.asarray(result.get("pred_rot"), dtype=np.float32).reshape(-1, 64, 3, 3)[0]
    return xyz, rot


def _unwrap_singleton_text(value: Any) -> str:
    while isinstance(value, list) and value:
        value = value[0]
    return str(value or "").strip()


def teacher_cot_text(item: dict[str, Any]) -> str:
    row = item["row"]
    for section_name in ("teacher_target", "hard_target"):
        text = _unwrap_singleton_text((row.get(section_name) or {}).get("cot_text"))
        if text:
            return text
    try:
        payload = json.loads(Path(item["raw_json"]).read_text(encoding="utf-8"))
        result = (payload.get("results") or [None])[0]
        if isinstance(result, dict):
            text = _unwrap_singleton_text((result.get("extra") or {}).get("cot"))
            if text:
                return text
    except Exception:  # noqa: BLE001
        pass
    raise ValueError(f"Missing teacher CoT for sample {item.get('sample_id')}")


def normalize_history_rot(rot: np.ndarray) -> np.ndarray:
    """Normalize materialized history rotations to [T, 3, 3]."""
    arr = np.asarray(rot, dtype=np.float32)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-2:] != (3, 3):
        raise ValueError(f"Expected ego_history_rot as [T,3,3] after squeeze, got shape={arr.shape}")
    return arr


def _to_device_batch(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {key: _to_device_batch(value, device) for key, value in batch.items()}
    return batch


def unwrap_backbone(backbone: nn.Module) -> nn.Module:
    if hasattr(backbone, "get_base_model"):
        return backbone.get_base_model()
    return backbone


def get_rope_deltas(backbone: nn.Module) -> torch.Tensor:
    candidates = []
    base = unwrap_backbone(backbone)
    candidates.extend([base, getattr(base, "model", None), getattr(getattr(base, "base_model", None), "model", None)])
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "rope_deltas"):
            value = getattr(candidate, "rope_deltas")
            if value is not None:
                return value
    raise AttributeError("Could not find Qwen/Cosmos rope_deltas after student generate().")


def _student_logits_and_cache(output: Any) -> tuple[torch.Tensor, Any]:
    if isinstance(output, Mapping):
        logits = output["logits"]
        backbone_outputs = output.get("backbone_outputs")
        cache = getattr(backbone_outputs, "past_key_values", None)
        if cache is None:
            cache = output.get("past_key_values")
        return logits, cache
    return output.logits, getattr(output, "past_key_values", None)


def _past_seq_len(past_key_values: Any) -> int | None:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    try:
        return int(past_key_values[0][0].shape[-2])
    except Exception:  # noqa: BLE001
        return None


def _position_delta_from_prefill(batch: dict[str, Any]) -> torch.Tensor | None:
    position_ids = batch.get("position_ids")
    attention_mask = batch.get("attention_mask")
    if not isinstance(position_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        return None
    row_positions = position_ids[0] if position_ids.ndim == 3 else position_ids
    valid = attention_mask.to(dtype=torch.bool)
    if row_positions.ndim != 2 or valid.ndim != 2 or tuple(row_positions.shape) != tuple(valid.shape):
        return None
    deltas: list[torch.Tensor] = []
    for row_index in range(int(row_positions.shape[0])):
        row_valid = valid[row_index]
        if bool(row_valid.any().item()):
            max_position = row_positions[row_index, row_valid].max() + 1
            valid_len = row_valid.long().sum()
            deltas.append(max_position.to(dtype=torch.long) - valid_len.to(dtype=torch.long))
        else:
            deltas.append(torch.zeros((), dtype=torch.long, device=row_positions.device))
    return torch.stack(deltas).to(device=row_positions.device, dtype=torch.long)


def manual_flex_generate_with_cache(
    *,
    student: Any,
    input_ids: torch.Tensor,
    model_kwargs: dict[str, Any],
    max_new_tokens: int,
    stopping_criteria: StoppingCriteriaList,
) -> ManualGenerateOutput:
    """Greedy generation through StudentWrapper FLEX so compressed prompts are used."""
    generated = input_ids.clone()
    attention_mask = model_kwargs["attention_mask"].clone()
    position_delta = _position_delta_from_prefill(model_kwargs)
    prefill_keys = (
        "attention_mask",
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
        "flex_selection_strategy",
        "flex_scene_deepstack",
    )
    prefill_kwargs = {key: model_kwargs[key] for key in prefill_keys if key in model_kwargs}
    prefill_output = student(
        input_ids=input_ids,
        **prefill_kwargs,
        use_cache=True,
        return_dict=True,
        logits_to_keep=1,
        return_hidden_states=False,
        compute_meta_action=False,
        compute_traj_aux=False,
    )
    logits, cache = _student_logits_and_cache(prefill_output)
    if cache is None:
        raise RuntimeError("FLEX student-free prefill did not return past_key_values.")

    for _ in range(max(int(max_new_tokens), 0)):
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)],
            dim=1,
        )
        should_stop = stopping_criteria(generated, logits[:, -1, :])
        if isinstance(should_stop, torch.Tensor):
            should_stop = bool(torch.all(should_stop).item())
        if bool(should_stop):
            break

        decode_kwargs: dict[str, Any] = {
            "input_ids": next_token,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "use_cache": True,
            "return_dict": True,
            "logits_to_keep": 1,
            "return_hidden_states": False,
            "compute_meta_action": False,
            "compute_traj_aux": False,
        }
        past_len = _past_seq_len(cache)
        if past_len is not None:
            cache_position = torch.arange(
                past_len,
                past_len + int(next_token.shape[1]),
                device=next_token.device,
                dtype=torch.long,
            )
            decode_kwargs["cache_position"] = cache_position
            if isinstance(position_delta, torch.Tensor):
                batch_size = int(next_token.shape[0])
                delta = position_delta.to(device=next_token.device, dtype=torch.long).reshape(-1)
                if int(delta.numel()) == 1 and batch_size > 1:
                    delta = delta.expand(batch_size)
                if int(delta.numel()) == batch_size:
                    decode_kwargs["position_ids"] = (
                        cache_position.view(1, 1, -1).expand(3, batch_size, -1)
                        + delta.view(1, batch_size, 1)
                    )
        decode_output = student(**decode_kwargs)
        logits, cache = _student_logits_and_cache(decode_output)
        if cache is None:
            raise RuntimeError("FLEX student-free decode did not return past_key_values.")

    return ManualGenerateOutput(sequences=generated, past_key_values=cache)


def disable_qwen_deepstack(module: Any) -> list[dict[str, Any]]:
    disabled: list[dict[str, Any]] = []
    seen: set[int] = set()

    def visit(name: str, obj: Any) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        if hasattr(obj, "deepstack_visual_indexes"):
            old = list(getattr(obj, "deepstack_visual_indexes") or [])
            setattr(obj, "deepstack_visual_indexes", [])
            disabled.append({"target": name, "old_indexes": old})
        cfg = getattr(obj, "config", None)
        if cfg is not None and hasattr(cfg, "deepstack_visual_indexes"):
            old = list(getattr(cfg, "deepstack_visual_indexes") or [])
            setattr(cfg, "deepstack_visual_indexes", [])
            disabled.append({"target": f"{name}.config", "old_indexes": old})
        vision_cfg = getattr(cfg, "vision_config", None) if cfg is not None else None
        if vision_cfg is not None and hasattr(vision_cfg, "deepstack_visual_indexes"):
            old = list(getattr(vision_cfg, "deepstack_visual_indexes") or [])
            setattr(vision_cfg, "deepstack_visual_indexes", [])
            disabled.append({"target": f"{name}.config.vision_config", "old_indexes": old})

    candidates = [
        ("student", module),
        ("student.backbone", getattr(module, "backbone", None)),
        ("student.backbone.model", getattr(getattr(module, "backbone", None), "model", None)),
        (
            "student.backbone.model.visual",
            getattr(getattr(getattr(module, "backbone", None), "model", None), "visual", None),
        ),
        (
            "student.backbone.model.model.visual",
            getattr(getattr(getattr(getattr(module, "backbone", None), "model", None), "model", None), "visual", None),
        ),
    ]
    for name, obj in candidates:
        visit(name, obj)
    return disabled


def load_student(args: argparse.Namespace):
    checkpoint_dir = args.student_checkpoint_dir
    train_config_path = checkpoint_dir / "train_config.json"
    train_config = json.loads(train_config_path.read_text(encoding="utf-8")) if train_config_path.exists() else {}
    checkpoint_manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    checkpoint_manifest = (
        json.loads(checkpoint_manifest_path.read_text(encoding="utf-8")) if checkpoint_manifest_path.exists() else {}
    )
    base_model = str((train_config.get("args") or {}).get("student_model") or args.student_model)
    use_lora = not bool((train_config.get("args") or {}).get("disable_lora", False))
    tokenizer_dir = checkpoint_dir / "tokenizer"
    processor_dir = checkpoint_dir / "processor"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir if tokenizer_dir.exists() else base_model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(processor_dir if processor_dir.exists() else base_model, local_files_only=True)
    processor.tokenizer = tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    data_view = train_config.get("data_view") or {}
    wrapper_cfg = StudentWrapperConfig(
        student_model_name=base_model,
        max_length=int((train_config.get("trainer_config") or {}).get("max_length", args.max_length)),
        torch_dtype=torch_dtype_from_name(args.student_dtype),
        local_files_only=Path(base_model).expanduser().exists(),
        attn_implementation=args.attn_implementation,
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
    print(json.dumps({"event": "load_student_start", "checkpoint": str(checkpoint_dir), "base_model": base_model}), flush=True)
    model = build_student_model(wrapper_cfg, tokenizer)
    checkpoint_format = detect_checkpoint_format(checkpoint_dir)
    if checkpoint_format == "full_state_dict" and use_lora:
        model.backbone = maybe_apply_lora(
            model.backbone,
            LoraConfigSpec(trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer))),
            enabled=True,
        )
    load_info = load_student_checkpoint(checkpoint_dir, model, use_lora=use_lora, adapter_trainable=False)
    if bool(getattr(args, "disable_student_deepstack", False)):
        disabled = disable_qwen_deepstack(model)
        print(json.dumps({"event": "student_deepstack_disabled", "targets": disabled}), flush=True)

    # ── QAT: Re-apply ModelOpt quantization for AE training on quantized backbone ──
    _ae_qat = str(getattr(args, "qat_quantization", "") or "").strip().lower()
    if _ae_qat:
        import modelopt.torch.quantization as mtq

        import copy as _copy
        _INT4_FFN_ONLY_CFG = _copy.deepcopy(mtq.INT4_AWQ_CFG)
        _INT4_FFN_ONLY_CFG["quant_cfg"]["*q_proj*"] = {"enable": False}
        _INT4_FFN_ONLY_CFG["quant_cfg"]["*k_proj*"] = {"enable": False}
        _INT4_FFN_ONLY_CFG["quant_cfg"]["*v_proj*"] = {"enable": False}
        _INT4_FFN_ONLY_CFG["quant_cfg"]["*o_proj*"] = {"enable": False}
        _INT4_FFN_ONLY_CFG["quant_cfg"]["*q_norm*"] = {"enable": False}
        _INT4_FFN_ONLY_CFG["quant_cfg"]["*k_norm*"] = {"enable": False}
        _ae_qat_cfgs = {
            "int4_awq": mtq.INT4_AWQ_CFG,
            "int4_blockwise": getattr(mtq, "INT4_BLOCKWISE_WEIGHT_ONLY_CFG", mtq.INT4_AWQ_CFG),
            "int4_ffn_only": _INT4_FFN_ONLY_CFG,
        }
        _ae_qat_cfg = _ae_qat_cfgs.get(_ae_qat)
        if _ae_qat_cfg is None:
            raise ValueError(f"Unknown --qat-quantization for AE: {_ae_qat!r}")

        print(json.dumps({"event": "ae_qat_quantize_start", "quantization": _ae_qat}), flush=True)

        # Navigate to language_model only
        _ae_backbone = model.backbone
        _ae_unwrapped = _ae_backbone
        if hasattr(_ae_unwrapped, "base_model"):
            _ae_unwrapped = _ae_unwrapped.base_model
        if hasattr(_ae_unwrapped, "model"):
            _ae_unwrapped = _ae_unwrapped.model
        _ae_qwen = getattr(_ae_unwrapped, "model", _ae_unwrapped)
        _ae_lang = getattr(_ae_qwen, "language_model", None)
        if _ae_lang is None:
            raise RuntimeError("Could not find language_model for AE QAT.")

        _ae_calib_samples = int(getattr(args, "qat_calib_samples", 256) or 256)
        _ae_calib_ok = 0

        def _ae_calib_fn(_lang):
            nonlocal _ae_calib_ok
            # Minimal calibration — AE student is frozen, just need quantizer ranges
            _lang.eval()

        # Save VisionAttention's original attention functions BEFORE quantize
        # ModelOpt patches ALL_ATTENTION_FUNCTIONS at the class level, which bleeds
        # into VisionAttention if it shares the same dict via inheritance.
        _visual_model = getattr(_ae_qwen, "visual", None)
        _vision_attn_originals: list[tuple[Any, dict]] = []
        if _visual_model is not None:
            for _vm_name, _vm_mod in _visual_model.named_modules():
                _aaf = getattr(_vm_mod, "ALL_ATTENTION_FUNCTIONS", None)
                if isinstance(_aaf, dict) and _aaf:
                    _vision_attn_originals.append((_vm_mod, dict(_aaf)))

        _ae_qwen.language_model = mtq.quantize(_ae_lang, _ae_qat_cfg, forward_loop=_ae_calib_fn)

        # Restore VisionAttention's original attention functions
        # (ModelOpt's class-level patch may have overwritten them)
        for _vm_mod, _orig_aaf in _vision_attn_originals:
            if hasattr(_vm_mod, "ALL_ATTENTION_FUNCTIONS"):
                _vm_mod.ALL_ATTENTION_FUNCTIONS.update(_orig_aaf)

        # Disable LoRA quantizers
        for _n, _m in _ae_backbone.named_modules():
            if "lora_" in _n:
                for _a in ("weight_quantizer", "input_quantizer", "output_quantizer"):
                    _q = getattr(_m, _a, None)
                    if _q is not None and hasattr(_q, "disable"):
                        _q.disable()

        _ae_lang_q = sum(
            1 for _, m in _ae_backbone.named_modules()
            if hasattr(m, "weight_quantizer")
            and getattr(m.weight_quantizer, "is_enabled", False)
            and "language_model" in str(type(m).__module__ or "")
        )
        print(json.dumps({"event": "ae_qat_quantize_done", "language_quantizers": _ae_lang_q}), flush=True)

    model.to(args.device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(
        json.dumps(
            {
                "event": "load_student_done",
                "checkpoint_format": checkpoint_format,
                "load_format": load_info.get("format"),
                "device": args.device,
            }
        ),
        flush=True,
    )
    return model, tokenizer, processor, base_model


def reset_module_parameters(module: nn.Module) -> None:
    """Reset trainable parameters while preserving structural buffers."""
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()
        elif hasattr(child, "weight") and child.__class__.__name__ == "RMSNorm":
            with torch.no_grad():
                child.weight.fill_(1.0)


def set_module_requires_grad(module: nn.Module, enabled: bool = True) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def trainable_module_summary(module: nn.Module, *, prefix: str) -> dict[str, Any]:
    rows = [
        {"name": f"{prefix}.{name}", "numel": int(param.numel())}
        for name, param in module.named_parameters()
        if param.requires_grad
    ]
    return {
        "module": prefix,
        "trainable_param_tensors": len(rows),
        "trainable_params": int(sum(row["numel"] for row in rows)),
        "trainable_names": [row["name"] for row in rows],
    }


def optimizer_membership_summary(
    opt_groups: list[dict[str, Any]],
    modules: dict[str, nn.Module],
) -> dict[str, Any]:
    param_to_lrs: dict[int, list[float]] = {}
    for group in opt_groups:
        lr = float(group["lr"])
        for param in group["params"]:
            param_to_lrs.setdefault(id(param), []).append(lr)
    out: dict[str, Any] = {}
    for module_name, module in modules.items():
        rows = []
        total = 0
        included = 0
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            numel = int(param.numel())
            total += numel
            lrs = sorted(set(param_to_lrs.get(id(param), [])))
            if lrs:
                included += numel
            rows.append(
                {
                    "name": f"{module_name}.{name}",
                    "numel": numel,
                    "optimizer_lrs": lrs,
                }
            )
        out[module_name] = {
            "trainable_params": total,
            "optimizer_included_params": included,
            "missing_from_optimizer_params": total - included,
            "params": rows,
        }
    return out


def snapshot_trainable_params(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().float().cpu().clone()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def param_delta_summary(before: dict[str, torch.Tensor], module: nn.Module) -> dict[str, Any]:
    rows = []
    changed_elems = 0
    total_elems = 0
    max_abs_delta = 0.0
    for name, old_value in before.items():
        param = dict(module.named_parameters())[name].detach().float().cpu()
        diff = (param - old_value).abs()
        changed = int((diff > 0).sum().item())
        total = int(diff.numel())
        max_delta = float(diff.max().item()) if total else 0.0
        changed_elems += changed
        total_elems += total
        max_abs_delta = max(max_abs_delta, max_delta)
        rows.append(
            {
                "name": name,
                "numel": total,
                "changed_elems": changed,
                "max_abs_delta": max_delta,
                "mean_abs_delta": float(diff.mean().item()) if total else 0.0,
            }
        )
    return {
        "tensor_count": len(rows),
        "changed_elems": changed_elems,
        "total_elems": total_elems,
        "max_abs_delta": max_abs_delta,
        "rows": rows,
    }


def build_scratch_expert(
    *,
    teacher_expert: nn.Module,
    compressed_layers: int,
    dtype: torch.dtype,
    device: str,
    attn_implementation: str,
) -> nn.Module:
    new_config = copy.deepcopy(teacher_expert.config)
    new_config.num_hidden_layers = int(compressed_layers)
    if hasattr(new_config, "layer_types") and getattr(new_config, "layer_types") is not None:
        new_config.layer_types = list(getattr(new_config, "layer_types"))[: int(compressed_layers)]
    if hasattr(new_config, "_attn_implementation"):
        new_config._attn_implementation = attn_implementation
    if hasattr(new_config, "attn_implementation"):
        new_config.attn_implementation = attn_implementation

    expert = AutoModel.from_config(new_config)
    if hasattr(expert, "embed_tokens"):
        del expert.embed_tokens
    expert = expert.to(device=device, dtype=dtype).train()
    force_attention(expert, attn_implementation)
    return expert



def _merged_layer_state_dict(layer: nn.Module) -> dict:
    """Extract state dict from a (possibly LoRA-wrapped) transformer layer.

    For LoRA layers, merges base_layer.weight + lora_B @ lora_A * scaling
    to produce the effective weight. Non-LoRA parameters are copied as-is.
    """
    merged: dict = {}
    for param_name, param in layer.named_parameters():
        parts = param_name.split(".")
        # e.g. self_attn.q_proj.base_layer.weight -> canonical: self_attn.q_proj.weight
        if "base_layer" in parts:
            idx = parts.index("base_layer")
            canonical = ".".join(parts[:idx] + parts[idx + 1:])
            # find the parent LoRA module to compute merged weight
            parent = layer
            for p in parts[:idx]:
                parent = getattr(parent, p)
            # parent is now the lora Linear module
            if hasattr(parent, "lora_A") and hasattr(parent, "lora_B"):
                # compute merged: base + lora_B @ lora_A * scaling
                base_w = param.data.float()
                for adapter_name in parent.lora_A:
                    scale = parent.scaling.get(adapter_name, 1.0)
                    lora_a = parent.lora_A[adapter_name].weight.data.float()
                    lora_b = parent.lora_B[adapter_name].weight.data.float()
                    base_w = base_w + (lora_b @ lora_a) * scale
                merged[canonical] = base_w.to(param.dtype)
            else:
                merged[canonical] = param.data
        elif any(x in parts for x in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "scaling")):
            # skip raw LoRA delta tensors — already merged above via base_layer path
            pass
        else:
            merged[param_name] = param.data
    return merged


def build_student_backbone_expert(
    *,
    student: Any,
    dtype: torch.dtype,
    device: str,
    attn_implementation: str,
) -> nn.Module:
    """Init AE expert from student backbone transformer layers.

    Student backbone has 28 layers / hidden_size=2048 matching teacher expert dims.
    Q/K/V weights are calibrated for student KV, avoiding uniform-attention collapse
    that random init causes over 3000+ token KV caches.
    LoRA adapters (if present) are merged into base weights during copy.
    """
    student_lm = student.backbone.model.language_model
    new_config = copy.deepcopy(student_lm.config)
    if hasattr(new_config, "_attn_implementation"):
        new_config._attn_implementation = attn_implementation
    if hasattr(new_config, "attn_implementation"):
        new_config.attn_implementation = attn_implementation
    expert = AutoModel.from_config(new_config)
    if hasattr(expert, "embed_tokens"):
        del expert.embed_tokens
    with torch.no_grad():
        for i, src_layer in enumerate(student_lm.layers):
            sd = _merged_layer_state_dict(src_layer)
            expert.layers[i].load_state_dict(sd, strict=True)
        # norm has no LoRA — direct copy
        norm_sd = {k: v.data for k, v in student_lm.norm.named_parameters()}
        expert.norm.load_state_dict(norm_sd, strict=True)
    expert = expert.to(device=device, dtype=dtype).train()
    force_attention(expert, attn_implementation)
    return expert


def build_bundle(teacher_model: Any, args: argparse.Namespace, student: Any = None) -> tuple[AE28Bundle, list[int]]:
    ae_dtype = torch_dtype_from_name(args.ae_dtype)
    selected = layer_mapping(
        int(teacher_model.expert.config.num_hidden_layers),
        int(args.compressed_layers),
        args.mapping,
    )
    expert_attn = str(args.attn_implementation)
    if args.ae_init_mode == "teacher_compressed":
        expert = build_28layer_expert(
            teacher_expert=teacher_model.expert,
            selected_old_indices=selected,
            dtype=ae_dtype,
            device=args.device,
            attn_implementation=expert_attn,
        ).train()
        action_in_proj = copy.deepcopy(teacher_model.action_in_proj).to(device=args.device, dtype=ae_dtype).train()
        action_out_proj = copy.deepcopy(teacher_model.action_out_proj).to(device=args.device, dtype=ae_dtype).train()
    elif args.ae_init_mode == "scratch":
        expert = build_scratch_expert(
            teacher_expert=teacher_model.expert,
            compressed_layers=int(args.compressed_layers),
            dtype=ae_dtype,
            device=args.device,
            attn_implementation=expert_attn,
        )
        action_in_proj = copy.deepcopy(teacher_model.action_in_proj).to(device=args.device, dtype=ae_dtype).train()
        action_out_proj = copy.deepcopy(teacher_model.action_out_proj).to(device=args.device, dtype=ae_dtype).train()
        reset_module_parameters(action_in_proj)
        reset_module_parameters(action_out_proj)
    elif args.ae_init_mode == "student_backbone_init":
        if student is None:
            raise ValueError("student_backbone_init requires student model passed to build_bundle()")
        expert = build_student_backbone_expert(
            student=student,
            dtype=ae_dtype,
            device=args.device,
            attn_implementation=expert_attn,
        ).train()
        action_in_proj = copy.deepcopy(teacher_model.action_in_proj).to(device=args.device, dtype=ae_dtype).train()
        action_out_proj = copy.deepcopy(teacher_model.action_out_proj).to(device=args.device, dtype=ae_dtype).train()
        reset_module_parameters(action_in_proj)
        reset_module_parameters(action_out_proj)
    elif args.ae_init_mode == "student_backbone_init_teacher_q":
        if student is None:
            raise ValueError("student_backbone_init_teacher_q requires student model passed to build_bundle()")
        expert = build_student_backbone_expert(
            student=student,
            dtype=ae_dtype,
            device=args.device,
            attn_implementation=expert_attn,
        ).train()
        # Override q_proj from teacher expert layers (first_n mapping)
        teacher_layers = teacher_model.expert.layers
        n_layers = len(expert.layers)
        if len(teacher_layers) < n_layers:
            raise RuntimeError(
                f"Teacher expert has {len(teacher_layers)} layers, need at least {n_layers}"
            )
        with torch.no_grad():
            for new_idx in range(n_layers):
                t_q = teacher_layers[new_idx].self_attn.q_proj
                s_q = expert.layers[new_idx].self_attn.q_proj
                if t_q.weight.shape != s_q.weight.shape:
                    raise RuntimeError(
                        f"Q proj shape mismatch at layer {new_idx}: "
                        f"teacher={tuple(t_q.weight.shape)} student={tuple(s_q.weight.shape)}"
                    )
                s_q.weight.copy_(t_q.weight.to(device=args.device, dtype=ae_dtype))
                if getattr(t_q, "bias", None) is not None and getattr(s_q, "bias", None) is not None:
                    s_q.bias.copy_(t_q.bias.to(device=args.device, dtype=ae_dtype))
        action_in_proj = copy.deepcopy(teacher_model.action_in_proj).to(device=args.device, dtype=ae_dtype).train()
        action_out_proj = copy.deepcopy(teacher_model.action_out_proj).to(device=args.device, dtype=ae_dtype).train()
        reset_module_parameters(action_in_proj)
        reset_module_parameters(action_out_proj)
    elif args.ae_init_mode == "ae_checkpoint_compressed":
        source_ckpt_path = str(getattr(args, "init_ae_source_checkpoint", ""))
        if not source_ckpt_path:
            raise ValueError("ae_checkpoint_compressed requires --init-ae-source-checkpoint")
        # 1. Build a full-size bundle matching AE28 architecture to load the source checkpoint
        #    AE28 was built from student backbone (intermediate_size=6144), not teacher (8256).
        if student is not None:
            source_expert = build_student_backbone_expert(
                student=student, dtype=ae_dtype, device="cpu",
                attn_implementation=expert_attn,
            )
        else:
            source_selected = layer_mapping(
                int(teacher_model.expert.config.num_hidden_layers), 28, args.mapping)
            source_expert = build_28layer_expert(
                teacher_expert=teacher_model.expert,
                selected_old_indices=source_selected,
                dtype=ae_dtype, device="cpu", attn_implementation=expert_attn,
            )
        source_action_in = copy.deepcopy(teacher_model.action_in_proj).to(device="cpu", dtype=ae_dtype)
        source_action_out = copy.deepcopy(teacher_model.action_out_proj).to(device="cpu", dtype=ae_dtype)
        source_bundle = AE28Bundle(expert=source_expert, action_in_proj=source_action_in, action_out_proj=source_action_out)
        load_bundle_checkpoint(Path(source_ckpt_path), bundle=source_bundle)
        print(json.dumps({"event": "ae_source_loaded", "checkpoint": source_ckpt_path,
                          "source_layers": len(source_expert.layers)}), flush=True)
        # 2. Select layers from source expert (28 -> compressed_layers)
        source_n = len(source_expert.layers)
        target_n = int(args.compressed_layers)
        compress_selected = layer_mapping(source_n, target_n, args.mapping)
        new_config = copy.deepcopy(source_expert.config)
        new_config.num_hidden_layers = target_n
        if hasattr(new_config, "_attn_implementation"):
            new_config._attn_implementation = expert_attn
        if hasattr(new_config, "attn_implementation"):
            new_config.attn_implementation = expert_attn
        from transformers import AutoModel as _AM
        expert = _AM.from_config(new_config)
        if hasattr(expert, "embed_tokens"):
            del expert.embed_tokens
        with torch.no_grad():
            for new_idx, old_idx in enumerate(compress_selected):
                expert.layers[new_idx].load_state_dict(
                    source_expert.layers[old_idx].state_dict(), strict=True)
            expert.norm.load_state_dict(source_expert.norm.state_dict(), strict=True)
        expert = expert.to(device=args.device, dtype=ae_dtype).train()
        # 3. Copy action projections from source
        action_in_proj = source_action_in.to(device=args.device, dtype=ae_dtype).train()
        action_out_proj = source_action_out.to(device=args.device, dtype=ae_dtype).train()
        # Update selected to reflect backbone KV layer mapping
        # AE14 layer i corresponds to source AE layer compress_selected[i],
        # which corresponds to backbone layer compress_selected[i] (since AE28 is 1:1 with backbone)
        selected = compress_selected
        del source_bundle, source_expert, source_action_in, source_action_out
        print(json.dumps({"event": "ae_compressed", "source_layers": source_n,
                          "target_layers": target_n, "selected_from_source": compress_selected}), flush=True)
    else:
        raise ValueError(f"Unsupported ae-init-mode: {args.ae_init_mode}")
    set_module_requires_grad(action_in_proj, True)
    set_module_requires_grad(action_out_proj, True)
    force_attention(expert, expert_attn)
    return AE28Bundle(expert=expert, action_in_proj=action_in_proj, action_out_proj=action_out_proj).train(), selected


def build_batch(
    *,
    args: argparse.Namespace,
    student: Any,
    student_processor: Any,
    student_tokenizer: Any,
    teacher_model: Any,
    batch_items: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [item["row"] for item in batch_items]
    image_batch = [load_sample_images(row, PROJECT_ROOT) for row in rows]
    histories_xyz = [load_ego_history_xyz(row, PROJECT_ROOT).astype(np.float32) for row in rows]
    histories_rot = [normalize_history_rot(load_ego_history_rot(row, PROJECT_ROOT)) for row in rows]
    prompt_messages = []
    teacher_cot_texts = [teacher_cot_text(item) for item in batch_items]
    for row, images, hist_xyz, cot_text in zip(rows, image_batch, histories_xyz, teacher_cot_texts):
        camera_indices = resolve_camera_indices(row, PROJECT_ROOT, image_count=len(images))
        frames_per_camera = max(len(images) // max(len(camera_indices), 1), 1)
        prompt_text = build_user_prompt(
            row,
            PROJECT_ROOT,
            ego_history_xyz=hist_xyz,
            prompt_text_style="official_alpamayo",
        )
        if args.prefix_mode == "teacher_forced":
            completion_text = f"{cot_text}<|cot_end|><|traj_future_start|>"
        elif args.prefix_mode == "student_free":
            completion_text = None
        else:
            raise ValueError(f"Unsupported prefix mode: {args.prefix_mode}")
        prompt_messages.append(
            build_messages(
                prompt_text,
                len(images),
                completion_text=completion_text,
                assistant_prefix="<|cot_start|>",
                image_prompt_style="camera_labeled",
                camera_indices=camera_indices,
                num_frames_per_camera=frames_per_camera,
            )
        )
    encoded = _encode_messages(
        student_processor,
        prompt_messages,
        image_batch,
        args.max_length,
        continue_final_message=True,
    )
    encoded["input_ids"] = fuse_history_tokens_in_input_ids(
        encoded["input_ids"],
        student_tokenizer,
        histories_xyz,
    )
    device = torch.device(args.device)
    flex_enabled = bool(hasattr(student, "flex_enabled") and student.flex_enabled())
    if flex_enabled:
        flex_cfg = getattr(student, "flex_scene_config", None)
        image_token_id = getattr(student, "image_token_id", None)
        if flex_cfg is None or image_token_id is None:
            raise RuntimeError("FLEX student is enabled but missing flex_scene_config or image_token_id.")
        if bool(getattr(args, "preserve_flex_positions", False)):
            encoded = attach_qwen_mrope_position_ids(encoded, student)
            conditional = student._conditional_backbone() if hasattr(student, "_conditional_backbone") else student
            qwen_model = getattr(conditional, "model", None)
            get_rope_index = getattr(qwen_model, "get_rope_index", None)
            if get_rope_index is not None:
                _, rope_deltas = get_rope_index(
                    input_ids=encoded["input_ids"],
                    image_grid_thw=encoded["image_grid_thw"],
                    video_grid_thw=None,
                    attention_mask=encoded["attention_mask"],
                )
                encoded["flex_rope_deltas"] = rope_deltas
        encoded = compress_batch_for_flex(
            encoded,
            image_token_id=int(image_token_id),
            tokens_per_image=int(getattr(flex_cfg, "tokens_per_image")),
            pad_token_id=int(student_tokenizer.pad_token_id or 0),
            preserve_original_position_ids=bool(getattr(args, "preserve_flex_positions", False)),
            selection_strategy=str(getattr(args, "flex_selection_strategy", "first") or "first"),
        )
        encoded["flex_selection_strategy"] = str(getattr(args, "flex_selection_strategy", "first") or "first")
        if bool(getattr(args, "flex_scene_deepstack", False)):
            encoded["flex_scene_deepstack"] = True
    encoded = _to_device_batch(encoded, device)
    prompt_lengths = encoded["attention_mask"].sum(dim=1).to(dtype=torch.long).tolist()

    target_xyz_np: list[np.ndarray] = []
    target_rot_np: list[np.ndarray] = []
    target_source = str(getattr(args, "target_source", "teacher"))
    expected_wp = 64
    for item, row in zip(batch_items, rows):
        if target_source == "gt":
            xyz = load_ego_future_xyz(row, PROJECT_ROOT).astype(np.float32)
            rot = load_ego_future_rot(row, PROJECT_ROOT).astype(np.float32)
            if not (xyz.ndim == 2 and xyz.shape[-1] == 3):
                raise ValueError(
                    f"GT future xyz unexpected shape {xyz.shape} for {item.get('sample_id')}"
                )
            if not (rot.ndim == 3 and rot.shape[-2:] == (3, 3)):
                raise ValueError(
                    f"GT future rot unexpected shape {rot.shape} for {item.get('sample_id')}"
                )
            if xyz.shape[0] < expected_wp or rot.shape[0] < expected_wp:
                raise ValueError(
                    f"GT future too short (xyz={xyz.shape[0]} wp, rot={rot.shape[0]} wp) "
                    f"for {item.get('sample_id')}; expected >= {expected_wp}"
                )
            xyz = xyz[:expected_wp]
            rot = rot[:expected_wp]
        else:  # teacher (default)
            xyz, rot = raw_teacher_pred(Path(item["raw_json"]))
        target_xyz_np.append(xyz)
        target_rot_np.append(rot)
    target_xyz = torch.from_numpy(np.stack(target_xyz_np, axis=0)).to(device=device, dtype=torch.float32)
    target_rot = torch.from_numpy(np.stack(target_rot_np, axis=0)).to(device=device, dtype=torch.float32)
    ego_history_xyz = torch.from_numpy(np.stack(histories_xyz, axis=0)).to(device=device, dtype=torch.float32)
    ego_history_rot = torch.from_numpy(np.stack(histories_rot, axis=0)).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        target_action = teacher_model.action_space.traj_to_action(
            ego_history_xyz,
            ego_history_rot,
            target_xyz,
            target_rot,
        )
    expected_action_dims = tuple(teacher_model.action_space.get_action_space_dims())
    if tuple(target_action.shape[1:]) != expected_action_dims:
        raise AssertionError(
            f"target_action shape {tuple(target_action.shape)} != expected (B, *{expected_action_dims}); "
            f"target_source={target_source}"
        )

    traj_start_id = student_tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
    if not isinstance(traj_start_id, int) or traj_start_id < 0:
        raise ValueError("Student tokenizer is missing <|traj_future_start|>")

    model_kwargs = dict(encoded)
    input_ids = model_kwargs.pop("input_ids")
    flex_rope_deltas = model_kwargs.pop("flex_rope_deltas", None)
    if args.prefix_mode == "student_free":
        generation_config = copy.deepcopy(student.backbone.generation_config)
        generation_config.do_sample = False
        generation_config.num_return_sequences = 1
        generation_config.num_beams = 1
        generation_config.top_p = 1.0
        generation_config.top_k = None
        generation_config.temperature = 1.0
        generation_config.max_new_tokens = int(args.max_new_tokens)
        generation_config.output_logits = False
        generation_config.output_scores = False
        generation_config.output_hidden_states = False
        generation_config.return_dict_in_generate = True
        generation_config.pad_token_id = student_tokenizer.pad_token_id
        stopping = StoppingCriteriaList([StopAfterToken(traj_start_id, prompt_lengths)])
        with torch.no_grad(), torch.autocast(
            "cuda",
            dtype=torch_dtype_from_name(args.student_dtype),
            enabled=device.type == "cuda" and torch.cuda.is_available(),
        ):
            if flex_enabled:
                outputs = manual_flex_generate_with_cache(
                    student=student,
                    input_ids=input_ids,
                    model_kwargs=model_kwargs,
                    max_new_tokens=int(args.max_new_tokens),
                    stopping_criteria=stopping,
                )
            else:
                outputs = student.backbone.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    stopping_criteria=stopping,
                    **model_kwargs,
                )
        if flex_enabled:
            rope_deltas = flex_rope_deltas
            if rope_deltas is None:
                # FLEX without --preserve-flex-positions uses sequential 0..N positions,
                # so rope_deltas is effectively 0.  get_rope_deltas() would crash because
                # _forward_flex does not set conditional.model.rope_deltas in this path.
                try:
                    rope_deltas = get_rope_deltas(student.backbone)
                except (AttributeError, RuntimeError):
                    rope_deltas = torch.zeros(
                        int(input_ids.shape[0]), 1, dtype=torch.long, device=device,
                    )
        else:
            rope_deltas = get_rope_deltas(student.backbone)
        sequences = outputs.sequences
        generated_ids = sequences[:, int(input_ids.shape[1]) :]
        generated_texts = student_tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens=False)
        prefix_attention_mask = encoded.get("attention_mask")
        cache = outputs.past_key_values
    else:
        backbone_grad_ctx = nullcontext() if bool(getattr(args, "train_backbone_lora", False)) else torch.no_grad()
        with backbone_grad_ctx, torch.autocast(
            "cuda",
            dtype=torch_dtype_from_name(args.student_dtype),
            enabled=device.type == "cuda" and torch.cuda.is_available(),
        ):
            if flex_enabled:
                student_outputs = student(
                    input_ids=input_ids,
                    **model_kwargs,
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=1,
                    return_hidden_states=False,
                    compute_meta_action=False,
                    compute_traj_aux=False,
                )
                outputs = student_outputs["backbone_outputs"]
            else:
                try:
                    outputs = student.backbone(
                        input_ids=input_ids,
                        **model_kwargs,
                        use_cache=True,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                except TypeError:
                    outputs = student.backbone(
                        input_ids=input_ids,
                        **model_kwargs,
                        use_cache=True,
                        return_dict=True,
                    )
        rope_deltas = getattr(outputs, "rope_deltas", None)
        if rope_deltas is None and flex_rope_deltas is not None:
            rope_deltas = flex_rope_deltas
        if rope_deltas is None:
            try:
                rope_deltas = get_rope_deltas(student.backbone)
            except (AttributeError, RuntimeError):
                rope_deltas = torch.zeros(
                    int(input_ids.shape[0]), 1, dtype=torch.long, device=device,
                )
        sequences = input_ids
        generated_texts = [f"{text}<|cot_end|><|traj_future_start|>" for text in teacher_cot_texts]
        prefix_attention_mask = encoded.get("attention_mask")
        cache = outputs.past_key_values

    offset = teacher_model._find_eos_offset(
        sequences=sequences,
        eos_token_id=int(traj_start_id),
        device=device,
        warn=False,
    )
    kv_cache_seq_len = int(cache.get_seq_length())
    n_diffusion_tokens = int(teacher_model.action_space.get_action_space_dims()[0])
    position_ids, attention_mask = teacher_model._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=rope_deltas.to(device),
        kv_cache_seq_len=kv_cache_seq_len,
        n_diffusion_tokens=n_diffusion_tokens,
        b_star=int(sequences.shape[0]),
        device=device,
        prefix_mask=prefix_attention_mask,
    )
    # --- BUG FIX: FLEX rope_deltas correction ---
    # When --preserve-flex-positions is used, flex_rope_deltas is computed from
    # the ORIGINAL (pre-compression) sequence via get_rope_index().  It encodes:
    #   rope_deltas = max_mrope_position - original_seq_len + 1
    # But kv_cache_seq_len is the COMPRESSED KV length (~768 instead of ~3086).
    # Without correction the expert RoPE positions are off by
    # (original_seq_len - compressed_seq_len) ≈ 2318 positions.
    # Correct formula: expert_start_position = max_mrope_position + 1
    #   = rope_deltas + original_seq_len = rope_deltas + kv_cache_seq_len + deficit
    if flex_enabled and flex_rope_deltas is not None and rope_deltas is flex_rope_deltas:
        flex_stats = encoded.get("flex_stats", {})
        orig_seq_len = int(flex_stats.get("flex_original_seq_len", kv_cache_seq_len))
        flex_position_deficit = orig_seq_len - kv_cache_seq_len
        rope_deltas = rope_deltas + flex_position_deficit

    if str(args.stage2_attention_mode) == "official_none":
        position_ids = (
            torch.arange(n_diffusion_tokens, dtype=torch.long, device=device)
            .view(1, 1, -1)
            .repeat(3, int(sequences.shape[0]), 1)
            + rope_deltas.to(device)
            + kv_cache_seq_len
        )
        attention_mask = None
    return {
        "sample_ids": [item["sample_id"] for item in batch_items],
        "prefix_mode": str(args.prefix_mode),
        "cache": cache,
        "context": {
            "kv_cache_seq_len": kv_cache_seq_len,
            "n_diffusion_tokens": n_diffusion_tokens,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "stage2_attention_mode": str(args.stage2_attention_mode),
        },
        "target_action": target_action.detach(),
        "target_xyz": target_xyz.detach(),
        "ego_history_xyz": ego_history_xyz.detach(),
        "ego_history_rot": ego_history_rot.detach(),
        "generated_texts": list(generated_texts),
        "generated_text_preview": generated_texts[0][:240] if generated_texts else "",
        "traj_start_hit_rate": float(
            sum("<|traj_future_start|>" in text for text in generated_texts) / max(len(generated_texts), 1)
        ),
    }


def load_batch_from_kv_cache(
    *,
    kv_cache_dir: str | Path,
    batch_items: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Load pre-computed KV cache from disk instead of running student forward."""
    kv_cache_dir = Path(kv_cache_dir)
    all_kv: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    all_target_action = []
    all_target_xyz = []
    all_ego_history_xyz = []
    all_ego_history_rot = []
    sample_ids = []
    ctx = None

    for item in batch_items:
        sample_id = item["sample_id"]
        safe_id = sample_id.replace("/", "_").replace("\\", "_")
        pt_path = kv_cache_dir / f"{safe_id}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"KV cache not found: {pt_path}")

        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        all_kv.append(data["kv_cache"])
        all_target_action.append(data["target_action"])
        all_target_xyz.append(data["target_xyz"])
        all_ego_history_xyz.append(data["ego_history_xyz"])
        all_ego_history_rot.append(data["ego_history_rot"])
        sample_ids.append(sample_id)

        if ctx is None:
            ctx = {
                "kv_cache_seq_len": data["kv_cache_seq_len"],
                "n_diffusion_tokens": data["n_diffusion_tokens"],
                "stage2_attention_mode": data["stage2_attention_mode"],
            }

    # Stack KV cache into DynamicCache
    from transformers.cache_utils import DynamicCache

    batch_size = len(batch_items)
    num_layers = len(all_kv[0])
    cache = DynamicCache()
    for layer_idx in range(num_layers):
        k = torch.cat([all_kv[b][layer_idx][0] for b in range(batch_size)], dim=0).to(device)
        v = torch.cat([all_kv[b][layer_idx][1] for b in range(batch_size)], dim=0).to(device)
        cache.update(k, v, layer_idx)

    # Stack position_ids
    position_ids = torch.cat(
        [torch.load(kv_cache_dir / f"{item['sample_id'].replace('/', '_').replace(chr(92), '_')}.pt",
                     map_location="cpu", weights_only=False)["position_ids"]
         for item in batch_items],
        dim=1,
    ).to(device)

    return {
        "sample_ids": sample_ids,
        "prefix_mode": "offline_kv_cache",
        "cache": cache,
        "context": {
            "kv_cache_seq_len": ctx["kv_cache_seq_len"],
            "n_diffusion_tokens": ctx["n_diffusion_tokens"],
            "position_ids": position_ids,
            "attention_mask": None,
            "stage2_attention_mode": ctx["stage2_attention_mode"],
        },
        "target_action": torch.cat(all_target_action, dim=0).to(device),
        "target_xyz": torch.cat(all_target_xyz, dim=0).to(device),
        "ego_history_xyz": torch.cat(all_ego_history_xyz, dim=0).to(device),
        "ego_history_rot": torch.cat(all_ego_history_rot, dim=0).to(device),
        "generated_text_preview": "(offline KV cache)",
        "traj_start_hit_rate": 1.0,
    }


def repeat_context(context: dict[str, Any], repeats: int) -> dict[str, Any]:
    if int(repeats) <= 1:
        return context
    repeated = dict(context)
    repeated["position_ids"] = context["position_ids"].repeat_interleave(int(repeats), dim=1)
    if context.get("attention_mask") is not None:
        repeated["attention_mask"] = context["attention_mask"].repeat_interleave(int(repeats), dim=0)
    return repeated


def repeat_eval_batch_for_paths(batch: dict[str, Any], repeats: int) -> dict[str, Any]:
    if int(repeats) <= 1:
        return batch
    repeated = dict(batch)
    prompt_cache = copy.deepcopy(batch["cache"])
    prompt_cache.batch_repeat_interleave(int(repeats))
    repeated["cache"] = prompt_cache
    repeated["context"] = repeat_context(batch["context"], int(repeats))
    for key in ("ego_history_xyz", "ego_history_rot", "target_action", "target_xyz"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            repeated[key] = value.repeat_interleave(int(repeats), dim=0)
    return repeated


def maybe_cuda_cleanup(every: int, index: int) -> None:
    if int(every) <= 0:
        return
    if int(index) % int(every) != 0:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_fm_timesteps(
    *,
    batch_size: int,
    sampler: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if sampler == "uniform":
        t = torch.rand((batch_size,), device=device, dtype=dtype)
    elif sampler == "beta":
        # Matches alpamayo_base/src/alpamayo_r1/diffusion/flow_matching.py.
        beta = torch.distributions.beta.Beta(
            torch.tensor(1.5, dtype=torch.float32, device=device),
            torch.tensor(1.0, dtype=torch.float32, device=device),
        )
        t = 0.999 - beta.sample((batch_size,)).to(device=device, dtype=dtype) * 0.999
    else:
        raise ValueError(f"Unknown train timestep sampler: {sampler}")
    return t.view(batch_size, 1, 1)


def train_step(
    *,
    bundle: AE28Bundle,
    teacher_model: Any,
    batch: dict[str, Any],
    num_time_samples: int,
    train_timestep_sampler: str,
    device: torch.device,
    allow_cache_mutation: bool = False,
    kv_layer_indices: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    dtype = next(bundle.parameters()).dtype
    repeats = max(int(num_time_samples), 1)
    prompt_cache = batch["cache"]
    if kv_layer_indices is not None:
        prompt_cache = select_kv_cache_layers(prompt_cache, kv_layer_indices)
    context = batch["context"]
    target_action = batch["target_action"]
    if repeats > 1:
        # batch_repeat_interleave mutates DynamicCache in-place. Keep the original
        # batch cache intact unless the caller guarantees the batch cache will not
        # be reused after this train_step.
        if not bool(allow_cache_mutation):
            prompt_cache = copy.deepcopy(prompt_cache)
        prompt_cache.batch_repeat_interleave(repeats)
        context = repeat_context(context, repeats)
        target_action = target_action.repeat_interleave(repeats, dim=0)

    x1 = target_action.to(device=device, dtype=dtype)
    x0 = torch.randn_like(x1)
    t = sample_fm_timesteps(
        batch_size=int(x1.shape[0]),
        sampler=str(train_timestep_sampler),
        device=device,
        dtype=dtype,
    )
    x_t = (1.0 - t) * x0 + t * x1
    target_v = x1 - x0

    prefill_seq_len = int(context["kv_cache_seq_len"])
    n_diffusion_tokens = int(context["n_diffusion_tokens"])
    action_dims = teacher_model.action_space.get_action_space_dims()
    kwargs: dict[str, Any] = {}
    if bool(getattr(teacher_model.config, "expert_non_causal_attention", False)):
        kwargs["is_causal"] = False
    future_token_embeds = bundle.action_in_proj(x_t, t)
    if future_token_embeds.dim() == 2:
        future_token_embeds = future_token_embeds.view(x_t.shape[0], n_diffusion_tokens, -1)
    expert_attention_mask = context.get("attention_mask")
    if expert_attention_mask is not None:
        expert_attention_mask = expert_attention_mask.to(dtype=future_token_embeds.dtype)
    out = bundle.expert(
        inputs_embeds=future_token_embeds,
        position_ids=context["position_ids"],
        past_key_values=prompt_cache,
        attention_mask=expert_attention_mask,
        use_cache=True,
        **kwargs,
    )
    prompt_cache.crop(prefill_seq_len)
    last_hidden = out.last_hidden_state[:, -n_diffusion_tokens:]
    pred_v = bundle.action_out_proj(last_hidden).view(-1, *action_dims)
    loss = F.mse_loss(pred_v.float(), target_v.float())
    return loss, {
        "num_time_samples": float(repeats),
        "effective_fm_batch": float(x1.shape[0]),
        "target_action_abs_mean": float(x1.detach().abs().mean().cpu()),
        "target_v_abs_mean": float(target_v.detach().abs().mean().cpu()),
        "pred_v_abs_mean": float(pred_v.detach().abs().mean().cpu()),
        "train_t_mean": float(t.detach().float().mean().cpu()),
        "train_cache_deepcopy": float(not bool(allow_cache_mutation) and repeats > 1),
    }


def sample_paths(
    *,
    bundle: AE28Bundle,
    teacher_model: Any,
    batch: dict[str, Any],
    seed: int,
    device: torch.device,
    inference_steps: int | None = None,
    temperature: float = 1.0,
    kv_layer_indices: list[int] | None = None,
) -> dict[str, np.ndarray]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    dtype = next(bundle.parameters()).dtype
    prompt_cache = batch["cache"]
    if kv_layer_indices is not None:
        prompt_cache = select_kv_cache_layers(prompt_cache, kv_layer_indices)
    context = batch["context"]
    batch_size = int(batch["ego_history_xyz"].shape[0])
    prefill_seq_len = int(context["kv_cache_seq_len"])
    n_diffusion_tokens = int(context["n_diffusion_tokens"])
    action_dims = teacher_model.action_space.get_action_space_dims()
    kwargs: dict[str, Any] = {}
    if bool(getattr(teacher_model.config, "expert_non_causal_attention", False)):
        kwargs["is_causal"] = False

    def step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        future_token_embeds = bundle.action_in_proj(x.to(dtype=dtype), t.to(dtype=dtype))
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(x.shape[0], n_diffusion_tokens, -1)
        expert_attention_mask = context.get("attention_mask")
        if expert_attention_mask is not None:
            expert_attention_mask = expert_attention_mask.to(dtype=future_token_embeds.dtype)
        out = bundle.expert(
            inputs_embeds=future_token_embeds,
            position_ids=context["position_ids"],
            past_key_values=prompt_cache,
            attention_mask=expert_attention_mask,
            use_cache=True,
            **kwargs,
        )
        prompt_cache.crop(prefill_seq_len)
        last_hidden = out.last_hidden_state[:, -n_diffusion_tokens:]
        return bundle.action_out_proj(last_hidden).view(-1, *action_dims)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        action = teacher_model.diffusion.sample(
            batch_size=batch_size,
            step_fn=step_fn,
            device=device,
            inference_step=inference_steps,
            temperature=float(temperature),
        )
        pred_xyz, pred_rot = teacher_model.action_space.action_to_traj(
            action,
            batch["ego_history_xyz"].to(device),
            batch["ego_history_rot"].to(device),
        )
    return {
        "action": action.detach().float().cpu().numpy(),
        "pred_xyz": pred_xyz.detach().float().cpu().numpy(),
        "pred_rot": pred_rot.detach().float().cpu().numpy(),
    }


def trajectory_medoid_index(paths: np.ndarray) -> int:
    """Return the path with minimum mean pointwise distance to other paths."""
    diff = paths[:, None, :, :] - paths[None, :, :, :]
    dist = np.linalg.norm(diff, axis=-1).mean(axis=-1)
    return int(np.argmin(dist.sum(axis=1)))


def iter_batches(items: list[dict[str, Any]], batch_size: int):
    width = max(int(batch_size), 1)
    for index in range(0, len(items), width):
        yield items[index : index + width]


def evaluate(
    *,
    args: argparse.Namespace,
    bundle: AE28Bundle,
    student: Any,
    student_processor: Any,
    student_tokenizer: Any,
    teacher_model: Any,
    items: list[dict[str, Any]],
    step: int,
    kv_layer_indices: list[int] | None = None,
) -> dict[str, Any]:
    bundle.eval()
    rows: list[dict[str, Any]] = []
    device = torch.device(args.device)
    horizon_specs = (("h1p6_16wp", 16), ("h3p2_32wp", 32), ("h6p4_64wp", 64))
    horizon_names = tuple(name for name, _ in horizon_specs)
    # Aggregates from path 0 only (backward-compatible "single sample" view).
    horizon_values: dict[str, dict[str, list[float]]] = {
        name: {"ade": [], "fde": []} for name in horizon_names
    }
    # Best-of-N aggregates (populated only when num_paths > 1).
    horizon_best_values: dict[str, dict[str, list[float]]] = {
        name: {"ade": [], "fde": []} for name in horizon_names
    }
    ades_best: list[float] = []
    fdes_best: list[float] = []
    mean_paths_all: list[float] = []
    std_paths_all: list[float] = []
    num_paths = max(1, int(getattr(args, "eval_num_paths", 1)))
    eval_temperature = float(getattr(args, "eval_temperature", 1.0))
    eval_selection_method = str(getattr(args, "eval_selection_method", "single"))
    eval_seed_base = int(args.seed) + 1000 + (0 if str(args.eval_seed_mode) == "fixed" else int(step))
    for batch_index, batch_items in enumerate(iter_batches(items[: int(args.eval_samples)], int(args.eval_batch_size))):
        batch = build_batch(
            args=args,
            student=student,
            student_processor=student_processor,
            student_tokenizer=student_tokenizer,
            teacher_model=teacher_model,
            batch_items=batch_items,
        )
        target_xyz = batch["target_xyz"].detach().cpu().numpy()
        sample_ids = list(batch["sample_ids"])
        n_samples_batch = len(sample_ids)
        per_sample_ades: list[list[float]] = [[] for _ in range(n_samples_batch)]
        per_sample_fdes: list[list[float]] = [[] for _ in range(n_samples_batch)]
        per_sample_h_ades: list[dict[str, list[float]]] = [
            {name: [] for name in horizon_names} for _ in range(n_samples_batch)
        ]
        per_sample_h_fdes: list[dict[str, list[float]]] = [
            {name: [] for name in horizon_names} for _ in range(n_samples_batch)
        ]
        per_sample_pred_xyz: list[list[np.ndarray]] = [[] for _ in range(n_samples_batch)]
        first_path_pred_xyz: list[np.ndarray | None] = [None] * n_samples_batch
        if bool(getattr(args, "eval_vectorize_paths", False)) and num_paths > 1:
            path_chunk_size = int(getattr(args, "eval_path_batch_size", 0))
            if path_chunk_size <= 0:
                path_chunk_size = num_paths
            path_offset = 0
            chunk_index = 0
            while path_offset < num_paths:
                chunk_paths = min(path_chunk_size, num_paths - path_offset)
                path_seed = eval_seed_base + batch_index * num_paths + path_offset
                repeated_batch = repeat_eval_batch_for_paths(batch, chunk_paths)
                pred = sample_paths(
                    bundle=bundle,
                    teacher_model=teacher_model,
                    batch=repeated_batch,
                    seed=path_seed,
                    device=device,
                    temperature=eval_temperature,
                    kv_layer_indices=kv_layer_indices,
                )
                pred_xyz_chunk = pred["pred_xyz"].reshape(n_samples_batch, chunk_paths, *pred["pred_xyz"].shape[1:])
                for row_index in range(n_samples_batch):
                    target_xyz_row = target_xyz[row_index]
                    for local_path_idx in range(chunk_paths):
                        pred_xyz_row = pred_xyz_chunk[row_index, local_path_idx]
                        global_path_idx = path_offset + local_path_idx
                        per_sample_pred_xyz[row_index].append(pred_xyz_row.copy())
                        ade, fde = ade_fde(pred_xyz_row, target_xyz_row)
                        per_sample_ades[row_index].append(float(ade))
                        per_sample_fdes[row_index].append(float(fde))
                        for name, horizon in horizon_specs:
                            n = min(horizon, int(pred_xyz_row.shape[0]), int(target_xyz_row.shape[0]))
                            h_ade, h_fde = ade_fde(pred_xyz_row[:n], target_xyz_row[:n])
                            per_sample_h_ades[row_index][name].append(float(h_ade))
                            per_sample_h_fdes[row_index][name].append(float(h_fde))
                        if global_path_idx == 0:
                            first_path_pred_xyz[row_index] = pred_xyz_row.copy()
                del pred, repeated_batch
                chunk_index += 1
                maybe_cuda_cleanup(int(getattr(args, "eval_cleanup_every", 1)), chunk_index)
                path_offset += chunk_paths
        else:
            for path_idx in range(num_paths):
                # N=1 → seed = eval_seed_base + batch_index (matches legacy formula).
                path_seed = eval_seed_base + batch_index * num_paths + path_idx
                pred = sample_paths(
                    bundle=bundle,
                    teacher_model=teacher_model,
                    batch=batch,
                    seed=path_seed,
                    device=device,
                    temperature=eval_temperature,
                    kv_layer_indices=kv_layer_indices,
                )
                for row_index in range(n_samples_batch):
                    pred_xyz_row = pred["pred_xyz"][row_index]
                    target_xyz_row = target_xyz[row_index]
                    per_sample_pred_xyz[row_index].append(pred_xyz_row.copy())
                    ade, fde = ade_fde(pred_xyz_row, target_xyz_row)
                    per_sample_ades[row_index].append(float(ade))
                    per_sample_fdes[row_index].append(float(fde))
                    for name, horizon in horizon_specs:
                        n = min(horizon, int(pred_xyz_row.shape[0]), int(target_xyz_row.shape[0]))
                        h_ade, h_fde = ade_fde(pred_xyz_row[:n], target_xyz_row[:n])
                        per_sample_h_ades[row_index][name].append(float(h_ade))
                        per_sample_h_fdes[row_index][name].append(float(h_fde))
                    if path_idx == 0:
                        first_path_pred_xyz[row_index] = pred_xyz_row.copy()
                del pred
                maybe_cuda_cleanup(int(getattr(args, "eval_cleanup_every", 1)), path_idx + 1)
        for row_index, sample_id in enumerate(sample_ids):
            ades_n = per_sample_ades[row_index]
            fdes_n = per_sample_fdes[row_index]
            single_ade = ades_n[0]
            single_fde = fdes_n[0]
            paths_n = np.stack(per_sample_pred_xyz[row_index], axis=0)
            target_xyz_row = target_xyz[row_index]
            best_idx = int(np.argmin(ades_n))
            selected_path_idx: int | None = 0
            if eval_selection_method == "single":
                selected_path = paths_n[0]
                selected_ade = single_ade
                selected_fde = single_fde
            elif eval_selection_method == "oracle_best":
                selected_path_idx = best_idx
                selected_path = paths_n[best_idx]
                selected_ade = ades_n[best_idx]
                selected_fde = fdes_n[best_idx]
            elif eval_selection_method == "medoid":
                selected_path_idx = trajectory_medoid_index(paths_n)
                selected_path = paths_n[selected_path_idx]
                selected_ade, selected_fde = ade_fde(selected_path, target_xyz_row)
            elif eval_selection_method == "mean_traj":
                selected_path_idx = None
                selected_path = paths_n.mean(axis=0)
                selected_ade, selected_fde = ade_fde(selected_path, target_xyz_row)
            else:
                raise ValueError(f"Unknown eval_selection_method={eval_selection_method!r}")

            # Main eval aggregates follow eval_selection_method. Default single is backward-compatible.
            horizon_metrics: dict[str, float] = {}
            for name in horizon_names:
                horizon = next(width for spec_name, width in horizon_specs if spec_name == name)
                n = min(horizon, int(selected_path.shape[0]), int(target_xyz_row.shape[0]))
                h_ade, h_fde = ade_fde(selected_path[:n], target_xyz_row[:n])
                horizon_values[name]["ade"].append(float(h_ade))
                horizon_values[name]["fde"].append(float(h_fde))
                horizon_metrics[f"{name}_ade_m"] = float(h_ade)
                horizon_metrics[f"{name}_fde_m"] = float(h_fde)
            row = {
                "sample_id": sample_id,
                "ade_m": float(selected_ade),
                "fde_m": float(selected_fde),
                **horizon_metrics,
                "eval_selection_method": eval_selection_method,
                "selected_path_idx": selected_path_idx,
                "ade_single_m": single_ade,
                "fde_single_m": single_fde,
                "pred_path_length_m": path_len(selected_path),
                "single_path_length_m": path_len(first_path_pred_xyz[row_index]),
                "target_path_length_m": path_len(target_xyz[row_index]),
            }
            if num_paths > 1:
                best_ade = ades_n[best_idx]
                best_fde = fdes_n[best_idx]
                ades_best.append(best_ade)
                fdes_best.append(best_fde)
                mean_paths_all.append(float(np.mean(ades_n)))
                std_paths_all.append(float(np.std(ades_n)))
                horizon_best_metrics: dict[str, float] = {}
                for name in horizon_names:
                    h_ades_n = per_sample_h_ades[row_index][name]
                    h_fdes_n = per_sample_h_fdes[row_index][name]
                    h_best_idx = int(np.argmin(h_ades_n))
                    horizon_best_values[name]["ade"].append(h_ades_n[h_best_idx])
                    horizon_best_values[name]["fde"].append(h_fdes_n[h_best_idx])
                    horizon_best_metrics[f"{name}_ade_best_of_n_m"] = h_ades_n[h_best_idx]
                    horizon_best_metrics[f"{name}_fde_best_of_n_m"] = h_fdes_n[h_best_idx]
                row.update(
                    {
                        "ade_best_of_n_m": best_ade,
                        "fde_best_of_n_m": best_fde,
                        "ade_mean_over_paths_m": float(np.mean(ades_n)),
                        "ade_std_over_paths_m": float(np.std(ades_n)),
                        "ade_all_paths_m": [float(a) for a in ades_n],
                        "best_path_idx": best_idx,
                        **horizon_best_metrics,
                    }
                )
            rows.append(row)
        del batch
        maybe_cuda_cleanup(int(getattr(args, "eval_cleanup_every", 1)), batch_index + 1)
    ades = [row["ade_m"] for row in rows]
    fdes = [row["fde_m"] for row in rows]
    out = {
        "event": "eval",
        "step": int(step),
        "eval_num_paths": int(num_paths),
        "eval_temperature": float(eval_temperature),
        "eval_selection_method": eval_selection_method,
        "eval_seed_mode": str(args.eval_seed_mode),
        "eval_seed_base": int(eval_seed_base),
        "eval_count": len(rows),
        "ade_mean_m": float(np.mean(ades)) if ades else None,
        "ade_p50_m": float(np.percentile(ades, 50)) if ades else None,
        "fde_mean_m": float(np.mean(fdes)) if fdes else None,
        "fde_p50_m": float(np.percentile(fdes, 50)) if fdes else None,
        "horizon": {
            name: {
                "ade_mean_m": float(np.mean(values["ade"])) if values["ade"] else None,
                "ade_p50_m": float(np.percentile(values["ade"], 50)) if values["ade"] else None,
                "fde_mean_m": float(np.mean(values["fde"])) if values["fde"] else None,
                "fde_p50_m": float(np.percentile(values["fde"], 50)) if values["fde"] else None,
            }
            for name, values in horizon_values.items()
        },
    }
    eval_log_rows = int(getattr(args, "eval_log_rows", -1))
    if eval_log_rows < 0:
        out["rows"] = rows
    elif eval_log_rows > 0:
        out["rows"] = rows[:eval_log_rows]
        out["rows_truncated_count"] = max(len(rows) - eval_log_rows, 0)
    else:
        out["rows"] = []
        out["rows_truncated_count"] = len(rows)
    if num_paths > 1:
        out["ade_best_of_n_mean_m"] = float(np.mean(ades_best)) if ades_best else None
        out["ade_best_of_n_p50_m"] = float(np.percentile(ades_best, 50)) if ades_best else None
        out["fde_best_of_n_mean_m"] = float(np.mean(fdes_best)) if fdes_best else None
        out["fde_best_of_n_p50_m"] = float(np.percentile(fdes_best, 50)) if fdes_best else None
        out["minade_at_n"] = int(num_paths)
        out["minade_at_n_mean_m"] = out["ade_best_of_n_mean_m"]
        out["minade_at_n_p50_m"] = out["ade_best_of_n_p50_m"]
        out["minfde_at_n_mean_m"] = out["fde_best_of_n_mean_m"]
        out["minfde_at_n_p50_m"] = out["fde_best_of_n_p50_m"]
        if int(num_paths) == 6:
            out["minade_at_6_mean_m"] = out["ade_best_of_n_mean_m"]
            out["minade_at_6_p50_m"] = out["ade_best_of_n_p50_m"]
            out["minfde_at_6_mean_m"] = out["fde_best_of_n_mean_m"]
            out["minfde_at_6_p50_m"] = out["fde_best_of_n_p50_m"]
        out["ade_mean_over_paths_mean_m"] = float(np.mean(mean_paths_all)) if mean_paths_all else None
        out["ade_std_over_paths_mean_m"] = float(np.mean(std_paths_all)) if std_paths_all else None
        out["horizon_best_of_n"] = {
            name: {
                "ade_mean_m": float(np.mean(values["ade"])) if values["ade"] else None,
                "ade_p50_m": float(np.percentile(values["ade"], 50)) if values["ade"] else None,
                "fde_mean_m": float(np.mean(values["fde"])) if values["fde"] else None,
                "fde_p50_m": float(np.percentile(values["fde"], 50)) if values["fde"] else None,
            }
            for name, values in horizon_best_values.items()
        }
    bundle.train()
    return out


def save_checkpoint(path: Path, *, bundle: AE28Bundle, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"bundle_state_dict": bundle.state_dict(), "payload": payload}, path)


def load_bundle_checkpoint(path: Path, *, bundle: AE28Bundle) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if "bundle_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {path} does not contain bundle_state_dict")
    bundle.load_state_dict(checkpoint["bundle_state_dict"], strict=True)
    payload = checkpoint.get("payload") or {}
    return payload if isinstance(payload, dict) else {"payload": payload}


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    summary_path = args.output_dir / "summary.json"
    print(
        json.dumps(
            {
                "event": "stage1_boot",
                "output_dir": str(args.output_dir),
                "num_samples": int(args.num_samples),
                "val_samples": int(args.val_samples),
                "eval_samples": int(args.eval_samples),
                "device_arg": str(args.device),
            }
        ),
        flush=True,
    )
    print(json.dumps({"event": "train_val_split_start"}), flush=True)
    train_items, val_items, split_summary = select_train_val_items(args)
    train_eval_items = train_items[: int(args.eval_train_samples)] if int(args.eval_train_samples) > 0 else []
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    print(json.dumps({"event": "stage1_seeded", "device": str(device)}), flush=True)

    summary: dict[str, Any] = {
        "created_at_unix": time.time(),
        "args": vars(args) | {
            "corpus_jsonl": str(args.corpus_jsonl),
            "student_checkpoint_dir": str(args.student_checkpoint_dir),
            "teacher_checkpoint_path": str(args.teacher_checkpoint_path),
            "output_dir": str(args.output_dir),
        },
        "status": "running",
    }
    try:
        summary["split_summary"] = split_summary
        summary["selected_count"] = len(train_items)
        summary["train_selected_count"] = len(train_items)
        summary["val_selected_count"] = len(val_items)
        summary["selected_sample_ids_head"] = [item["sample_id"] for item in train_items[:16]]
        summary["val_sample_ids_head"] = [item["sample_id"] for item in val_items[:16]]

        student, student_tokenizer, student_processor, base_model = load_student(args)
        summary["student_base_model"] = str(base_model)

        print(json.dumps({"event": "target_source", "mode": str(args.target_source)}), flush=True)
        print(json.dumps({"event": "load_teacher_action_modules_start", "device": args.teacher_load_device}), flush=True)
        teacher_model, _teacher_processor, _cfg, _cfg_path, _runtime = load_model_and_processor(
            checkpoint_path=args.teacher_checkpoint_path,
            dtype=torch_dtype_from_name(args.ae_dtype),
            device=args.teacher_load_device,
            config_json=None,
            runtime_support=None,
            attn_implementation=args.attn_implementation,
            min_pixels=163840,
            max_pixels=196608,
        )
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        force_attention(teacher_model.expert, str(args.attn_implementation))
        bundle, selected_layers = build_bundle(teacher_model, args, student=student)
        resume_payload: dict[str, Any] | None = None
        if args.resume_ae_checkpoint is not None:
            resume_payload = load_bundle_checkpoint(Path(args.resume_ae_checkpoint), bundle=bundle)
            resume_event = {
                "event": "resume_ae_checkpoint_loaded",
                "checkpoint": str(args.resume_ae_checkpoint),
                "payload_step": resume_payload.get("step"),
                "optimizer_state_restored": False,
            }
            print(json.dumps(resume_event), flush=True)
            summary["resume_ae_checkpoint"] = resume_event
        summary["ae28_selected_teacher_layers"] = selected_layers
        # KV layer selection: when expert has fewer layers than backbone (e.g. AE14 vs 28L backbone)
        expert_n_layers = int(bundle.expert.config.num_hidden_layers)
        backbone_n_layers = 28  # student backbone always has 28 layers
        if expert_n_layers < backbone_n_layers:
            kv_layer_indices = selected_layers  # maps expert layer i -> backbone KV layer selected_layers[i]
            print(json.dumps({"event": "kv_layer_selection_enabled",
                              "expert_layers": expert_n_layers,
                              "backbone_layers": backbone_n_layers,
                              "kv_indices": kv_layer_indices}), flush=True)
        else:
            kv_layer_indices = None
        summary["trainable_params"] = int(sum(p.numel() for p in bundle.parameters() if p.requires_grad))
        trainable_summary = {
            "event": "bundle_trainable_summary",
            "total_trainable_params": summary["trainable_params"],
            "modules": {
                "expert": trainable_module_summary(bundle.expert, prefix="expert"),
                "action_in_proj": trainable_module_summary(bundle.action_in_proj, prefix="action_in_proj"),
                "action_out_proj": trainable_module_summary(bundle.action_out_proj, prefix="action_out_proj"),
            },
        }
        print(json.dumps(trainable_summary), flush=True)
        summary["bundle_trainable_summary"] = trainable_summary
        # Free teacher VLM weights from memory; action_space/diffusion/mask helpers stay on the parent.
        if hasattr(teacher_model, "vlm"):
            delattr(teacher_model, "vlm")
        maybe_cuda_cleanup(1, 1)

        def _split_decay_params(mod: nn.Module, lr_val: float) -> list[dict[str, Any]]:
            if not args.no_norm_bias_decay:
                return [{"params": list(mod.parameters()), "lr": lr_val,
                         "weight_decay": float(args.weight_decay)}]
            decay, no_decay = [], []
            for pname, p in mod.named_parameters():
                if not p.requires_grad:
                    continue
                lname = pname.lower()
                is_norm = ("norm" in lname or "layernorm" in lname or "rmsnorm" in lname or "ln_" in lname)
                if p.dim() <= 1 or pname.endswith(".bias") or is_norm:
                    no_decay.append(p)
                else:
                    decay.append(p)
            groups: list[dict[str, Any]] = []
            if decay:
                groups.append({"params": decay, "lr": lr_val, "weight_decay": float(args.weight_decay)})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr_val, "weight_decay": 0.0})
            return groups

        opt_groups: list[dict[str, Any]] = []
        opt_groups.extend(_split_decay_params(bundle.expert, float(args.expert_lr)))
        opt_groups.extend(_split_decay_params(bundle.action_in_proj, float(args.proj_lr)))
        opt_groups.extend(_split_decay_params(bundle.action_out_proj, float(args.proj_lr)))
        optimizer_summary = {
            "event": "optimizer_membership_summary",
            "membership": optimizer_membership_summary(
                opt_groups,
                {
                    "action_in_proj": bundle.action_in_proj,
                    "action_out_proj": bundle.action_out_proj,
                },
            ),
        }
        print(json.dumps(optimizer_summary), flush=True)
        summary["optimizer_membership_summary"] = optimizer_summary

        # Joint-train student backbone LoRA params (only meaningful in teacher_forced mode).
        backbone_lora_trainable_count = 0
        if bool(args.train_backbone_lora):
            if str(args.prefix_mode) != "teacher_forced":
                raise ValueError("--train-backbone-lora requires --prefix-mode teacher_forced "
                                 "(stochastic generate() in student_free blocks gradient).")
            backbone_lora_params: list[nn.Parameter] = []
            for pname, p in student.backbone.named_parameters():
                lname = pname.lower()
                if ("lora_a" in lname or "lora_b" in lname or "lora_embedding_a" in lname or "lora_embedding_b" in lname):
                    p.requires_grad = True
                    backbone_lora_params.append(p)
                else:
                    p.requires_grad = False
            backbone_lora_trainable_count = sum(int(p.numel()) for p in backbone_lora_params)
            if backbone_lora_params:
                opt_groups.append({"params": backbone_lora_params,
                                   "lr": float(args.backbone_lora_lr),
                                   "weight_decay": 0.0})
            print(json.dumps({
                "event": "backbone_lora_unfrozen",
                "param_count": backbone_lora_trainable_count,
                "module_count": len(backbone_lora_params),
                "lr": float(args.backbone_lora_lr),
            }), flush=True)
        if bool(args.allow_train_cache_mutation) and int(getattr(args, "train_ade_every", 0)) > 0:
            raise ValueError(
                "--allow-train-cache-mutation is incompatible with --train-ade-every > 0 "
                "because train_inb_ade reuses the same batch cache after train_step."
            )

        optimizer_kwargs: dict[str, Any] = {}
        if bool(args.fused_adamw):
            if torch.device(args.device).type != "cuda":
                raise ValueError("--fused-adamw requires a CUDA device")
            optimizer_kwargs["fused"] = True
        optimizer = torch.optim.AdamW(opt_groups, **optimizer_kwargs)
        optimizer_created = {
            "event": "optimizer_created",
            "optimizer": "AdamW",
            "fused": bool(args.fused_adamw),
        }
        print(json.dumps(optimizer_created), flush=True)
        summary["optimizer_created"] = optimizer_created

        # Cosine LR schedule with warmup (matches alpamayo_base SFT lr_scheduler_type=cosine_warmup_with_min_lr).
        # Only enabled when --lr-warmup-steps > 0.
        scheduler = None
        if int(args.lr_warmup_steps) > 0:
            import math as _math
            warmup_steps_local = int(args.lr_warmup_steps)
            total_steps_local = int(args.steps)
            min_lr_local = float(args.min_lr)

            def _make_lambda(base_lr: float):
                min_ratio = min(1.0, min_lr_local / max(base_lr, 1e-12))
                def _lr_lambda(step_idx: int) -> float:
                    if step_idx < warmup_steps_local:
                        return float(step_idx) / max(1, warmup_steps_local)
                    progress = (step_idx - warmup_steps_local) / max(1, total_steps_local - warmup_steps_local)
                    cosine = 0.5 * (1.0 + _math.cos(_math.pi * progress))
                    return max(min_ratio, cosine * (1.0 - min_ratio) + min_ratio)
                return _lr_lambda

            lambdas = [_make_lambda(g["lr"]) for g in opt_groups]
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
        log_handle = log_path.open("a", encoding="utf-8")
        best_eval: dict[str, Any] | None = None
        start_step = int(
            args.start_step
            if args.start_step is not None
            else ((resume_payload or {}).get("step") or 0)
        )
        if start_step < 0:
            raise ValueError(f"--start-step must be >= 0, got {start_step}")
        summary["start_step"] = int(start_step)
        print(
            json.dumps(
                {
                    "event": "train_loop_start",
                    "start_step": int(start_step),
                    "end_step": int(args.steps),
                    "resume_optimizer_state": False,
                }
            ),
            flush=True,
        )

        if bool(args.eval_only):
            sweep_configs: list[dict[str, Any]]
            if args.eval_sweep_json:
                raw_sweep = str(args.eval_sweep_json)
                if raw_sweep.startswith("@"):
                    raw_sweep = Path(raw_sweep[1:]).read_text(encoding="utf-8")
                parsed_sweep = json.loads(raw_sweep)
                if not isinstance(parsed_sweep, list):
                    raise ValueError("--eval-sweep-json must decode to a list of objects")
                sweep_configs = []
                for idx, entry in enumerate(parsed_sweep):
                    if not isinstance(entry, dict):
                        raise ValueError(f"--eval-sweep-json item {idx} is not an object")
                    sweep_configs.append(dict(entry))
            else:
                sweep_configs = [{}]

            val_evals = []
            train_evals = []
            for sweep_idx, sweep_cfg in enumerate(sweep_configs):
                eval_args = copy.copy(args)
                for key in ("eval_temperature", "eval_num_paths", "eval_selection_method"):
                    if key in sweep_cfg:
                        setattr(eval_args, key, sweep_cfg[key])
                label = str(sweep_cfg.get("label") or f"sweep_{sweep_idx:02d}")
                ev = evaluate(
                    args=eval_args,
                    bundle=bundle,
                    student=student,
                    student_processor=student_processor,
                    student_tokenizer=student_tokenizer,
                    teacher_model=teacher_model,
                    items=val_items,
                    step=start_step,
                    kv_layer_indices=kv_layer_indices,
                )
                ev["event"] = "val_eval"
                ev["sweep_label"] = label
                ev["sweep_index"] = int(sweep_idx)
                print(json.dumps(ev), flush=True)
                log_handle.write(json.dumps(ev) + "\n")
                log_handle.flush()
                val_evals.append(ev)
                if train_eval_items:
                    train_ev = evaluate(
                        args=eval_args,
                        bundle=bundle,
                        student=student,
                        student_processor=student_processor,
                        student_tokenizer=student_tokenizer,
                        teacher_model=teacher_model,
                        items=train_eval_items,
                        step=start_step,
                        kv_layer_indices=kv_layer_indices,
                    )
                    train_ev["event"] = "train_eval"
                    train_ev["sweep_label"] = label
                    train_ev["sweep_index"] = int(sweep_idx)
                    print(json.dumps(train_ev), flush=True)
                    log_handle.write(json.dumps(train_ev) + "\n")
                    log_handle.flush()
                    train_evals.append(train_ev)
            summary.update(
                {
                    "status": "ok",
                    "eval_only": True,
                    "eval_step": int(start_step),
                    "val_eval": val_evals[-1] if val_evals else None,
                    "train_eval": train_evals[-1] if train_evals else None,
                    "eval_sweep": val_evals,
                    "train_eval_sweep": train_evals,
                }
            )
            log_handle.close()
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            return

        if not args.skip_initial_eval and start_step == 0:
            ev = evaluate(
                args=args,
                bundle=bundle,
                student=student,
                student_processor=student_processor,
                student_tokenizer=student_tokenizer,
                teacher_model=teacher_model,
                items=val_items,
                step=0,
                kv_layer_indices=kv_layer_indices,
            )
            ev["event"] = "val_eval"
            print(json.dumps(ev), flush=True)
            log_handle.write(json.dumps(ev) + "\n")
            log_handle.flush()
            best_eval = ev
            if train_eval_items:
                train_ev = evaluate(
                    args=args,
                    bundle=bundle,
                    student=student,
                    student_processor=student_processor,
                    student_tokenizer=student_tokenizer,
                    teacher_model=teacher_model,
                    items=train_eval_items,
                    step=0,
                    kv_layer_indices=kv_layer_indices,
                )
                train_ev["event"] = "train_eval"
                print(json.dumps(train_ev), flush=True)
                log_handle.write(json.dumps(train_ev) + "\n")
                log_handle.flush()

        started = time.perf_counter()
        batches = list(iter_batches(train_items, int(args.batch_size)))

        # --- Prefetch: prepare next batch on CPU/IO while GPU trains ---
        # build_batch has two phases:
        #   1) CPU/IO: image loading, tokenization, FLEX compression (~7s)
        #   2) GPU: student forward for CoT generation + KV cache (~2s)
        # Phase 1 can overlap with the main thread's GPU train_step.
        # Phase 2 serializes on the CUDA default stream (safe, read-only student).
        from concurrent.futures import ThreadPoolExecutor as _PrefetchPool

        _prefetch_exec = _PrefetchPool(max_workers=1)

        _kv_cache_dir = str(getattr(args, "kv_cache_dir", "") or "").strip()
        _use_offline_kv = bool(_kv_cache_dir) and Path(_kv_cache_dir).is_dir()
        if _use_offline_kv:
            print(json.dumps({"event": "using_offline_kv_cache", "dir": _kv_cache_dir}), flush=True)

        def _submit_prefetch(step_idx: int) -> Any:
            items = batches[(step_idx - 1) % len(batches)]
            if _use_offline_kv:
                return _prefetch_exec.submit(
                    load_batch_from_kv_cache,
                    kv_cache_dir=_kv_cache_dir,
                    batch_items=items,
                    device=device,
                )
            return _prefetch_exec.submit(
                build_batch,
                args=args,
                student=student,
                student_processor=student_processor,
                student_tokenizer=student_tokenizer,
                teacher_model=teacher_model,
                batch_items=items,
            )

        _next_future = _submit_prefetch(start_step + 1)

        for step in range(start_step + 1, int(args.steps) + 1):
            batch = _next_future.result()
            # prefetch next batch while GPU runs train_step
            if step < int(args.steps):
                _next_future = _submit_prefetch(step + 1)
            projection_before_step1 = None
            if step == 1:
                projection_before_step1 = {
                    "action_in_proj": snapshot_trainable_params(bundle.action_in_proj),
                    "action_out_proj": snapshot_trainable_params(bundle.action_out_proj),
                }
            optimizer.zero_grad(set_to_none=True)
            loss, stats = train_step(
                bundle=bundle,
                teacher_model=teacher_model,
                batch=batch,
                num_time_samples=int(args.num_time_samples),
                train_timestep_sampler=str(args.train_timestep_sampler),
                device=device,
                allow_cache_mutation=bool(args.allow_train_cache_mutation),
                kv_layer_indices=kv_layer_indices,
            )
            loss.backward()
            if bool(args.train_backbone_lora):
                params_for_clip = [p for p in bundle.parameters() if p.requires_grad]
                params_for_clip += [p for p in student.backbone.parameters() if p.requires_grad]
            else:
                params_for_clip = list(bundle.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clip, float(args.grad_clip_norm))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            projection_delta_step1 = None
            if projection_before_step1 is not None:
                projection_delta_step1 = {
                    "action_in_proj": param_delta_summary(
                        projection_before_step1["action_in_proj"],
                        bundle.action_in_proj,
                    ),
                    "action_out_proj": param_delta_summary(
                        projection_before_step1["action_out_proj"],
                        bundle.action_out_proj,
                    ),
                }
            if step == 1 or step % int(args.log_every) == 0:
                row = {
                    "event": "train_step",
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "grad_norm": float(grad_norm.detach().cpu() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                    "elapsed_sec": round(time.perf_counter() - started, 3),
                    "traj_start_hit_rate": batch["traj_start_hit_rate"],
                    "generated_text_preview": batch["generated_text_preview"],
                    **stats,
                }
                if projection_delta_step1 is not None:
                    row["projection_delta_step1"] = projection_delta_step1
                print(json.dumps(row), flush=True)
                log_handle.write(json.dumps(row) + "\n")
                log_handle.flush()
            if int(getattr(args, "train_ade_every", 0)) > 0 and step % int(args.train_ade_every) == 0:
                # In-batch ADE diagnostic: did the model memorize the training samples it just saw?
                _torch_rng = torch.get_rng_state()
                _cuda_rng = (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                )
                bundle.eval()
                try:
                    train_seed = int(args.seed) + 2_000_000 + int(step)
                    with torch.no_grad():
                        pred_tr = sample_paths(
                            bundle=bundle,
                            teacher_model=teacher_model,
                            batch=batch,
                            seed=train_seed,
                            device=device,
                            temperature=float(getattr(args, "eval_temperature", 1.0)),
                            kv_layer_indices=kv_layer_indices,
                        )
                finally:
                    bundle.train()
                    torch.set_rng_state(_torch_rng)
                    if _cuda_rng is not None:
                        torch.cuda.set_rng_state_all(_cuda_rng)
                target_xyz_tr = batch["target_xyz"].detach().cpu().numpy()
                in_batch_ades: list[float] = []
                in_batch_fdes: list[float] = []
                in_batch_h: dict[str, dict[str, list[float]]] = {
                    "h1p6_16wp": {"ade": [], "fde": []},
                    "h3p2_32wp": {"ade": [], "fde": []},
                    "h6p4_64wp": {"ade": [], "fde": []},
                }
                per_sample_target_xyz_abs: list[float] = []
                for r_idx in range(pred_tr["pred_xyz"].shape[0]):
                    p_xyz = pred_tr["pred_xyz"][r_idx]
                    t_xyz = target_xyz_tr[r_idx]
                    ade_tr, fde_tr = ade_fde(p_xyz, t_xyz)
                    in_batch_ades.append(float(ade_tr))
                    in_batch_fdes.append(float(fde_tr))
                    for name, horizon in (("h1p6_16wp", 16), ("h3p2_32wp", 32), ("h6p4_64wp", 64)):
                        n_h = min(horizon, int(p_xyz.shape[0]), int(t_xyz.shape[0]))
                        h_ade, h_fde = ade_fde(p_xyz[:n_h], t_xyz[:n_h])
                        in_batch_h[name]["ade"].append(float(h_ade))
                        in_batch_h[name]["fde"].append(float(h_fde))
                    per_sample_target_xyz_abs.append(float(np.abs(t_xyz).mean()))
                train_inb_row = {
                    "event": "train_inb_ade",
                    "step": int(step),
                    "train_inb_ade_m": float(np.mean(in_batch_ades)) if in_batch_ades else None,
                    "train_inb_ade_p50_m": float(np.percentile(in_batch_ades, 50)) if in_batch_ades else None,
                    "train_inb_fde_m": float(np.mean(in_batch_fdes)) if in_batch_fdes else None,
                    "train_inb_horizon": {
                        name: {
                            "ade_mean_m": float(np.mean(v["ade"])) if v["ade"] else None,
                            "fde_mean_m": float(np.mean(v["fde"])) if v["fde"] else None,
                        }
                        for name, v in in_batch_h.items()
                    },
                    "batch_sample_ids": list(batch["sample_ids"]),
                    "batch_target_xyz_abs_mean_per_sample": per_sample_target_xyz_abs,
                    "batch_loss": float(loss.detach().cpu()) if loss is not None else None,
                    "train_seed_used": int(train_seed),
                }
                print(json.dumps(train_inb_row), flush=True)
                log_handle.write(json.dumps(train_inb_row) + "\n")
                log_handle.flush()
                del pred_tr
                maybe_cuda_cleanup(int(args.cleanup_every), step)
            del batch, loss
            maybe_cuda_cleanup(int(args.cleanup_every), step)

            should_eval = step % int(args.eval_every) == 0 or step == int(args.steps)
            if should_eval:
                ev = evaluate(
                    args=args,
                    bundle=bundle,
                    student=student,
                    student_processor=student_processor,
                    student_tokenizer=student_tokenizer,
                    teacher_model=teacher_model,
                    items=val_items,
                    step=step,
                    kv_layer_indices=kv_layer_indices,
                )
                ev["event"] = "val_eval"
                print(json.dumps(ev), flush=True)
                log_handle.write(json.dumps(ev) + "\n")
                log_handle.flush()
                if train_eval_items:
                    train_ev = evaluate(
                        args=args,
                        bundle=bundle,
                        student=student,
                        student_processor=student_processor,
                        student_tokenizer=student_tokenizer,
                        teacher_model=teacher_model,
                        items=train_eval_items,
                        step=step,
                        kv_layer_indices=kv_layer_indices,
                    )
                    train_ev["event"] = "train_eval"
                    print(json.dumps(train_ev), flush=True)
                    log_handle.write(json.dumps(train_ev) + "\n")
                    log_handle.flush()
                if best_eval is None or float(ev.get("ade_mean_m") or 1e9) < float(best_eval.get("ade_mean_m") or 1e9):
                    best_eval = ev
                    save_checkpoint(args.output_dir / "best.pt", bundle=bundle, payload={"step": step, "eval": ev, "args": vars(args)})
            if args.save_every and step % int(args.save_every) == 0:
                save_checkpoint(args.output_dir / f"step_{step:06d}.pt", bundle=bundle, payload={"step": step, "args": vars(args)})

        _prefetch_exec.shutdown(wait=False)
        save_checkpoint(args.output_dir / "final.pt", bundle=bundle, payload={"step": int(args.steps), "args": vars(args)})
        summary.update(
            {
                "status": "ok",
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "best_eval": best_eval,
            }
        )
        log_handle.close()
    except Exception as exc:  # noqa: BLE001
        summary.update({"status": "failed", "error": repr(exc)})
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        raise
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"event": "done", "summary_json": str(summary_path), "status": summary["status"]}), flush=True)


if __name__ == "__main__":
    main()
