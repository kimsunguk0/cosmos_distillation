#!/usr/bin/env python3
"""Evaluate AE28 student end-to-end with the plain Alpamayo-1.5 VLM rollout protocol.

This script intentionally imports scripts/84_train_student_ae28_official.py as
the source of truth for student loading, AE bundle construction, split
selection, batch/collator helper calls, checkpoint loading, and trajectory
decode helpers. The eval path here only replaces the non-official parts of the
old evaluator: stochastic backbone rollout, six independent CoT samples, and
official flow-matching sampling defaults.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AE84_PATH = PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py"


def _import_ae84() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ae84_train_student_ae28_official", AE84_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {AE84_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AE84 = _import_ae84()

from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor  # noqa: E402
from alpamayo1_5.models.token_utils import replace_padding_after_eos  # noqa: E402


DEFAULT_AE_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "action_expert"
    / "ae_formatfix_444k_studentfree_20260727"
    / "main"
    / "step_030000.pt"
)
DEFAULT_STUDENT_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "stepb_ceonly_444k_formatfix_20260726"
    / "formatfix_e0"
    / "best_decode"
)
DEFAULT_CORPUS_JSONL = PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_full444k.jsonl"
DEFAULT_SPLIT_CACHE_JSON = PROJECT_ROOT / "outputs" / "action_expert" / "split_cache_444k_10k_seed42.json"
DEFAULT_TRAIN_LOG_JSONL = (
    PROJECT_ROOT
    / "outputs"
    / "action_expert"
    / "ae_formatfix_444k_studentfree_20260727"
    / "main"
    / "train_log.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "action_expert"
    / "ae_formatfix_444k_studentfree_20260727"
    / "official_protocol_step30000"
)

OFFICIAL_BACKBONE_TEMPERATURE = 0.6
OFFICIAL_BACKBONE_TOP_P = 0.98
OFFICIAL_NUM_TRAJ_SAMPLES = 6
OFFICIAL_DIFFUSION_TEMPERATURE = 1.0
OFFICIAL_DIFFUSION_STEPS = 10
OFFICIAL_TOKENS_PER_FUTURE_TRAJ = 128
OFFICIAL_TRAJ_TOKEN_START_IDX = 151669
OFFICIAL_TRAJ_VOCAB_SIZE = 4000

HORIZON_SPECS = (("h1p6_16wp", 16), ("h3p2_32wp", 32), ("h6p4_64wp", 64))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ae-checkpoint", type=Path, default=DEFAULT_AE_CHECKPOINT)
    parser.add_argument("--student-checkpoint-dir", type=Path, default=DEFAULT_STUDENT_CHECKPOINT_DIR)
    parser.add_argument("--teacher-checkpoint-path", type=Path, default=AE84.DEFAULT_TEACHER)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS_JSONL)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=391586)
    parser.add_argument("--val-samples", type=int, default=10000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--split-scan-all", action="store_true", default=True)
    parser.add_argument("--no-split-scan-all", action="store_false", dest="split_scan_all")
    parser.add_argument("--split-cache-json", type=Path, default=DEFAULT_SPLIT_CACHE_JSON)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--geometry-reference", choices=("teacher", "gt"), default="teacher")
    parser.add_argument("--train-log-jsonl", type=Path, default=DEFAULT_TRAIN_LOG_JSONL)
    parser.add_argument("--expected-log-step", type=int, default=30000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reserve-vram-gib", type=float, default=60.0)

    parser.add_argument("--backbone-temperature", type=float, default=OFFICIAL_BACKBONE_TEMPERATURE)
    parser.add_argument("--backbone-top-p", type=float, default=OFFICIAL_BACKBONE_TOP_P)
    parser.add_argument("--num-traj-samples", type=int, default=OFFICIAL_NUM_TRAJ_SAMPLES)
    parser.add_argument("--diffusion-temperature", type=float, default=OFFICIAL_DIFFUSION_TEMPERATURE)
    parser.add_argument("--diffusion-steps", type=int, default=OFFICIAL_DIFFUSION_STEPS)

    parser.add_argument("--student-model", default=AE84.resolve_student_model_path())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--ae-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"), default="flash_attention_2")
    parser.add_argument("--teacher-load-device", default="cpu")
    parser.add_argument("--mapping", choices=("linspace_round", "first_n"), default="linspace_round")
    parser.add_argument("--compressed-layers", type=int, default=28)
    parser.add_argument(
        "--ae-init-mode",
        choices=(
            "teacher_compressed",
            "scratch",
            "student_backbone_init",
            "student_backbone_init_teacher_q",
            "ae_checkpoint_compressed",
        ),
        default="teacher_compressed",
    )
    parser.add_argument("--init-ae-source-checkpoint", default="")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--disable-student-deepstack", action="store_true")
    parser.add_argument("--preserve-flex-positions", action="store_true")
    parser.add_argument("--flex-selection-strategy", choices=("first", "uniform"), default="first")
    parser.add_argument("--flex-scene-deepstack", action="store_true")
    parser.add_argument("--qat-quantization", choices=("", "int4_awq", "int4_blockwise", "int4_ffn_only"), default="")
    parser.add_argument("--qat-calib-samples", type=int, default=256)
    parser.add_argument("--log-every-samples", type=int, default=16)
    return parser.parse_args()


def emit(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False, default=str), flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def seed_all(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_seed_pair(base_seed: int, eval_index: int) -> dict[str, int]:
    stem = int(base_seed) + int(eval_index) * 1009
    return {"backbone_seed": stem, "diffusion_seed": stem + 1}


def make_ae84_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the Namespace shape expected by helpers imported from script 84."""
    return argparse.Namespace(
        corpus_jsonl=args.corpus_jsonl,
        split=args.split,
        num_samples=int(args.num_samples),
        steps=0,
        batch_size=1,
        eval_samples=int(args.eval_samples),
        eval_num_paths=int(args.num_traj_samples),
        eval_temperature=float(args.diffusion_temperature),
        eval_selection_method="single",
        eval_vectorize_paths=False,
        eval_path_batch_size=0,
        eval_log_rows=-1,
        train_ade_every=0,
        eval_batch_size=1,
        eval_every=0,
        log_every=0,
        val_fraction=float(args.val_fraction),
        val_samples=int(args.val_samples),
        eval_train_samples=0,
        split_seed=args.split_seed,
        split_scan_all=bool(args.split_scan_all),
        split_cache_json=args.split_cache_json,
        student_model=args.student_model,
        student_checkpoint_dir=args.student_checkpoint_dir,
        teacher_checkpoint_path=args.teacher_checkpoint_path,
        output_dir=args.output_dir,
        disable_student_deepstack=bool(args.disable_student_deepstack),
        preserve_flex_positions=bool(args.preserve_flex_positions),
        flex_selection_strategy=str(args.flex_selection_strategy),
        flex_scene_deepstack=bool(args.flex_scene_deepstack),
        qat_quantization=str(args.qat_quantization),
        qat_calib_samples=int(args.qat_calib_samples),
        kv_cache_dir="",
        device=str(args.device),
        student_dtype=str(args.student_dtype),
        ae_dtype=str(args.ae_dtype),
        attn_implementation=str(args.attn_implementation),
        teacher_load_device=str(args.teacher_load_device),
        mapping=str(args.mapping),
        compressed_layers=int(args.compressed_layers),
        prefix_mode="student_free",
        ae_init_mode=str(args.ae_init_mode),
        init_ae_source_checkpoint=str(args.init_ae_source_checkpoint),
        target_source=str(args.geometry_reference),
        max_new_tokens=OFFICIAL_TOKENS_PER_FUTURE_TRAJ,
        num_time_samples=1,
        train_timestep_sampler="beta",
        stage2_attention_mode="official_vlm_rollout",
        expert_lr=0.0,
        proj_lr=0.0,
        weight_decay=0.0,
        grad_clip_norm=0.0,
        lr_warmup_steps=0,
        min_lr=0.0,
        no_norm_bias_decay=True,
        allow_train_cache_mutation=False,
        fused_adamw=False,
        train_backbone_lora=False,
        backbone_lora_lr=0.0,
        seed=int(args.seed),
        eval_seed_mode="fixed",
        save_every=0,
        resume_ae_checkpoint=args.ae_checkpoint,
        start_step=int(args.expected_log_step),
        cleanup_every=0,
        eval_cleanup_every=0,
        reserve_vram_gib=float(args.reserve_vram_gib),
        eval_only=True,
        eval_sweep_json=None,
        skip_initial_eval=True,
        max_length=int(args.max_length),
    )


def read_official_config(args: argparse.Namespace) -> dict[str, int]:
    config_path = Path(args.teacher_checkpoint_path) / "config.json"
    values = {
        "tokens_per_future_traj": OFFICIAL_TOKENS_PER_FUTURE_TRAJ,
        "traj_token_start_idx": OFFICIAL_TRAJ_TOKEN_START_IDX,
        "traj_vocab_size": OFFICIAL_TRAJ_VOCAB_SIZE,
    }
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        for key in tuple(values):
            if key in payload:
                values[key] = int(payload[key])
    return values


def load_expected_val_sample_ids(path: Path, expected_step: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Expected train log not found: {path}")
    for line_index, line in enumerate(path.open("r", encoding="utf-8"), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("event") != "val_eval" or int(row.get("step", -1)) != int(expected_step):
            continue
        rows = row.get("rows")
        if not isinstance(rows, list):
            raise RuntimeError(f"val_eval step {expected_step} at line {line_index} has no rows list")
        return [str(item["sample_id"]) for item in rows]
    raise RuntimeError(f"No val_eval event found for step {expected_step} in {path}")


def verify_val_sample_ids(eval_items: list[dict[str, Any]], expected_ids: list[str]) -> dict[str, Any]:
    actual_ids = [str(item["sample_id"]) for item in eval_items]
    actual_set = set(actual_ids)
    expected_set = set(expected_ids)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing or unexpected or len(actual_ids) != len(expected_ids):
        raise RuntimeError(
            "Val sample_id set mismatch against step-30000 val_eval rows: "
            f"actual_count={len(actual_ids)} expected_count={len(expected_ids)} "
            f"missing_head={missing[:8]} unexpected_head={unexpected[:8]}"
        )
    return {
        "event": "val_sample_id_check_done",
        "actual_count": len(actual_ids),
        "expected_count": len(expected_ids),
        "set_equal": True,
        "order_equal": actual_ids == expected_ids,
        "sample_ids_head": actual_ids[:8],
    }


def prepare_official_prompt_batch(
    *,
    args: argparse.Namespace,
    student: Any,
    student_processor: Any,
    student_tokenizer: Any,
    teacher_model: Any,
    batch_items: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    rows = [item["row"] for item in batch_items]
    image_batch = [AE84.load_sample_images(row, AE84.PROJECT_ROOT) for row in rows]
    histories_xyz = [AE84.load_ego_history_xyz(row, AE84.PROJECT_ROOT).astype(np.float32) for row in rows]
    histories_rot = [
        AE84.normalize_history_rot(AE84.load_ego_history_rot(row, AE84.PROJECT_ROOT))
        for row in rows
    ]
    prompt_messages = []
    for row, images, hist_xyz in zip(rows, image_batch, histories_xyz):
        camera_indices = AE84.resolve_camera_indices(row, AE84.PROJECT_ROOT, image_count=len(images))
        frames_per_camera = max(len(images) // max(len(camera_indices), 1), 1)
        prompt_text = AE84.build_user_prompt(
            row,
            AE84.PROJECT_ROOT,
            ego_history_xyz=hist_xyz,
            prompt_text_style="official_alpamayo",
        )
        prompt_messages.append(
            AE84.build_messages(
                prompt_text,
                len(images),
                completion_text=None,
                assistant_prefix="<|cot_start|>",
                image_prompt_style="camera_labeled",
                camera_indices=camera_indices,
                num_frames_per_camera=frames_per_camera,
            )
        )

    encoded = AE84._encode_messages(
        student_processor,
        prompt_messages,
        image_batch,
        int(args.max_length),
        continue_final_message=True,
    )
    encoded["input_ids"] = AE84.fuse_history_tokens_in_input_ids(
        encoded["input_ids"],
        student_tokenizer,
        histories_xyz,
    )

    flex_rope_deltas = None
    flex_enabled = bool(hasattr(student, "flex_enabled") and student.flex_enabled())
    if flex_enabled:
        flex_cfg = getattr(student, "flex_scene_config", None)
        image_token_id = getattr(student, "image_token_id", None)
        if flex_cfg is None or image_token_id is None:
            raise RuntimeError("FLEX student is enabled but missing flex_scene_config or image_token_id.")
        if bool(getattr(args, "preserve_flex_positions", False)):
            encoded = AE84.attach_qwen_mrope_position_ids(encoded, student)
            conditional = student._conditional_backbone() if hasattr(student, "_conditional_backbone") else student
            qwen_model = getattr(conditional, "model", None)
            get_rope_index = getattr(qwen_model, "get_rope_index", None)
            if get_rope_index is not None:
                _, flex_rope_deltas = get_rope_index(
                    input_ids=encoded["input_ids"],
                    image_grid_thw=encoded["image_grid_thw"],
                    video_grid_thw=None,
                    attention_mask=encoded["attention_mask"],
                )
                encoded["flex_rope_deltas"] = flex_rope_deltas
        encoded = AE84.compress_batch_for_flex(
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

    encoded = AE84._to_device_batch(encoded, device)
    prompt_lengths = encoded["attention_mask"].sum(dim=1).to(dtype=torch.long).tolist()

    target_xyz_np: list[np.ndarray] = []
    target_rot_np: list[np.ndarray] = []
    expected_wp = 64
    for item, row in zip(batch_items, rows):
        if str(args.geometry_reference) == "gt":
            xyz = AE84.load_ego_future_xyz(row, AE84.PROJECT_ROOT).astype(np.float32)
            rot = AE84.load_ego_future_rot(row, AE84.PROJECT_ROOT).astype(np.float32)
            if xyz.shape[0] < expected_wp or rot.shape[0] < expected_wp:
                raise ValueError(
                    f"GT future too short for {item.get('sample_id')}: "
                    f"xyz={xyz.shape[0]} rot={rot.shape[0]} expected>={expected_wp}"
                )
            xyz = xyz[:expected_wp]
            rot = rot[:expected_wp]
        else:
            xyz, rot = AE84.raw_teacher_pred(Path(item["raw_json"]))
        target_xyz_np.append(np.asarray(xyz, dtype=np.float32))
        target_rot_np.append(np.asarray(rot, dtype=np.float32))

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

    traj_start_id = student_tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
    if not isinstance(traj_start_id, int) or traj_start_id < 0:
        raise ValueError("Student tokenizer is missing <|traj_future_start|>")

    model_kwargs = dict(encoded)
    input_ids = model_kwargs.pop("input_ids")
    flex_rope_deltas = model_kwargs.pop("flex_rope_deltas", flex_rope_deltas)
    return {
        "sample_ids": [str(item["sample_id"]) for item in batch_items],
        "encoded": encoded,
        "input_ids": input_ids,
        "model_kwargs": model_kwargs,
        "prompt_lengths": [int(x) for x in prompt_lengths],
        "flex_enabled": flex_enabled,
        "flex_rope_deltas": flex_rope_deltas,
        "target_action": target_action.detach(),
        "target_xyz": target_xyz.detach(),
        "ego_history_xyz": ego_history_xyz.detach(),
        "ego_history_rot": ego_history_rot.detach(),
        "traj_start_id": int(traj_start_id),
    }


def expanded_rope_deltas(
    rope_deltas: torch.Tensor,
    *,
    original_batch_size: int,
    num_traj_samples: int,
    b_star: int,
    device: torch.device,
) -> torch.Tensor:
    rope = rope_deltas.to(device)
    if rope.ndim == 1:
        rope = rope.view(-1, 1)
    if int(rope.shape[0]) == int(b_star):
        return rope
    if int(rope.shape[0]) == int(original_batch_size):
        return rope.repeat_interleave(int(num_traj_samples), dim=0)
    if int(rope.shape[0]) == 1:
        return rope.repeat(int(b_star), *([1] * (rope.ndim - 1)))
    raise RuntimeError(
        f"Cannot align rope_deltas shape={tuple(rope.shape)} to expanded batch {b_star}"
    )


def shape_list(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return [int(dim) for dim in value.shape]
    return None


def tensor_batch_dim(value: torch.Tensor, *, name: str) -> int:
    if name == "position_ids" and value.ndim == 3 and int(value.shape[0]) in (3, 4):
        return int(value.shape[1])
    if value.ndim == 0:
        raise RuntimeError(f"AE shape mismatch: {name} is scalar, expected batch dimension")
    return int(value.shape[0])


def prompt_cache_batch_size(prompt_cache: Any) -> int | None:
    layers = getattr(prompt_cache, "layers", None)
    if layers is not None:
        for layer in layers:
            for attr in ("keys", "values"):
                value = getattr(layer, attr, None)
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    return int(value.shape[0])
    try:
        first_layer = prompt_cache[0]
    except Exception:  # noqa: BLE001
        return None
    if isinstance(first_layer, (tuple, list)) and first_layer:
        value = first_layer[0]
    else:
        value = first_layer
    if isinstance(value, torch.Tensor) and value.ndim > 0:
        return int(value.shape[0])
    return None


def repeat_tensor_to_batch(
    value: torch.Tensor,
    *,
    name: str,
    original_batch_size: int,
    num_traj_samples: int,
    b_star: int,
) -> torch.Tensor:
    batch = int(value.shape[0])
    if batch == int(b_star):
        return value
    if batch == int(original_batch_size):
        return value.repeat_interleave(int(num_traj_samples), dim=0)
    if batch == 1:
        repeats = [1] * value.ndim
        repeats[0] = int(b_star)
        return value.repeat(*repeats)
    raise RuntimeError(
        f"Cannot align {name} shape={tuple(value.shape)} to expanded batch {b_star}"
    )


def repeat_position_ids_to_batch(
    position_ids: torch.Tensor,
    *,
    original_batch_size: int,
    num_traj_samples: int,
    b_star: int,
) -> torch.Tensor:
    batch = tensor_batch_dim(position_ids, name="position_ids")
    if batch == int(b_star):
        return position_ids
    if position_ids.ndim == 3 and int(position_ids.shape[0]) in (3, 4):
        dim = 1
    else:
        dim = 0
    if batch == int(original_batch_size):
        return position_ids.repeat_interleave(int(num_traj_samples), dim=dim)
    if batch == 1:
        repeats = [1] * position_ids.ndim
        repeats[dim] = int(b_star)
        return position_ids.repeat(*repeats)
    raise RuntimeError(
        f"Cannot align position_ids shape={tuple(position_ids.shape)} to expanded batch {b_star}"
    )


def expert_attention_implementation(expert: Any) -> str:
    config = getattr(expert, "config", None)
    if config is None:
        return ""
    return str(
        getattr(
            config,
            "_attn_implementation",
            getattr(config, "attn_implementation", ""),
        )
        or ""
    )


def add_protocol_deviation_once(
    protocol_deviations: list[dict[str, str]],
    *,
    name: str,
    reason: str,
) -> None:
    if any(str(item.get("name")) == str(name) for item in protocol_deviations):
        return
    protocol_deviations.append({"name": str(name), "reason": str(reason)})


def build_fa2_expert_attention_mask(
    *,
    sequences: torch.Tensor,
    prefix_mask: torch.Tensor | None,
    offset: torch.Tensor,
    kv_cache_seq_len: int,
    n_diffusion_tokens: int,
    b_star: int,
    pad_token_id: int | None,
    device: torch.device,
) -> torch.Tensor:
    kv_width = int(kv_cache_seq_len)
    mask = torch.ones((int(b_star), kv_width), dtype=torch.long, device=device)
    seq_width = min(kv_width, int(sequences.shape[1]))
    if seq_width > 0 and pad_token_id is not None:
        mask[:, :seq_width] = sequences[:, :seq_width].ne(int(pad_token_id)).to(dtype=mask.dtype)
    if prefix_mask is not None:
        prefix = prefix_mask.to(device=device)
        prefix_width = min(kv_width, int(prefix.shape[1]))
        if prefix_width > 0:
            mask[:, :prefix_width] = prefix[:, :prefix_width].to(dtype=mask.dtype)
    for row in range(int(b_star)):
        gap_start = int(offset[row].item())
        if gap_start < kv_width:
            mask[row, max(gap_start, 0) :] = 0
    diffusion_mask = torch.ones((int(b_star), int(n_diffusion_tokens)), dtype=mask.dtype, device=device)
    return torch.cat([mask, diffusion_mask], dim=1)


def assert_ae_input_shapes(
    *,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_cache: Any,
    b_star: int,
    attention_implementation: str,
) -> None:
    checks = (
        ("inputs_embeds", inputs_embeds),
        ("position_ids", position_ids),
    )
    for name, value in checks:
        batch = tensor_batch_dim(value, name=name)
        if batch != int(b_star):
            raise RuntimeError(
                f"AE shape mismatch: {name} batch={batch} shape={tuple(value.shape)} expected_b_star={b_star}"
            )
    if attention_mask is not None:
        mask_batch = tensor_batch_dim(attention_mask, name="attention_mask")
        if mask_batch != int(b_star):
            raise RuntimeError(
                "AE shape mismatch: attention_mask "
                f"batch={mask_batch} shape={tuple(attention_mask.shape)} expected_b_star={b_star}"
            )
        if str(attention_implementation) == "flash_attention_2" and attention_mask.ndim != 2:
            raise RuntimeError(
                "AE shape mismatch: attention_mask rank "
                f"shape={tuple(attention_mask.shape)} expected_rank=2 for flash_attention_2"
            )
    cache_batch = prompt_cache_batch_size(prompt_cache)
    if cache_batch != int(b_star):
        raise RuntimeError(
            "AE shape mismatch: prompt_cache "
            f"batch={cache_batch} expected_b_star={b_star}"
        )


def official_backbone_rollout(
    *,
    args: argparse.Namespace,
    student: Any,
    student_tokenizer: Any,
    eval_batch: dict[str, Any],
    official_config: dict[str, int],
    device: torch.device,
    seed: int,
    protocol_deviations: list[dict[str, str]],
) -> dict[str, Any]:
    if bool(eval_batch["flex_enabled"]):
        deviation = {
            "name": "flex_sampled_generate",
            "reason": "script 84 only implements manual FLEX cache generation for greedy decoding, not sampled num_return_sequences rollouts",
        }
        protocol_deviations.append(deviation)
        raise RuntimeError(deviation["reason"])

    num_traj_samples = int(args.num_traj_samples)
    input_ids = eval_batch["input_ids"]
    model_kwargs = eval_batch["model_kwargs"]
    prompt_lengths = [
        length
        for length in eval_batch["prompt_lengths"]
        for _ in range(num_traj_samples)
    ]
    generation_config = copy.deepcopy(student.backbone.generation_config)
    generation_config.do_sample = True
    generation_config.temperature = float(args.backbone_temperature)
    generation_config.top_p = float(args.backbone_top_p)
    generation_config.top_k = None
    generation_config.num_return_sequences = num_traj_samples
    generation_config.num_beams = 1
    generation_config.max_new_tokens = int(official_config["tokens_per_future_traj"])
    generation_config.output_logits = True
    generation_config.output_scores = False
    generation_config.output_hidden_states = False
    generation_config.return_dict_in_generate = True
    generation_config.pad_token_id = student_tokenizer.pad_token_id
    generation_config.use_cache = True

    stopping_criteria = StoppingCriteriaList(
        [AE84.StopAfterToken(int(eval_batch["traj_start_id"]), prompt_lengths)]
    )
    logits_processor = LogitsProcessorList(
        [
            ExpertLogitsProcessor(
                traj_token_offset=int(official_config["traj_token_start_idx"]),
                traj_vocab_size=int(official_config["traj_vocab_size"]),
            )
        ]
    )

    seed_all(seed)
    with torch.no_grad(), torch.autocast(
        "cuda",
        dtype=AE84.torch_dtype_from_name(args.student_dtype),
        enabled=device.type == "cuda" and torch.cuda.is_available(),
    ):
        outputs = student.backbone.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            use_cache=True,
            **model_kwargs,
        )

    if getattr(outputs, "past_key_values", None) is None:
        raise RuntimeError("Official backbone rollout did not return past_key_values.")
    outputs.sequences = replace_padding_after_eos(
        token_ids=outputs.sequences,
        eos_token_id=int(eval_batch["traj_start_id"]),
        pad_token_id=student_tokenizer.pad_token_id,
    )
    generated_ids = outputs.sequences[:, int(input_ids.shape[1]) :]
    generated_texts = student_tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens=False)
    hit_rate = float(
        sum("<|traj_future_start|>" in text for text in generated_texts)
        / max(len(generated_texts), 1)
    )
    return {
        "outputs": outputs,
        "generated_texts": generated_texts,
        "traj_start_hit_rate": hit_rate,
    }


def official_ae_flow_sample(
    *,
    args: argparse.Namespace,
    bundle: Any,
    student: Any,
    student_tokenizer: Any,
    teacher_model: Any,
    eval_batch: dict[str, Any],
    rollout: dict[str, Any],
    kv_layer_indices: list[int] | None,
    device: torch.device,
    seed: int,
    protocol_deviations: list[dict[str, str]],
) -> dict[str, np.ndarray]:
    seed_all(seed)
    dtype = next(bundle.parameters()).dtype
    outputs = rollout["outputs"]
    sequences = outputs.sequences
    prompt_cache = outputs.past_key_values
    if kv_layer_indices is not None:
        prompt_cache = AE84.select_kv_cache_layers(prompt_cache, kv_layer_indices)

    b_star = int(sequences.shape[0])
    original_batch_size = int(eval_batch["input_ids"].shape[0])
    num_traj_samples = int(args.num_traj_samples)
    prefix_mask = eval_batch["encoded"].get("attention_mask")
    if prefix_mask is not None:
        prefix_mask = repeat_tensor_to_batch(
            prefix_mask,
            name="prefix_mask",
            original_batch_size=original_batch_size,
            num_traj_samples=num_traj_samples,
            b_star=b_star,
        )

    rope_deltas = AE84.get_rope_deltas(student.backbone)
    rope_deltas = expanded_rope_deltas(
        rope_deltas,
        original_batch_size=original_batch_size,
        num_traj_samples=num_traj_samples,
        b_star=b_star,
        device=device,
    )
    offset = teacher_model._find_eos_offset(
        sequences=sequences,
        eos_token_id=int(eval_batch["traj_start_id"]),
        device=device,
        warn=False,
    )
    prefill_seq_len = int(prompt_cache.get_seq_length())
    n_diffusion_tokens = int(teacher_model.action_space.get_action_space_dims()[0])
    position_ids, attention_mask_4d = teacher_model._build_expert_pos_ids_and_attn_mask(
        offset=offset,
        rope_deltas=rope_deltas,
        kv_cache_seq_len=prefill_seq_len,
        n_diffusion_tokens=n_diffusion_tokens,
        b_star=b_star,
        device=device,
        prefix_mask=prefix_mask,
    )
    position_ids = repeat_position_ids_to_batch(
        position_ids,
        original_batch_size=original_batch_size,
        num_traj_samples=num_traj_samples,
        b_star=b_star,
    )
    attention_impl = expert_attention_implementation(bundle.expert)
    expert_attention_mask = attention_mask_4d
    attention_mask_kind = "4d_additive"
    if attention_impl == "flash_attention_2":
        expert_attention_mask = build_fa2_expert_attention_mask(
            sequences=sequences,
            prefix_mask=prefix_mask,
            offset=offset,
            kv_cache_seq_len=prefill_seq_len,
            n_diffusion_tokens=n_diffusion_tokens,
            b_star=b_star,
            pad_token_id=getattr(student_tokenizer, "pad_token_id", None),
            device=device,
        )
        attention_mask_kind = "2d_padding_gap_for_flash_attention_2"
        add_protocol_deviation_once(
            protocol_deviations,
            name="expert_attention_mask_flash_attention_2_2d",
            reason=(
                "Qwen3-VL flash_attention_2 consumes a 2D padding mask; the official Alpamayo helper "
                "builds a 4D additive expert mask. This script preserves b*=B*num_return_sequences "
                "and masks prefix padding plus post-<|traj_future_start|> generated KV tokens with a 2D mask."
            ),
        )

    action_dims = teacher_model.action_space.get_action_space_dims()
    forward_kwargs: dict[str, Any] = {}
    if bool(getattr(teacher_model.config, "expert_non_causal_attention", False)):
        forward_kwargs["is_causal"] = False

    ae_shapes_emitted = False

    def step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        nonlocal ae_shapes_emitted
        future_token_embeds = bundle.action_in_proj(x.to(dtype=dtype), t.to(dtype=dtype))
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(x.shape[0], n_diffusion_tokens, -1)
        call_attention_mask = expert_attention_mask
        if call_attention_mask is not None and call_attention_mask.ndim == 4:
            call_attention_mask = call_attention_mask.to(dtype=future_token_embeds.dtype)
        if not ae_shapes_emitted:
            emit(
                {
                    "event": "ae_shapes",
                    "b_star": int(b_star),
                    "original_batch_size": int(original_batch_size),
                    "num_traj_samples": int(num_traj_samples),
                    "sequences_shape": shape_list(sequences),
                    "inputs_embeds_shape": shape_list(future_token_embeds),
                    "position_ids_shape": shape_list(position_ids),
                    "attention_mask_shape": shape_list(call_attention_mask),
                    "attention_mask_kind": attention_mask_kind,
                    "attention_implementation": attention_impl,
                    "prompt_cache_batch_size": prompt_cache_batch_size(prompt_cache),
                    "prompt_cache_seq_len": int(prompt_cache.get_seq_length()),
                    "prefix_mask_shape": shape_list(prefix_mask),
                    "rope_deltas_shape": shape_list(rope_deltas),
                    "offset_shape": shape_list(offset),
                }
            )
            ae_shapes_emitted = True
        assert_ae_input_shapes(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            attention_mask=call_attention_mask,
            prompt_cache=prompt_cache,
            b_star=b_star,
            attention_implementation=attention_impl,
        )
        expert_out = bundle.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=prompt_cache,
            attention_mask=call_attention_mask,
            use_cache=True,
            **forward_kwargs,
        )
        prompt_cache.crop(prefill_seq_len)
        last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
        return bundle.action_out_proj(last_hidden).view(-1, *action_dims)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        action = teacher_model.diffusion.sample(
            batch_size=b_star,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
            inference_step=int(args.diffusion_steps),
            int_method="euler",
            use_classifier_free_guidance=False,
            temperature=float(args.diffusion_temperature),
        )
        hist_xyz = eval_batch["ego_history_xyz"].repeat_interleave(num_traj_samples, dim=0).to(device)
        hist_rot = eval_batch["ego_history_rot"].repeat_interleave(num_traj_samples, dim=0).to(device)
        pred_xyz, pred_rot = teacher_model.action_space.action_to_traj(action, hist_xyz, hist_rot)

    return {
        "action": action.detach().float().cpu().numpy(),
        "pred_xyz": pred_xyz.detach().float().cpu().numpy(),
        "pred_rot": pred_rot.detach().float().cpu().numpy(),
        "kv_cache_seq_len": np.asarray([prefill_seq_len], dtype=np.int64),
        "offset": offset.detach().cpu().numpy(),
    }


def ade_fde_np(pred_xyz: np.ndarray, target_xyz: np.ndarray, horizon: int) -> tuple[float, float]:
    n = min(int(horizon), int(pred_xyz.shape[0]), int(target_xyz.shape[0]))
    ade, fde = AE84.ade_fde(pred_xyz[:n], target_xyz[:n])
    return float(ade), float(fde)


def compute_row_metrics(
    *,
    sample_id: str,
    pred_xyz_paths: np.ndarray,
    target_xyz: np.ndarray,
    num_traj_samples: int,
    seeds: dict[str, int],
    traj_start_hit_rate: float,
    generated_texts: list[str],
) -> dict[str, Any]:
    per_path: dict[str, list[float]] = {"ade": [], "fde": []}
    horizon_per_path: dict[str, dict[str, list[float]]] = {
        name: {"ade": [], "fde": []} for name, _ in HORIZON_SPECS
    }
    for path_index in range(int(num_traj_samples)):
        pred_xyz = pred_xyz_paths[path_index]
        ade, fde = ade_fde_np(pred_xyz, target_xyz, 64)
        per_path["ade"].append(float(ade))
        per_path["fde"].append(float(fde))
        for name, horizon in HORIZON_SPECS:
            h_ade, h_fde = ade_fde_np(pred_xyz, target_xyz, horizon)
            horizon_per_path[name]["ade"].append(float(h_ade))
            horizon_per_path[name]["fde"].append(float(h_fde))

    best_idx = int(np.argmin(per_path["ade"]))
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "num_traj_samples": int(num_traj_samples),
        "ade_single_m": float(per_path["ade"][0]),
        "fde_single_m": float(per_path["fde"][0]),
        "minADE_at_n_m": float(per_path["ade"][best_idx]),
        "fde_minade_at_n_m": float(per_path["fde"][best_idx]),
        "minade_winner_idx": best_idx,
        "ade_all_rollouts_m": [float(x) for x in per_path["ade"]],
        "fde_all_rollouts_m": [float(x) for x in per_path["fde"]],
        "traj_start_hit_rate": float(traj_start_hit_rate),
        "generated_text_preview": [text[:240] for text in generated_texts],
        **seeds,
    }
    if int(num_traj_samples) == 6:
        row["minADE6_m"] = row["minADE_at_n_m"]
        row["fde_minade6_m"] = row["fde_minade_at_n_m"]

    horizon_rows: dict[str, dict[str, Any]] = {}
    for name, _horizon in HORIZON_SPECS:
        h_ades = horizon_per_path[name]["ade"]
        h_fdes = horizon_per_path[name]["fde"]
        h_best_idx = int(np.argmin(h_ades))
        h_row: dict[str, Any] = {
            "ade_single_m": float(h_ades[0]),
            "fde_single_m": float(h_fdes[0]),
            "minADE_at_n_m": float(h_ades[h_best_idx]),
            "fde_minade_at_n_m": float(h_fdes[h_best_idx]),
            "minade_winner_idx": h_best_idx,
            "ade_all_rollouts_m": [float(x) for x in h_ades],
            "fde_all_rollouts_m": [float(x) for x in h_fdes],
        }
        if int(num_traj_samples) == 6:
            h_row["minADE6_m"] = h_row["minADE_at_n_m"]
            h_row["fde_minade6_m"] = h_row["fde_minade_at_n_m"]
        horizon_rows[name] = h_row
        for key, value in h_row.items():
            if isinstance(value, (float, int)):
                row[f"{name}_{key}"] = value
    row["horizon"] = horizon_rows
    return row


def mean_or_none(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def p50_or_none(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.percentile(values, 50)) if values else None


def aggregate_metrics(rows: list[dict[str, Any]], num_traj_samples: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "eval_count": len(rows),
        "num_traj_samples": int(num_traj_samples),
        "ade_single_m": mean_or_none(rows, "ade_single_m"),
        "ade_single_p50_m": p50_or_none(rows, "ade_single_m"),
        "fde_single_m": mean_or_none(rows, "fde_single_m"),
        "fde_single_p50_m": p50_or_none(rows, "fde_single_m"),
        "minADE_at_n_m": mean_or_none(rows, "minADE_at_n_m"),
        "minADE_at_n_p50_m": p50_or_none(rows, "minADE_at_n_m"),
        "fde_minade_at_n_m": mean_or_none(rows, "fde_minade_at_n_m"),
        "fde_minade_at_n_p50_m": p50_or_none(rows, "fde_minade_at_n_m"),
    }
    if int(num_traj_samples) == 6:
        out["minADE6_m"] = out["minADE_at_n_m"]
        out["minADE6_p50_m"] = out["minADE_at_n_p50_m"]
        out["fde_minade6_m"] = out["fde_minade_at_n_m"]
        out["fde_minade6_p50_m"] = out["fde_minade_at_n_p50_m"]
    out["horizon"] = {}
    for name, _horizon in HORIZON_SPECS:
        h = {
            "ade_single_m": mean_or_none(rows, f"{name}_ade_single_m"),
            "ade_single_p50_m": p50_or_none(rows, f"{name}_ade_single_m"),
            "fde_single_m": mean_or_none(rows, f"{name}_fde_single_m"),
            "fde_single_p50_m": p50_or_none(rows, f"{name}_fde_single_m"),
            "minADE_at_n_m": mean_or_none(rows, f"{name}_minADE_at_n_m"),
            "minADE_at_n_p50_m": p50_or_none(rows, f"{name}_minADE_at_n_m"),
            "fde_minade_at_n_m": mean_or_none(rows, f"{name}_fde_minade_at_n_m"),
            "fde_minade_at_n_p50_m": p50_or_none(rows, f"{name}_fde_minade_at_n_m"),
        }
        if int(num_traj_samples) == 6:
            h["minADE6_m"] = h["minADE_at_n_m"]
            h["minADE6_p50_m"] = h["minADE_at_n_p50_m"]
            h["fde_minade6_m"] = h["fde_minade_at_n_m"]
            h["fde_minade6_p50_m"] = h["fde_minade_at_n_p50_m"]
        out["horizon"][name] = h
    return out


def resolve_kv_layer_indices(bundle: Any, selected_layers: list[int]) -> list[int] | None:
    expert_n_layers = int(bundle.expert.config.num_hidden_layers)
    backbone_n_layers = 28
    if expert_n_layers < backbone_n_layers:
        return list(selected_layers)
    return None


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    rows_path = args.output_dir / "rows.jsonl"
    protocol_deviations: list[dict[str, str]] = []
    summary: dict[str, Any] = {
        "created_at_unix": time.time(),
        "status": "running",
        "args": jsonable(vars(args)),
        "protocol_deviations": protocol_deviations,
        "official_reference_line_sanity": {
            "plain_sample_trajectories_from_data_with_vlm_rollout": 218,
            "cfg_nav_sample_trajectories_from_data_with_vlm_rollout_cfg_nav": 408,
            "flow_matching_num_inference_steps_default": 35,
            "flow_matching_temperature_arg": 64,
            "flow_matching_initial_noise_scale": 171,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    try:
        official_config = read_official_config(args)
        resolved_protocol = {
            "backbone_do_sample": True,
            "backbone_temperature": float(args.backbone_temperature),
            "backbone_top_p": float(args.backbone_top_p),
            "backbone_top_k": None,
            "num_traj_samples": int(args.num_traj_samples),
            "backbone_num_return_sequences": int(args.num_traj_samples),
            "max_new_tokens": int(official_config["tokens_per_future_traj"]),
            "return_dict_in_generate": True,
            "use_cache": True,
            "logits_processor": {
                "class": "ExpertLogitsProcessor",
                "traj_token_offset": int(official_config["traj_token_start_idx"]),
                "traj_vocab_size": int(official_config["traj_vocab_size"]),
            },
            "stopping_criteria": "scripts/84 StopAfterToken on <|traj_future_start|>",
            "diffusion_int_method": "euler",
            "diffusion_steps": int(args.diffusion_steps),
            "diffusion_temperature": float(args.diffusion_temperature),
            "use_classifier_free_guidance": False,
            "geometry_reference": str(args.geometry_reference),
        }
        summary["resolved_protocol"] = resolved_protocol
        summary["seed_policy"] = {
            "base_seed": int(args.seed),
            "per_sample_formula": "backbone_seed=seed+eval_index*1009; diffusion_seed=backbone_seed+1",
        }
        emit({"event": "official_protocol_boot", "output_dir": str(args.output_dir), **resolved_protocol})

        helper_args = make_ae84_args(args)
        emit({"event": "train_val_split_start"})
        train_items, val_items, split_summary = AE84.select_train_val_items(helper_args)
        del train_items
        eval_items = val_items[: int(args.eval_samples)]
        expected_ids = load_expected_val_sample_ids(Path(args.train_log_jsonl), int(args.expected_log_step))
        id_check = verify_val_sample_ids(eval_items, expected_ids)
        emit(id_check)
        summary["split_summary"] = split_summary
        summary["val_sample_id_check"] = id_check
        summary["eval_sample_ids_head"] = [item["sample_id"] for item in eval_items[:16]]

        seed_all(int(args.seed))
        device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
        emit({"event": "seeded", "seed": int(args.seed), "device": str(device)})

        student, student_tokenizer, student_processor, base_model = AE84.load_student(helper_args)
        summary["student_base_model"] = str(base_model)

        student_vocab_size = None
        try:
            student_vocab_size = int(len(student_tokenizer))
        except TypeError:
            student_vocab_size = None
        if student_vocab_size is not None:
            mask_end = int(official_config["traj_token_start_idx"]) + int(official_config["traj_vocab_size"])
            if mask_end > student_vocab_size:
                protocol_deviations.append(
                    {
                        "name": "expert_logits_processor_vocab_range",
                        "reason": (
                            "official trajectory-token mask range exceeds the student tokenizer size: "
                            f"mask_end={mask_end} tokenizer_size={student_vocab_size}"
                        ),
                    }
                )
                raise RuntimeError(
                    "ExpertLogitsProcessor mask range exceeds student tokenizer size: "
                    f"mask_end={mask_end} tokenizer_size={student_vocab_size}"
                )

        emit({"event": "load_teacher_action_modules_start", "device": str(args.teacher_load_device)})
        teacher_model, _teacher_processor, _cfg, _cfg_path, _runtime = AE84.load_model_and_processor(
            checkpoint_path=args.teacher_checkpoint_path,
            dtype=AE84.torch_dtype_from_name(args.ae_dtype),
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
        AE84.force_attention(teacher_model.expert, str(args.attn_implementation))

        bundle, selected_layers = AE84.build_bundle(teacher_model, helper_args, student=student)
        checkpoint_payload = AE84.load_bundle_checkpoint(Path(args.ae_checkpoint), bundle=bundle)
        emit(
            {
                "event": "ae_checkpoint_loaded",
                "checkpoint": str(args.ae_checkpoint),
                "payload_step": checkpoint_payload.get("step"),
            }
        )
        bundle.eval()
        kv_layer_indices = resolve_kv_layer_indices(bundle, selected_layers)
        if kv_layer_indices is not None:
            emit(
                {
                    "event": "kv_layer_selection_enabled",
                    "expert_layers": int(bundle.expert.config.num_hidden_layers),
                    "backbone_layers": 28,
                    "kv_indices": kv_layer_indices,
                }
            )
        summary["ae_checkpoint_payload_step"] = checkpoint_payload.get("step")
        summary["ae28_selected_teacher_layers"] = list(selected_layers)
        summary["kv_layer_indices"] = kv_layer_indices

        if hasattr(teacher_model, "vlm"):
            delattr(teacher_model, "vlm")

        reserve_event, reserve_warnings = AE84.reserve_vram_cache(float(args.reserve_vram_gib), device)
        for warning in reserve_warnings:
            emit(warning)
        if reserve_event is not None:
            emit(reserve_event)
        summary["vram_reserved"] = reserve_event
        summary["vram_reserve_warnings"] = reserve_warnings

        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        with rows_path.open("w", encoding="utf-8") as rows_handle:
            for eval_index, item in enumerate(eval_items):
                seeds = sample_seed_pair(int(args.seed), eval_index)
                eval_batch = prepare_official_prompt_batch(
                    args=args,
                    student=student,
                    student_processor=student_processor,
                    student_tokenizer=student_tokenizer,
                    teacher_model=teacher_model,
                    batch_items=[item],
                    device=device,
                )
                rollout = official_backbone_rollout(
                    args=args,
                    student=student,
                    student_tokenizer=student_tokenizer,
                    eval_batch=eval_batch,
                    official_config=official_config,
                    device=device,
                    seed=seeds["backbone_seed"],
                    protocol_deviations=protocol_deviations,
                )
                sampled = official_ae_flow_sample(
                    args=args,
                    bundle=bundle,
                    student=student,
                    student_tokenizer=student_tokenizer,
                    teacher_model=teacher_model,
                    eval_batch=eval_batch,
                    rollout=rollout,
                    kv_layer_indices=kv_layer_indices,
                    device=device,
                    seed=seeds["diffusion_seed"],
                    protocol_deviations=protocol_deviations,
                )
                pred_xyz_paths = sampled["pred_xyz"].reshape(
                    int(args.num_traj_samples),
                    *sampled["pred_xyz"].shape[1:],
                )
                target_xyz = eval_batch["target_xyz"].detach().cpu().numpy()[0]
                row = compute_row_metrics(
                    sample_id=str(item["sample_id"]),
                    pred_xyz_paths=pred_xyz_paths,
                    target_xyz=target_xyz,
                    num_traj_samples=int(args.num_traj_samples),
                    seeds=seeds,
                    traj_start_hit_rate=float(rollout["traj_start_hit_rate"]),
                    generated_texts=list(rollout["generated_texts"]),
                )
                row["kv_cache_seq_len"] = int(sampled["kv_cache_seq_len"][0])
                row["eos_offsets"] = [int(x) for x in sampled["offset"].reshape(-1).tolist()]
                rows_handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                rows_handle.flush()
                rows.append(row)

                done = eval_index + 1
                if done == 1 or done == len(eval_items) or done % max(int(args.log_every_samples), 1) == 0:
                    emit(
                        {
                            "event": "eval_progress",
                            "done": done,
                            "total": len(eval_items),
                            "elapsed_sec": round(time.perf_counter() - started, 3),
                            "last_sample_id": str(item["sample_id"]),
                            "last_ade_single_m": row["ade_single_m"],
                            "last_minADE_at_n_m": row["minADE_at_n_m"],
                        }
                    )
                del eval_batch, rollout, sampled

        metrics = aggregate_metrics(rows, int(args.num_traj_samples))
        summary.update(
            {
                "status": "ok",
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "rows_jsonl": str(rows_path),
                "summary_json": str(summary_path),
                "metrics": metrics,
                "protocol_deviations": protocol_deviations,
            }
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        emit({"event": "done", "status": "ok", "summary_json": str(summary_path), "rows_jsonl": str(rows_path), **metrics})
    except Exception as exc:  # noqa: BLE001
        summary.update(
            {
                "status": "failed",
                "error": repr(exc),
                "protocol_deviations": protocol_deviations,
            }
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        emit({"event": "failed", "summary_json": str(summary_path), "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
