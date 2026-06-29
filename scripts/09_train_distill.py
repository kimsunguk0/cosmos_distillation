#!/usr/bin/env python3
"""WP9-WP13 entrypoint: v3.2 distillation training."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import socket
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.schema_versions import active_versions
from src.inference.checkpoint_eval import DecodeEvalConfig, evaluate_decode_subset, resolve_traj_tokenizer_config_path
from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint, save_student_checkpoint
from src.model.lm_head_adapter import enable_lm_head_token_rows, get_lm_head_token_row_count, get_output_lm_head
from src.model.flex_scene_encoder import FlexSceneConfig
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora
from src.model.student_wrapper import (
    StudentWrapperConfig,
    build_student_model,
    load_student_processor,
    load_student_tokenizer,
)
from src.model.tokenizer_ext import distill_trainable_token_ids
from src.training.collator import DistillationCollator
from src.training.collator import _teacher_traj15_signal_from_sample
from src.training.losses import (
    DistillationLossWeights,
    TrajectoryAuxInterfaceConfig,
    TrajectoryDecodeConfig,
    export_loss_weights,
    get_stage_weights,
    resolve_optional_loss_weight_value,
    resolve_loss_weight_value,
)
from src.training.trainer import (
    OnPolicyGKDConfig,
    ScheduledSamplingConfig,
    TrainerConfig,
    move_batch_to_device,
    prepare_flex_batch_for_model,
    run_train_step,
)
from src.utils.runtime_paths import remap_external_path, resolve_student_model_path
from src.utils.seeds import set_seed
from src.utils.traj_tokens import discrete_traj_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "distill_v3_2.jsonl",
    )
    parser.add_argument(
        "--stage-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train" / "stage_b.yaml",
    )
    parser.add_argument(
        "--student-model",
        default=resolve_student_model_path(),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("COSMOS_DATALOADER_NUM_WORKERS", "0")),
        help="DataLoader worker count. Use >0 to overlap image/token preprocessing with GPU work.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=int(os.environ.get("COSMOS_DATALOADER_PREFETCH_FACTOR", "2")),
        help="DataLoader prefetch factor when --num-workers > 0.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("COSMOS_DATALOADER_PIN_MEMORY", "0") == "1",
        help="Pin DataLoader tensors before device transfer.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("COSMOS_DATALOADER_PERSISTENT_WORKERS", "0") == "1",
        help="Keep DataLoader workers alive between epochs when --num-workers > 0.",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=1,
        help="Print rank-0 training progress to stdout every N optimizer steps. Set <=0 to disable step logs.",
    )
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument(
        "--init-checkpoint-dir",
        type=Path,
        default=None,
        help="Optional checkpoint directory to resume/continue from before training.",
    )
    parser.add_argument(
        "--eval-every-epochs",
        type=float,
        default=0.5,
        help="Run validation every N epochs worth of optimizer steps.",
    )
    parser.add_argument(
        "--save-every-epochs",
        type=float,
        default=1.0,
        help="Save checkpoint every N epochs worth of optimizer steps.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=2,
        help="Stop after this many consecutive validation regressions on the target stage.",
    )
    parser.add_argument(
        "--early-stop-stage",
        default="stage_b",
        help="Stage name on which validation early stopping should be active.",
    )
    parser.add_argument(
        "--multi-gpu",
        choices=("auto", "off", "ddp"),
        default="auto",
        help="Use multiple visible CUDA devices. 'auto' enables local DDP when 2+ GPUs are visible.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "stage_b_v3_2",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "train_summary_v3_2.json",
    )
    parser.add_argument(
        "--data-only-dry-run",
        action="store_true",
        help="Inspect and batch the corpus without loading the student model.",
    )
    parser.add_argument(
        "--skip-asset-check",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("COSMOS_SKIP_ASSET_CHECK", "0") == "1",
        help="Trust corpus paths and skip the startup stat pass over materialized assets.",
    )
    parser.add_argument(
        "--skip-final-save",
        action="store_true",
        default=os.environ.get("COSMOS_SKIP_FINAL_SAVE", "0") == "1",
        help="Run training without writing the final checkpoint. Useful for one-step smoke tests.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load the checkpoint and run one batched validation pass without training or saving.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional cap on validation records for eval-only/probe runs.",
    )
    parser.add_argument(
        "--eval-log-every-batches",
        type=int,
        default=int(os.environ.get("COSMOS_EVAL_LOG_EVERY_BATCHES", "10")),
        help="Print eval-only progress every N validation batches. Set <=0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    # QAT (Quantization-Aware Training) options
    parser.add_argument(
        "--qat-quantization",
        type=str,
        default="",
        choices=["", "int4_awq", "int4_blockwise", "int4_ffn_only", "fp8", "fp8_pcpt", "fp8_vit", "fp8_pcpt_vit"],
        help="Apply ModelOpt fake-quantization to backbone before training. "
             "int4_ffn_only: only FFN (gate/up/down_proj) in INT4, attention QKV/O stays FP16. "
             "fp8_pcpt: FP8 per-channel weights and per-token dynamic activations. "
             "fp8_vit/fp8_pcpt_vit: language FP8 plus visual tower FP8, while keeping "
             "patch/embed/rotary/merger/norm paths unquantized. "
             "Empty = disabled.",
    )
    parser.add_argument(
        "--qat-calib-samples",
        type=int,
        default=512,
        help="Number of samples for QAT calibration forward pass.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL corpus file."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _path_exists(raw_path: str | Path | None) -> bool:
    if raw_path in (None, ""):
        return False
    path_str = remap_external_path(raw_path)
    if path_str is None:
        return False
    return Path(path_str).exists()


def has_required_materialized_assets(record: dict) -> bool:
    """Return True when the record's required v3.2 artifacts exist locally."""
    sample_input = record.get("input") or {}
    hard_target = record.get("hard_target") or {}
    required_paths = [
        sample_input.get("materialized_sample_path"),
        sample_input.get("metadata_path"),
        sample_input.get("ego_history_path"),
        hard_target.get("traj_future_token_ids_path"),
    ]
    if not all(_path_exists(path) for path in required_paths):
        return False
    image_paths = list(sample_input.get("image_paths") or [])
    if image_paths:
        return all(_path_exists(path) for path in image_paths)
    if sample_input.get("image_layout") == "materialized_4x4_png":
        sample_root = remap_external_path(sample_input.get("materialized_sample_path"))
        if sample_root is None:
            return False
        image_names = sample_input.get("image_names") or [
            f"cam{camera_idx}_f{frame_idx}.png" for camera_idx in range(4) for frame_idx in range(4)
        ]
        image_dir = Path(sample_root) / "images"
        return all((image_dir / str(name)).exists() for name in image_names)
    return False


def _record_traj_future_token_ids(record: dict) -> list[int]:
    hard_target = record.get("hard_target") or {}
    raw_token_ids = hard_target.get("traj_future_token_ids") or []
    if raw_token_ids:
        return [int(token_idx) for token_idx in raw_token_ids]
    raw_path = hard_target.get("traj_future_token_ids_path")
    resolved = remap_external_path(raw_path)
    if resolved is None:
        return []
    try:
        return [int(token_idx) for token_idx in np.load(Path(resolved), mmap_mode="r").reshape(-1).tolist()]
    except Exception:  # noqa: BLE001
        return []


def stage_weights_from_yaml(path: Path) -> tuple[TrainerConfig, DistillationLossWeights, dict[str, object]]:
    """Load stage config YAML."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = get_stage_weights(str(config["stage_name"]))
    weights = config.get("loss_weights") or {}

    def _resolve_weights(weights_map: dict[str, object]) -> DistillationLossWeights:
        return DistillationLossWeights(
            hard_cot_ce=resolve_loss_weight_value(weights_map, "hard_cot_ce", defaults.hard_cot_ce),
            teacher_seq_ce=resolve_loss_weight_value(weights_map, "teacher_seq_ce", defaults.teacher_seq_ce),
            teacher_logit_kd=resolve_loss_weight_value(weights_map, "teacher_logit_kd", defaults.teacher_logit_kd),
            traj_ce=resolve_loss_weight_value(weights_map, "traj_ce", defaults.traj_ce),
            traj_aux_reg=resolve_loss_weight_value(weights_map, "traj_aux_reg", defaults.traj_aux_reg),
            format_ce=resolve_loss_weight_value(weights_map, "format_ce", defaults.format_ce),
            action_aux=resolve_loss_weight_value(weights_map, "action_aux", defaults.action_aux),
            feat_align=resolve_loss_weight_value(weights_map, "feat_align", defaults.feat_align),
            teacher_traj_ce=resolve_optional_loss_weight_value(weights_map, "teacher_traj_ce"),
            teacher_traj_topk_kd=resolve_loss_weight_value(
                weights_map,
                "teacher_traj_topk_kd",
                defaults.teacher_traj_topk_kd,
            ),
            teacher_traj_hidden_align=resolve_loss_weight_value(
                weights_map,
                "teacher_traj_hidden_align",
                defaults.teacher_traj_hidden_align,
            ),
            teacher_boundary_hidden_align=resolve_loss_weight_value(
                weights_map,
                "teacher_boundary_hidden_align",
                defaults.teacher_boundary_hidden_align,
            ),
            traj_xyz_reg=resolve_loss_weight_value(weights_map, "traj_xyz_reg", defaults.traj_xyz_reg),
            traj_delta_reg=resolve_loss_weight_value(weights_map, "traj_delta_reg", defaults.traj_delta_reg),
            traj_final_reg=resolve_loss_weight_value(weights_map, "traj_final_reg", defaults.traj_final_reg),
            traj_control_reg=resolve_loss_weight_value(weights_map, "traj_control_reg", defaults.traj_control_reg),
            traj_control_delta_reg=resolve_loss_weight_value(
                weights_map,
                "traj_control_delta_reg",
                defaults.traj_control_delta_reg,
            ),
            traj_aux_xyz_reg=resolve_loss_weight_value(weights_map, "traj_aux_xyz_reg", defaults.traj_aux_xyz_reg),
            traj_aux_final_reg=resolve_loss_weight_value(weights_map, "traj_aux_final_reg", defaults.traj_aux_final_reg),
            traj_aux_guided_kd=resolve_loss_weight_value(weights_map, "traj_aux_guided_kd", defaults.traj_aux_guided_kd),
            traj_aux_pseudo_ce=resolve_loss_weight_value(weights_map, "traj_aux_pseudo_ce", defaults.traj_aux_pseudo_ce),
            boundary_action_xyz=resolve_loss_weight_value(
                weights_map,
                "boundary_action_xyz",
                defaults.boundary_action_xyz,
            ),
        )

    trainer_config = TrainerConfig(
        stage_name=str(config["stage_name"]),
        epochs=float(config.get("epochs", 1.0)),
        max_length=int(config.get("max_length", 4096)),
        bf16=bool(config.get("bf16", True)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        batch_size=int(config.get("batch_size", 1)),
        learning_rate=float(config.get("learning_rate", 2e-5)),
    )
    loss_weights = _resolve_weights(weights)
    stage_options = {
        "data_view": dict(config.get("data_view") or {}),
        "lora": dict(config.get("lora") or {}),
        "attn_implementation": config.get("attn_implementation"),
        "traj_token_reweighting": dict(config.get("traj_token_reweighting") or {}),
        "decode_eval": dict(config.get("decode_eval") or {}),
        "traj_decode_reg": dict(config.get("traj_decode_reg") or {}),
        "traj_aux_interface": dict(config.get("traj_aux_interface") or {}),
        "traj_hidden_bridge": dict(config.get("traj_hidden_bridge") or {}),
        "optimization": dict(config.get("optimization") or {}),
        "curriculum": dict(config.get("curriculum") or {}),
        "interface_loss_weights": dict(config.get("interface_loss_weights") or {}),
        "scheduled_sampling": dict(config.get("scheduled_sampling") or {}),
        "on_policy_gkd": dict(config.get("on_policy_gkd") or {}),
        "flex": dict(config.get("flex") or {}),
        "force_trainable_fp32": bool(config.get("force_trainable_fp32", True)),
        "gradient_checkpointing_use_reentrant": bool(config.get("gradient_checkpointing_use_reentrant", False)),
    }
    return trainer_config, loss_weights, stage_options


def unwrap_model(model):
    """Return the underlying model when wrapped by DataParallel/DDP."""
    return getattr(model, "module", model)


def apply_optimization_policy(model, optimization_cfg: dict[str, object] | None) -> dict[str, object]:
    """Apply stage-specific trainability rules and return a small summary."""
    cfg = dict(optimization_cfg or {})
    summary = {
        "freeze_all_but_traj_aux_head": False,
        "freeze_visual_tower": False,
        "freeze_traj_aux_head": False,
        "freeze_meta_action_head": False,
        "freeze_boundary_action_head": False,
        "freeze_traj_hidden_projector": False,
        "freeze_all_parameters": False,
    }

    explicit_layers = cfg.get("unfreeze_language_layers")
    layer_ids: set[int] = set()
    if isinstance(explicit_layers, (list, tuple)):
        layer_ids = {int(value) for value in explicit_layers}
    raw_layer_from = cfg.get("language_layers_from")
    layer_from = (
        None
        if raw_layer_from is None or raw_layer_from == "" or raw_layer_from is False
        else int(raw_layer_from)
    )
    unfreeze_lora_only = bool(cfg.get("unfreeze_lora_only", False))

    def is_visual_tower_parameter(name: str) -> bool:
        if ".visual." not in name and not name.startswith("visual."):
            return False
        # Qwen3-VL exposes visual mergers as projector/bridge parameters. Keep
        # those trainable when freezing the ViT body.
        if "visual.merger" in name or "visual.deepstack_merger" in name:
            return False
        return True

    def should_unfreeze(name: str) -> bool:
        layer_match = re.search(r"language_model\.layers\.(\d+)\.", name)
        if layer_match is not None:
            layer_index = int(layer_match.group(1))
            if layer_index in layer_ids:
                return (not unfreeze_lora_only) or ("lora_" in name)
            if layer_from is not None and layer_index >= layer_from:
                return (not unfreeze_lora_only) or ("lora_" in name)
        if bool(cfg.get("unfreeze_language_norm", False)) and "language_model.norm" in name:
            return True
        if bool(cfg.get("unfreeze_token_embeddings", False)) and "language_model.embed_tokens" in name:
            return True
        if bool(cfg.get("unfreeze_lm_head", False)) and "lm_head" in name:
            return True
        if bool(cfg.get("unfreeze_multimodal_projector", False)) and (
            "multi_modal_projector" in name or "visual.merger" in name
        ):
            return True
        if bool(cfg.get("unfreeze_flex_scene_encoder", False)) and "flex_scene_encoder" in name:
            return True
        return False

    def apply_explicit_unfreeze_rules() -> list[str]:
        trainable_modules: list[str] = []
        for name, parameter in model.named_parameters():
            if should_unfreeze(name):
                parameter.requires_grad = True

        if bool(cfg.get("unfreeze_lm_head", False)):
            root_model = unwrap_model(model)
            backbone = getattr(root_model, "backbone", root_model)
            try:
                output_head = get_output_lm_head(backbone)
            except Exception:  # noqa: BLE001
                output_head = None
            if output_head is not None:
                for parameter in output_head.parameters():
                    parameter.requires_grad = True

        if bool(cfg.get("unfreeze_traj_aux_head", False)):
            traj_aux_head = getattr(model, "traj_aux_head", None)
            if traj_aux_head is None:
                raise RuntimeError("unfreeze_traj_aux_head requires model.traj_aux_head to exist.")
            for parameter in traj_aux_head.parameters():
                parameter.requires_grad = True
            trainable_modules.append("traj_aux_head")
        if bool(cfg.get("unfreeze_meta_action_head", False)):
            meta_action_head = getattr(model, "meta_action_head", None)
            if meta_action_head is None:
                raise RuntimeError("unfreeze_meta_action_head requires model.meta_action_head to exist.")
            for parameter in meta_action_head.parameters():
                parameter.requires_grad = True
            trainable_modules.append("meta_action_head")
        if bool(cfg.get("unfreeze_boundary_action_head", False)):
            boundary_action_head = getattr(model, "boundary_action_head", None)
            if boundary_action_head is None:
                raise RuntimeError("unfreeze_boundary_action_head requires model.boundary_action_head to exist.")
            for parameter in boundary_action_head.parameters():
                parameter.requires_grad = True
            trainable_modules.append("boundary_action_head")
        if bool(cfg.get("unfreeze_traj_hidden_projector", False)):
            traj_hidden_projector = getattr(model, "traj_hidden_projector", None)
            if traj_hidden_projector is None:
                raise RuntimeError(
                    "unfreeze_traj_hidden_projector requires model.traj_hidden_projector to exist."
                )
            for parameter in traj_hidden_projector.parameters():
                parameter.requires_grad = True
            trainable_modules.append("traj_hidden_projector")
        if bool(cfg.get("unfreeze_traj_hidden_bridge", False)):
            traj_hidden_bridge_student = getattr(model, "traj_hidden_bridge_student", None)
            traj_hidden_bridge_teacher = getattr(model, "traj_hidden_bridge_teacher", None)
            if traj_hidden_bridge_student is None or traj_hidden_bridge_teacher is None:
                raise RuntimeError(
                    "unfreeze_traj_hidden_bridge requires model.traj_hidden_bridge_student/teacher to exist."
                )
            for parameter in traj_hidden_bridge_student.parameters():
                parameter.requires_grad = True
            for parameter in traj_hidden_bridge_teacher.parameters():
                parameter.requires_grad = True
            trainable_modules.append("traj_hidden_bridge")
        if bool(cfg.get("unfreeze_flex_scene_encoder", False)):
            flex_scene_encoder = getattr(model, "flex_scene_encoder", None)
            if flex_scene_encoder is None:
                raise RuntimeError("unfreeze_flex_scene_encoder requires model.flex_scene_encoder to exist.")
            for parameter in flex_scene_encoder.parameters():
                parameter.requires_grad = True
            trainable_modules.append("flex_scene_encoder")

        for layer_index in sorted(layer_ids):
            trainable_modules.append(f"language_layers.{layer_index}")
        if layer_from is not None:
            trainable_modules.append(f"language_layers_from.{layer_from}")
        if unfreeze_lora_only:
            trainable_modules.append("unfreeze_lora_only")
        if bool(cfg.get("unfreeze_language_norm", False)):
            trainable_modules.append("language_norm")
        if bool(cfg.get("unfreeze_token_embeddings", False)):
            trainable_modules.append("token_embeddings")
        if bool(cfg.get("unfreeze_lm_head", False)):
            trainable_modules.append("lm_head")
        if bool(cfg.get("unfreeze_multimodal_projector", False)):
            trainable_modules.append("multimodal_projector")
        return trainable_modules

    if not bool(cfg.get("freeze_all_but_traj_aux_head", False)):
        if bool(cfg.get("freeze_all_parameters", False)):
            for parameter in model.parameters():
                parameter.requires_grad = False
            trainable_modules = apply_explicit_unfreeze_rules()

            return {
                "freeze_all_but_traj_aux_head": False,
                "freeze_traj_aux_head": False,
                "freeze_meta_action_head": False,
                "freeze_boundary_action_head": False,
                "freeze_all_parameters": True,
                "unfreeze_lora_only": unfreeze_lora_only,
                "trainable_modules": trainable_modules,
            }
        trainable_modules = apply_explicit_unfreeze_rules()
        if bool(cfg.get("freeze_visual_tower", cfg.get("freeze_vit", False))):
            frozen_tensors = 0
            frozen_params = 0
            for name, parameter in model.named_parameters():
                if is_visual_tower_parameter(name):
                    if parameter.requires_grad:
                        frozen_tensors += 1
                        frozen_params += int(parameter.numel())
                    parameter.requires_grad = False
            summary["freeze_visual_tower"] = True
            summary["frozen_visual_tensors"] = frozen_tensors
            summary["frozen_visual_params"] = frozen_params
        if bool(cfg.get("freeze_traj_aux_head", False)):
            traj_aux_head = getattr(model, "traj_aux_head", None)
            if traj_aux_head is None:
                raise RuntimeError("freeze_traj_aux_head requires model.traj_aux_head to exist.")
            for parameter in traj_aux_head.parameters():
                parameter.requires_grad = False
            summary["freeze_traj_aux_head"] = True
        if bool(cfg.get("freeze_meta_action_head", False)):
            meta_action_head = getattr(model, "meta_action_head", None)
            if meta_action_head is None:
                raise RuntimeError("freeze_meta_action_head requires model.meta_action_head to exist.")
            for parameter in meta_action_head.parameters():
                parameter.requires_grad = False
            summary["freeze_meta_action_head"] = True
        if bool(cfg.get("freeze_boundary_action_head", False)):
            boundary_action_head = getattr(model, "boundary_action_head", None)
            if boundary_action_head is None:
                raise RuntimeError("freeze_boundary_action_head requires model.boundary_action_head to exist.")
            for parameter in boundary_action_head.parameters():
                parameter.requires_grad = False
            summary["freeze_boundary_action_head"] = True
        if bool(cfg.get("freeze_traj_hidden_projector", False)):
            traj_hidden_projector = getattr(model, "traj_hidden_projector", None)
            if traj_hidden_projector is None:
                raise RuntimeError("freeze_traj_hidden_projector requires model.traj_hidden_projector to exist.")
            for parameter in traj_hidden_projector.parameters():
                parameter.requires_grad = False
            summary["freeze_traj_hidden_projector"] = True
        if trainable_modules:
            summary["trainable_modules"] = trainable_modules
        return summary

    for parameter in model.parameters():
        parameter.requires_grad = False
    traj_aux_head = getattr(model, "traj_aux_head", None)
    if traj_aux_head is None:
        raise RuntimeError("freeze_all_but_traj_aux_head requires model.traj_aux_head to exist.")
    for parameter in traj_aux_head.parameters():
        parameter.requires_grad = True
    return {
        "freeze_all_but_traj_aux_head": True,
        "freeze_traj_aux_head": False,
        "freeze_meta_action_head": False,
        "freeze_boundary_action_head": False,
        "freeze_traj_hidden_projector": False,
        "trainable_modules": ["traj_aux_head"],
    }


def build_optimizer_param_groups(model, base_learning_rate: float, optimization_cfg: dict[str, object] | None):
    """Build simple per-module LR groups when a stage requests them."""
    cfg = dict(optimization_cfg or {})
    named_trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not named_trainable:
        return [], {}
    lm_head_token_row_param_ids: set[int] = set()
    lm_head_base_param_ids: set[int] = set()
    root_model = unwrap_model(model)
    backbone = getattr(root_model, "backbone", root_model)
    try:
        output_head = get_output_lm_head(backbone)
    except Exception:  # noqa: BLE001
        output_head = None
    if output_head is not None:
        for head_name, head_param in output_head.named_parameters(recurse=True):
            if "trainable_tokens_delta" in head_name:
                lm_head_token_row_param_ids.add(id(head_param))
            else:
                lm_head_base_param_ids.add(id(head_param))
    token_row_scale = float(cfg.get("token_row_lr_scale", 1.0) or 1.0)
    lm_head_token_row_scale = float(cfg.get("lm_head_token_row_lr_scale", token_row_scale) or token_row_scale)
    language_full_scale = float(cfg.get("language_full_lr_scale", 1.0) or 1.0)
    language_lora_scale = float(cfg.get("language_lora_lr_scale", language_full_scale) or language_full_scale)
    language_norm_scale = float(cfg.get("language_norm_lr_scale", language_full_scale) or language_full_scale)
    token_embedding_scale = float(cfg.get("token_embedding_lr_scale", language_full_scale) or language_full_scale)
    lm_head_scale = float(cfg.get("lm_head_lr_scale", language_full_scale) or language_full_scale)
    multimodal_projector_scale = float(
        cfg.get("multimodal_projector_lr_scale", 1.0) or 1.0
    )
    projector_scale = float(cfg.get("traj_hidden_projector_lr_scale", 1.0) or 1.0)
    hidden_bridge_scale = float(cfg.get("traj_hidden_bridge_lr_scale", language_full_scale) or language_full_scale)
    meta_action_head_scale = float(cfg.get("meta_action_head_lr_scale", 1.0) or 1.0)
    traj_aux_head_scale = float(cfg.get("traj_aux_head_lr_scale", 1.0) or 1.0)
    boundary_action_head_scale = float(cfg.get("boundary_action_head_lr_scale", 1.0) or 1.0)
    flex_scene_scale = float(cfg.get("flex_scene_lr_scale", 1.0) or 1.0)

    def _group_name_and_scale(name: str, param: torch.nn.Parameter) -> tuple[str, float]:
        if id(param) in lm_head_token_row_param_ids:
            return "shared_embed_lm_head_token_rows", lm_head_token_row_scale
        if id(param) in lm_head_base_param_ids and bool(cfg.get("unfreeze_lm_head", False)):
            return "lm_head", lm_head_scale
        if "embed_tokens.token_adapter.trainable_tokens_delta" in name:
            return "token_rows", token_row_scale
        if bool(getattr(param, "_distill_lm_head_token_rows", False)):
            return "lm_head_token_rows", lm_head_token_row_scale
        if "traj_hidden_projector" in name:
            return "traj_hidden_projector", projector_scale
        if "traj_hidden_bridge_" in name:
            return "traj_hidden_bridge", hidden_bridge_scale
        if "meta_action_head" in name:
            return "meta_action_head", meta_action_head_scale
        if "traj_aux_head" in name:
            return "traj_aux_head", traj_aux_head_scale
        if "boundary_action_head" in name:
            return "boundary_action_head", boundary_action_head_scale
        if "flex_scene_encoder" in name:
            return "flex_scene_encoder", flex_scene_scale
        if "multi_modal_projector" in name or "visual.merger" in name:
            return "multimodal_projector", multimodal_projector_scale
        if ".language_model.layers." in name and "lora_" in name:
            return "language_lora", language_lora_scale
        if ".language_model.layers." in name and "lora_" not in name:
            return "language_full", language_full_scale
        if "language_model.norm" in name and "lora_" not in name:
            return "language_norm", language_norm_scale
        if "language_model.embed_tokens" in name and "token_adapter" not in name and "lora_" not in name:
            return "token_embeddings", token_embedding_scale
        if "lm_head" in name and "token_adapter" not in name and "lora_" not in name:
            return "lm_head", lm_head_scale
        return "default", 1.0

    grouped_params: dict[tuple[str, float], list[torch.nn.Parameter]] = {}
    summary: dict[str, int] = {}
    for name, param in named_trainable:
        group_name, scale = _group_name_and_scale(name, param)
        grouped_params.setdefault((group_name, scale), []).append(param)
        summary[group_name] = summary.get(group_name, 0) + 1

    param_groups = []
    for (group_name, scale), params in sorted(grouped_params.items(), key=lambda item: (item[0][1], item[0][0])):
        group = {"params": params, "lr": float(base_learning_rate) * float(scale)}
        if group_name == "lm_head_token_rows":
            group["weight_decay"] = 0.0
        param_groups.append(group)
        if abs(scale - 1.0) >= 1e-8:
            summary[f"{group_name}_lr_scale"] = float(scale)
    return param_groups, summary


def resolve_parallelism(multi_gpu: str) -> tuple[str, list[int]]:
    """Decide how to use the available CUDA devices."""
    if not torch.cuda.is_available():
        return "cpu", []
    visible = list(range(torch.cuda.device_count()))
    if multi_gpu == "off":
        return "single", visible[:1]
    if multi_gpu in {"auto", "ddp"} and len(visible) >= 2:
        return "ddp", visible
    return "single", visible[:1]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _is_rank_zero(rank: int, world_size: int) -> bool:
    return world_size <= 1 or rank == 0


def _average_scalar_logs(logs: dict[str, float], device: torch.device) -> dict[str, float]:
    if not dist.is_available() or not dist.is_initialized():
        return logs
    keys = list(logs.keys())
    values = torch.tensor([float(logs[key]) for key in keys], device=device, dtype=torch.float32)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {key: float(values[index].item()) for index, key in enumerate(keys)}


def _average_scalar(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def _maybe_init_distributed(rank: int, world_size: int, master_port: int) -> None:
    if world_size <= 1:
        return
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _maybe_cleanup_distributed(world_size: int) -> None:
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def preferred_model_dtype(*, bf16: bool) -> torch.dtype | None:
    """Choose the backbone load dtype for the current runtime."""
    if torch.cuda.is_available() and bf16:
        return torch.bfloat16
    return None


def torch_dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[str(name).strip().lower()]


def load_on_policy_gkd_teacher(
    config: dict[str, object] | None,
    *,
    tokenizer,
    device: torch.device,
    is_rank_zero: bool,
) -> tuple[Any | None, dict[str, object]]:
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return None, {"enabled": False}

    workspace_root = PROJECT_ROOT.parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import load_model_and_processor

    teacher_model_path = Path(
        str(
            cfg.get("teacher_model_path")
            or cfg.get("teacher_checkpoint_path")
            or workspace_root / "base_weights" / "Alpamayo-1.5-10B"
        )
    ).expanduser()
    teacher_dtype = torch_dtype_from_name(str(cfg.get("teacher_dtype", "bfloat16")))
    teacher_device = str(cfg.get("teacher_device") or device)
    config_json = cfg.get("teacher_config_json", cfg.get("config_json", None))
    runtime_support = cfg.get("teacher_runtime_support", cfg.get("runtime_support", None))
    attn_implementation = str(cfg.get("teacher_attn_implementation", cfg.get("attn_implementation", "sdpa")))
    min_pixels = cfg.get("teacher_min_pixels", cfg.get("min_pixels", None))
    max_pixels = cfg.get("teacher_max_pixels", cfg.get("max_pixels", None))

    if is_rank_zero:
        print(
            json.dumps(
                {
                    "event": "on_policy_gkd_teacher_load_start",
                    "teacher_model_path": str(teacher_model_path),
                    "teacher_dtype": str(teacher_dtype).replace("torch.", ""),
                    "teacher_device": teacher_device,
                    "runtime_support": runtime_support,
                    "attn_implementation": attn_implementation,
                }
            ),
            flush=True,
        )

    container_model, processor, teacher_config, config_path, runtime_support_path = load_model_and_processor(
        teacher_model_path,
        dtype=teacher_dtype,
        device=teacher_device,
        config_json=str(config_json) if config_json not in (None, "") else None,
        runtime_support=str(runtime_support) if runtime_support not in (None, "") else None,
        attn_implementation=attn_implementation,
        min_pixels=int(min_pixels) if min_pixels not in (None, "") else None,
        max_pixels=int(max_pixels) if max_pixels not in (None, "") else None,
    )
    teacher_tokenizer = getattr(container_model, "tokenizer", getattr(processor, "tokenizer", None))
    codebook_tokens = ["<i0>", "<i2999>"]
    remap_tokens = [
        "<|cot_start|>",
        "<|cot_end|>",
        "<|traj_future_start|>",
        "<|traj_future_end|>",
        "<|traj_history_start|>",
        "<|traj_history_end|>",
        "<|traj_history|>",
        "<|route_start|>",
        "<|route_end|>",
        "<|question_start|>",
        "<|question_end|>",
    ]
    checked_tokens = [*codebook_tokens, *remap_tokens]
    token_ids: dict[str, dict[str, int]] = {}
    token_id_map: dict[int, int] = {}
    for token in checked_tokens:
        student_id = tokenizer.convert_tokens_to_ids(token)
        teacher_id = teacher_tokenizer.convert_tokens_to_ids(token) if teacher_tokenizer is not None else None
        token_ids[token] = {"student": int(student_id), "teacher": int(teacher_id)}
        if not isinstance(student_id, int) or not isinstance(teacher_id, int):
            raise RuntimeError(
                f"on-policy GKD token missing for {token!r}: student={student_id}, teacher={teacher_id}"
            )
        if token in codebook_tokens and int(student_id) != int(teacher_id):
            raise RuntimeError(
                f"on-policy GKD codebook token-id mismatch for {token!r}: student={student_id}, teacher={teacher_id}"
            )
        if int(student_id) != int(teacher_id):
            token_id_map[int(student_id)] = int(teacher_id)

    teacher_vlm = container_model.vlm
    teacher_vlm.eval()
    for parameter in teacher_vlm.parameters():
        parameter.requires_grad_(False)
    for attr in ("expert", "action_in_proj", "action_out_proj", "diffusion"):
        if hasattr(container_model, attr):
            delattr(container_model, attr)
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "enabled": True,
        "teacher_model_path": str(teacher_model_path),
        "teacher_dtype": str(teacher_dtype).replace("torch.", ""),
        "teacher_device": teacher_device,
        "teacher_config_path": str(config_path),
        "runtime_support": str(runtime_support_path) if runtime_support_path is not None else None,
        "attn_implementation": attn_implementation,
        "token_ids": token_ids,
        "token_id_map": token_id_map,
    }
    if is_rank_zero:
        print(json.dumps({"event": "on_policy_gkd_teacher_load_done", **summary}), flush=True)
    return teacher_vlm, summary


def resolve_requested_epochs(args: argparse.Namespace, trainer_cfg: TrainerConfig) -> float:
    """Resolve the requested epoch count from CLI or stage config."""
    if args.epochs is not None:
        return float(args.epochs)
    return float(trainer_cfg.epochs)


def resolve_requested_steps(
    *,
    args: argparse.Namespace,
    steps_per_epoch: int,
    requested_epochs: float,
) -> int | None:
    """Resolve the training horizon in optimizer steps."""
    if args.max_steps is not None:
        return int(args.max_steps)
    if steps_per_epoch <= 0:
        return 0
    return int(math.ceil(max(requested_epochs, 0.0) * steps_per_epoch))


def interval_steps_from_epochs(interval_epochs: float, steps_per_epoch: int) -> int | None:
    """Convert an epoch interval into optimizer steps."""
    if interval_epochs <= 0 or steps_per_epoch <= 0:
        return None
    return max(1, int(round(interval_epochs * steps_per_epoch)))


def build_traj_token_weight_map(
    records: list[dict],
    tokenizer,
    config: dict[str, object] | None,
) -> tuple[dict[int, float] | None, dict[str, object]]:
    """Build a capped inverse-sqrt label-weight map for discrete traj tokens."""
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return None, {"enabled": False}

    min_weight = float(cfg.get("min_weight", 0.5))
    max_weight = float(cfg.get("max_weight", 2.5))
    power = float(cfg.get("power", 0.5))
    counts: Counter[int] = Counter()
    for record in records:
        for token_idx in _record_traj_future_token_ids(record):
            counts[int(token_idx)] += 1
    if not counts:
        return None, {"enabled": True, "num_tokens": 0}

    sorted_counts = sorted(counts.values())
    reference = float(sorted_counts[len(sorted_counts) // 2])
    weight_map: dict[int, float] = {}
    exported_weights: list[float] = []
    for traj_token_idx, count in counts.items():
        raw_weight = (reference / max(int(count), 1)) ** power
        clipped_weight = min(max(raw_weight, min_weight), max_weight)
        tokenizer_id = tokenizer.convert_tokens_to_ids(discrete_traj_token(int(traj_token_idx)))
        if isinstance(tokenizer_id, int) and tokenizer_id >= 0:
            weight_map[int(tokenizer_id)] = float(clipped_weight)
            exported_weights.append(float(clipped_weight))
    summary = {
        "enabled": True,
        "num_tokens": len(weight_map),
        "reference_count": reference,
        "min_weight": min(exported_weights) if exported_weights else None,
        "max_weight": max(exported_weights) if exported_weights else None,
        "power": power,
        "cap_min": min_weight,
        "cap_max": max_weight,
    }
    return weight_map, summary


def infer_teacher_traj_hidden_size(
    cache_dir: Path | None,
    *,
    source: str = "hidden",
    latent_suffix: str = "lat32",
    records: list[dict] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> int | None:
    """Infer the teacher trajectory hidden width from the imported cache."""
    if cache_dir is not None:
        resolved_source = str(source).strip().lower()
        if resolved_source == "latent":
            hidden_dir = cache_dir / "latent"
            pattern = f"*.teacher_traj15.{latent_suffix}.npy"
        else:
            hidden_dir = cache_dir / "hidden"
            pattern = "*.npy"
        if hidden_dir.exists():
            for path in sorted(hidden_dir.glob(pattern)):
                try:
                    hidden = np.load(path, mmap_mode="r")
                except Exception:  # noqa: BLE001
                    continue
                if hidden.ndim >= 2:
                    return int(hidden.shape[-1])
    for record in records or []:
        signal = _teacher_traj15_signal_from_sample(
            record,
            teacher_traj_cache_dir=cache_dir,
            teacher_traj_hidden_source=source,
            teacher_traj_latent_suffix=latent_suffix,
            project_root=project_root,
        )
        if signal is None or "hidden" not in signal:
            continue
        hidden = np.asarray(signal["hidden"])
        if hidden.ndim >= 2:
            return int(hidden.shape[-1])
    for record in records or []:
        boundary_paths = ((record.get("teacher_cache") or {}).get("boundary_hidden_paths") or {})
        for raw_path in boundary_paths.values():
            resolved = remap_external_path(raw_path)
            if resolved is None:
                continue
            try:
                hidden = np.load(Path(resolved), mmap_mode="r")
            except Exception:  # noqa: BLE001
                continue
            if hidden.ndim >= 1 and int(hidden.shape[-1]) > 0:
                return int(hidden.shape[-1])
    return None


def load_traj_decode_config(
    student_model: str | Path | None,
    tokenizer,
    overrides: dict[str, object] | None = None,
) -> tuple[TrajectoryDecodeConfig | None, dict[str, object]]:
    """Load the Alpamayo trajectory-token decode contract used for geometry regularization."""
    config_path = resolve_traj_tokenizer_config_path(student_model)
    if config_path is None:
        return None, {"enabled": False, "reason": "missing_traj_tokenizer_config"}
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    traj_cfg = payload.get("traj_tokenizer_cfg")
    action_cfg = (traj_cfg or {}).get("action_space_cfg") or {}
    if not traj_cfg:
        return None, {"enabled": False, "reason": "missing_traj_tokenizer_cfg", "config_path": str(config_path)}

    traj_token_start_idx = getattr(tokenizer, "traj_token_start_idx", None)
    if not isinstance(traj_token_start_idx, int) or traj_token_start_idx < 0:
        token_id = tokenizer.convert_tokens_to_ids("<i0>")
        traj_token_start_idx = int(token_id) if isinstance(token_id, int) and token_id >= 0 else None
    if traj_token_start_idx is None:
        return None, {"enabled": False, "reason": "missing_traj_token_start_idx", "config_path": str(config_path)}

    config = TrajectoryDecodeConfig(
        traj_token_start_idx=int(traj_token_start_idx),
        num_bins=int(traj_cfg["num_bins"]),
        dims_min=tuple(float(value) for value in traj_cfg["dims_min"]),
        dims_max=tuple(float(value) for value in traj_cfg["dims_max"]),
        accel_mean=float(action_cfg["accel_mean"]),
        accel_std=float(action_cfg["accel_std"]),
        curvature_mean=float(action_cfg["curvature_mean"]),
        curvature_std=float(action_cfg["curvature_std"]),
        dt=float(action_cfg["dt"]),
        n_waypoints=int(action_cfg["n_waypoints"]),
    )
    override_cfg = dict(overrides or {})
    if "short_horizon_steps" in override_cfg:
        config.short_horizon_steps = int(override_cfg["short_horizon_steps"])
    if "short_horizon_weight" in override_cfg:
        config.short_horizon_weight = float(override_cfg["short_horizon_weight"])
    return config, {
        "enabled": True,
        "config_path": str(config_path),
        "num_bins": config.num_bins,
        "n_waypoints": config.n_waypoints,
        "short_horizon_steps": config.short_horizon_steps,
        "short_horizon_weight": config.short_horizon_weight,
    }


def build_teacher_traj_aux_target_stats(
    records: list[dict],
    *,
    teacher_traj_cache_dir: Path | None,
    traj_decode_config: TrajectoryDecodeConfig | None,
    num_buckets: int,
) -> tuple[TrajectoryAuxInterfaceConfig | None, dict[str, object]]:
    """Estimate bucket/channel control statistics from teacher discrete trajectory bodies."""
    if teacher_traj_cache_dir is None or traj_decode_config is None:
        return None, {"enabled": False, "reason": "missing_teacher_cache_or_decode_config"}
    num_buckets = max(int(num_buckets), 1)
    counts = np.zeros((num_buckets, 2), dtype=np.int64)
    sums = np.zeros((num_buckets, 2), dtype=np.float64)
    sq_sums = np.zeros((num_buckets, 2), dtype=np.float64)
    dims_min = np.asarray(traj_decode_config.dims_min, dtype=np.float64)
    dims_max = np.asarray(traj_decode_config.dims_max, dtype=np.float64)
    used_records = 0

    for record in records:
        signal = _teacher_traj15_signal_from_sample(
            record,
            teacher_traj_cache_dir=teacher_traj_cache_dir,
        )
        if signal is None or "token_ids" not in signal:
            continue
        token_ids = np.asarray(signal["token_ids"], dtype=np.int64).reshape(-1)
        usable_count = min(int(token_ids.shape[0]), int(traj_decode_config.n_waypoints * 2))
        usable_count -= usable_count % 2
        if usable_count <= 0:
            continue
        token_ids = np.clip(token_ids[:usable_count], 0, int(traj_decode_config.num_bins) - 1).astype(np.float64)
        token_indices = np.arange(usable_count, dtype=np.int64)
        waypoint_ids = token_indices // 2
        parity = token_indices % 2
        bucket_ids = np.clip(
            (waypoint_ids * num_buckets) // max(int(traj_decode_config.n_waypoints), 1),
            0,
            num_buckets - 1,
        )
        dim_min = dims_min[parity]
        dim_max = dims_max[parity]
        controls = token_ids / float(max(int(traj_decode_config.num_bins) - 1, 1)) * (dim_max - dim_min) + dim_min
        for bucket_index in range(num_buckets):
            for channel_index in range(2):
                mask = (bucket_ids == bucket_index) & (parity == channel_index)
                if not np.any(mask):
                    continue
                values = controls[mask]
                counts[bucket_index, channel_index] += int(values.shape[0])
                sums[bucket_index, channel_index] += float(values.sum())
                sq_sums[bucket_index, channel_index] += float(np.square(values).sum())
        used_records += 1

    if used_records <= 0 or not np.all(counts > 0):
        return None, {
            "enabled": False,
            "reason": "insufficient_teacher_traj_stats",
            "used_records": used_records,
            "counts": counts.tolist(),
        }

    means = sums / counts
    variances = np.maximum((sq_sums / counts) - np.square(means), 1e-4)
    stds = np.sqrt(variances)
    aux_config = TrajectoryAuxInterfaceConfig(
        num_buckets=num_buckets,
        normalize_targets=True,
        tanh_bound=3.0,
        huber_delta=1.0,
        target_means=tuple(tuple(float(value) for value in row) for row in means.tolist()),
        target_stds=tuple(tuple(float(value) for value in row) for row in stds.tolist()),
    )
    return aux_config, {
        "enabled": True,
        "used_records": used_records,
        "num_buckets": num_buckets,
        "target_means": means.tolist(),
        "target_stds": stds.tolist(),
        "counts": counts.tolist(),
    }


def decode_eval_config_from_yaml(config: dict[str, object] | None, *, fallback_split: str = "val") -> DecodeEvalConfig:
    cfg = dict(config or {})
    return DecodeEvalConfig(
        enabled=bool(cfg.get("enabled", False)),
        split=str(cfg.get("split", fallback_split)),
        num_samples=int(cfg.get("num_samples", 8)),
        max_new_tokens=int(cfg.get("max_new_tokens", 160)),
        prompt_mode=str(cfg.get("prompt_mode", "joint")),
        target_mode=str(cfg.get("target_mode", "joint")),
        image_prompt_style=str(cfg.get("image_prompt_style", "compact")),
        prompt_text_style=str(cfg.get("prompt_text_style", "numeric_history_question")),
        fuse_history_tokens=bool(cfg.get("fuse_history_tokens", False)),
        metric_name=str(cfg.get("metric_name", "anti_collapse_score")),
    )


def scheduled_sampling_config_from_yaml(config: dict[str, object] | None, tokenizer) -> tuple[ScheduledSamplingConfig | None, dict[str, object]]:
    cfg = dict(config or {})
    enabled = bool(cfg.get("enabled", False))
    probability = float(cfg.get("p", cfg.get("probability", 0.0)) or 0.0)
    probability_start_raw = cfg.get("p_start", cfg.get("probability_start", None))
    probability_start = None if probability_start_raw is None else float(probability_start_raw)
    start_step = int(cfg.get("start_step", cfg.get("start_steps", 0)) or 0)
    ramp_steps = int(cfg.get("ramp_steps", cfg.get("p_ramp_steps", 0)) or 0)
    mode = str(cfg.get("mode", cfg.get("sampling_mode", "token")) or "token").strip().lower()
    if mode in {"generated", "generated-prefix", "sample-prefix"}:
        mode = "generated_prefix"
    prefix_tokens = int(cfg.get("prefix_tokens", cfg.get("generated_prefix_tokens", 0)) or 0)
    autoregressive_prefix = bool(
        cfg.get("autoregressive_prefix", cfg.get("ar_prefix", cfg.get("true_autoregressive", False)))
    )
    generated_prefix_topk_kd_scale = float(
        cfg.get(
            "generated_prefix_topk_kd_scale",
            cfg.get("scheduled_topk_kd_scale", cfg.get("topk_kd_scale_on_generated_prefix", 1.0)),
        )
        or 0.0
    )
    vocab_spec = str(cfg.get("replacement_vocab", "<i0>~<i2999>")).strip()

    traj_start = tokenizer.convert_tokens_to_ids("<i0>")
    replacement_vocab_size = int(cfg.get("replacement_vocab_size", 3000) or 3000)
    match = re.fullmatch(r"<i(\d+)>\s*~\s*<i(\d+)>", vocab_spec)
    if match:
        start_index = int(match.group(1))
        end_index = int(match.group(2))
        traj_start = tokenizer.convert_tokens_to_ids(discrete_traj_token(start_index))
        replacement_vocab_size = max(end_index - start_index + 1, 0)

    if not enabled or probability <= 0:
        return None, {
            "enabled": False,
            "p": probability,
            "p_start": probability_start,
            "start_step": start_step,
            "ramp_steps": ramp_steps,
            "mode": mode,
            "prefix_tokens": prefix_tokens,
            "autoregressive_prefix": autoregressive_prefix,
            "generated_prefix_topk_kd_scale": generated_prefix_topk_kd_scale,
            "replacement_vocab": vocab_spec,
            "replacement_vocab_size": replacement_vocab_size,
        }
    if not isinstance(traj_start, int) or traj_start < 0 or replacement_vocab_size <= 0:
        raise ValueError(f"Invalid scheduled_sampling replacement_vocab={vocab_spec!r}")

    config_obj = ScheduledSamplingConfig(
        enabled=True,
        probability=probability,
        probability_start=probability_start,
        start_step=start_step,
        ramp_steps=ramp_steps,
        traj_token_start_id=int(traj_start),
        replacement_vocab_size=int(replacement_vocab_size),
        mode=mode,
        prefix_tokens=prefix_tokens,
        autoregressive_prefix=autoregressive_prefix,
        generated_prefix_topk_kd_scale=generated_prefix_topk_kd_scale,
    )
    return config_obj, {
        "enabled": True,
        "p": probability,
        "p_start": probability_start,
        "start_step": start_step,
        "ramp_steps": ramp_steps,
        "mode": mode,
        "prefix_tokens": prefix_tokens,
        "autoregressive_prefix": autoregressive_prefix,
        "generated_prefix_topk_kd_scale": generated_prefix_topk_kd_scale,
        "replacement_vocab": vocab_spec,
        "traj_token_start_id": int(traj_start),
        "replacement_vocab_size": int(replacement_vocab_size),
    }


def on_policy_gkd_config_from_yaml(config: dict[str, object] | None, tokenizer) -> tuple[OnPolicyGKDConfig | None, dict[str, object]]:
    cfg = dict(config or {})
    enabled = bool(cfg.get("enabled", False))
    probability = float(cfg.get("p", cfg.get("probability", 0.0)) or 0.0)
    probability_start_raw = cfg.get("p_start", cfg.get("probability_start", None))
    probability_start = None if probability_start_raw is None else float(probability_start_raw)
    start_step = int(cfg.get("start_step", cfg.get("start_steps", 0)) or 0)
    ramp_steps = int(cfg.get("ramp_steps", cfg.get("p_ramp_steps", 0)) or 0)
    rollout_tokens = int(cfg.get("rollout_tokens", cfg.get("prefix_tokens", 128)) or 128)
    loss_weight = float(cfg.get("loss_weight", cfg.get("weight", 1.0)) or 0.0)
    kd_temperature = float(cfg.get("kd_temperature", 1.5) or 1.5)
    do_sample = bool(cfg.get("do_sample", cfg.get("sample", True)))
    generation_temperature = float(cfg.get("generation_temperature", cfg.get("sample_temperature", 0.6)) or 0.6)
    top_p = float(cfg.get("top_p", 0.98) or 0.98)
    top_k = int(cfg.get("top_k", 0) or 0)
    vocab_spec = str(cfg.get("replacement_vocab", "<i0>~<i2999>")).strip()

    traj_start = tokenizer.convert_tokens_to_ids("<i0>")
    replacement_vocab_size = int(cfg.get("replacement_vocab_size", 3000) or 3000)
    match = re.fullmatch(r"<i(\d+)>\s*~\s*<i(\d+)>", vocab_spec)
    if match:
        start_index = int(match.group(1))
        end_index = int(match.group(2))
        traj_start = tokenizer.convert_tokens_to_ids(discrete_traj_token(start_index))
        replacement_vocab_size = max(end_index - start_index + 1, 0)

    summary = {
        "enabled": enabled,
        "p": probability,
        "p_start": probability_start,
        "start_step": start_step,
        "ramp_steps": ramp_steps,
        "rollout_tokens": rollout_tokens,
        "loss_weight": loss_weight,
        "kd_temperature": kd_temperature,
        "do_sample": do_sample,
        "generation_temperature": generation_temperature,
        "top_p": top_p,
        "top_k": top_k,
        "replacement_vocab": vocab_spec,
        "replacement_vocab_size": replacement_vocab_size,
    }
    if not enabled or probability <= 0 or loss_weight <= 0:
        summary["enabled"] = False
        return None, summary
    if not isinstance(traj_start, int) or traj_start < 0 or replacement_vocab_size <= 0:
        raise ValueError(f"Invalid on_policy_gkd replacement_vocab={vocab_spec!r}")

    config_obj = OnPolicyGKDConfig(
        enabled=True,
        probability=probability,
        probability_start=probability_start,
        start_step=start_step,
        ramp_steps=ramp_steps,
        traj_token_start_id=int(traj_start),
        replacement_vocab_size=int(replacement_vocab_size),
        rollout_tokens=max(int(rollout_tokens), 1),
        loss_weight=loss_weight,
        kd_temperature=kd_temperature,
        do_sample=do_sample,
        generation_temperature=generation_temperature,
        top_p=top_p,
        top_k=top_k,
    )
    summary["traj_token_start_id"] = int(traj_start)
    return config_obj, summary


def build_dataloader(
    records: list[dict],
    *,
    batch_size: int,
    collator,
    rank: int,
    world_size: int,
    shuffle: bool,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            records,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": (sampler is None and shuffle),
        "sampler": sampler,
        "collate_fn": collator,
        "num_workers": max(int(num_workers), 0),
        "pin_memory": bool(pin_memory),
    }
    if dataloader_kwargs["num_workers"] > 0:
        dataloader_kwargs["prefetch_factor"] = max(int(prefetch_factor), 1)
        dataloader_kwargs["persistent_workers"] = bool(persistent_workers)
    dataloader = DataLoader(records, **dataloader_kwargs)
    return dataloader, sampler


@torch.inference_mode()
def evaluate_model(
    model,
    dataloader: DataLoader,
    *,
    device: torch.device,
    bf16: bool,
    world_size: int,
    loss_weights: DistillationLossWeights,
    traj_decode_config: TrajectoryDecodeConfig | None,
    traj_aux_interface_config: TrajectoryAuxInterfaceConfig | None,
    traj_body_prefix_tokens: int | None = None,
    traj_hidden_bridge_config: dict[str, float] | None = None,
    log_every_batches: int = 0,
) -> dict[str, float]:
    """Run a validation pass and return mean scalar metrics."""
    metric_sums: dict[str, float] = {}
    local_batches = 0
    model.eval()
    total_batches = len(dataloader)
    started_at = time.time()
    for batch_index, batch in enumerate(dataloader, start=1):
        batch = prepare_flex_batch_for_model(batch, model)
        batch = move_batch_to_device(batch, device)
        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if bf16 and device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad():
            with autocast_context:
                _, logs = run_train_step(
                    model,
                    batch,
                    loss_weights,
                    traj_decode_config=traj_decode_config,
                    traj_aux_interface_config=traj_aux_interface_config,
                    traj_body_prefix_tokens=traj_body_prefix_tokens,
                    traj_hidden_bridge_config=traj_hidden_bridge_config,
                )
        local_batches += 1
        for key, value in logs.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        if log_every_batches > 0 and (batch_index == 1 or batch_index % log_every_batches == 0 or batch_index == total_batches):
            elapsed = max(time.time() - started_at, 1e-6)
            print(
                json.dumps(
                    {
                        "event": "eval_batch",
                        "batch": batch_index,
                        "num_batches": total_batches,
                        "elapsed_sec": round(elapsed, 3),
                        "batches_per_sec": round(batch_index / elapsed, 4),
                    }
                ),
                flush=True,
            )

    if world_size > 1:
        keys = sorted(metric_sums.keys())
        values = torch.tensor(
            [metric_sums.get(key, 0.0) for key in keys] + [float(local_batches)],
            device=device,
            dtype=torch.float32,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        total_batches = max(int(values[-1].item()), 1)
        averaged = {key: float(values[index].item() / total_batches) for index, key in enumerate(keys)}
        return averaged

    denom = max(local_batches, 1)
    return {key: value / denom for key, value in metric_sums.items()}


def save_training_checkpoint(
    checkpoint_dir: Path,
    *,
    model,
    tokenizer,
    processor,
    use_lora: bool,
    train_config_payload: dict,
) -> dict[str, object]:
    """Persist a training checkpoint plus the matching run metadata."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = save_student_checkpoint(
        checkpoint_dir,
        model,
        tokenizer,
        processor,
        use_lora=use_lora,
    )
    payload = dict(train_config_payload)
    payload["checkpoint"] = checkpoint_payload
    (checkpoint_dir / "train_config.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    return checkpoint_payload


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def run_training(args: argparse.Namespace, *, rank: int = 0, world_size: int = 1, master_port: int | None = None) -> None:
    if master_port is not None:
        _maybe_init_distributed(rank, world_size, master_port)
    set_seed(args.seed + rank)
    is_rank_zero = _is_rank_zero(rank, world_size)
    if is_rank_zero:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records = load_jsonl(args.corpus_jsonl)
    all_train_records = [record for record in all_records if record.get("split") == "train"]
    all_val_records = [record for record in all_records if record.get("split") == "val"]
    if args.skip_asset_check:
        train_records = list(all_train_records)
        val_records = list(all_val_records)
    else:
        train_records = [record for record in all_train_records if has_required_materialized_assets(record)]
        val_records = [record for record in all_val_records if has_required_materialized_assets(record)]
    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_records = val_records[: args.max_val_samples]
    if not train_records:
        raise RuntimeError("No train records with required materialized assets were found.")

    trainer_cfg, loss_weights, stage_options = stage_weights_from_yaml(args.stage_config)
    trainer_cfg.batch_size = args.batch_size
    trainer_cfg.epochs = resolve_requested_epochs(args, trainer_cfg)
    if args.learning_rate is not None:
        trainer_cfg.learning_rate = float(args.learning_rate)
    student_model = resolve_student_model_path(args.student_model)
    data_view_cfg = stage_options.get("data_view") or {}
    prompt_mode = str(data_view_cfg.get("prompt_mode", "joint"))
    target_mode = str(data_view_cfg.get("target_mode", "joint"))
    image_prompt_style = str(data_view_cfg.get("image_prompt_style", "compact")).strip().lower()
    prompt_text_style = str(data_view_cfg.get("prompt_text_style", "numeric_history_question")).strip().lower()
    fuse_history_tokens = bool(data_view_cfg.get("fuse_history_tokens", False))
    if prompt_mode not in {"joint", "traj_only"}:
        raise ValueError(f"Unsupported data_view.prompt_mode={prompt_mode!r}")
    if target_mode not in {"joint", "traj_only"}:
        raise ValueError(f"Unsupported data_view.target_mode={target_mode!r}")
    if image_prompt_style in {"plain", "unlabeled"}:
        image_prompt_style = "compact"
    elif image_prompt_style in {"alpamayo_camera_labeled", "official"}:
        image_prompt_style = "camera_labeled"
    if image_prompt_style not in {"compact", "camera_labeled"}:
        raise ValueError(f"Unsupported data_view.image_prompt_style={image_prompt_style!r}")
    if prompt_text_style in {"official", "alpamayo"}:
        prompt_text_style = "official_alpamayo"
    elif prompt_text_style in {"numeric", "numeric_history", "question"}:
        prompt_text_style = "numeric_history_question"
    if prompt_text_style not in {"numeric_history_question", "official_alpamayo"}:
        raise ValueError(f"Unsupported data_view.prompt_text_style={prompt_text_style!r}")
    if prompt_text_style == "official_alpamayo" and "fuse_history_tokens" not in data_view_cfg:
        fuse_history_tokens = True
    enable_teacher_view = bool(data_view_cfg.get("enable_teacher_view", True))
    enable_action_aux = bool(data_view_cfg.get("enable_action_aux", True))
    teacher_pair_target = bool(data_view_cfg.get("teacher_pair_target", False))
    hard_view_uses_teacher_cot = bool(data_view_cfg.get("hard_view_uses_teacher_cot", False))
    teacher_view_force_enable = bool(data_view_cfg.get("teacher_view_force_enable", False))
    teacher_view_uses_teacher_traj = bool(data_view_cfg.get("teacher_view_uses_teacher_traj", False))
    teacher_view_default_traj_weight = float(data_view_cfg.get("teacher_view_default_traj_weight", 0.0) or 0.0)
    teacher_traj_topk_on_teacher_view = bool(data_view_cfg.get("teacher_traj_topk_on_teacher_view", False))
    teacher_traj_cache_dir_raw = data_view_cfg.get("teacher_traj_cache_dir")
    teacher_traj_cache_dir = (
        Path(str(remap_external_path(teacher_traj_cache_dir_raw)))
        if teacher_traj_cache_dir_raw not in (None, "")
        else None
    )
    teacher_traj_hidden_source = str(data_view_cfg.get("teacher_traj_hidden_source", "hidden")).strip().lower()
    teacher_traj_latent_suffix = str(data_view_cfg.get("teacher_traj_latent_suffix", "lat32")).strip()
    teacher_traj_hidden_size = None
    if loss_weights.teacher_traj_hidden_align > 0 or loss_weights.teacher_boundary_hidden_align > 0:
        teacher_traj_hidden_size = infer_teacher_traj_hidden_size(
            teacher_traj_cache_dir,
            source=teacher_traj_hidden_source,
            latent_suffix=teacher_traj_latent_suffix,
            records=train_records,
            project_root=PROJECT_ROOT,
        )
        if teacher_traj_hidden_size is None:
            raise RuntimeError(
                "teacher hidden alignment is enabled but no teacher hidden cache dimension could be inferred."
            )
    traj_aux_interface_cfg = dict(stage_options.get("traj_aux_interface") or {})
    traj_hidden_bridge_cfg = dict(stage_options.get("traj_hidden_bridge") or {})
    lora_cfg = dict(stage_options.get("lora") or {})
    train_lm_head_token_rows = bool(lora_cfg.get("train_lm_head_token_rows", True))
    traj_aux_num_buckets = max(int(traj_aux_interface_cfg.get("num_buckets", 1) or 1), 1)
    traj_hidden_bridge_size = (
        int(traj_hidden_bridge_cfg.get("shared_size"))
        if traj_hidden_bridge_cfg.get("shared_size") not in (None, "", 0)
        else None
    )
    optimization_cfg = dict(stage_options.get("optimization") or {})
    curriculum_cfg = dict(stage_options.get("curriculum") or {})
    flex_cfg = dict(stage_options.get("flex") or {})
    flex_scene_config = None
    if bool(flex_cfg.get("enabled", False)):
        flex_architecture = str(flex_cfg.get("architecture", "single_level") or "single_level")
        default_flex_compression = (
            "per_image" if flex_architecture in {"multi_level", "ml_flex", "ml-flex"} else "global"
        )
        flex_scene_config = FlexSceneConfig(
            enabled=True,
            architecture=flex_architecture,
            tokens_per_image=int(flex_cfg.get("tokens_per_image", 32) or 32),
            expected_images_per_sample=int(flex_cfg.get("expected_images_per_sample", 16) or 16),
            input_hidden_size=int(flex_cfg.get("input_hidden_size", 2048) or 2048),
            hidden_size=int(flex_cfg.get("hidden_size", 1024) or 1024),
            num_layers=int(flex_cfg.get("num_layers", 2) or 2),
            num_heads=int(flex_cfg.get("num_heads", 8) or 8),
            mlp_ratio=float(flex_cfg.get("mlp_ratio", 4.0) or 4.0),
            dropout=float(flex_cfg.get("dropout", 0.0) or 0.0),
            use_camera_time_embeddings=bool(flex_cfg.get("use_camera_time_embeddings", False)),
            use_local_slot_embeddings=bool(flex_cfg.get("use_local_slot_embeddings", True)),
            max_camera_types=int(flex_cfg.get("max_camera_types", 16) or 16),
            compression_mode=str(flex_cfg.get("compression_mode", default_flex_compression) or default_flex_compression),
            selection_strategy=str(flex_cfg.get("selection_strategy", "first") or "first"),
            num_deepstack_levels=int(flex_cfg.get("num_deepstack_levels", 3) or 3),
        )
    traj_body_prefix_tokens = (
        int(curriculum_cfg.get("traj_body_prefix_tokens"))
        if curriculum_cfg.get("traj_body_prefix_tokens") not in (None, "", 0)
        else None
    )
    interface_loss_weights_cfg = dict(stage_options.get("interface_loss_weights") or {})
    wrapper_cfg = StudentWrapperConfig(
        student_model_name=student_model,
        max_length=trainer_cfg.max_length,
        torch_dtype=preferred_model_dtype(bf16=trainer_cfg.bf16),
        local_files_only=Path(student_model).expanduser().exists(),
        attn_implementation=stage_options.get("attn_implementation"),
        traj_teacher_hidden_size=teacher_traj_hidden_size,
        traj_aux_num_buckets=traj_aux_num_buckets,
        traj_hidden_bridge_size=traj_hidden_bridge_size,
        flex_scene=flex_scene_config,
    )
    tokenizer = load_student_tokenizer(wrapper_cfg)
    processor = load_student_processor(wrapper_cfg, tokenizer=tokenizer)
    lora_spec = LoraConfigSpec(
        r=int(lora_cfg.get("rank", 32) or 32),
        alpha=int(lora_cfg.get("alpha", 64) or 64),
        dropout=float(lora_cfg.get("dropout", 0.05) or 0.05),
        trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer)),
    )
    scheduled_sampling_config, scheduled_sampling_summary = scheduled_sampling_config_from_yaml(
        stage_options.get("scheduled_sampling"),
        tokenizer,
    )
    on_policy_gkd_config, on_policy_gkd_summary = on_policy_gkd_config_from_yaml(
        stage_options.get("on_policy_gkd"),
        tokenizer,
    )
    traj_decode_config, traj_decode_summary = load_traj_decode_config(
        student_model,
        tokenizer,
        stage_options.get("traj_decode_reg"),
    )
    if (
        loss_weights.traj_xyz_reg > 0
        or loss_weights.traj_delta_reg > 0
        or loss_weights.traj_final_reg > 0
    ) and traj_decode_config is None:
        raise RuntimeError("Decoded trajectory regularization is enabled but no traj tokenizer config could be loaded.")
    traj_aux_interface_runtime_config = None
    traj_aux_interface_summary = {"enabled": False}
    if bool(traj_aux_interface_cfg.get("normalize_targets", False)):
        traj_aux_interface_runtime_config, traj_aux_interface_summary = build_teacher_traj_aux_target_stats(
            train_records,
            teacher_traj_cache_dir=teacher_traj_cache_dir,
            traj_decode_config=traj_decode_config,
            num_buckets=traj_aux_num_buckets,
        )
        if traj_aux_interface_runtime_config is None:
            raise RuntimeError(
                "traj_aux_interface.normalize_targets is enabled but teacher trajectory target stats could not be built."
            )
        traj_aux_interface_runtime_config.tanh_bound = float(traj_aux_interface_cfg.get("tanh_bound", 3.0))
        traj_aux_interface_runtime_config.huber_delta = float(traj_aux_interface_cfg.get("huber_delta", 1.0))
        traj_aux_interface_runtime_config.decode_tanh_scale = float(traj_aux_interface_cfg.get("decode_tanh_scale", 1.0))
        traj_aux_interface_runtime_config.decode_edge_margin = int(traj_aux_interface_cfg.get("decode_edge_margin", 0))
        traj_aux_interface_runtime_config.decode_prior_sigma = float(traj_aux_interface_cfg.get("decode_prior_sigma", 96.0))
        traj_aux_interface_summary["decode_tanh_scale"] = traj_aux_interface_runtime_config.decode_tanh_scale
        traj_aux_interface_summary["decode_edge_margin"] = traj_aux_interface_runtime_config.decode_edge_margin
        traj_aux_interface_summary["decode_prior_sigma"] = traj_aux_interface_runtime_config.decode_prior_sigma
    else:
        traj_aux_interface_runtime_config = TrajectoryAuxInterfaceConfig(
            num_buckets=traj_aux_num_buckets,
            normalize_targets=False,
            tanh_bound=float(traj_aux_interface_cfg.get("tanh_bound", 3.0)),
            huber_delta=float(traj_aux_interface_cfg.get("huber_delta", 1.0)),
            decode_tanh_scale=float(traj_aux_interface_cfg.get("decode_tanh_scale", 1.0)),
            decode_edge_margin=int(traj_aux_interface_cfg.get("decode_edge_margin", 0)),
            decode_prior_sigma=float(traj_aux_interface_cfg.get("decode_prior_sigma", 96.0)),
        )
        traj_aux_interface_summary = {
            "enabled": True,
            "num_buckets": traj_aux_num_buckets,
            "normalize_targets": False,
            "tanh_bound": traj_aux_interface_runtime_config.tanh_bound,
            "huber_delta": traj_aux_interface_runtime_config.huber_delta,
            "decode_tanh_scale": traj_aux_interface_runtime_config.decode_tanh_scale,
            "decode_edge_margin": traj_aux_interface_runtime_config.decode_edge_margin,
            "decode_prior_sigma": traj_aux_interface_runtime_config.decode_prior_sigma,
        }
    interface_loss_weights = None
    if interface_loss_weights_cfg:
        defaults = get_stage_weights(trainer_cfg.stage_name)
        interface_loss_weights = DistillationLossWeights(
            hard_cot_ce=resolve_loss_weight_value(interface_loss_weights_cfg, "hard_cot_ce", defaults.hard_cot_ce),
            teacher_seq_ce=resolve_loss_weight_value(interface_loss_weights_cfg, "teacher_seq_ce", defaults.teacher_seq_ce),
            teacher_logit_kd=resolve_loss_weight_value(interface_loss_weights_cfg, "teacher_logit_kd", defaults.teacher_logit_kd),
            traj_ce=resolve_loss_weight_value(interface_loss_weights_cfg, "traj_ce", defaults.traj_ce),
            traj_aux_reg=resolve_loss_weight_value(interface_loss_weights_cfg, "traj_aux_reg", defaults.traj_aux_reg),
            format_ce=resolve_loss_weight_value(interface_loss_weights_cfg, "format_ce", defaults.format_ce),
            action_aux=resolve_loss_weight_value(interface_loss_weights_cfg, "action_aux", defaults.action_aux),
            feat_align=resolve_loss_weight_value(interface_loss_weights_cfg, "feat_align", defaults.feat_align),
            teacher_traj_ce=resolve_optional_loss_weight_value(interface_loss_weights_cfg, "teacher_traj_ce"),
            teacher_traj_topk_kd=resolve_loss_weight_value(interface_loss_weights_cfg, "teacher_traj_topk_kd", defaults.teacher_traj_topk_kd),
            teacher_traj_hidden_align=resolve_loss_weight_value(interface_loss_weights_cfg, "teacher_traj_hidden_align", defaults.teacher_traj_hidden_align),
            teacher_boundary_hidden_align=resolve_loss_weight_value(
                interface_loss_weights_cfg,
                "teacher_boundary_hidden_align",
                defaults.teacher_boundary_hidden_align,
            ),
            traj_xyz_reg=resolve_loss_weight_value(interface_loss_weights_cfg, "traj_xyz_reg", defaults.traj_xyz_reg),
            traj_delta_reg=resolve_loss_weight_value(interface_loss_weights_cfg, "traj_delta_reg", defaults.traj_delta_reg),
            traj_final_reg=resolve_loss_weight_value(interface_loss_weights_cfg, "traj_final_reg", defaults.traj_final_reg),
            traj_control_reg=resolve_loss_weight_value(interface_loss_weights_cfg, "traj_control_reg", defaults.traj_control_reg),
            traj_control_delta_reg=resolve_loss_weight_value(
                interface_loss_weights_cfg,
                "traj_control_delta_reg",
                defaults.traj_control_delta_reg,
            ),
            traj_aux_xyz_reg=resolve_loss_weight_value(interface_loss_weights_cfg, "traj_aux_xyz_reg", defaults.traj_aux_xyz_reg),
            traj_aux_final_reg=resolve_loss_weight_value(interface_loss_weights_cfg, "traj_aux_final_reg", defaults.traj_aux_final_reg),
            traj_aux_guided_kd=resolve_loss_weight_value(interface_loss_weights_cfg, "traj_aux_guided_kd", defaults.traj_aux_guided_kd),
            traj_aux_pseudo_ce=resolve_loss_weight_value(interface_loss_weights_cfg, "traj_aux_pseudo_ce", defaults.traj_aux_pseudo_ce),
            boundary_action_xyz=resolve_loss_weight_value(
                interface_loss_weights_cfg,
                "boundary_action_xyz",
                defaults.boundary_action_xyz,
            ),
        )
    traj_token_weight_map, traj_token_reweight_summary = build_traj_token_weight_map(
        train_records,
        tokenizer,
        stage_options.get("traj_token_reweighting"),
    )
    decode_eval_cfg = decode_eval_config_from_yaml(
        stage_options.get("decode_eval"),
        fallback_split="val",
    )
    raw_decode_eval_cfg = dict(stage_options.get("decode_eval") or {})
    if "prompt_mode" not in raw_decode_eval_cfg:
        decode_eval_cfg.prompt_mode = prompt_mode
    if "target_mode" not in raw_decode_eval_cfg:
        decode_eval_cfg.target_mode = target_mode
    if "image_prompt_style" not in raw_decode_eval_cfg:
        decode_eval_cfg.image_prompt_style = image_prompt_style
    if "prompt_text_style" not in raw_decode_eval_cfg:
        decode_eval_cfg.prompt_text_style = prompt_text_style
    if "fuse_history_tokens" not in raw_decode_eval_cfg:
        decode_eval_cfg.fuse_history_tokens = fuse_history_tokens
    collator = DistillationCollator(
        tokenizer=tokenizer,
        processor=processor,
        project_root=PROJECT_ROOT,
        max_length=trainer_cfg.max_length,
        prompt_mode=prompt_mode,
        target_mode=target_mode,
        teacher_pair_target=teacher_pair_target,
        enable_teacher_view=enable_teacher_view,
        enable_action_aux=enable_action_aux,
        traj_token_weight_map=traj_token_weight_map,
        teacher_traj_cache_dir=teacher_traj_cache_dir,
        teacher_traj_hidden_source=teacher_traj_hidden_source,
        teacher_traj_latent_suffix=teacher_traj_latent_suffix,
        hard_view_uses_teacher_cot=hard_view_uses_teacher_cot,
        teacher_view_force_enable=teacher_view_force_enable,
        teacher_view_uses_teacher_traj=teacher_view_uses_teacher_traj,
        teacher_view_default_traj_weight=teacher_view_default_traj_weight,
        teacher_traj_topk_on_teacher_view=teacher_traj_topk_on_teacher_view,
        image_prompt_style=image_prompt_style,
        prompt_text_style=prompt_text_style,
        fuse_history_tokens=fuse_history_tokens,
    )
    dataloader_pin_memory = bool(args.pin_memory and torch.cuda.is_available())
    train_dataloader, train_sampler = build_dataloader(
        train_records,
        batch_size=args.batch_size,
        collator=collator,
        rank=rank,
        world_size=world_size,
        shuffle=not args.data_only_dry_run,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=dataloader_pin_memory,
        persistent_workers=args.persistent_workers,
    )
    val_dataloader, val_sampler = build_dataloader(
        val_records,
        batch_size=args.batch_size,
        collator=collator,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=dataloader_pin_memory,
        persistent_workers=args.persistent_workers,
    )
    steps_per_epoch = len(train_dataloader)
    resolved_max_steps = resolve_requested_steps(
        args=args,
        steps_per_epoch=steps_per_epoch,
        requested_epochs=trainer_cfg.epochs,
    )
    eval_every_steps = interval_steps_from_epochs(args.eval_every_epochs, steps_per_epoch)
    save_every_steps = interval_steps_from_epochs(args.save_every_epochs, steps_per_epoch)

    teacher_ready = sum(
        1
        for record in train_records
        if bool((record.get("teacher_target") or {}).get("teacher_view_allowed"))
    )
    action_aux_ready = sum(
        1
        for record in train_records
        if bool((record.get("gate") or {}).get("action_aux_allowed"))
    )
    val_teacher_ready = sum(
        1
        for record in val_records
        if bool((record.get("teacher_target") or {}).get("teacher_view_allowed"))
    )
    val_action_aux_ready = sum(
        1
        for record in val_records
        if bool((record.get("gate") or {}).get("action_aux_allowed"))
    )

    if args.data_only_dry_run:
        first_batch = next(iter(train_dataloader), None)
        summary = {
            "mode": "data_only_dry_run",
            "corpus_jsonl": str(args.corpus_jsonl),
            "train_records": len(train_records),
            "train_records_with_materialized_assets": len(train_records),
            "all_train_records": len(all_train_records),
            "val_records": len(val_records),
            "val_records_with_materialized_assets": len(val_records),
            "all_val_records": len(all_val_records),
            "skip_asset_check": bool(args.skip_asset_check),
            "teacher_ready_records": teacher_ready,
            "action_aux_ready_records": action_aux_ready,
            "val_teacher_ready_records": val_teacher_ready,
            "val_action_aux_ready_records": val_action_aux_ready,
            "stage_name": trainer_cfg.stage_name,
            "batch_size": args.batch_size,
            "epochs": trainer_cfg.epochs,
            "max_length": trainer_cfg.max_length,
            "steps_per_epoch": steps_per_epoch,
            "resolved_max_steps": resolved_max_steps,
            "eval_every_steps": eval_every_steps,
            "save_every_steps": save_every_steps,
            "student_model": student_model,
            "prompt_mode": prompt_mode,
            "target_mode": target_mode,
            "image_prompt_style": image_prompt_style,
            "prompt_text_style": prompt_text_style,
            "fuse_history_tokens": fuse_history_tokens,
            "teacher_pair_target": teacher_pair_target,
            "hard_view_uses_teacher_cot": hard_view_uses_teacher_cot,
            "enable_teacher_view": enable_teacher_view,
            "teacher_view_force_enable": teacher_view_force_enable,
            "teacher_view_uses_teacher_traj": teacher_view_uses_teacher_traj,
            "teacher_view_default_traj_weight": teacher_view_default_traj_weight,
            "teacher_traj_topk_on_teacher_view": teacher_traj_topk_on_teacher_view,
            "enable_action_aux": enable_action_aux,
            "teacher_traj_cache_dir": str(teacher_traj_cache_dir) if teacher_traj_cache_dir is not None else None,
            "teacher_traj_hidden_size": teacher_traj_hidden_size,
            "traj_hidden_bridge_size": traj_hidden_bridge_size,
            "traj_aux_num_buckets": traj_aux_num_buckets,
            "traj_aux_interface": traj_aux_interface_summary,
            "traj_hidden_bridge": traj_hidden_bridge_cfg,
            "optimization": optimization_cfg,
            "flex": flex_cfg,
            "curriculum": curriculum_cfg,
            "scheduled_sampling": scheduled_sampling_summary,
            "on_policy_gkd": on_policy_gkd_summary,
            "traj_body_prefix_tokens": traj_body_prefix_tokens,
            "interface_loss_weights": export_loss_weights(interface_loss_weights) if interface_loss_weights is not None else None,
            "traj_token_reweighting": traj_token_reweight_summary,
            "traj_decode": traj_decode_summary,
            "decode_eval": {
                "enabled": decode_eval_cfg.enabled,
                "split": decode_eval_cfg.split,
                "num_samples": decode_eval_cfg.num_samples,
                "image_prompt_style": decode_eval_cfg.image_prompt_style,
                "prompt_text_style": decode_eval_cfg.prompt_text_style,
                "fuse_history_tokens": decode_eval_cfg.fuse_history_tokens,
            },
            "first_batch_shapes": {
                "input_ids": list(first_batch["input_ids"].shape) if first_batch is not None else None,
                "labels": list(first_batch["labels"].shape) if first_batch is not None else None,
                "cot_span_mask": list(first_batch["cot_span_mask"].shape) if first_batch is not None else None,
                "traj_span_mask": list(first_batch["traj_span_mask"].shape) if first_batch is not None else None,
                "pixel_values": (
                    list(first_batch["pixel_values"].shape)
                    if first_batch is not None and first_batch.get("pixel_values") is not None
                    else None
                ),
                "image_grid_thw": (
                    list(first_batch["image_grid_thw"].shape)
                    if first_batch is not None and first_batch.get("image_grid_thw") is not None
                    else None
                ),
                "teacher_view_input_ids": (
                    list(first_batch["teacher_view"]["input_ids"].shape)
                    if first_batch is not None and first_batch.get("teacher_view") is not None
                    else None
                ),
                "teacher_topk_indices": (
                    list(first_batch["teacher_view"]["teacher_topk_indices"].shape)
                    if first_batch is not None
                    and first_batch.get("teacher_view") is not None
                    and first_batch["teacher_view"].get("teacher_topk_indices") is not None
                    else None
                ),
                "hard_teacher_topk_indices": (
                    list(first_batch["teacher_topk_indices"].shape)
                    if first_batch is not None and first_batch.get("teacher_topk_indices") is not None
                    else None
                ),
                "hard_teacher_topk_positions": (
                    list(first_batch["teacher_topk_positions"].shape)
                    if first_batch is not None and first_batch.get("teacher_topk_positions") is not None
                    else None
                ),
                "teacher_traj_topk_indices": (
                    list(first_batch["teacher_traj_topk_indices"].shape)
                    if first_batch is not None and first_batch.get("teacher_traj_topk_indices") is not None
                    else None
                ),
                "teacher_traj_hidden": (
                    list(first_batch["teacher_traj_hidden"].shape)
                    if first_batch is not None and first_batch.get("teacher_traj_hidden") is not None
                    else None
                ),
                "teacher_text_boundary_hidden": (
                    list(first_batch["teacher_text_boundary_hidden"].shape)
                    if first_batch is not None and first_batch.get("teacher_text_boundary_hidden") is not None
                    else None
                ),
                "teacher_text_boundary_hidden_positions": (
                    list(first_batch["teacher_text_boundary_hidden_positions"].shape)
                    if first_batch is not None
                    and first_batch.get("teacher_text_boundary_hidden_positions") is not None
                    else None
                ),
                "teacher_action_traj_xyz": (
                    list(first_batch["teacher_action_traj_xyz"].shape)
                    if first_batch is not None and first_batch.get("teacher_action_traj_xyz") is not None
                    else None
                ),
                "teacher_action_traj_available": (
                    list(first_batch["teacher_action_traj_available"].shape)
                    if first_batch is not None and first_batch.get("teacher_action_traj_available") is not None
                    else None
                ),
                "teacher_traj_token_weights": (
                    list(first_batch["teacher_traj_token_weights"].shape)
                    if first_batch is not None and first_batch.get("teacher_traj_token_weights") is not None
                    else None
                ),
                "teacher_traj_label_token_weights": (
                    list(first_batch["teacher_traj_label_token_weights"].shape)
                    if first_batch is not None and first_batch.get("teacher_traj_label_token_weights") is not None
                    else None
                ),
            },
        }
        if is_rank_zero:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))
        _maybe_cleanup_distributed(world_size)
        return

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    parallel_mode = "ddp" if world_size > 1 else "single"
    device_ids = list(range(world_size)) if world_size > 1 else ([rank] if torch.cuda.is_available() else [])
    model = build_student_model(wrapper_cfg, tokenizer)
    use_lora = not args.disable_lora
    init_checkpoint_dir = Path(args.init_checkpoint_dir).expanduser() if args.init_checkpoint_dir is not None else None
    checkpoint_format = detect_checkpoint_format(init_checkpoint_dir) if init_checkpoint_dir is not None else None
    if checkpoint_format == "lora_adapter" and not use_lora:
        raise ValueError("Cannot load a LoRA adapter checkpoint when --disable-lora is set.")
    if trainer_cfg.gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        gc_use_reentrant = bool(stage_options.get("gradient_checkpointing_use_reentrant", False))
        try:
            model.backbone.gradient_checkpointing_enable({"use_reentrant": gc_use_reentrant})
        except Exception:  # noqa: BLE001
            try:
                model.backbone.gradient_checkpointing_enable()
            except Exception:  # noqa: BLE001
                pass
    elif not trainer_cfg.gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_disable"):
        try:
            model.backbone.gradient_checkpointing_disable()
        except Exception:  # noqa: BLE001
            pass
    if use_lora and checkpoint_format != "lora_adapter":
        model.backbone = maybe_apply_lora(
            model.backbone,
            lora_spec,
            enabled=True,
        )
    elif not use_lora:
        model.backbone = maybe_apply_lora(
            model.backbone,
            lora_spec,
            enabled=False,
        )
    if init_checkpoint_dir is not None:
        load_student_checkpoint(
            init_checkpoint_dir,
            model,
            use_lora=use_lora,
            adapter_trainable=use_lora,
        )
    if use_lora and train_lm_head_token_rows and get_lm_head_token_row_count(model.backbone) == 0:
        enable_lm_head_token_rows(
            model.backbone,
            lora_spec.trainable_token_indices or (),
        )
    if getattr(model, "traj_aux_num_buckets", 1) != traj_aux_num_buckets:
        model.configure_traj_aux_head(traj_aux_num_buckets)
    optimization_summary = apply_optimization_policy(model, optimization_cfg)
    if bool(stage_options.get("force_trainable_fp32", True)):
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
    if hasattr(model.backbone, "enable_input_require_grads"):
        try:
            model.backbone.enable_input_require_grads()
        except Exception:  # noqa: BLE001
            pass
    model = model.to(device)

    # ── QAT: ModelOpt INT4 AWQ fake-quantization (optional) ─────────
    qat_quantization = str(getattr(args, "qat_quantization", "") or "").strip().lower()
    if qat_quantization:
        import modelopt.torch.quantization as mtq
        qat_quantization_requested = qat_quantization
        quantize_visual = qat_quantization in {"fp8_vit", "fp8_pcpt_vit"}
        if qat_quantization == "fp8_vit":
            qat_quantization = "fp8"
        elif qat_quantization == "fp8_pcpt_vit":
            qat_quantization = "fp8_pcpt"

        # INT4 FFN-only: start from INT4_AWQ_CFG and disable attention projections.
        # Target deployment spec:
        #   QK, PV, softmax, RMSNorm → FP16
        #   gate_proj, up_proj, down_proj → INT4
        import copy as _copy
        _INT4_FFN_ONLY_CFG = _copy.deepcopy(mtq.INT4_AWQ_CFG)
        _INT4_FFN_ONLY_CFG["quant_cfg"]["*q_proj*"] = {"enable": False}
        _INT4_FFN_ONLY_CFG["quant_cfg"]["*k_proj*"] = {"enable": False}
        _INT4_FFN_ONLY_CFG["quant_cfg"]["*v_proj*"] = {"enable": False}
        _INT4_FFN_ONLY_CFG["quant_cfg"]["*o_proj*"] = {"enable": False}
        _INT4_FFN_ONLY_CFG["quant_cfg"]["*q_norm*"] = {"enable": False}
        _INT4_FFN_ONLY_CFG["quant_cfg"]["*k_norm*"] = {"enable": False}
        _FP8_PCPT_CFG = _copy.deepcopy(mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG)
        _FP8_DEFAULT_CFG = _copy.deepcopy(mtq.FP8_DEFAULT_CFG)

        qat_configs = {
            "int4_awq": mtq.INT4_AWQ_CFG,
            "int4_blockwise": getattr(mtq, "INT4_BLOCKWISE_WEIGHT_ONLY_CFG", mtq.INT4_AWQ_CFG),
            "int4_ffn_only": _INT4_FFN_ONLY_CFG,
            "fp8": _FP8_DEFAULT_CFG,
            "fp8_pcpt": _FP8_PCPT_CFG,
        }
        qat_cfg = _copy.deepcopy(qat_configs.get(qat_quantization))
        if qat_cfg is None:
            raise ValueError(f"Unknown --qat-quantization: {qat_quantization_requested!r}")

        qat_calib_samples = int(getattr(args, "qat_calib_samples", 512) or 512)
        if is_rank_zero:
            print(
                json.dumps(
                    {
                        "event": "qat_quantize_start",
                        "quantization": qat_quantization_requested,
                        "base_quantization": qat_quantization,
                        "calib_samples": qat_calib_samples,
                        "quantize_visual": quantize_visual,
                    }
                ),
                flush=True,
            )

        # FIX #5: Calibration with proper success tracking. Each quantized subtree
        # needs its own loop so visual calibration cannot reuse language counters.
        def _make_qat_calib_forward_loop(calib_target: str):
            def _qat_calib_forward_loop(_target_model):
                _target_model.eval()
                raw_m = model.module if hasattr(model, "module") else model
                raw_m.eval()
                _qat_calib_success = 0
                _qat_calib_fail = 0
                _qat_calib_first_error = None
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=trainer_cfg.bf16):
                    for batch in train_dataloader:
                        if _qat_calib_success >= qat_calib_samples:
                            break
                        # FLEX compression must happen before model forward.
                        batch = prepare_flex_batch_for_model(batch, raw_m)
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        try:
                            raw_m(
                                input_ids=batch["input_ids"],
                                attention_mask=batch.get("attention_mask"),
                                pixel_values=batch.get("pixel_values"),
                                image_grid_thw=batch.get("image_grid_thw"),
                                return_hidden_states=False,
                                compute_meta_action=False,
                                compute_traj_aux=False,
                            )
                            _qat_calib_success += int(batch["input_ids"].shape[0])
                        except Exception as _calib_exc:
                            _qat_calib_fail += 1
                            if _qat_calib_first_error is None:
                                _qat_calib_first_error = repr(_calib_exc)
                if is_rank_zero:
                    print(json.dumps({
                        "event": "qat_calib_done",
                        "target": calib_target,
                        "success": _qat_calib_success,
                        "failed": _qat_calib_fail,
                        "first_error": _qat_calib_first_error,
                    }), flush=True)
                if _qat_calib_success == 0:
                    raise RuntimeError(
                        f"QAT calibration target={calib_target!r} produced 0 successful forwards "
                        f"out of {_qat_calib_fail} attempts. First error: {_qat_calib_first_error}"
                    )

            return _qat_calib_forward_loop

        # FIX #3: Quantize the language model by default. The *_vit variants also
        # quantize the visual tower, while keeping visual boundary/projection paths
        # BF16 so the multimodal projector can learn the feature shift.
        # Qwen3-VL structure:
        #   model.backbone (PeftModel)
        #     .base_model.model (Qwen3VLForConditionalGeneration)
        #       .model.visual (407M) ← quantize only for *_vit modes
        #       .model.language_model (1720M) ← quantize by default
        #       .lm_head (311M) ← keep BF16 for trainable/new token rows
        raw_model = model.module if hasattr(model, "module") else model
        _backbone = raw_model.backbone

        # Navigate to the actual Qwen3VL model inside PeftModel
        _unwrapped = _backbone
        if hasattr(_unwrapped, "base_model"):
            _unwrapped = _unwrapped.base_model
        if hasattr(_unwrapped, "model"):
            _unwrapped = _unwrapped.model
        # _unwrapped is now Qwen3VLForConditionalGeneration
        _qwen_model = getattr(_unwrapped, "model", _unwrapped)
        # _qwen_model is now Qwen3VLModel with .visual and .language_model
        _language_model = getattr(_qwen_model, "language_model", None)
        if _language_model is None:
            raise RuntimeError(
                "Could not find language_model submodule for QAT. "
                f"Available: {[n for n, _ in _qwen_model.named_children()]}"
            )
        _visual_model = getattr(_qwen_model, "visual", None)
        if quantize_visual and _visual_model is None:
            raise RuntimeError("Requested visual FP8 quantization, but the Qwen3-VL visual module was not found.")

        language_qat_cfg = _copy.deepcopy(qat_cfg)
        layer_modules = getattr(_language_model, "layers", None)
        if layer_modules is not None and hasattr(layer_modules, "__len__"):
            last_layer_idx = max(int(len(layer_modules)) - 1, 0)
            for excluded_idx in sorted({0, last_layer_idx}):
                language_qat_cfg.setdefault("quant_cfg", {})[f"*layers.{excluded_idx}.*"] = {"enable": False}
        language_qat_cfg.setdefault("quant_cfg", {})["*embed_tokens*"] = {"enable": False}
        language_qat_cfg.setdefault("quant_cfg", {})["*lm_head*"] = {"enable": False}

        visual_qat_cfg = _copy.deepcopy(qat_cfg)
        if quantize_visual:
            visual_excludes = {
                "*patch_embed*": {"enable": False},
                "*embed*": {"enable": False},
                "*pos_embed*": {"enable": False},
                "*rotary*": {"enable": False},
                "*merger*": {"enable": False},
                "*norm*": {"enable": False},
                "*ln*": {"enable": False},
            }
            visual_qat_cfg.setdefault("quant_cfg", {}).update(visual_excludes)

        if is_rank_zero:
            _lang_params = sum(p.numel() for p in _language_model.parameters())
            _vis_params = sum(p.numel() for p in (_visual_model or torch.nn.Module()).parameters())
            print(json.dumps({
                "event": "qat_scope",
                "quantize_target": "language_visual" if quantize_visual else "language_model",
                "quantization": qat_quantization_requested,
                "base_quantization": qat_quantization,
                "language_params_M": round(_lang_params / 1e6, 1),
                "visual_params_M": round(_vis_params / 1e6, 1),
                "visual_excluded": not quantize_visual,
                "excluded_language_layers": [0, max(int(len(layer_modules)) - 1, 0)] if layer_modules is not None and hasattr(layer_modules, "__len__") else [],
                "visual_exclude_patterns": sorted(visual_qat_cfg.get("quant_cfg", {}).keys()) if quantize_visual else [],
            }), flush=True)

        # Save VisionAttention's original attention functions BEFORE quantize.
        # ModelOpt patches ALL_ATTENTION_FUNCTIONS at the class level, which can
        # bleed into VisionAttention if it shares the same dict via inheritance.
        _vision_attn_originals: list[tuple[Any, dict]] = []
        if _visual_model is not None:
            for _vm_name, _vm_mod in _visual_model.named_modules():
                _aaf = getattr(_vm_mod, "ALL_ATTENTION_FUNCTIONS", None)
                if isinstance(_aaf, dict) and _aaf:
                    _vision_attn_originals.append((_vm_mod, dict(_aaf)))

        # Apply quantization to language_model first. If requested, quantize the
        # visual tower second against the already-quantized language path.
        _quantized_lang = mtq.quantize(
            _language_model,
            language_qat_cfg,
            forward_loop=_make_qat_calib_forward_loop("language_model"),
        )
        _qwen_model.language_model = _quantized_lang
        if quantize_visual and _visual_model is not None:
            _quantized_visual = mtq.quantize(
                _visual_model,
                visual_qat_cfg,
                forward_loop=_make_qat_calib_forward_loop("visual"),
            )
            _qwen_model.visual = _quantized_visual

        # Restore VisionAttention's original attention functions
        for _vm_mod, _orig_aaf in _vision_attn_originals:
            if hasattr(_vm_mod, "ALL_ATTENTION_FUNCTIONS"):
                _vm_mod.ALL_ATTENTION_FUNCTIONS.update(_orig_aaf)

        # Disable quantizers on LoRA adapter weights (keep FP16 trainable)
        _lora_q_disabled = 0
        for _name, _mod in _backbone.named_modules():
            if "lora_" in _name:
                for _attr in ("weight_quantizer", "input_quantizer", "output_quantizer"):
                    _q = getattr(_mod, _attr, None)
                    if _q is not None and hasattr(_q, "disable"):
                        _q.disable()
                        _lora_q_disabled += 1

        # FIX #4: Count quantizers by module family for verification
        if is_rank_zero:
            _counts = {"language": 0, "visual": 0, "lm_head": 0, "lora": 0, "flex": 0, "other": 0}
            for _name, _mod in _backbone.named_modules():
                if not hasattr(_mod, "weight_quantizer"):
                    continue
                _wq = getattr(_mod, "weight_quantizer", None)
                _enabled = _wq is not None and hasattr(_wq, "is_enabled") and _wq.is_enabled
                if not _enabled:
                    continue
                if "lora_" in _name:
                    _counts["lora"] += 1
                elif "visual" in _name:
                    _counts["visual"] += 1
                elif "lm_head" in _name:
                    _counts["lm_head"] += 1
                elif "language_model" in _name or "layers." in _name:
                    _counts["language"] += 1
                else:
                    _counts["other"] += 1

            _flex_ok = not any(
                hasattr(m, "weight_quantizer")
                and getattr(m.weight_quantizer, "is_enabled", False)
                for m in (getattr(raw_model, "flex_scene_encoder", None) or torch.nn.Module()).modules()
            )
            print(json.dumps({
                "event": "qat_quantize_done",
                "quantizers_by_family": _counts,
                "lora_quantizers_disabled": _lora_q_disabled,
                "flex_clean": _flex_ok,
                "visual_quantized": quantize_visual,
                "visual_clean": (not quantize_visual and _counts["visual"] == 0),
            }), flush=True)
            if not quantize_visual and _counts["visual"] > 0:
                raise RuntimeError(f"QAT quantized {_counts['visual']} visual modules — this should not happen.")
            if quantize_visual and _counts["visual"] == 0:
                raise RuntimeError("QAT visual quantization requested, but no enabled visual quantizers were found.")
            if _counts["lora"] > 0:
                raise RuntimeError(f"QAT left {_counts['lora']} LoRA quantizers enabled — disable failed.")

    on_policy_teacher_model = None
    on_policy_teacher_summary: dict[str, object] = {"enabled": False}
    if on_policy_gkd_config is not None and not args.eval_only:
        if world_size > 1:
            raise RuntimeError("on_policy_gkd currently supports single-GPU training only.")
        on_policy_teacher_model, on_policy_teacher_summary = load_on_policy_gkd_teacher(
            stage_options.get("on_policy_gkd"),
            tokenizer=tokenizer,
            device=device,
            is_rank_zero=is_rank_zero,
        )
        on_policy_gkd_config.teacher_token_id_map = {
            int(student_id): int(teacher_id)
            for student_id, teacher_id in dict(on_policy_teacher_summary.get("token_id_map") or {}).items()
        }

    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, static_graph=True)
    if args.eval_only:
        eval_started_at = time.time()
        if val_sampler is not None:
            val_sampler.set_epoch(0)
        val_logs = evaluate_model(
            model,
            val_dataloader,
            device=device,
            bf16=trainer_cfg.bf16,
            world_size=world_size,
            loss_weights=loss_weights,
            traj_decode_config=traj_decode_config,
            traj_aux_interface_config=traj_aux_interface_runtime_config,
            traj_body_prefix_tokens=traj_body_prefix_tokens,
            traj_hidden_bridge_config=traj_hidden_bridge_cfg,
            log_every_batches=args.eval_log_every_batches if args.eval_only else 0,
        )
        if is_rank_zero:
            summary = {
                "mode": "eval_only",
                "student_model": student_model,
                "checkpoint_dir": str(init_checkpoint_dir) if init_checkpoint_dir is not None else None,
                "stage_name": trainer_cfg.stage_name,
                "corpus_jsonl": str(args.corpus_jsonl),
                "train_records": len(train_records),
                "all_train_records": len(all_train_records),
                "val_records": len(val_records),
                "all_val_records": len(all_val_records),
                "max_val_samples": args.max_val_samples,
                "effective_batch_size": trainer_cfg.batch_size * max(world_size, 1),
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
                "pin_memory": dataloader_pin_memory,
                "persistent_workers": bool(args.persistent_workers and args.num_workers > 0),
                "eval_log_every_batches": args.eval_log_every_batches,
                "gradient_checkpointing": trainer_cfg.gradient_checkpointing,
                "bf16": trainer_cfg.bf16,
                "loss_weights": export_loss_weights(loss_weights),
                "data_view": {
                    "prompt_mode": prompt_mode,
                    "target_mode": target_mode,
                    "teacher_pair_target": teacher_pair_target,
                    "enable_teacher_view": enable_teacher_view,
                    "enable_action_aux": enable_action_aux,
                },
                "logs": val_logs,
                "elapsed_sec": round(time.time() - eval_started_at, 3),
            }
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[eval-done] stage={trainer_cfg.stage_name} summary={args.summary_json}", flush=True)
            print(json.dumps(summary, indent=2), flush=True)
        _maybe_cleanup_distributed(world_size)
        return
    optimizer_param_groups, optimizer_group_summary = build_optimizer_param_groups(
        model,
        trainer_cfg.learning_rate,
        optimization_cfg,
    )
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after applying the optimization policy.")
    optimizer = torch.optim.AdamW(optimizer_param_groups, lr=trainer_cfg.learning_rate)
    if is_rank_zero:
        print(
            (
                f"[train-start] stage={trainer_cfg.stage_name} "
                f"train_records={len(train_records)} val_records={len(val_records)} "
                f"steps_per_epoch={steps_per_epoch} max_steps={resolved_max_steps} "
                f"eval_every_steps={eval_every_steps} save_every_steps={save_every_steps} "
                f"batch_per_gpu={args.batch_size} effective_batch={trainer_cfg.batch_size * max(world_size, 1)} "
                f"devices={device_ids} use_lora={use_lora} "
                f"trainable_token_rows={len(lora_spec.trainable_token_indices or ())} "
                f"train_lm_head_token_rows={train_lm_head_token_rows} "
                f"prompt_mode={prompt_mode} target_mode={target_mode} "
                f"teacher_pair_target={teacher_pair_target} "
                f"teacher_traj_hidden_size={teacher_traj_hidden_size} "
                f"scheduled_sampling={scheduled_sampling_summary} "
                f"on_policy_gkd={on_policy_gkd_summary} "
                f"flex={flex_cfg} "
                f"traj_aux_num_buckets={traj_aux_num_buckets} "
                f"freeze_aux_only={bool(optimization_cfg.get('freeze_all_but_traj_aux_head', False))} "
                f"interface_every_n={curriculum_cfg.get('interface_every_n_steps')} "
                f"optimizer_groups={optimizer_group_summary}"
            ),
            flush=True,
        )
        if init_checkpoint_dir is not None:
            print(f"[resume] loading_checkpoint={init_checkpoint_dir}", flush=True)

    if is_rank_zero:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    global_step = 0
    epoch_index = 0
    started_at = time.time()
    eval_history: list[dict[str, object]] = []
    saved_checkpoints: list[dict[str, object]] = []
    best_val_total_loss: float | None = None
    best_decode_score: float | None = None
    best_decode_checkpoint_dir: str | None = None
    consecutive_val_regressions = 0
    early_stop_triggered = False
    early_stop_reason: str | None = None
    early_stop_enabled = (
        trainer_cfg.stage_name == args.early_stop_stage
        and args.early_stop_patience > 0
        and len(val_records) > 0
        and eval_every_steps is not None
    )
    train_config_payload_base = {
        "init_checkpoint_dir": str(init_checkpoint_dir) if init_checkpoint_dir is not None else None,
        "args": {**vars(args), "student_model": student_model},
        "trainer_config": {
            "stage_name": trainer_cfg.stage_name,
            "epochs": trainer_cfg.epochs,
            "max_length": trainer_cfg.max_length,
            "bf16": trainer_cfg.bf16,
            "gradient_checkpointing": trainer_cfg.gradient_checkpointing,
            "learning_rate": trainer_cfg.learning_rate,
            "batch_size": trainer_cfg.batch_size,
            "max_steps": resolved_max_steps,
            "steps_per_epoch": steps_per_epoch,
            "eval_every_steps": eval_every_steps,
            "save_every_steps": save_every_steps,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
            "pin_memory": args.pin_memory and device.type == "cuda",
            "persistent_workers": args.persistent_workers and args.num_workers > 0,
        },
        "parallelism": {
            "multi_gpu_mode": parallel_mode,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_count": max(world_size, torch.cuda.device_count() if torch.cuda.is_available() else 0),
            "device_ids": device_ids,
        },
        "optimizer_groups": optimizer_group_summary,
        "lora": {
            "enabled": use_lora,
            "rank": int(lora_spec.r),
            "alpha": int(lora_spec.alpha),
            "dropout": float(lora_spec.dropout),
            "trainable_token_rows": len(lora_spec.trainable_token_indices or ()),
            "train_lm_head_token_rows": train_lm_head_token_rows,
            "lm_head_trainable_token_rows": get_lm_head_token_row_count(model.backbone),
        },
        "loss_weights": export_loss_weights(loss_weights),
        "data_view": {
            "prompt_mode": prompt_mode,
            "target_mode": target_mode,
            "image_prompt_style": image_prompt_style,
            "prompt_text_style": prompt_text_style,
            "fuse_history_tokens": fuse_history_tokens,
            "teacher_pair_target": teacher_pair_target,
            "enable_teacher_view": enable_teacher_view,
            "enable_action_aux": enable_action_aux,
            "teacher_traj_cache_dir": str(teacher_traj_cache_dir) if teacher_traj_cache_dir is not None else None,
            "teacher_traj_hidden_size": teacher_traj_hidden_size,
            "traj_aux_num_buckets": traj_aux_num_buckets,
            "traj_aux_interface": traj_aux_interface_cfg,
            "traj_aux_interface_runtime": traj_aux_interface_summary,
            "optimization": {**optimization_cfg, **optimization_summary},
            "curriculum": curriculum_cfg,
            "interface_loss_weights": export_loss_weights(interface_loss_weights) if interface_loss_weights is not None else None,
        },
        "scheduled_sampling": scheduled_sampling_summary,
        "on_policy_gkd": {
            **on_policy_gkd_summary,
            "teacher": on_policy_teacher_summary,
        },
        "flex": flex_cfg,
        "traj_token_reweighting": traj_token_reweight_summary,
        "traj_decode": traj_decode_summary,
        "decode_eval": {
            "enabled": decode_eval_cfg.enabled,
            "split": decode_eval_cfg.split,
            "num_samples": decode_eval_cfg.num_samples,
            "max_new_tokens": decode_eval_cfg.max_new_tokens,
            "prompt_mode": decode_eval_cfg.prompt_mode,
            "target_mode": decode_eval_cfg.target_mode,
            "image_prompt_style": decode_eval_cfg.image_prompt_style,
            "prompt_text_style": decode_eval_cfg.prompt_text_style,
            "fuse_history_tokens": decode_eval_cfg.fuse_history_tokens,
            "metric_name": decode_eval_cfg.metric_name,
        },
        "versions": active_versions(),
    }
    metrics_handle = metrics_path.open("w", encoding="utf-8") if is_rank_zero else nullcontext()
    with metrics_handle as opened_metrics_handle:
        while resolved_max_steps is None or global_step < resolved_max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch_index)
            for batch in train_dataloader:
                if resolved_max_steps is not None and global_step >= resolved_max_steps:
                    break
                batch = prepare_flex_batch_for_model(batch, model)
                batch = move_batch_to_device(batch, device)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                next_step = global_step + 1
                interface_every_n_steps = int(curriculum_cfg.get("interface_every_n_steps", 0) or 0)
                use_interface_step = (
                    interface_loss_weights is not None
                    and interface_every_n_steps > 0
                    and next_step % interface_every_n_steps == 0
                )
                active_loss_weights = interface_loss_weights if use_interface_step else loss_weights
                active_phase = "interface" if use_interface_step else "pair_lm"
                autocast_context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if trainer_cfg.bf16 and device.type == "cuda"
                    else nullcontext()
                )
                with autocast_context:
                    loss, logs = run_train_step(
                        model,
                        batch,
                        active_loss_weights,
                        traj_decode_config=traj_decode_config,
                        traj_aux_interface_config=traj_aux_interface_runtime_config,
                        traj_body_prefix_tokens=traj_body_prefix_tokens,
                        traj_hidden_bridge_config=traj_hidden_bridge_cfg,
                        scheduled_sampling_config=scheduled_sampling_config,
                        on_policy_teacher_model=on_policy_teacher_model,
                        on_policy_gkd_config=on_policy_gkd_config,
                        global_step=next_step,
                    )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
                optimizer.step()
                global_step += 1

                logs = _average_scalar_logs(logs, device)
                grad_norm_value = _average_scalar(float(grad_norm.detach().cpu()), device)

                if is_rank_zero:
                    row = {
                        "timestamp": time.time(),
                        "phase": "train",
                        "train_subphase": active_phase,
                        "epoch_index": epoch_index,
                        "global_step": global_step,
                        "logs": {**logs, "grad_norm": grad_norm_value},
                    }
                    opened_metrics_handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                    opened_metrics_handle.flush()
                    if args.log_every_steps > 0 and (
                        global_step == 1
                        or global_step % args.log_every_steps == 0
                        or (resolved_max_steps is not None and global_step == resolved_max_steps)
                    ):
                        epoch_progress = (global_step / steps_per_epoch) if steps_per_epoch else 0.0
                        gt_cot_raw = float(logs.get("gt_cot_loss") or 0.0)
                        teacher_text_kd_raw = float(logs.get("teacher_topk_kd_loss") or 0.0)
                        gt_traj_raw = float(logs.get("traj_loss") or 0.0)
                        teacher_traj_ce_raw = float(logs.get("teacher_traj_loss") or 0.0)
                        teacher_traj_kd_raw = float(logs.get("teacher_traj_topk_kd_loss") or 0.0)
                        teacher_traj_kd_scale = float(logs.get("teacher_traj_topk_kd_scale") or 1.0)
                        format_raw = float(logs.get("output_format_loss") or 0.0)
                        ss_rate = float(logs.get("scheduled_sampling_rate") or 0.0)
                        ss_replaced = float(logs.get("scheduled_sampling_replaced") or 0.0)
                        ss_candidates = float(logs.get("scheduled_sampling_candidates") or 0.0)
                        ss_probability = float(logs.get("scheduled_sampling_probability") or 0.0)
                        gkd_active = float(logs.get("on_policy_gkd_active") or 0.0)
                        gkd_rows = float(logs.get("on_policy_gkd_rows") or 0.0)
                        gkd_tokens = float(logs.get("on_policy_gkd_tokens") or 0.0)
                        gkd_loss = float(logs.get("on_policy_gkd_loss") or 0.0)
                        gkd_prob = float(logs.get("on_policy_gkd_probability") or 0.0)
                        gt_cot_weighted = float(active_loss_weights.hard_cot_ce) * gt_cot_raw
                        teacher_text_kd_weighted = float(active_loss_weights.teacher_logit_kd) * teacher_text_kd_raw
                        gt_traj_weighted = float(active_loss_weights.traj_ce) * gt_traj_raw
                        teacher_traj_ce_weighted = (
                            float(active_loss_weights.teacher_traj_ce or 0.0) * teacher_traj_ce_raw
                        )
                        teacher_traj_kd_weighted = (
                            float(active_loss_weights.teacher_traj_topk_kd) * teacher_traj_kd_raw
                        )
                        format_weighted = float(active_loss_weights.format_ce) * format_raw
                        print(
                            (
                                f"[train][{trainer_cfg.stage_name}] "
                                f"mode={active_phase} "
                                f"step={global_step}/{resolved_max_steps or '?'} "
                                f"epoch={epoch_progress:.3f} "
                                f"total={_format_metric(logs.get('total_loss'))} "
                                f"raw(cot/text_kd/gtraj/ttraj_ce/ttraj_kd/fmt)="
                                f"{gt_cot_raw:.4f}/{teacher_text_kd_raw:.4f}/{gt_traj_raw:.4f}/{teacher_traj_ce_raw:.4f}/"
                                f"{teacher_traj_kd_raw:.4f}/{format_raw:.4f} "
                                f"w(cot/text_kd/gtraj/ttraj_ce/ttraj_kd/fmt)="
                                f"{gt_cot_weighted:.4f}/{teacher_text_kd_weighted:.4f}/{gt_traj_weighted:.4f}/"
                                f"{teacher_traj_ce_weighted:.4f}/{teacher_traj_kd_weighted:.4f}/"
                                f"{format_weighted:.4f} "
                                f"acc(cot/traj)="
                                f"{_format_metric(logs.get('gt_cot_token_acc'))}/"
                                f"{_format_metric(logs.get('traj_token_acc'))} "
                                f"ss={ss_replaced:.0f}/{ss_candidates:.0f}@{ss_rate:.3f}/p{ss_probability:.3f} "
                                f"gkd={gkd_active:.0f}:{gkd_rows:.0f}r/{gkd_tokens:.0f}t/"
                                f"p{gkd_prob:.3f}/loss{gkd_loss:.4f} "
                                f"traj_kd_scale={teacher_traj_kd_scale:.3f} "
                                f"grad={_format_metric(grad_norm_value)}"
                            ),
                            flush=True,
                        )

                should_eval = (
                    val_dataloader is not None
                    and len(val_records) > 0
                    and eval_every_steps is not None
                    and global_step % eval_every_steps == 0
                )
                if should_eval:
                    if val_sampler is not None:
                        val_sampler.set_epoch(global_step)
                    val_logs = evaluate_model(
                            model,
                            val_dataloader,
                            device=device,
                            bf16=trainer_cfg.bf16,
                            world_size=world_size,
                            loss_weights=loss_weights,
                            traj_decode_config=traj_decode_config,
                            traj_aux_interface_config=traj_aux_interface_runtime_config,
                            traj_body_prefix_tokens=traj_body_prefix_tokens,
                            traj_hidden_bridge_config=traj_hidden_bridge_cfg,
                        )
                    val_total_loss = float(val_logs.get("total_loss", float("inf")))
                    if best_val_total_loss is None or val_total_loss <= best_val_total_loss:
                        best_val_total_loss = val_total_loss
                        consecutive_val_regressions = 0
                    else:
                        consecutive_val_regressions += 1
                    eval_row = {
                        "phase": "val",
                        "epoch_progress": (global_step / steps_per_epoch) if steps_per_epoch else 0.0,
                        "global_step": global_step,
                        "logs": val_logs,
                        "best_val_total_loss": best_val_total_loss,
                        "consecutive_val_regressions": consecutive_val_regressions,
                    }
                    if is_rank_zero and decode_eval_cfg.enabled:
                        decode_records = val_records if decode_eval_cfg.split == "val" else train_records
                        decode_eval = evaluate_decode_subset(
                            unwrap_model(model),
                            tokenizer=tokenizer,
                            processor=processor,
                            records=decode_records,
                            device=device,
                            project_root=PROJECT_ROOT,
                            config=decode_eval_cfg,
                            student_model=student_model,
                        )
                        eval_row["decode_eval"] = decode_eval
                        decode_score = float(decode_eval.get(decode_eval_cfg.metric_name, float("-inf")))
                        if best_decode_score is None or decode_score >= best_decode_score:
                            best_decode_score = decode_score
                            best_decode_dir = args.output_dir / "best_decode"
                            checkpoint_payload = save_training_checkpoint(
                                best_decode_dir,
                                model=unwrap_model(model),
                                tokenizer=tokenizer,
                                processor=processor,
                                use_lora=use_lora,
                                train_config_payload={
                                    **train_config_payload_base,
                                    "run_state": {
                                        "global_step": global_step,
                                        "epoch_index": epoch_index,
                                        "completed_epochs": (global_step / steps_per_epoch) if steps_per_epoch else 0.0,
                                        "best_val_total_loss": best_val_total_loss,
                                        "best_decode_score": best_decode_score,
                                        "decode_eval_metric_name": decode_eval_cfg.metric_name,
                                    },
                                },
                            )
                            best_decode_checkpoint_dir = str(best_decode_dir)
                            saved_checkpoints.append(
                                {
                                    "global_step": global_step,
                                    "checkpoint_dir": str(best_decode_dir),
                                    "checkpoint": checkpoint_payload,
                                    "kind": "best_decode",
                                }
                            )
                    eval_history.append(eval_row)
                    if is_rank_zero:
                        opened_metrics_handle.write(
                            json.dumps(
                                {
                                    "timestamp": time.time(),
                                    "epoch_index": epoch_index,
                                    **eval_row,
                                },
                                ensure_ascii=True,
                            )
                            + "\n"
                        )
                        opened_metrics_handle.flush()
                        print(
                            (
                                f"[val][{trainer_cfg.stage_name}] "
                                f"step={global_step}/{resolved_max_steps or '?'} "
                                f"epoch={(global_step / steps_per_epoch) if steps_per_epoch else 0.0:.3f} "
                                f"total={_format_metric(val_total_loss)} "
                                f"best={_format_metric(best_val_total_loss)} "
                                f"regressions={consecutive_val_regressions}/{args.early_stop_patience} "
                                f"decode={_format_metric(best_decode_score if decode_eval_cfg.enabled else None)}"
                            ),
                            flush=True,
                        )
                    if (
                        early_stop_enabled
                        and consecutive_val_regressions >= args.early_stop_patience
                    ):
                        early_stop_triggered = True
                        early_stop_reason = (
                            f"{trainer_cfg.stage_name} validation regressed "
                            f"{consecutive_val_regressions} consecutive evals"
                        )
                        if is_rank_zero:
                            print(
                                f"[early-stop][{trainer_cfg.stage_name}] reason={early_stop_reason}",
                                flush=True,
                            )

                should_save = save_every_steps is not None and global_step % save_every_steps == 0
                if should_save:
                    if world_size > 1:
                        dist.barrier()
                    if is_rank_zero:
                        checkpoint_dir = args.output_dir / f"step_{global_step:06d}"
                        checkpoint_payload = save_training_checkpoint(
                            checkpoint_dir,
                            model=unwrap_model(model),
                            tokenizer=tokenizer,
                            processor=processor,
                            use_lora=use_lora,
                            train_config_payload={
                                **train_config_payload_base,
                                "run_state": {
                                    "global_step": global_step,
                                    "epoch_index": epoch_index,
                                    "completed_epochs": (global_step / steps_per_epoch) if steps_per_epoch else 0.0,
                                    "best_val_total_loss": best_val_total_loss,
                                    "best_decode_score": best_decode_score,
                                    "consecutive_val_regressions": consecutive_val_regressions,
                                    "early_stop_triggered": early_stop_triggered,
                                },
                            },
                        )
                        saved_checkpoints.append(
                            {
                                "global_step": global_step,
                                "checkpoint_dir": str(checkpoint_dir),
                                "checkpoint": checkpoint_payload,
                            }
                        )
                        print(
                            f"[checkpoint][{trainer_cfg.stage_name}] step={global_step} dir={checkpoint_dir}",
                            flush=True,
                        )
                    if world_size > 1:
                        dist.barrier()

                if early_stop_triggered:
                    break
            epoch_index += 1
            if steps_per_epoch == 0 or early_stop_triggered:
                break

    summary = {
        "mode": "train",
        "student_model": student_model,
        "stage_name": trainer_cfg.stage_name,
        "train_records": len(train_records),
        "all_train_records": len(all_train_records),
        "val_records": len(val_records),
        "all_val_records": len(all_val_records),
        "teacher_ready_records": teacher_ready,
        "action_aux_ready_records": action_aux_ready,
        "val_teacher_ready_records": val_teacher_ready,
        "val_action_aux_ready_records": val_action_aux_ready,
        "global_steps": global_step,
        "steps_per_epoch": steps_per_epoch,
        "requested_epochs": trainer_cfg.epochs,
        "completed_epochs": (global_step / steps_per_epoch) if steps_per_epoch else 0.0,
        "resolved_max_steps": resolved_max_steps,
        "eval_every_steps": eval_every_steps,
        "save_every_steps": save_every_steps,
        "effective_batch_size": trainer_cfg.batch_size * max(world_size, 1),
        "learning_rate": trainer_cfg.learning_rate,
        "use_lora": use_lora,
        "multi_gpu_mode": parallel_mode,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count": max(world_size, torch.cuda.device_count() if torch.cuda.is_available() else 0),
        "device_ids": device_ids,
        "best_val_total_loss": best_val_total_loss,
        "best_decode_score": best_decode_score,
        "best_decode_checkpoint_dir": best_decode_checkpoint_dir,
        "consecutive_val_regressions": consecutive_val_regressions,
        "early_stop_enabled": early_stop_enabled,
        "early_stop_stage": args.early_stop_stage,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_triggered": early_stop_triggered,
        "early_stop_reason": early_stop_reason,
        "eval_history": eval_history,
        "saved_checkpoints": saved_checkpoints,
        "on_policy_gkd": {
            **on_policy_gkd_summary,
            "teacher": on_policy_teacher_summary,
        },
        "versions": active_versions(),
        "elapsed_sec": round(time.time() - started_at, 3),
        "metrics_path": str(metrics_path),
        "output_dir": str(args.output_dir),
    }
    if world_size > 1:
        dist.barrier()
    if is_rank_zero and not args.skip_final_save:
        checkpoint_dir = args.output_dir / "final"
        checkpoint_payload = save_training_checkpoint(
            checkpoint_dir,
            model=unwrap_model(model),
            tokenizer=tokenizer,
            processor=processor,
            use_lora=use_lora,
            train_config_payload={
                **train_config_payload_base,
                "run_state": {
                    "global_step": global_step,
                    "epoch_index": epoch_index,
                    "completed_epochs": (global_step / steps_per_epoch) if steps_per_epoch else 0.0,
                    "best_val_total_loss": best_val_total_loss,
                    "best_decode_score": best_decode_score,
                    "consecutive_val_regressions": consecutive_val_regressions,
                    "early_stop_triggered": early_stop_triggered,
                },
            },
        )
        summary["checkpoint_dir"] = str(checkpoint_dir)
        summary["checkpoint"] = checkpoint_payload
    elif is_rank_zero:
        summary["checkpoint_dir"] = None
        summary["checkpoint"] = None
        summary["skip_final_save"] = True
    if is_rank_zero:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[train-done] stage={trainer_cfg.stage_name} summary={args.summary_json}", flush=True)
        print(json.dumps(summary, indent=2))
    _maybe_cleanup_distributed(world_size)


def _spawn_worker(rank: int, world_size: int, master_port: int, args: argparse.Namespace) -> None:
    run_training(args, rank=rank, world_size=world_size, master_port=master_port)


def main() -> None:
    args = parse_args()
    parallel_mode, device_ids = resolve_parallelism(args.multi_gpu)
    if parallel_mode == "ddp" and not args.data_only_dry_run:
        master_port = _find_free_port()
        mp.spawn(
            _spawn_worker,
            nprocs=len(device_ids),
            args=(len(device_ids), master_port, args),
            join=True,
        )
        return
    run_training(args)


if __name__ == "__main__":
    main()
