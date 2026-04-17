#!/usr/bin/env python3
"""WP9-WP13 entrypoint: v3.2 distillation training."""

from __future__ import annotations

import argparse
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
from src.training.trainer import TrainerConfig, move_batch_to_device, run_train_step
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
    parser.add_argument("--seed", type=int, default=42)
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
    return bool(image_paths) and all(_path_exists(path) for path in image_paths)


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
        )

    trainer_config = TrainerConfig(
        stage_name=str(config["stage_name"]),
        epochs=float(config.get("epochs", 1.0)),
        max_length=int(config.get("max_length", 4096)),
        bf16=bool(config.get("bf16", True)),
        batch_size=int(config.get("batch_size", 1)),
        learning_rate=float(config.get("learning_rate", 2e-5)),
    )
    loss_weights = _resolve_weights(weights)
    stage_options = {
        "data_view": dict(config.get("data_view") or {}),
        "traj_token_reweighting": dict(config.get("traj_token_reweighting") or {}),
        "decode_eval": dict(config.get("decode_eval") or {}),
        "traj_decode_reg": dict(config.get("traj_decode_reg") or {}),
        "traj_aux_interface": dict(config.get("traj_aux_interface") or {}),
        "traj_hidden_bridge": dict(config.get("traj_hidden_bridge") or {}),
        "optimization": dict(config.get("optimization") or {}),
        "curriculum": dict(config.get("curriculum") or {}),
        "interface_loss_weights": dict(config.get("interface_loss_weights") or {}),
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
        "freeze_traj_aux_head": False,
        "freeze_all_parameters": False,
    }
    if not bool(cfg.get("freeze_all_but_traj_aux_head", False)):
        if bool(cfg.get("freeze_all_parameters", False)):
            for parameter in model.parameters():
                parameter.requires_grad = False

            trainable_modules: list[str] = []
            explicit_layers = cfg.get("unfreeze_language_layers")
            layer_ids: set[int] = set()
            if isinstance(explicit_layers, (list, tuple)):
                layer_ids = {int(value) for value in explicit_layers}
            layer_from = cfg.get("language_layers_from")
            layer_from = int(layer_from) if layer_from not in (None, "", False) else None

            def should_unfreeze(name: str) -> bool:
                layer_match = re.search(r"language_model\.layers\.(\d+)\.", name)
                if layer_match is not None:
                    layer_index = int(layer_match.group(1))
                    if layer_index in layer_ids:
                        return True
                    if layer_from is not None and layer_index >= layer_from:
                        return True
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
                return False

            for name, parameter in model.named_parameters():
                if should_unfreeze(name):
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

            for layer_index in sorted(layer_ids):
                trainable_modules.append(f"language_layers.{layer_index}")
            if layer_from is not None:
                trainable_modules.append(f"language_layers_from.{layer_from}")
            if bool(cfg.get("unfreeze_language_norm", False)):
                trainable_modules.append("language_norm")
            if bool(cfg.get("unfreeze_token_embeddings", False)):
                trainable_modules.append("token_embeddings")
            if bool(cfg.get("unfreeze_lm_head", False)):
                trainable_modules.append("lm_head")
            if bool(cfg.get("unfreeze_multimodal_projector", False)):
                trainable_modules.append("multimodal_projector")

            return {
                "freeze_all_but_traj_aux_head": False,
                "freeze_traj_aux_head": False,
                "freeze_all_parameters": True,
                "trainable_modules": trainable_modules,
            }
        if bool(cfg.get("freeze_traj_aux_head", False)):
            traj_aux_head = getattr(model, "traj_aux_head", None)
            if traj_aux_head is None:
                raise RuntimeError("freeze_traj_aux_head requires model.traj_aux_head to exist.")
            for parameter in traj_aux_head.parameters():
                parameter.requires_grad = False
            summary["freeze_traj_aux_head"] = True
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
        "trainable_modules": ["traj_aux_head"],
    }


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
        for token_idx in (record.get("hard_target") or {}).get("traj_future_token_ids") or []:
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


def infer_teacher_traj_hidden_size(cache_dir: Path | None) -> int | None:
    """Infer the teacher trajectory hidden width from the imported cache."""
    if cache_dir is None:
        return None
    hidden_dir = cache_dir / "hidden"
    if not hidden_dir.exists():
        return None
    for path in sorted(hidden_dir.glob("*.npy")):
        try:
            hidden = np.load(path, mmap_mode="r")
        except Exception:  # noqa: BLE001
            continue
        if hidden.ndim >= 2:
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
        metric_name=str(cfg.get("metric_name", "anti_collapse_score")),
    )


def build_dataloader(
    records: list[dict],
    *,
    batch_size: int,
    collator,
    rank: int,
    world_size: int,
    shuffle: bool,
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
    dataloader = DataLoader(
        records,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        collate_fn=collator,
    )
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
) -> dict[str, float]:
    """Run a validation pass and return mean scalar metrics."""
    metric_sums: dict[str, float] = {}
    local_batches = 0
    model.eval()
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        autocast_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if bf16 and device.type == "cuda"
            else nullcontext()
        )
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
    train_records = [record for record in all_train_records if has_required_materialized_assets(record)]
    val_records = [record for record in all_val_records if has_required_materialized_assets(record)]
    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]
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
    if prompt_mode not in {"joint", "traj_only"}:
        raise ValueError(f"Unsupported data_view.prompt_mode={prompt_mode!r}")
    if target_mode not in {"joint", "traj_only"}:
        raise ValueError(f"Unsupported data_view.target_mode={target_mode!r}")
    enable_teacher_view = bool(data_view_cfg.get("enable_teacher_view", True))
    enable_action_aux = bool(data_view_cfg.get("enable_action_aux", True))
    teacher_pair_target = bool(data_view_cfg.get("teacher_pair_target", False))
    teacher_traj_cache_dir_raw = data_view_cfg.get("teacher_traj_cache_dir")
    teacher_traj_cache_dir = (
        Path(str(remap_external_path(teacher_traj_cache_dir_raw)))
        if teacher_traj_cache_dir_raw not in (None, "")
        else None
    )
    teacher_traj_hidden_size = None
    if loss_weights.teacher_traj_hidden_align > 0:
        teacher_traj_hidden_size = infer_teacher_traj_hidden_size(teacher_traj_cache_dir)
        if teacher_traj_hidden_size is None:
            raise RuntimeError(
                "teacher_traj_hidden_align is enabled but no teacher trajectory hidden cache dimension could be inferred."
            )
    traj_aux_interface_cfg = dict(stage_options.get("traj_aux_interface") or {})
    traj_hidden_bridge_cfg = dict(stage_options.get("traj_hidden_bridge") or {})
    traj_aux_num_buckets = max(int(traj_aux_interface_cfg.get("num_buckets", 1) or 1), 1)
    traj_hidden_bridge_size = (
        int(traj_hidden_bridge_cfg.get("shared_size"))
        if traj_hidden_bridge_cfg.get("shared_size") not in (None, "", 0)
        else None
    )
    optimization_cfg = dict(stage_options.get("optimization") or {})
    curriculum_cfg = dict(stage_options.get("curriculum") or {})
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
        traj_teacher_hidden_size=teacher_traj_hidden_size,
        traj_aux_num_buckets=traj_aux_num_buckets,
        traj_hidden_bridge_size=traj_hidden_bridge_size,
    )
    tokenizer = load_student_tokenizer(wrapper_cfg)
    processor = load_student_processor(wrapper_cfg, tokenizer=tokenizer)
    lora_spec = LoraConfigSpec(trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer)))
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
    )
    train_dataloader, train_sampler = build_dataloader(
        train_records,
        batch_size=args.batch_size,
        collator=collator,
        rank=rank,
        world_size=world_size,
        shuffle=not args.data_only_dry_run,
    )
    val_dataloader, val_sampler = build_dataloader(
        val_records,
        batch_size=args.batch_size,
        collator=collator,
        rank=rank,
        world_size=world_size,
        shuffle=False,
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
            "teacher_pair_target": teacher_pair_target,
            "enable_teacher_view": enable_teacher_view,
            "enable_action_aux": enable_action_aux,
            "teacher_traj_cache_dir": str(teacher_traj_cache_dir) if teacher_traj_cache_dir is not None else None,
            "teacher_traj_hidden_size": teacher_traj_hidden_size,
            "traj_hidden_bridge_size": traj_hidden_bridge_size,
            "traj_aux_num_buckets": traj_aux_num_buckets,
            "traj_aux_interface": traj_aux_interface_summary,
            "traj_hidden_bridge": traj_hidden_bridge_cfg,
            "optimization": optimization_cfg,
            "curriculum": curriculum_cfg,
            "traj_body_prefix_tokens": traj_body_prefix_tokens,
            "interface_loss_weights": export_loss_weights(interface_loss_weights) if interface_loss_weights is not None else None,
            "traj_token_reweighting": traj_token_reweight_summary,
            "traj_decode": traj_decode_summary,
            "decode_eval": {
                "enabled": decode_eval_cfg.enabled,
                "split": decode_eval_cfg.split,
                "num_samples": decode_eval_cfg.num_samples,
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
    if hasattr(model.backbone, "gradient_checkpointing_enable"):
        try:
            model.backbone.gradient_checkpointing_enable()
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
    if getattr(model, "traj_aux_num_buckets", 1) != traj_aux_num_buckets:
        model.configure_traj_aux_head(traj_aux_num_buckets)
    optimization_summary = apply_optimization_policy(model, optimization_cfg)
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    if hasattr(model.backbone, "enable_input_require_grads"):
        try:
            model.backbone.enable_input_require_grads()
        except Exception:  # noqa: BLE001
            pass
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, static_graph=True)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after applying the optimization policy.")
    optimizer = torch.optim.AdamW(trainable_params, lr=trainer_cfg.learning_rate)
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
                f"prompt_mode={prompt_mode} target_mode={target_mode} "
                f"teacher_pair_target={teacher_pair_target} "
                f"teacher_traj_hidden_size={teacher_traj_hidden_size} "
                f"traj_aux_num_buckets={traj_aux_num_buckets} "
                f"freeze_aux_only={bool(optimization_cfg.get('freeze_all_but_traj_aux_head', False))} "
                f"interface_every_n={curriculum_cfg.get('interface_every_n_steps')}"
            ),
            flush=True,
        )
        if init_checkpoint_dir is not None:
            print(f"[resume] loading_checkpoint={init_checkpoint_dir}", flush=True)

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
        "args": {**vars(args), "student_model": student_model},
        "trainer_config": {
            "stage_name": trainer_cfg.stage_name,
            "epochs": trainer_cfg.epochs,
            "max_length": trainer_cfg.max_length,
            "bf16": trainer_cfg.bf16,
            "learning_rate": trainer_cfg.learning_rate,
            "batch_size": trainer_cfg.batch_size,
            "max_steps": resolved_max_steps,
            "steps_per_epoch": steps_per_epoch,
            "eval_every_steps": eval_every_steps,
            "save_every_steps": save_every_steps,
        },
        "parallelism": {
            "multi_gpu_mode": parallel_mode,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_count": max(world_size, torch.cuda.device_count() if torch.cuda.is_available() else 0),
            "device_ids": device_ids,
        },
        "loss_weights": export_loss_weights(loss_weights),
        "data_view": {
            "prompt_mode": prompt_mode,
            "target_mode": target_mode,
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
        "traj_token_reweighting": traj_token_reweight_summary,
        "traj_decode": traj_decode_summary,
        "decode_eval": {
            "enabled": decode_eval_cfg.enabled,
            "split": decode_eval_cfg.split,
            "num_samples": decode_eval_cfg.num_samples,
            "max_new_tokens": decode_eval_cfg.max_new_tokens,
            "prompt_mode": decode_eval_cfg.prompt_mode,
            "target_mode": decode_eval_cfg.target_mode,
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
                        print(
                            (
                                f"[train][{trainer_cfg.stage_name}] "
                                f"mode={active_phase} "
                                f"step={global_step}/{resolved_max_steps or '?'} "
                                f"epoch={epoch_progress:.3f} "
                                f"total={_format_metric(logs.get('total_loss'))} "
                                f"gt_cot={_format_metric(logs.get('gt_cot_loss'))} "
                                f"teacher_cot={_format_metric(logs.get('teacher_cot_loss'))} "
                                f"topk_kd={_format_metric(logs.get('teacher_topk_kd_loss'))} "
                                f"traj={_format_metric(logs.get('traj_loss'))} "
                                f"traj_aux={_format_metric(logs.get('traj_aux_loss'))} "
                                f"traj_aux_abs={_format_metric(logs.get('traj_aux_abs_max'))} "
                                f"traj_xyz={_format_metric(logs.get('traj_xyz_loss'))} "
                                f"traj_delta={_format_metric(logs.get('traj_delta_loss'))} "
                                f"meta_action={_format_metric(logs.get('meta_action_loss'))} "
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
        "versions": active_versions(),
        "elapsed_sec": round(time.time() - started_at, 3),
        "metrics_path": str(metrics_path),
        "output_dir": str(args.output_dir),
    }
    if world_size > 1:
        dist.barrier()
    if is_rank_zero:
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
