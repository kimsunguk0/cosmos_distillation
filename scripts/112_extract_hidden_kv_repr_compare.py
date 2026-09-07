#!/usr/bin/env python3
"""Compare Step-B student hidden states and KV cache representations.

AE-reads-KV analysis target:
    Did trajectory top-k KD, or FullFT vs LoRA, move the backbone
    representation consumed by the Action Expert, independent of the LM head?

This script intentionally reuses local conventions from:
  * scripts/21_probe_hidden_latent.py for hidden geometry metrics.
  * scripts/82_eval_test_b_teacher_forced.py for Step-B checkpoint loading.
  * scripts/54_probe_no_nav_hidden_qc.py and
    scripts/55_probe_no_nav_prefill_hidden_qc.py for hidden-output access
    conventions.
  * scripts/50_train_stage1_ae28_teacher_kv_overfit.py and
    scripts/51_train_stage1_ae28_teacher_kv_scale.py for AE KV cache shape
    conventions.
  * scripts/17_profile_action_pre_state.py for the AE action-pre boundary note.
  * scripts/68-70 hidden-to-action probe scripts for the optional cheap
    hidden-to-teacher-trajectory R2 probe framing.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "reports" / "hidden_kv_repr_compare_val512.json"
TRAJ_TOKEN_COUNT = 128
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Repeatable checkpoint spec, for example CE=outputs/.../best_decode.",
    )
    parser.add_argument(
        "--ce-baseline-dir",
        default="CE",
        help="CE baseline checkpoint label or path. Defaults to label CE.",
    )
    parser.add_argument("--student-model", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--image-prompt-style",
        choices=("compact", "camera_labeled"),
        default="camera_labeled",
    )
    parser.add_argument(
        "--prompt-text-style",
        choices=("numeric_history_question", "official_alpamayo"),
        default="official_alpamayo",
    )
    parser.add_argument("--fuse-history-tokens", action="store_true")
    parser.add_argument(
        "--metric-token-cap",
        type=int,
        default=0,
        help=(
            "Optional deterministic token cap for exact scripts/21 Gram metrics. "
            "0 uses every extracted trajectory token."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compute-hidden-action-r2",
        action="store_true",
        help="Run a cheap linear ridge R2 probe from mean traj hidden to teacher action trajectory xyz.",
    )
    parser.add_argument("--r2-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--r2-val-fraction", type=float, default=0.25)
    parser.add_argument(
        "--dump-hidden-action-npz",
        type=str,
        default=None,
        help="If set, dump per-label pooled features + teacher-action targets to this .npz for offline probing.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic shape tests for metrics/KV aggregation without loading checkpoints.",
    )
    return parser.parse_args()


def load_script_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_metric_module():
    return load_script_module("hidden_latent_21", PROJECT_ROOT / "scripts" / "21_probe_hidden_latent.py")


def load_teacher_forced_module():
    return load_script_module("teacher_forced_82", PROJECT_ROOT / "scripts" / "82_eval_test_b_teacher_forced.py")


def parse_checkpoint_specs(raw_specs: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"--checkpoint-dir must be LABEL=PATH, got {raw!r}")
        label, value = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty checkpoint label in {raw!r}")
        path = Path(value.strip())
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        specs.append((label, path))
    if not specs:
        raise ValueError("At least one --checkpoint-dir LABEL=PATH is required.")
    labels = [label for label, _path in specs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate checkpoint labels are not allowed: {labels}")
    return specs


def resolve_baseline_label(specs: list[tuple[str, Path]], raw_baseline: str) -> str:
    labels = {label for label, _path in specs}
    if raw_baseline in labels:
        return raw_baseline
    baseline_path = Path(raw_baseline)
    if not baseline_path.is_absolute():
        baseline_path = PROJECT_ROOT / baseline_path
    matches = [label for label, path in specs if path.resolve() == baseline_path.resolve()]
    if len(matches) == 1:
        return matches[0]
    if "CE" in labels:
        return "CE"
    raise ValueError(f"Could not resolve CE baseline from {raw_baseline!r}; labels={sorted(labels)}")


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def maybe_cap_tokens(x: np.ndarray, cap: int, *, seed: int) -> np.ndarray:
    if cap <= 0 or int(x.shape[0]) <= cap:
        return x
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(int(x.shape[0]), size=int(cap), replace=False))
    return x[indices]


def safe_token_cosine(metric_mod: Any, left: np.ndarray, right: np.ndarray) -> tuple[float | None, str | None]:
    if left.ndim != 2 or right.ndim != 2:
        return None, f"expected 2D arrays, got {left.shape} vs {right.shape}"
    if int(left.shape[0]) != int(right.shape[0]):
        return None, f"token count mismatch {left.shape[0]} vs {right.shape[0]}"
    if int(left.shape[1]) != int(right.shape[1]):
        return None, f"hidden dim mismatch {left.shape[1]} vs {right.shape[1]}"
    return finite_float(metric_mod.token_cosine_mean(left, right)), None


def hidden_geometry_metrics(
    metric_mod: Any,
    left: np.ndarray,
    right: np.ndarray,
    *,
    metric_token_cap: int,
    seed: int,
) -> dict[str, Any]:
    if int(left.shape[0]) != int(right.shape[0]):
        raise ValueError(f"Hidden token count mismatch: {left.shape} vs {right.shape}")
    if metric_token_cap > 0 and int(left.shape[0]) > metric_token_cap:
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(int(left.shape[0]), size=int(metric_token_cap), replace=False))
        left_metric = left[indices]
        right_metric = right[indices]
    else:
        left_metric = left
        right_metric = right
    token_cos, token_cos_skip = safe_token_cosine(metric_mod, left_metric, right_metric)
    return {
        "token_count_total": int(left.shape[0]),
        "metric_token_count": int(left_metric.shape[0]),
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "token_cosine_mean": token_cos,
        "token_cosine_skip_reason": token_cos_skip,
        "gram_corr": finite_float(metric_mod.gram_corr(left_metric, right_metric)),
        "centered_gram_corr": finite_float(metric_mod.centered_gram_corr(left_metric, right_metric)),
    }


def load_rows(tf_mod: Any, path: Path, split: str, num_samples: int) -> list[dict[str, Any]]:
    rows = [row for row in tf_mod.load_jsonl(path) if row.get("split") == split]
    if num_samples > 0:
        rows = rows[: int(num_samples)]
    if not rows:
        raise RuntimeError(f"No rows selected from {path} for split={split!r}")
    return rows


def make_load_args(args: argparse.Namespace, checkpoint_dir: Path, tf_mod: Any) -> argparse.Namespace:
    student_model = args.student_model
    if student_model is None:
        student_model = tf_mod.resolve_student_model_path()
    return argparse.Namespace(
        checkpoint_dir=checkpoint_dir,
        student_model=student_model,
        device=args.device,
    )


def load_model_bundle(args: argparse.Namespace, checkpoint_dir: Path, tf_mod: Any):
    model, tokenizer, processor, device, base_model, train_config = tf_mod.load_model(
        make_load_args(args, checkpoint_dir, tf_mod)
    )
    # Bypass optional traj-hidden bridge/projector heads (present in hidden-align
    # checkpoints). The R2 probe only reads raw backbone hidden_states; these heads
    # are unused here and can crash forward on dtype mismatch (float32 head vs bf16
    # hidden). Nulling them leaves result["hidden_states"] (raw 2048-d) untouched.
    for _attr in ("traj_hidden_projector", "traj_hidden_bridge_student", "traj_hidden_bridge_teacher"):
        if getattr(model, _attr, None) is not None:
            try:
                setattr(model, _attr, None)
                print(json.dumps({"event": "disabled_bridge_head_for_probe", "head": _attr}), flush=True)
            except Exception:
                pass
    return {
        "model": model,
        "tokenizer": tokenizer,
        "processor": processor,
        "device": device,
        "base_model": base_model,
        "train_config": train_config,
    }


def build_teacher_pair_collator(args: argparse.Namespace, tf_mod: Any, bundle: dict[str, Any]):
    train_config = bundle["train_config"]
    data_view = train_config.get("data_view") or {}
    trainer_config = train_config.get("trainer_config") or {}
    return tf_mod.DistillationCollator(
        tokenizer=bundle["tokenizer"],
        processor=bundle["processor"],
        project_root=PROJECT_ROOT,
        teacher_pair_target=True,
        enable_teacher_view=False,
        enable_action_aux=False,
        teacher_traj_hidden_source="hidden",
        prompt_mode=str(data_view.get("prompt_mode") or "joint"),
        target_mode=str(data_view.get("target_mode") or "joint"),
        image_prompt_style=args.image_prompt_style,
        prompt_text_style=args.prompt_text_style,
        fuse_history_tokens=bool(args.fuse_history_tokens),
        max_length=int(trainer_config.get("max_length", 4096)),
    )


def batched(items: list[Any], batch_size: int):
    width = max(int(batch_size), 1)
    for index in range(0, len(items), width):
        yield items[index : index + width]


def move_batch_to_device(batch: dict[str, Any], *, device: torch.device, model_dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=model_dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def forward_representations(model: torch.nn.Module, moved: dict[str, Any]) -> tuple[torch.Tensor, Any]:
    kwargs = {
        "input_ids": moved["input_ids"],
        "attention_mask": moved["attention_mask"],
        "return_hidden_states": True,
        "compute_meta_action": False,
        "compute_traj_aux": False,
        "use_cache": True,
    }
    for optional_key in ("pixel_values", "image_grid_thw"):
        if optional_key in moved and moved[optional_key] is not None:
            kwargs[optional_key] = moved[optional_key]
    try:
        with torch.inference_mode():
            outputs = model(**kwargs, logits_to_keep=1)
    except TypeError:
        with torch.inference_mode():
            outputs = model(**kwargs)
    final_hidden = outputs.get("hidden_states")
    if final_hidden is None:
        backbone_outputs = outputs.get("backbone_outputs")
        hidden_states = getattr(backbone_outputs, "hidden_states", None)
        if hidden_states is None and hasattr(backbone_outputs, "language_model_outputs"):
            hidden_states = getattr(backbone_outputs.language_model_outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Student forward did not return hidden states.")
        final_hidden = hidden_states[-1]
    cache = getattr(outputs.get("backbone_outputs"), "past_key_values", None)
    if cache is None:
        raise RuntimeError("Student forward did not return past_key_values with use_cache=True.")
    return final_hidden, cache


def traj_positions_from_batch(batch: dict[str, Any], *, token_count: int = TRAJ_TOKEN_COUNT) -> torch.Tensor:
    labels = batch["labels"]
    traj_mask = batch["traj_token_mask"].bool() & (labels != -100)
    positions: list[torch.Tensor] = []
    bad_rows: list[tuple[int, int]] = []
    for row_index in range(int(traj_mask.shape[0])):
        row_pos = torch.nonzero(traj_mask[row_index], as_tuple=False).flatten()
        if int(row_pos.numel()) < int(token_count):
            bad_rows.append((row_index, int(row_pos.numel())))
            continue
        positions.append(row_pos[: int(token_count)].long())
    if bad_rows:
        raise RuntimeError(f"Rows with fewer than {token_count} trajectory tokens: {bad_rows[:8]}")
    return torch.stack(positions, dim=0)


def gather_hidden_tokens(final_hidden: torch.Tensor, positions: torch.Tensor) -> np.ndarray:
    pieces: list[torch.Tensor] = []
    positions = positions.to(device=final_hidden.device)
    for row_index in range(int(final_hidden.shape[0])):
        pieces.append(final_hidden[row_index].index_select(0, positions[row_index]))
    return torch.stack(pieces, dim=0).detach().float().cpu().numpy().astype(np.float32)


def gather_teacher_hidden(batch: dict[str, Any], *, token_count: int = TRAJ_TOKEN_COUNT) -> np.ndarray:
    teacher_hidden = batch.get("teacher_traj_hidden")
    teacher_mask = batch.get("teacher_traj_hidden_mask")
    if teacher_hidden is None or teacher_mask is None:
        raise RuntimeError("Batch is missing teacher_traj_hidden/teacher_traj_hidden_mask.")
    pieces: list[torch.Tensor] = []
    bad_rows: list[tuple[int, int]] = []
    for row_index in range(int(teacher_hidden.shape[0])):
        valid = torch.nonzero(teacher_mask[row_index].bool(), as_tuple=False).flatten()
        if int(valid.numel()) < int(token_count):
            bad_rows.append((row_index, int(valid.numel())))
            continue
        pieces.append(teacher_hidden[row_index].index_select(0, valid[: int(token_count)]))
    if bad_rows:
        raise RuntimeError(f"Rows with fewer than {token_count} teacher hidden tokens: {bad_rows[:8]}")
    return torch.stack(pieces, dim=0).float().cpu().numpy().astype(np.float32)


def cache_layers(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if cache is None:
        raise RuntimeError("Missing cache.")
    if hasattr(cache, "layers"):
        out = []
        for layer in list(cache.layers):
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None or value is None:
                raise RuntimeError("Cache layer is missing keys/values.")
            out.append((key, value))
        if out:
            return out
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        return [(key, value) for key, value in zip(list(key_cache), list(value_cache), strict=True)]
    if isinstance(cache, (list, tuple)):
        out = []
        for layer in cache:
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                out.append((layer.keys, layer.values))
            elif isinstance(layer, (list, tuple)) and len(layer) >= 2:
                out.append((layer[0], layer[1]))
            else:
                raise RuntimeError(f"Unsupported cache layer type: {type(layer)!r}")
        if out:
            return out
    raise RuntimeError(f"Unsupported cache type: {type(cache)!r}")


def infer_cache_seq_axis(tensor: torch.Tensor, positions: torch.Tensor) -> int:
    max_pos = int(positions.max().item())
    candidates: list[int] = []
    preferred = [2, 1, tensor.ndim - 2]
    for axis in preferred + list(range(1, tensor.ndim)):
        if axis < 0 or axis >= tensor.ndim or axis in candidates:
            continue
        if int(tensor.shape[axis]) > max_pos:
            candidates.append(axis)
    if not candidates:
        raise RuntimeError(f"Could not infer cache sequence axis for shape={tuple(tensor.shape)}, max_pos={max_pos}")
    return candidates[0]


def select_cache_positions(tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 3:
        raise RuntimeError(f"Expected cache tensor with batch/seq/features dims, got shape={tuple(tensor.shape)}")
    if int(tensor.shape[0]) != int(positions.shape[0]):
        raise RuntimeError(f"Cache batch mismatch: tensor={tuple(tensor.shape)}, positions={tuple(positions.shape)}")
    seq_axis = infer_cache_seq_axis(tensor, positions)
    positions = positions.to(device=tensor.device, dtype=torch.long)
    rows: list[torch.Tensor] = []
    for row_index in range(int(tensor.shape[0])):
        row_tensor = tensor[row_index]
        row_axis = seq_axis - 1
        selected = row_tensor.index_select(row_axis, positions[row_index])
        selected = selected.movedim(row_axis, 0).contiguous().reshape(int(positions.shape[1]), -1)
        rows.append(selected)
    return torch.cat(rows, dim=0).float()


@dataclass
class ScalarAccumulator:
    total: float = 0.0
    count: int = 0

    def add_tensor(self, values: torch.Tensor) -> None:
        finite = values.detach().float()
        finite = finite[torch.isfinite(finite)]
        if finite.numel() == 0:
            return
        self.total += float(finite.sum().cpu())
        self.count += int(finite.numel())

    def mean(self) -> float | None:
        if self.count <= 0:
            return None
        return finite_float(self.total / self.count)


@dataclass
class OneKVStats:
    relative_l2: ScalarAccumulator = field(default_factory=ScalarAccumulator)
    cosine: ScalarAccumulator = field(default_factory=ScalarAccumulator)

    def add(self, baseline: torch.Tensor, other: torch.Tensor) -> None:
        base = baseline.float().flatten(1)
        comp = other.float().flatten(1)
        if base.shape != comp.shape:
            raise RuntimeError(f"KV tensor shape mismatch: {tuple(base.shape)} vs {tuple(comp.shape)}")
        base_norm = torch.linalg.vector_norm(base, dim=1).clamp_min(EPS)
        comp_norm = torch.linalg.vector_norm(comp, dim=1).clamp_min(EPS)
        rel = torch.linalg.vector_norm(comp - base, dim=1) / base_norm
        cos = (base * comp).sum(dim=1) / (base_norm * comp_norm).clamp_min(EPS)
        self.relative_l2.add_tensor(rel)
        self.cosine.add_tensor(cos)

    def summary(self) -> dict[str, float | None]:
        return {
            "relative_l2_mean": self.relative_l2.mean(),
            "cosine_mean": self.cosine.mean(),
        }


@dataclass
class LayerKVStats:
    key: OneKVStats = field(default_factory=OneKVStats)
    value: OneKVStats = field(default_factory=OneKVStats)

    def add(self, base_key: torch.Tensor, other_key: torch.Tensor, base_value: torch.Tensor, other_value: torch.Tensor) -> None:
        self.key.add(base_key, other_key)
        self.value.add(base_value, other_value)

    def summary(self) -> dict[str, Any]:
        return {
            "key": self.key.summary(),
            "value": self.value.summary(),
        }


def update_kv_delta_stats(
    stats: dict[int, LayerKVStats],
    baseline_cache: Any,
    other_cache: Any,
    positions: torch.Tensor,
) -> None:
    base_layers = cache_layers(baseline_cache)
    other_layers = cache_layers(other_cache)
    if len(base_layers) != len(other_layers):
        raise RuntimeError(f"Cache layer count mismatch: {len(base_layers)} vs {len(other_layers)}")
    for layer_index, ((base_key, base_value), (other_key, other_value)) in enumerate(zip(base_layers, other_layers, strict=True)):
        base_key_pos = select_cache_positions(base_key, positions)
        other_key_pos = select_cache_positions(other_key, positions)
        base_value_pos = select_cache_positions(base_value, positions)
        other_value_pos = select_cache_positions(other_value, positions)
        stats[layer_index].add(base_key_pos, other_key_pos, base_value_pos, other_value_pos)
        del base_key_pos, other_key_pos, base_value_pos, other_value_pos


def zero_kv_summary(layer_count: int) -> dict[str, Any]:
    layers = []
    for layer_index in range(int(layer_count)):
        layers.append(
            {
                "layer_index": layer_index,
                "key": {"relative_l2_mean": 0.0, "cosine_mean": 1.0},
                "value": {"relative_l2_mean": 0.0, "cosine_mean": 1.0},
            }
        )
    return summarize_kv_layers(layers)


def summarize_kv_stats(stats: dict[int, LayerKVStats]) -> dict[str, Any]:
    layers = []
    for layer_index in sorted(stats):
        item = stats[layer_index].summary()
        layers.append({"layer_index": int(layer_index), **item})
    return summarize_kv_layers(layers)


def summarize_kv_layers(layers: list[dict[str, Any]]) -> dict[str, Any]:
    key_rel = [layer["key"]["relative_l2_mean"] for layer in layers if layer["key"]["relative_l2_mean"] is not None]
    value_rel = [layer["value"]["relative_l2_mean"] for layer in layers if layer["value"]["relative_l2_mean"] is not None]
    key_cos = [layer["key"]["cosine_mean"] for layer in layers if layer["key"]["cosine_mean"] is not None]
    value_cos = [layer["value"]["cosine_mean"] for layer in layers if layer["value"]["cosine_mean"] is not None]
    return {
        "layer_count": int(len(layers)),
        "key_relative_l2_mean_over_layers": finite_float(np.mean(key_rel)) if key_rel else None,
        "value_relative_l2_mean_over_layers": finite_float(np.mean(value_rel)) if value_rel else None,
        "key_cosine_mean_over_layers": finite_float(np.mean(key_cos)) if key_cos else None,
        "value_cosine_mean_over_layers": finite_float(np.mean(value_cos)) if value_cos else None,
        "layers": layers,
    }


def extract_hidden_for_label(
    *,
    label: str,
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    collator: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    model_dtype = next(model.backbone.parameters()).dtype
    hidden_batches: list[np.ndarray] = []
    teacher_batches: list[np.ndarray] = []
    sample_ids: list[str] = []
    layer_count = 0
    for row_batch in batched(rows, batch_size):
        batch = collator(row_batch)
        moved = move_batch_to_device(batch, device=device, model_dtype=model_dtype)
        positions = traj_positions_from_batch(moved)
        final_hidden, cache = forward_representations(model, moved)
        hidden_batches.append(gather_hidden_tokens(final_hidden, positions))
        teacher_batches.append(gather_teacher_hidden(batch))
        sample_ids.extend(str(value) for value in batch.get("sample_ids", []))
        if layer_count <= 0:
            layer_count = len(cache_layers(cache))
        del moved, final_hidden, cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"event": "hidden_batch_done", "label": label, "done": len(sample_ids), "total": len(rows)}), flush=True)
    return np.concatenate(hidden_batches, axis=0), np.concatenate(teacher_batches, axis=0), layer_count, sample_ids


def compare_candidate_to_baseline(
    *,
    baseline_label: str,
    candidate_label: str,
    baseline_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    rows: list[dict[str, Any]],
    collator: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    baseline_dtype = next(baseline_model.backbone.parameters()).dtype
    candidate_dtype = next(candidate_model.backbone.parameters()).dtype
    hidden_batches: list[np.ndarray] = []
    sample_ids: list[str] = []
    stats: dict[int, LayerKVStats] = defaultdict(LayerKVStats)
    for row_batch in batched(rows, batch_size):
        batch = collator(row_batch)
        baseline_moved = move_batch_to_device(batch, device=device, model_dtype=baseline_dtype)
        candidate_moved = move_batch_to_device(batch, device=device, model_dtype=candidate_dtype)
        positions = traj_positions_from_batch(baseline_moved)
        _baseline_hidden, baseline_cache = forward_representations(baseline_model, baseline_moved)
        candidate_hidden, candidate_cache = forward_representations(candidate_model, candidate_moved)
        hidden_batches.append(gather_hidden_tokens(candidate_hidden, positions))
        update_kv_delta_stats(stats, baseline_cache, candidate_cache, positions)
        sample_ids.extend(str(value) for value in batch.get("sample_ids", []))
        del baseline_moved, candidate_moved, _baseline_hidden, baseline_cache, candidate_hidden, candidate_cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "event": "kv_compare_batch_done",
                    "baseline": baseline_label,
                    "candidate": candidate_label,
                    "done": len(sample_ids),
                    "total": len(rows),
                }
            ),
            flush=True,
        )
    return np.concatenate(hidden_batches, axis=0), summarize_kv_stats(stats), sample_ids


def load_teacher_action_xyz_targets(rows: list[dict[str, Any]], tf_mod: Any) -> tuple[np.ndarray | None, list[str]]:
    targets: list[np.ndarray] = []
    kept_ids: list[str] = []
    for row in rows:
        xyz = tf_mod.load_teacher_action_xyz(row)
        if xyz is None:
            continue
        xyz = np.asarray(xyz, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[0] < 64 or xyz.shape[1] < 3:
            continue
        targets.append(xyz[:64, :3].reshape(-1))
        kept_ids.append(str(row.get("sample_id")))
    if not targets:
        return None, []
    return np.stack(targets, axis=0).astype(np.float32), kept_ids


def ridge_r2(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    val_fraction: float,
    seed: int,
) -> dict[str, Any]:
    if features.shape[0] != targets.shape[0]:
        raise ValueError(f"Feature/target row mismatch: {features.shape} vs {targets.shape}")
    n = int(features.shape[0])
    if n < 4:
        return {"skipped": True, "reason": f"need at least 4 rows for holdout R2, got {n}"}
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    val_count = min(max(int(round(n * float(val_fraction))), 1), n - 2)
    val_idx = order[:val_count]
    train_idx = order[val_count:]
    x_train = features[train_idx].astype(np.float64)
    x_val = features[val_idx].astype(np.float64)
    y_train = targets[train_idx].astype(np.float64)
    y_val = targets[val_idx].astype(np.float64)
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = np.clip(x_train.std(axis=0, keepdims=True), 1e-6, None)
    y_mean = y_train.mean(axis=0, keepdims=True)
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    y_train_centered = y_train - y_mean
    xtx = x_train.T @ x_train
    reg = float(alpha) * np.eye(xtx.shape[0], dtype=np.float64)
    weights = np.linalg.solve(xtx + reg, x_train.T @ y_train_centered)
    pred = x_val @ weights + y_mean
    sse = float(np.sum((y_val - pred) ** 2))
    sst = float(np.sum((y_val - y_val.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - (sse / max(sst, 1e-12))
    return {
        "skipped": False,
        "target": "teacher_cache.text_raw_json_path results[0].pred_xyz first 64 xyz",
        "feature": "mean final hidden over traj_body_128",
        "rows": n,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "ridge_alpha": float(alpha),
        "r2": finite_float(r2),
        "mse": finite_float(np.mean((y_val - pred) ** 2)),
    }


def compute_hidden_action_r2(
    hidden_by_label: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    tf_mod: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    targets, target_ids = load_teacher_action_xyz_targets(rows, tf_mod)
    if targets is None:
        return {"skipped": True, "reason": "No usable teacher action xyz targets found."}
    if len(target_ids) != len(rows):
        return {
            "skipped": True,
            "reason": f"Target availability mismatch: {len(target_ids)} targets for {len(rows)} rows.",
        }
    out: dict[str, Any] = {}
    dump_path = getattr(args, "dump_hidden_action_npz", None)
    dump_arrays: dict[str, Any] = {}
    for label, hidden in hidden_by_label.items():
        features = hidden.mean(axis=1).astype(np.float32)
        out[label] = ridge_r2(
            features,
            targets,
            alpha=float(args.r2_ridge_alpha),
            val_fraction=float(args.r2_val_fraction),
            seed=int(args.seed),
        )
        if dump_path:
            dump_arrays[f"features__{label}"] = features
    if dump_path and dump_arrays:
        dump_arrays["targets"] = targets.astype(np.float32)
        dump_arrays["seed"] = np.asarray(int(args.seed))
        dump_arrays["val_fraction"] = np.asarray(float(args.r2_val_fraction))
        dump_arrays["ridge_alpha"] = np.asarray(float(args.r2_ridge_alpha))
        Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(dump_path, **dump_arrays)
        print(json.dumps({"event": "hidden_action_npz_dumped", "path": str(dump_path),
                          "labels": list(hidden_by_label.keys()),
                          "n_rows": int(targets.shape[0]),
                          "target_dim": int(targets.shape[1]) if targets.ndim > 1 else 1}), flush=True)
    return out


def compact_table(
    labels: list[str],
    baseline_label: str,
    hidden_to_teacher: dict[str, Any],
    hidden_pairwise: dict[str, Any],
    kv_to_baseline: dict[str, Any],
    hidden_action_r2: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in labels:
        pair = hidden_pairwise.get(label, {})
        kv = kv_to_baseline.get(label, {})
        r2_item = None
        if isinstance(hidden_action_r2, dict):
            r2_value = hidden_action_r2.get(label)
            if isinstance(r2_value, dict):
                r2_item = r2_value.get("r2")
        rows.append(
            {
                "label": label,
                "hidden_token_cosine_to_teacher": hidden_to_teacher[label].get("token_cosine_mean"),
                "hidden_gram_corr_to_teacher": hidden_to_teacher[label].get("gram_corr"),
                "hidden_centered_gram_corr_to_teacher": hidden_to_teacher[label].get("centered_gram_corr"),
                "hidden_token_cosine_to_ce": 1.0 if label == baseline_label else pair.get("token_cosine_mean"),
                "hidden_centered_gram_corr_to_ce": 1.0 if label == baseline_label else pair.get("centered_gram_corr"),
                "kv_key_relative_l2_mean_to_ce": kv.get("key_relative_l2_mean_over_layers"),
                "kv_value_relative_l2_mean_to_ce": kv.get("value_relative_l2_mean_over_layers"),
                "kv_key_cosine_mean_to_ce": kv.get("key_cosine_mean_over_layers"),
                "kv_value_cosine_mean_to_ce": kv.get("value_cosine_mean_over_layers"),
                "hidden_action_r2": r2_item,
            }
        )
    return rows


def run_self_test(args: argparse.Namespace) -> None:
    metric_mod = load_metric_module()
    rng = np.random.default_rng(7)
    teacher = rng.normal(size=(6, 5)).astype(np.float32)
    student_a = teacher + (0.01 * rng.normal(size=(6, 5)).astype(np.float32))
    student_b = rng.normal(size=(6, 5)).astype(np.float32)
    _ = hidden_geometry_metrics(metric_mod, student_a, teacher, metric_token_cap=0, seed=1)
    _ = hidden_geometry_metrics(metric_mod, student_a, student_b, metric_token_cap=0, seed=1)
    positions = torch.tensor([[1, 3, 4], [0, 2, 5]], dtype=torch.long)
    key0 = torch.randn(2, 4, 6, 8)
    value0 = torch.randn(2, 4, 6, 8)
    key1 = key0 + 0.1 * torch.randn_like(key0)
    value1 = value0 + 0.1 * torch.randn_like(value0)
    stats: dict[int, LayerKVStats] = defaultdict(LayerKVStats)
    update_kv_delta_stats(stats, [(key0, value0)], [(key1, value1)], positions)
    summary = summarize_kv_stats(stats)
    payload = {
        "schema_version": "hidden_kv_repr_compare_self_test_v1",
        "status": "ok",
        "kv_summary": summary,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test(args)
        return 0

    metric_mod = load_metric_module()
    tf_mod = load_teacher_forced_module()
    specs = parse_checkpoint_specs(args.checkpoint_dir)
    baseline_label = resolve_baseline_label(specs, str(args.ce_baseline_dir))
    labels = [label for label, _path in specs]
    checkpoint_paths = dict(specs)
    rows = load_rows(tf_mod, args.corpus_jsonl, args.split, args.num_samples)

    baseline_bundle = load_model_bundle(args, checkpoint_paths[baseline_label], tf_mod)
    baseline_model = baseline_bundle["model"]
    device = baseline_bundle["device"]
    collator = build_teacher_pair_collator(args, tf_mod, baseline_bundle)
    hidden_by_label: dict[str, np.ndarray] = {}
    teacher_hidden_tokens: np.ndarray | None = None
    sample_ids_by_label: dict[str, list[str]] = {}

    baseline_hidden, teacher_hidden, layer_count, baseline_sample_ids = extract_hidden_for_label(
        label=baseline_label,
        model=baseline_model,
        rows=rows,
        collator=collator,
        device=device,
        batch_size=int(args.batch_size),
    )
    hidden_by_label[baseline_label] = baseline_hidden
    teacher_hidden_tokens = teacher_hidden
    sample_ids_by_label[baseline_label] = baseline_sample_ids

    kv_to_baseline: dict[str, Any] = {
        baseline_label: zero_kv_summary(layer_count),
    }

    for label, checkpoint_dir in specs:
        if label == baseline_label:
            continue
        print(json.dumps({"event": "candidate_load_start", "label": label, "checkpoint": str(checkpoint_dir)}), flush=True)
        candidate_bundle = load_model_bundle(args, checkpoint_dir, tf_mod)
        candidate_model = candidate_bundle["model"]
        candidate_hidden, kv_summary, candidate_sample_ids = compare_candidate_to_baseline(
            baseline_label=baseline_label,
            candidate_label=label,
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            rows=rows,
            collator=collator,
            device=device,
            batch_size=int(args.batch_size),
        )
        hidden_by_label[label] = candidate_hidden
        kv_to_baseline[label] = kv_summary
        sample_ids_by_label[label] = candidate_sample_ids
        del candidate_bundle, candidate_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"event": "candidate_done", "label": label}), flush=True)

    if teacher_hidden_tokens is None:
        raise RuntimeError("Internal error: teacher hidden was not collected.")
    teacher_flat = teacher_hidden_tokens.reshape(-1, teacher_hidden_tokens.shape[-1])
    hidden_to_teacher: dict[str, Any] = {}
    hidden_pairwise: dict[str, Any] = {}
    baseline_flat = hidden_by_label[baseline_label].reshape(-1, hidden_by_label[baseline_label].shape[-1])
    for label in labels:
        student_hidden = hidden_by_label[label]
        student_flat = student_hidden.reshape(-1, student_hidden.shape[-1])
        hidden_to_teacher[label] = hidden_geometry_metrics(
            metric_mod,
            student_flat,
            teacher_flat,
            metric_token_cap=int(args.metric_token_cap),
            seed=int(args.seed),
        )
        if label == baseline_label:
            hidden_pairwise[label] = {
                "comparison": f"{baseline_label}_self",
                "token_cosine_mean": 1.0,
                "gram_corr": 1.0,
                "centered_gram_corr": 1.0,
                "student_shape": list(student_flat.shape),
                "baseline_shape": list(baseline_flat.shape),
            }
        else:
            hidden_pairwise[label] = {
                "comparison": f"{label}_vs_{baseline_label}",
                **hidden_geometry_metrics(
                    metric_mod,
                    student_flat,
                    baseline_flat,
                    metric_token_cap=int(args.metric_token_cap),
                    seed=int(args.seed),
                ),
            }

    hidden_action_r2: dict[str, Any] | None = None
    if bool(args.compute_hidden_action_r2):
        hidden_action_r2 = compute_hidden_action_r2(hidden_by_label, rows, tf_mod, args)

    train_config = baseline_bundle["train_config"]
    data_view = train_config.get("data_view") or {}
    trainer_config = train_config.get("trainer_config") or {}
    report = {
        "schema_version": "hidden_kv_repr_compare_v1",
        "goal": (
            "Compare final pre-head hidden states and per-layer KV cache tensors "
            "at teacher-forced traj_body_128 positions, independent of the LM head."
        ),
        "corpus_jsonl": str(args.corpus_jsonl),
        "split": args.split,
        "num_samples_requested": int(args.num_samples),
        "num_samples_used": len(rows),
        "sample_ids": [str(row.get("sample_id")) for row in rows],
        "checkpoint_labels": labels,
        "ce_baseline_label": baseline_label,
        "checkpoints": {label: str(path) for label, path in specs},
        "input_contract": {
            "source": "scripts/82 loader + DistillationCollator teacher_pair_target=True",
            "prompt_mode": str(data_view.get("prompt_mode") or "joint"),
            "target_mode": str(data_view.get("target_mode") or "joint"),
            "image_prompt_style": args.image_prompt_style,
            "prompt_text_style": args.prompt_text_style,
            "fuse_history_tokens": bool(args.fuse_history_tokens),
            "max_length": int(trainer_config.get("max_length", 4096)),
            "teacher_traj_hidden_source": "hidden",
            "traj_token_count": TRAJ_TOKEN_COUNT,
        },
        "teacher_anchor": {
            "field": "teacher_traj_target.hidden_path",
            "hidden_position_type": "traj_body_128",
            "expected_shape_per_sample": [TRAJ_TOKEN_COUNT, 4096],
            "collected_shape": list(teacher_hidden_tokens.shape),
        },
        "ae_boundary_note": {
            "source": "scripts/17_profile_action_pre_state.py",
            "action_pre_boundary": "traj_future_start_plus_one_token_for_kv",
            "this_probe_positions": "teacher-forced traj_body_128 token positions from the collator mask",
        },
        "metric_notes": {
            "hidden_metrics_source": "scripts/21_probe_hidden_latent.py token_cosine_mean, gram_corr, centered_gram_corr",
            "student_teacher_token_cosine": "Only computed when hidden dimensions match; Gram metrics are dimension-agnostic.",
            "metric_token_cap": int(args.metric_token_cap),
            "kv_delta": "Per-token relative L2 and cosine over flattened per-layer K/V at traj_body_128 positions, averaged over samples and positions.",
        },
        "hidden_to_teacher": hidden_to_teacher,
        "hidden_delta_to_ce": hidden_pairwise,
        "kv_delta_to_ce": kv_to_baseline,
        "hidden_action_r2": hidden_action_r2 if hidden_action_r2 is not None else {"skipped": True, "reason": "Flag --compute-hidden-action-r2 not set."},
        "compact_table": compact_table(labels, baseline_label, hidden_to_teacher, hidden_pairwise, kv_to_baseline, hidden_action_r2),
        "sample_ids_by_label": sample_ids_by_label,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", "output_json": str(args.output_json), "num_samples": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
