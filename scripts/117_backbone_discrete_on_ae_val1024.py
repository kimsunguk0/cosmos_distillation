#!/usr/bin/env python3
"""Evaluate Step-B backbone discrete-token trajectories on the AE val1024 split.

This aligns the historical backbone discrete-token evaluator with the Action
Expert's 1024-sample official-protocol split. Heavy repo/model imports are kept
inside ``main`` so ``--help`` and import smoke tests do not touch CUDA.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AE84_PATH = PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py"
AE116_PATH = PROJECT_ROOT / "scripts" / "116_eval_official_protocol_e2e.py"
DECODE25_PATH = PROJECT_ROOT / "scripts" / "25_decode_checkpoint_overlays.py"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "backbone_discrete_on_ae_val1024_20260731"
DEFAULT_CORPUS_JSONL = PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_full444k.jsonl"
DEFAULT_SPLIT_CACHE_JSON = PROJECT_ROOT / "outputs" / "action_expert" / "split_cache_444k_10k_seed42.json"
DEFAULT_AE_ROWS_JSONL = (
    PROJECT_ROOT
    / "outputs"
    / "action_expert"
    / "ae_formatfix_444k_studentfree_20260727"
    / "official_protocol_step30000"
    / "rows.jsonl"
)

BACKBONE_CHECKPOINTS = {
    "formatfix": (
        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / "stepb_ceonly_444k_formatfix_20260726"
        / "formatfix_e0"
        / "best_decode"
    ),
    "ceonly_444k": (
        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / "stepb_ceonly_444k"
        / "ceonly_444k_20260718"
        / "fullft_lr3e5_ceonly_444k_e1"
        / "best_decode"
    ),
}

DECODE_MODES = {
    "greedy_n1": {
        "do_sample": False,
        "samples_per_row": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
    },
    "sampled_n6": {
        "do_sample": True,
        "samples_per_row": 6,
        "temperature": 0.6,
        "top_p": 0.98,
        "top_k": None,
    },
}

REFERENCE_NAMES = ("teacher_continuous", "teacher_discrete", "gt")
DIM_SPECS = (("3d_xyz", 3), ("2d_xy", 2))
HORIZON_SPECS = (("h1p6_16wp", 16), ("h3p2_32wp", 32), ("h6p4_64wp", 64))
EXPECTED_WAYPOINTS = 64
EXPECTED_TRAJ_TOKEN_COUNT = 128


@dataclass
class RuntimeModules:
    torch: Any
    ae84: ModuleType
    ae116: ModuleType
    decode25: ModuleType


@dataclass
class SampleContext:
    sample_id: str
    item: dict[str, Any]
    row: dict[str, Any]
    history_xyz: np.ndarray | None
    history_rot: np.ndarray | None
    target_traj_tokens: list[int]
    references: dict[str, np.ndarray | None]
    reference_errors: dict[str, str]
    input_error: str | None = None


def emit(event: dict[str, Any]) -> None:
    print(json.dumps(jsonable(event), ensure_ascii=True), flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def import_module_from_path(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime_modules() -> RuntimeModules:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    emit({"event": "runtime_import_start"})
    import torch  # noqa: PLC0415

    ae84 = import_module_from_path("ae84_train_student_ae28_official_for_117", AE84_PATH)
    ae116 = import_module_from_path("ae116_eval_official_protocol_for_117", AE116_PATH)
    decode25 = import_module_from_path("decode25_checkpoint_overlays_for_117", DECODE25_PATH)
    emit({"event": "runtime_import_done"})
    return RuntimeModules(torch=torch, ae84=ae84, ae116=ae116, decode25=decode25)


def parse_name_list(raw: list[str], *, choices: tuple[str, ...], flag_name: str) -> list[str]:
    values: list[str] = []
    for item in raw:
        values.extend(part.strip() for part in str(item).split(",") if part.strip())
    bad = [item for item in values if item not in choices]
    if bad:
        raise SystemExit(f"{flag_name} has invalid value(s) {bad}; choices={list(choices)}")
    deduped: list[str] = []
    for item in values:
        if item not in deduped:
            deduped.append(item)
    return deduped


def parse_top_k(raw: str) -> int | None:
    text = str(raw).strip().lower()
    if text in {"", "none", "null"}:
        return None
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("--sampled-top-k must be >=0 or none")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    parser.add_argument("--ae-rows-jsonl", type=Path, default=DEFAULT_AE_ROWS_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONE_CHECKPOINTS), help="Backbone labels to run.")
    parser.add_argument("--decode-modes", nargs="+", default=list(DECODE_MODES), help="Decode mode labels to run.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-model", default="")
    parser.add_argument("--student-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--reserve-vram-gib", type=float, default=60.0)
    parser.add_argument("--log-every-samples", type=int, default=16)
    parser.add_argument("--missing-reference-fail-rate", type=float, default=0.01)
    parser.add_argument("--prompt-mode", choices=("joint", "traj_only"), default="joint")
    parser.add_argument("--target-mode", choices=("joint", "traj_only"), default="joint")
    parser.add_argument("--prompt-text-style", choices=("numeric_history_question", "official_alpamayo"), default="official_alpamayo")
    parser.add_argument("--image-prompt-style", choices=("compact", "camera_labeled"), default="camera_labeled")
    parser.add_argument(
        "--fuse-history-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse Alpamayo history delta tokens into <|traj_history|> placeholders.",
    )
    parser.add_argument("--preserve-flex-positions", action="store_true")
    parser.add_argument("--flex-selection-strategy", choices=("first", "uniform"), default="first")
    parser.add_argument("--flex-scene-deepstack", action="store_true")
    parser.add_argument("--disable-qwen-deepstack", action="store_true")
    parser.add_argument("--qat-quantization", choices=("", "int4_awq", "int4_blockwise", "int4_ffn_only", "fp8", "fp8_pcpt", "fp8_vit", "fp8_pcpt_vit"), default="")
    parser.add_argument("--qat-calib-samples", type=int, default=128)
    parser.add_argument("--qat-calib-batch-size", type=int, default=0)
    parser.add_argument("--sampled-temperature", type=float, default=0.6)
    parser.add_argument("--sampled-top-p", type=float, default=0.98)
    parser.add_argument("--sampled-top-k", type=parse_top_k, default=None)
    parser.add_argument(
        "--greedy-max-new-tokens",
        type=int,
        default=None,
        help="Optional max-new-token override for greedy_n1. Default reuses --max-new-tokens.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_existing_file(path: Path, label: str) -> Path:
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def require_existing_dir(path: Path, label: str) -> Path:
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {resolved}")
    return resolved


def load_expected_sample_ids_from_rows(path: Path) -> list[str]:
    expected: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if sample_id in (None, ""):
                raise RuntimeError(f"Missing sample_id in AE rows {path} line {line_index}")
            expected.append(str(sample_id))
    if not expected:
        raise RuntimeError(f"No sample IDs loaded from AE rows: {path}")
    return expected


def make_split_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        corpus_jsonl=resolve_repo_path(args.corpus_jsonl),
        split=str(args.split),
        num_samples=int(args.num_samples),
        val_samples=int(args.val_samples),
        eval_samples=int(args.eval_samples),
        val_fraction=float(args.val_fraction),
        split_seed=args.split_seed,
        split_scan_all=bool(args.split_scan_all),
        split_cache_json=resolve_repo_path(args.split_cache_json),
        seed=int(args.seed),
    )


def select_and_verify_eval_items(
    *,
    args: argparse.Namespace,
    runtime: RuntimeModules,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    emit({"event": "train_val_split_start"})
    train_items, val_items, split_summary = runtime.ae84.select_train_val_items(make_split_args(args))
    del train_items
    eval_items = list(val_items[: int(args.eval_samples)])
    expected_ids = load_expected_sample_ids_from_rows(require_existing_file(args.ae_rows_jsonl, "AE rows.jsonl"))
    id_check = runtime.ae116.verify_val_sample_ids(eval_items, expected_ids)
    if not bool(id_check.get("order_equal")):
        actual_ids = [str(item["sample_id"]) for item in eval_items]
        first_mismatch = next(
            (
                {
                    "index": index,
                    "actual": actual,
                    "expected": expected,
                }
                for index, (actual, expected) in enumerate(zip(actual_ids, expected_ids), start=1)
                if actual != expected
            ),
            None,
        )
        raise RuntimeError(
            "Val sample_id order mismatch against AE official rows: "
            f"first_mismatch={first_mismatch}"
        )
    emit(id_check)
    return eval_items, split_summary, id_check


def read_train_config_base_model(checkpoint_dir: Path, fallback: str) -> str:
    train_config_path = checkpoint_dir / "train_config.json"
    if train_config_path.exists():
        train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
        value = (train_config.get("args") or {}).get("student_model")
        if value not in (None, ""):
            return str(value)
    return str(fallback)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoder_signature(decoder: Any, config_path: Path) -> dict[str, Any]:
    return {
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "num_bins": int(getattr(decoder, "num_bins")),
        "n_waypoints": int(getattr(decoder, "n_waypoints")),
        "dims_min": [float(x) for x in decoder.dims_min.detach().cpu().tolist()],
        "dims_max": [float(x) for x in decoder.dims_max.detach().cpu().tolist()],
    }


def load_decoder(
    *,
    runtime: RuntimeModules,
    first_checkpoint: Path,
    student_model: str,
) -> tuple[Any, Path, str, dict[str, Any]]:
    fallback_model = student_model or runtime.ae84.resolve_student_model_path()
    base_model = read_train_config_base_model(first_checkpoint, str(fallback_model))
    config_path = runtime.decode25.resolve_traj_tokenizer_config_path(base_model)
    if config_path is None:
        raise RuntimeError(f"Could not resolve trajectory tokenizer config for base_model={base_model}")
    decoder = runtime.decode25.TrajectoryTokenDecoder(config_path=config_path)
    if int(getattr(decoder, "n_waypoints")) != EXPECTED_WAYPOINTS:
        raise RuntimeError(
            f"Trajectory decoder waypoint count {int(getattr(decoder, 'n_waypoints'))} != {EXPECTED_WAYPOINTS}"
        )
    return decoder, Path(config_path), str(base_model), decoder_signature(decoder, Path(config_path))


def coerce_xyz(value: Any, *, expected_min_wp: int = EXPECTED_WAYPOINTS) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, arr.shape[-1] if arr.shape[-1] in (2, 3) else 1)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2 or arr.shape[-1] < 2:
        raise ValueError(f"Expected trajectory xyz as [T,D>=2], got shape={arr.shape}")
    if arr.shape[0] < expected_min_wp:
        raise ValueError(f"Expected at least {expected_min_wp} waypoints, got {arr.shape[0]}")
    if arr.shape[-1] < 3:
        padded = np.zeros((arr.shape[0], 3), dtype=np.float32)
        padded[:, : arr.shape[-1]] = arr
        arr = padded
    return arr[:expected_min_wp, :3].astype(np.float32)


def precompute_sample_contexts(
    *,
    eval_items: list[dict[str, Any]],
    runtime: RuntimeModules,
    decoder: Any,
) -> tuple[list[SampleContext], dict[str, Any]]:
    emit({"event": "reference_precompute_start", "sample_count": len(eval_items)})
    contexts: list[SampleContext] = []
    missing_by_ref: dict[str, list[dict[str, str]]] = {name: [] for name in REFERENCE_NAMES}
    input_failures: list[dict[str, str]] = []

    for item in eval_items:
        sample_id = str(item["sample_id"])
        row = item["row"]
        history_xyz: np.ndarray | None = None
        history_rot: np.ndarray | None = None
        target_tokens: list[int] = []
        references: dict[str, np.ndarray | None] = {name: None for name in REFERENCE_NAMES}
        errors: dict[str, str] = {}
        input_error: str | None = None

        try:
            history_xyz = runtime.ae84.load_ego_history_xyz(row, PROJECT_ROOT).astype(np.float32)
            history_rot = runtime.ae84.normalize_history_rot(
                runtime.ae84.load_ego_history_rot(row, PROJECT_ROOT)
            )
        except Exception as exc:  # noqa: BLE001
            input_error = f"{type(exc).__name__}: {exc}"
            input_failures.append({"sample_id": sample_id, "error": input_error})

        try:
            teacher_xyz, _teacher_rot = runtime.ae84.raw_teacher_pred(Path(item["raw_json"]))
            references["teacher_continuous"] = coerce_xyz(teacher_xyz)
        except Exception as exc:  # noqa: BLE001
            errors["teacher_continuous"] = f"{type(exc).__name__}: {exc}"
            missing_by_ref["teacher_continuous"].append({"sample_id": sample_id, "error": errors["teacher_continuous"]})

        try:
            gt_xyz = runtime.ae84.load_ego_future_xyz(row, PROJECT_ROOT)
            references["gt"] = coerce_xyz(gt_xyz)
        except Exception as exc:  # noqa: BLE001
            errors["gt"] = f"{type(exc).__name__}: {exc}"
            missing_by_ref["gt"].append({"sample_id": sample_id, "error": errors["gt"]})

        try:
            target_tokens = runtime.decode25.load_traj_future_token_ids(row.get("hard_target") or {}, PROJECT_ROOT)
            if len(target_tokens) != EXPECTED_TRAJ_TOKEN_COUNT:
                raise ValueError(
                    f"teacher discrete token count {len(target_tokens)} != {EXPECTED_TRAJ_TOKEN_COUNT}"
                )
            if history_xyz is None or history_rot is None:
                raise ValueError("history unavailable, cannot decode teacher discrete tokens")
            teacher_discrete = decoder.decode(history_xyz, history_rot, target_tokens)
            if teacher_discrete is None:
                raise ValueError("TrajectoryTokenDecoder.decode returned None")
            references["teacher_discrete"] = coerce_xyz(teacher_discrete)
        except Exception as exc:  # noqa: BLE001
            errors["teacher_discrete"] = f"{type(exc).__name__}: {exc}"
            missing_by_ref["teacher_discrete"].append({"sample_id": sample_id, "error": errors["teacher_discrete"]})

        contexts.append(
            SampleContext(
                sample_id=sample_id,
                item=item,
                row=row,
                history_xyz=history_xyz,
                history_rot=history_rot,
                target_traj_tokens=[int(v) for v in target_tokens],
                references=references,
                reference_errors=errors,
                input_error=input_error,
            )
        )

    availability = {
        name: {
            "available_count": int(len(eval_items) - len(missing)),
            "missing_count": int(len(missing)),
            "missing_rate": float(len(missing) / max(len(eval_items), 1)),
            "missing_samples_head": missing[:16],
        }
        for name, missing in missing_by_ref.items()
    }
    availability["sample_input"] = {
        "available_count": int(len(eval_items) - len(input_failures)),
        "missing_count": int(len(input_failures)),
        "missing_rate": float(len(input_failures) / max(len(eval_items), 1)),
        "missing_samples_head": input_failures[:16],
    }
    emit({"event": "reference_precompute_done", "availability": availability})
    return contexts, availability


def fail_if_reference_missing_too_high(availability: dict[str, Any], threshold: float) -> None:
    offenders = []
    for name in REFERENCE_NAMES:
        info = availability.get(name) or {}
        if float(info.get("missing_rate") or 0.0) > float(threshold):
            offenders.append(
                {
                    "reference": name,
                    "missing_count": int(info.get("missing_count") or 0),
                    "missing_rate": float(info.get("missing_rate") or 0.0),
                    "missing_samples_head": info.get("missing_samples_head") or [],
                }
            )
    if offenders:
        raise RuntimeError(
            "Reference availability failure: more than allowed missing reference rows; "
            f"threshold={threshold} offenders={json.dumps(offenders, ensure_ascii=True)}"
        )


def make_model_load_args(args: argparse.Namespace, checkpoint_dir: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint_dir=checkpoint_dir,
        student_model=args.student_model,
        device=str(args.device),
        output_dir=output_dir,
        disable_qwen_deepstack=bool(args.disable_qwen_deepstack),
        preserve_flex_positions=bool(args.preserve_flex_positions),
        flex_selection_strategy=str(args.flex_selection_strategy),
        flex_dummy_image_slots=False,
        flex_residual_image_slots=False,
        flex_residual_scale=1.0,
        flex_passthrough_image_slots=False,
        flex_scene_deepstack=bool(args.flex_scene_deepstack),
        qat_quantization=str(args.qat_quantization),
        qat_calib_samples=int(args.qat_calib_samples),
        qat_calib_batch_size=int(args.qat_calib_batch_size),
        batch_size=int(args.batch_size),
    )


def tokenizer_signature(tokenizer: Any) -> dict[str, Any]:
    names = ("<|cot_start|>", "<|cot_end|>", "<|traj_future_start|>", "<|traj_future_end|>", "<i0>", "<i3999>")
    ids: dict[str, int | None] = {}
    for name in names:
        try:
            value = tokenizer.convert_tokens_to_ids(name)
            ids[name] = int(value) if isinstance(value, int) and value >= 0 else None
        except Exception:  # noqa: BLE001
            ids[name] = None
    return {
        "vocab_size": int(len(tokenizer)),
        "pad_token_id": int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None,
        "eos_token_id": int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
        "special_token_ids": ids,
    }


def validate_checkpoint_tokenizer(
    *,
    label: str,
    tokenizer: Any,
    expected_signature: dict[str, Any] | None,
) -> dict[str, Any]:
    signature = tokenizer_signature(tokenizer)
    required_missing = [
        name
        for name in ("<|cot_end|>", "<|traj_future_start|>", "<|traj_future_end|>", "<i0>", "<i3999>")
        if signature["special_token_ids"].get(name) is None
    ]
    if required_missing:
        raise RuntimeError(f"Tokenizer for {label} is missing required tokens: {required_missing}")
    if expected_signature is not None and signature["special_token_ids"] != expected_signature["special_token_ids"]:
        raise RuntimeError(
            "Backbone tokenizer special-token IDs differ; cannot align discrete-token decode silently: "
            f"label={label} expected={expected_signature['special_token_ids']} actual={signature['special_token_ids']}"
        )
    return signature


def validate_checkpoint_decoder_config(
    *,
    runtime: RuntimeModules,
    backbone: str,
    base_model: str,
    expected_decoder_signature: dict[str, Any],
) -> dict[str, Any]:
    config_path = runtime.decode25.resolve_traj_tokenizer_config_path(base_model)
    if config_path is None:
        raise RuntimeError(f"Could not resolve trajectory tokenizer config for {backbone} base_model={base_model}")
    actual = {
        "config_path": str(config_path),
        "config_sha256": file_sha256(Path(config_path)),
    }
    expected_hash = str(expected_decoder_signature["config_sha256"])
    if actual["config_sha256"] != expected_hash:
        raise RuntimeError(
            "Backbone trajectory decoder configs differ; cannot align discrete-token geometry silently: "
            f"backbone={backbone} expected_sha256={expected_hash} actual={actual}"
        )
    return actual


def prepare_generation_batch(
    *,
    args: argparse.Namespace,
    runtime: RuntimeModules,
    model: Any,
    tokenizer: Any,
    processor: Any,
    device: Any,
    sample_contexts: list[SampleContext],
) -> dict[str, Any]:
    decode25 = runtime.decode25
    prepared: list[dict[str, Any]] = []
    texts: list[str] = []
    image_batches: list[list[Any]] = []
    target_token_count: int | None = None

    for context in sample_contexts:
        sample_id = context.sample_id
        if context.history_xyz is None or context.history_rot is None:
            raise RuntimeError(f"History unavailable for sample_id={sample_id}: {context.input_error}")
        target_tokens = list(context.target_traj_tokens)
        if target_token_count is None:
            target_token_count = len(target_tokens)
        elif len(target_tokens) != target_token_count:
            raise ValueError(
                f"Mixed trajectory target lengths in one batch: {target_token_count} and {len(target_tokens)}"
            )
        if len(target_tokens) != EXPECTED_TRAJ_TOKEN_COUNT:
            raise ValueError(f"sample_id={sample_id} target token count {len(target_tokens)} != {EXPECTED_TRAJ_TOKEN_COUNT}")

        prompt_text = (
            decode25.build_traj_only_prompt(context.row, PROJECT_ROOT, ego_history_xyz=context.history_xyz)
            if str(args.prompt_mode) == "traj_only"
            else decode25.build_user_prompt(
                context.row,
                PROJECT_ROOT,
                ego_history_xyz=context.history_xyz,
                prompt_text_style=str(args.prompt_text_style),
            )
        )
        assistant_prefix = "<|traj_future_start|>" if str(args.target_mode) == "traj_only" else "<|cot_start|>"
        images = decode25.load_sample_images(context.row, PROJECT_ROOT)
        camera_indices = decode25.resolve_camera_indices(context.row, PROJECT_ROOT, image_count=len(images))
        frames_per_camera = max(len(images) // max(len(camera_indices), 1), 1)
        messages = decode25.build_messages(
            prompt_text,
            len(images),
            assistant_prefix=assistant_prefix,
            image_prompt_style=str(args.image_prompt_style),
            camera_indices=camera_indices,
            num_frames_per_camera=frames_per_camera,
        )
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        prepared.append(
            {
                "context": context,
                "sample_id": sample_id,
                "history_xyz": context.history_xyz,
                "history_rot": context.history_rot,
                "target_tokens": target_tokens,
                "camera_indices": camera_indices,
                "frames_per_camera": frames_per_camera,
                "generation_target_mode": str(args.target_mode),
            }
        )
        texts.append(text)
        image_batches.append(images)

    batch = processor(
        text=texts,
        images=image_batches,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(args.max_length),
    )
    if bool(args.fuse_history_tokens):
        batch["input_ids"] = decode25.fuse_history_tokens_in_input_ids(
            batch["input_ids"],
            tokenizer,
            [item["history_xyz"] for item in prepared],
        )

    flex_enabled = bool(hasattr(model, "flex_enabled") and model.flex_enabled())
    if flex_enabled:
        max_cameras = max(len(item["camera_indices"]) for item in prepared)
        max_frames = max(int(item["frames_per_camera"]) for item in prepared)
        camera_indices_tensor = runtime.torch.zeros((len(prepared), max_cameras), dtype=runtime.torch.long)
        relative_timestamps_tensor = runtime.torch.zeros((len(prepared), max_cameras, max_frames), dtype=runtime.torch.float32)
        camera_counts = runtime.torch.zeros((len(prepared),), dtype=runtime.torch.long)
        frames_per_camera_tensor = runtime.torch.zeros((len(prepared),), dtype=runtime.torch.long)
        for row_index, item in enumerate(prepared):
            row_camera_indices = [int(value) for value in item["camera_indices"]]
            row_frames = int(item["frames_per_camera"])
            row_relative_times = decode25.resolve_image_relative_timestamps(
                item["context"].row,
                PROJECT_ROOT,
                camera_count=len(row_camera_indices),
                frames_per_camera=row_frames,
            )
            camera_count = len(row_camera_indices)
            camera_indices_tensor[row_index, :camera_count] = runtime.torch.tensor(row_camera_indices, dtype=runtime.torch.long)
            camera_counts[row_index] = camera_count
            frames_per_camera_tensor[row_index] = row_frames
            for camera_offset, row_times in enumerate(row_relative_times[:camera_count]):
                count = min(len(row_times), max_frames)
                if count > 0:
                    relative_timestamps_tensor[row_index, camera_offset, :count] = runtime.torch.tensor(
                        row_times[:count],
                        dtype=runtime.torch.float32,
                    )
        batch["camera_indices"] = camera_indices_tensor
        batch["relative_timestamps"] = relative_timestamps_tensor
        batch["camera_counts"] = camera_counts
        batch["frames_per_camera"] = frames_per_camera_tensor
        flex_cfg = getattr(model, "flex_scene_config")
        if bool(args.preserve_flex_positions):
            batch = decode25.attach_qwen_mrope_position_ids(batch, model)
        batch = decode25.compress_batch_for_flex(
            batch,
            image_token_id=int(getattr(model, "image_token_id")),
            tokens_per_image=int(getattr(flex_cfg, "tokens_per_image")),
            pad_token_id=int(getattr(tokenizer, "pad_token_id", 0) or 0),
            preserve_original_position_ids=bool(args.preserve_flex_positions),
            selection_strategy=str(args.flex_selection_strategy),
        )
        if str(args.flex_selection_strategy) != "first":
            batch["flex_selection_strategy"] = str(args.flex_selection_strategy)
        if bool(args.flex_scene_deepstack):
            batch["flex_scene_deepstack"] = True

    model_dtype = decode25._infer_visual_float_dtype(model)
    batch = {
        key: (
            value.to(device=device, dtype=model_dtype)
            if isinstance(value, runtime.torch.Tensor) and runtime.torch.is_floating_point(value)
            else value.to(device)
            if isinstance(value, runtime.torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }

    return {
        "prepared": prepared,
        "batch": batch,
        "target_token_count": int(target_token_count or 0),
        "flex_enabled": flex_enabled,
    }


def build_decode_contract(
    *,
    runtime: RuntimeModules,
    tokenizer: Any,
    batch: dict[str, Any],
    target_token_count: int,
    target_mode: str,
) -> tuple[Any, Any]:
    decode25 = runtime.decode25
    prompt_lengths = [int(batch["input_ids"].shape[1])]
    if target_mode == "traj_only":
        contract = decode25.TrajOnlyDecodingContract.from_tokenizer(
            tokenizer,
            prompt_lengths=prompt_lengths,
            traj_token_count=int(target_token_count),
        )
        return (
            decode25.LogitsProcessorList([decode25.TrajOnlyLogitsProcessor(contract)]),
            decode25.StoppingCriteriaList([decode25.StopOnTrajOnlyEndCriteria(contract)]),
        )
    contract = decode25.TrajDecodingContract.from_tokenizer(
        tokenizer,
        prompt_lengths=prompt_lengths,
        traj_token_count=int(target_token_count),
    )
    return (
        decode25.LogitsProcessorList([decode25.TrajSpanLogitsProcessor(contract)]),
        decode25.StoppingCriteriaList([decode25.StopOnTrajEndCriteria(contract)]),
    )


def apply_seed(runtime: RuntimeModules, seed: int, device: Any) -> None:
    runtime.torch.manual_seed(int(seed))
    if device.type == "cuda" and runtime.torch.cuda.is_available():
        runtime.torch.cuda.manual_seed_all(int(seed))


def generate_for_batch(
    *,
    args: argparse.Namespace,
    runtime: RuntimeModules,
    model: Any,
    tokenizer: Any,
    device: Any,
    prepared_batch: dict[str, Any],
    decode_mode: str,
    seed: int,
) -> Any:
    settings = DECODE_MODES[decode_mode].copy()
    if decode_mode == "sampled_n6":
        settings["temperature"] = float(args.sampled_temperature)
        settings["top_p"] = float(args.sampled_top_p)
        settings["top_k"] = args.sampled_top_k
    max_new_tokens = int(args.greedy_max_new_tokens or args.max_new_tokens) if decode_mode == "greedy_n1" else int(args.max_new_tokens)
    samples_per_row = int(settings["samples_per_row"])
    logits_processor, stopping_criteria = build_decode_contract(
        runtime=runtime,
        tokenizer=tokenizer,
        batch=prepared_batch["batch"],
        target_token_count=int(prepared_batch["target_token_count"]),
        target_mode=str(args.target_mode),
    )
    apply_seed(runtime, seed, device)
    batch = prepared_batch["batch"]
    with runtime.torch.inference_mode():
        if bool(prepared_batch["flex_enabled"]):
            generated = runtime.decode25._manual_flex_generate(
                model,
                batch,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                do_sample=bool(settings["do_sample"]),
                num_return_sequences=samples_per_row,
                temperature=float(settings["temperature"]),
                top_p=float(settings["top_p"]),
            )
        else:
            generation_config = copy.deepcopy(model.backbone.generation_config)
            generation_config.do_sample = bool(settings["do_sample"])
            generation_config.temperature = float(settings["temperature"])
            generation_config.top_p = float(settings["top_p"])
            generation_config.top_k = settings["top_k"]
            generation_config.num_return_sequences = samples_per_row
            generation_config.num_beams = 1
            generation_config.max_new_tokens = max_new_tokens
            generation_config.output_scores = False
            generation_config.output_logits = False
            generation_config.output_hidden_states = False
            generation_config.return_dict_in_generate = False
            generation_config.pad_token_id = tokenizer.pad_token_id
            generation_config.use_cache = True
            generated = model.backbone.generate(
                **batch,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                use_cache=True,
            )
    if hasattr(generated, "sequences"):
        return generated.sequences
    return generated


def token_repetition_stats(tokens: list[int]) -> dict[str, Any]:
    counter = Counter(int(v) for v in tokens)
    return {
        "unique_count": int(len(counter)),
        "max_same_token_run": int(max_same_token_run(tokens)),
        "top_tokens": [
            {"token": int(token), "count": int(count), "mass": float(count / max(len(tokens), 1))}
            for token, count in counter.most_common(10)
        ],
    }


def max_same_token_run(tokens: list[int]) -> int:
    best = 0
    current = 0
    previous: int | None = None
    for token in tokens:
        token = int(token)
        if previous is None or token != previous:
            current = 1
            previous = token
        else:
            current += 1
        best = max(best, current)
    return best


def extract_candidates_for_prepared(
    *,
    runtime: RuntimeModules,
    tokenizer: Any,
    decoder: Any,
    generated: Any,
    prepared_batch: dict[str, Any],
    sample_batch_start_index: int,
    decode_mode: str,
) -> list[dict[str, Any]]:
    samples_per_row = int(DECODE_MODES[decode_mode]["samples_per_row"])
    rows: list[dict[str, Any]] = []
    batch = prepared_batch["batch"]
    for row_index, item in enumerate(prepared_batch["prepared"]):
        context: SampleContext = item["context"]
        row_start = row_index * samples_per_row
        row_end = min(row_start + samples_per_row, int(generated.shape[0]))
        candidate_records: list[dict[str, Any]] = []
        decoded_paths: list[np.ndarray | None] = []
        for candidate_index, generated_row in enumerate(range(row_start, row_end), start=1):
            generated_text = runtime.decode25._extract_generated_text(
                tokenizer,
                batch["input_ids"],
                generated,
                row_index=generated_row,
            )
            generated_tokens = [int(v) for v in runtime.decode25._extract_generated_traj_tokens(generated_text)]
            decoded_xyz = (
                decoder.decode(context.history_xyz, context.history_rot, generated_tokens)
                if context.history_xyz is not None
                and context.history_rot is not None
                and len(generated_tokens) == EXPECTED_TRAJ_TOKEN_COUNT
                else None
            )
            decoded_paths.append(coerce_xyz(decoded_xyz) if decoded_xyz is not None else None)
            rep = token_repetition_stats(generated_tokens)
            candidate_records.append(
                {
                    "candidate_index": int(candidate_index),
                    "generated_traj_token_count": int(len(generated_tokens)),
                    "generated_traj_tokens": generated_tokens,
                    "generated_unique_token_count": int(rep["unique_count"]),
                    "generated_max_same_token_run": int(rep["max_same_token_run"]),
                    "generated_invalid_future_token_count_i3000_plus": int(
                        sum(1 for token in generated_tokens if int(token) < 0 or int(token) >= 3000)
                    ),
                    "decoded_xyz_available": decoded_xyz is not None,
                    "generated_text_preview": str(generated_text)[:240],
                    "generated_top_tokens": rep["top_tokens"],
                }
            )
        rows.append(
            {
                "sample_index": int(sample_batch_start_index + row_index + 1),
                "sample_id": context.sample_id,
                "context": context,
                "candidate_records": candidate_records,
                "decoded_paths": decoded_paths,
            }
        )
    return rows


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def ade_fde_for_dims(
    *,
    pred_xyz: np.ndarray | None,
    reference_xyz: np.ndarray | None,
    dims: int,
    horizon: int,
) -> tuple[float | None, float | None]:
    if pred_xyz is None or reference_xyz is None:
        return None, None
    pred = np.asarray(pred_xyz, dtype=np.float32)
    ref = np.asarray(reference_xyz, dtype=np.float32)
    if pred.ndim != 2 or ref.ndim != 2:
        return None, None
    if pred.shape[-1] < dims or ref.shape[-1] < dims:
        return None, None
    n = min(int(horizon), int(pred.shape[0]), int(ref.shape[0]))
    if n <= 0:
        return None, None
    dist = np.linalg.norm(pred[:n, :dims] - ref[:n, :dims], axis=-1)
    return finite_or_none(float(dist.mean())), finite_or_none(float(dist[-1]))


def metric_prefix(reference_name: str, dim_label: str, horizon_name: str) -> str:
    return f"{reference_name}_{dim_label}_{horizon_name}"


def compute_row_reference_metrics(
    *,
    decoded_paths: list[np.ndarray | None],
    reference_xyz: np.ndarray | None,
    reference_name: str,
    dim_label: str,
    dims: int,
    decode_mode: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for horizon_name, horizon in HORIZON_SPECS:
        prefix = metric_prefix(reference_name, dim_label, horizon_name)
        ade_all: list[float | None] = []
        fde_all: list[float | None] = []
        for path in decoded_paths:
            ade, fde = ade_fde_for_dims(
                pred_xyz=path,
                reference_xyz=reference_xyz,
                dims=dims,
                horizon=int(horizon),
            )
            ade_all.append(ade)
            fde_all.append(fde)
        out[f"{prefix}_ADE_all_candidates_m"] = ade_all
        out[f"{prefix}_FDE_all_candidates_m"] = fde_all
        out[f"{prefix}_ADE_single_m"] = ade_all[0] if ade_all else None
        out[f"{prefix}_FDE_single_m"] = fde_all[0] if fde_all else None
        if decode_mode == "sampled_n6":
            valid = [(idx, value) for idx, value in enumerate(ade_all) if value is not None]
            if valid:
                winner_idx, winner_ade = min(valid, key=lambda item: float(item[1]))
                out[f"{prefix}_minADE6_m"] = float(winner_ade)
                out[f"{prefix}_FDE_of_minADE6_winner_m"] = fde_all[winner_idx]
                out[f"{prefix}_minADE6_winner_idx"] = int(winner_idx)
            else:
                out[f"{prefix}_minADE6_m"] = None
                out[f"{prefix}_FDE_of_minADE6_winner_m"] = None
                out[f"{prefix}_minADE6_winner_idx"] = None
    return out


def append_metrics_to_accumulators(
    *,
    accumulators: dict[tuple[str, str, str, str], dict[str, list[float]]],
    backbone: str,
    decode_mode: str,
    row_metrics: dict[str, Any],
) -> None:
    for reference_name in REFERENCE_NAMES:
        for dim_label, _dims in DIM_SPECS:
            key = (backbone, decode_mode, reference_name, dim_label)
            bucket = accumulators.setdefault(key, {})
            for horizon_name, _horizon in HORIZON_SPECS:
                prefix = metric_prefix(reference_name, dim_label, horizon_name)
                for metric_name in (
                    "ADE_single_m",
                    "FDE_single_m",
                    "minADE6_m",
                    "FDE_of_minADE6_winner_m",
                ):
                    explicit_key = f"{prefix}_{metric_name}"
                    value = row_metrics.get(explicit_key)
                    if value is None:
                        continue
                    bucket.setdefault(explicit_key, []).append(float(value))


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
    }


def build_nested_metrics(
    *,
    accumulators: dict[tuple[str, str, str, str], dict[str, list[float]]],
    selected_backbones: list[str],
    selected_decode_modes: list[str],
    total_samples: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for backbone in selected_backbones:
        metrics[backbone] = {}
        for decode_mode in selected_decode_modes:
            metrics[backbone][decode_mode] = {}
            for reference_name in REFERENCE_NAMES:
                metrics[backbone][decode_mode][reference_name] = {}
                for dim_label, _dims in DIM_SPECS:
                    bucket = accumulators.get((backbone, decode_mode, reference_name, dim_label), {})
                    leaf: dict[str, Any] = {"total_samples": int(total_samples)}
                    for metric_key, values in sorted(bucket.items()):
                        summary = summarize_values(values)
                        leaf[f"{metric_key}_eval_count"] = summary["count"]
                        leaf[f"{metric_key}_mean"] = summary["mean"]
                        leaf[f"{metric_key}_p50"] = summary["p50"]
                    metrics[backbone][decode_mode][reference_name][dim_label] = leaf
    return metrics


def row_for_output(
    *,
    backbone: str,
    checkpoint_dir: Path,
    decode_mode: str,
    decode_settings: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> dict[str, Any]:
    context: SampleContext = candidate_payload["context"]
    decoded_paths: list[np.ndarray | None] = candidate_payload["decoded_paths"]
    row: dict[str, Any] = {
        "sample_id": context.sample_id,
        "sample_index": int(candidate_payload["sample_index"]),
        "backbone": backbone,
        "backbone_checkpoint_dir": str(checkpoint_dir),
        "decode_mode": decode_mode,
        "decode_settings": decode_settings,
        "generated_candidate_count": int(len(candidate_payload["candidate_records"])),
        "generated_candidate_records": candidate_payload["candidate_records"],
        "reference_available_teacher_continuous": context.references["teacher_continuous"] is not None,
        "reference_available_teacher_discrete": context.references["teacher_discrete"] is not None,
        "reference_available_gt": context.references["gt"] is not None,
        "reference_errors": dict(context.reference_errors),
        "target_traj_token_count": int(len(context.target_traj_tokens)),
        "target_traj_tokens": context.target_traj_tokens,
    }
    for reference_name in REFERENCE_NAMES:
        reference_xyz = context.references.get(reference_name)
        for dim_label, dims in DIM_SPECS:
            row.update(
                compute_row_reference_metrics(
                    decoded_paths=decoded_paths,
                    reference_xyz=reference_xyz,
                    reference_name=reference_name,
                    dim_label=dim_label,
                    dims=int(dims),
                    decode_mode=decode_mode,
                )
            )
    return row


def decode_settings_for_mode(args: argparse.Namespace, decode_mode: str) -> dict[str, Any]:
    settings = dict(DECODE_MODES[decode_mode])
    if decode_mode == "sampled_n6":
        settings["temperature"] = float(args.sampled_temperature)
        settings["top_p"] = float(args.sampled_top_p)
        settings["top_k"] = args.sampled_top_k
    settings["max_new_tokens"] = int(args.greedy_max_new_tokens or args.max_new_tokens) if decode_mode == "greedy_n1" else int(args.max_new_tokens)
    settings["num_beams"] = 1
    return settings


def combo_seed(base_seed: int, sample_zero_index: int) -> int:
    return int(base_seed) + int(sample_zero_index) * 1009


def run_combo(
    *,
    args: argparse.Namespace,
    runtime: RuntimeModules,
    backbone: str,
    checkpoint_dir: Path,
    decode_mode: str,
    model: Any,
    tokenizer: Any,
    processor: Any,
    device: Any,
    decoder: Any,
    sample_contexts: list[SampleContext],
    rows_handle: Any,
    accumulators: dict[tuple[str, str, str, str], dict[str, list[float]]],
) -> dict[str, Any]:
    combo_start = time.time()
    settings = decode_settings_for_mode(args, decode_mode)
    emit(
        {
            "event": "combo_start",
            "backbone": backbone,
            "decode_mode": decode_mode,
            "checkpoint_dir": str(checkpoint_dir),
            "settings": settings,
        }
    )
    if int(args.batch_size) != 1:
        emit(
            {
                "event": "protocol_warning",
                "backbone": backbone,
                "decode_mode": decode_mode,
                "reason": "batch_size != 1 uses one RNG seed per batch rather than the AE per-sample seed formula",
                "batch_size": int(args.batch_size),
            }
        )

    processed = 0
    failed_samples = 0
    for batch_start in range(0, len(sample_contexts), int(args.batch_size)):
        batch_contexts = sample_contexts[batch_start : batch_start + int(args.batch_size)]
        seed = combo_seed(int(args.seed), batch_start)
        try:
            prepared_batch = prepare_generation_batch(
                args=args,
                runtime=runtime,
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                device=device,
                sample_contexts=batch_contexts,
            )
            generated = generate_for_batch(
                args=args,
                runtime=runtime,
                model=model,
                tokenizer=tokenizer,
                device=device,
                prepared_batch=prepared_batch,
                decode_mode=decode_mode,
                seed=seed,
            )
            candidate_payloads = extract_candidates_for_prepared(
                runtime=runtime,
                tokenizer=tokenizer,
                decoder=decoder,
                generated=generated,
                prepared_batch=prepared_batch,
                sample_batch_start_index=batch_start,
                decode_mode=decode_mode,
            )
        except Exception as exc:  # noqa: BLE001
            failed_samples += len(batch_contexts)
            emit(
                {
                    "event": "batch_failed",
                    "backbone": backbone,
                    "decode_mode": decode_mode,
                    "batch_start_index": int(batch_start + 1),
                    "batch_size": int(len(batch_contexts)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise

        for payload in candidate_payloads:
            row = row_for_output(
                backbone=backbone,
                checkpoint_dir=checkpoint_dir,
                decode_mode=decode_mode,
                decode_settings=settings,
                candidate_payload=payload,
            )
            intended_seed = combo_seed(int(args.seed), int(row["sample_index"] - 1))
            row["seed_policy"] = {
                "base_seed": int(args.seed),
                "sample_zero_index": int(row["sample_index"] - 1),
                "generation_seed_used": int(seed),
                "intended_per_sample_generation_seed": int(intended_seed),
                "formula": "generation_seed=seed+sample_zero_index*1009 when batch_size=1",
                "batch_size_seed_note": (
                    "exact_per_sample_seed"
                    if int(args.batch_size) == 1
                    else "batch used first sample's seed; set --batch-size 1 for exact per-sample seed parity"
                ),
            }
            append_metrics_to_accumulators(
                accumulators=accumulators,
                backbone=backbone,
                decode_mode=decode_mode,
                row_metrics=row,
            )
            rows_handle.write(json.dumps(jsonable(row), ensure_ascii=True) + "\n")
            processed += 1
            if processed == 1 or processed % int(args.log_every_samples) == 0 or processed == len(sample_contexts):
                emit(
                    {
                        "event": "sample_done",
                        "backbone": backbone,
                        "decode_mode": decode_mode,
                        "done": int(processed),
                        "total": int(len(sample_contexts)),
                        "sample_id": row["sample_id"],
                        "teacher_continuous_3d_xyz_h6p4_64wp_ADE_single_m": row.get(
                            "teacher_continuous_3d_xyz_h6p4_64wp_ADE_single_m"
                        ),
                        "teacher_continuous_3d_xyz_h6p4_64wp_minADE6_m": row.get(
                            "teacher_continuous_3d_xyz_h6p4_64wp_minADE6_m"
                        ),
                    }
                )

    elapsed = time.time() - combo_start
    combo_summary = {
        "backbone": backbone,
        "decode_mode": decode_mode,
        "processed_samples": int(processed),
        "failed_samples": int(failed_samples),
        "elapsed_sec": float(elapsed),
        "sec_per_sample": float(elapsed / max(processed, 1)),
    }
    emit({"event": "combo_done", **combo_summary})
    return combo_summary


def main() -> int:
    args = parse_args()
    args.corpus_jsonl = require_existing_file(args.corpus_jsonl, "corpus JSONL")
    args.split_cache_json = require_existing_file(args.split_cache_json, "split cache JSON")
    args.ae_rows_jsonl = require_existing_file(args.ae_rows_jsonl, "AE rows.jsonl")
    args.output_dir = resolve_repo_path(args.output_dir)
    selected_backbones = parse_name_list(
        list(args.backbones),
        choices=tuple(BACKBONE_CHECKPOINTS),
        flag_name="--backbones",
    )
    selected_decode_modes = parse_name_list(
        list(args.decode_modes),
        choices=tuple(DECODE_MODES),
        flag_name="--decode-modes",
    )
    checkpoint_by_backbone = {
        name: require_existing_dir(BACKBONE_CHECKPOINTS[name], f"backbone checkpoint {name}")
        for name in selected_backbones
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "rows.jsonl"
    summary_path = args.output_dir / "summary.json"

    emit(
        {
            "event": "boot",
            "output_dir": str(args.output_dir),
            "rows_jsonl": str(rows_path),
            "summary_json": str(summary_path),
            "backbones": selected_backbones,
            "decode_modes": selected_decode_modes,
        }
    )
    runtime = load_runtime_modules()

    eval_items, split_summary, id_check = select_and_verify_eval_items(args=args, runtime=runtime)
    decoder, decoder_config_path, decoder_base_model, decoder_sig = load_decoder(
        runtime=runtime,
        first_checkpoint=checkpoint_by_backbone[selected_backbones[0]],
        student_model=str(args.student_model),
    )
    emit(
        {
            "event": "trajectory_decoder_ready",
            "base_model": decoder_base_model,
            "decoder_config_path": str(decoder_config_path),
            "decoder_signature": decoder_sig,
        }
    )
    sample_contexts, reference_availability = precompute_sample_contexts(
        eval_items=eval_items,
        runtime=runtime,
        decoder=decoder,
    )
    fail_if_reference_missing_too_high(reference_availability, float(args.missing_reference_fail_rate))

    summary: dict[str, Any] = {
        "settings": {
            "corpus_jsonl": str(args.corpus_jsonl),
            "split": str(args.split),
            "num_samples": int(args.num_samples),
            "val_samples": int(args.val_samples),
            "val_fraction": float(args.val_fraction),
            "split_seed": args.split_seed,
            "split_scan_all": bool(args.split_scan_all),
            "split_cache_json": str(args.split_cache_json),
            "eval_samples": int(args.eval_samples),
            "seed": int(args.seed),
            "ae_rows_jsonl": str(args.ae_rows_jsonl),
            "output_dir": str(args.output_dir),
            "selected_backbones": selected_backbones,
            "selected_decode_modes": selected_decode_modes,
            "device": str(args.device),
            "student_model": str(args.student_model),
            "student_dtype": str(args.student_dtype),
            "max_length": int(args.max_length),
            "max_new_tokens": int(args.max_new_tokens),
            "batch_size": int(args.batch_size),
            "reserve_vram_gib": float(args.reserve_vram_gib),
            "prompt_mode": str(args.prompt_mode),
            "target_mode": str(args.target_mode),
            "prompt_text_style": str(args.prompt_text_style),
            "image_prompt_style": str(args.image_prompt_style),
            "fuse_history_tokens": bool(args.fuse_history_tokens),
            "primary_reference": "teacher_continuous",
            "dims": [label for label, _ in DIM_SPECS],
            "horizons": [{"name": name, "waypoints": wp} for name, wp in HORIZON_SPECS],
            "decode_mode_settings": {
                mode: decode_settings_for_mode(args, mode) for mode in selected_decode_modes
            },
        },
        "reference_resolution": {
            "teacher_continuous": {
                "source": "scripts/84_train_student_ae28_official.py raw_teacher_pred(Path(item['raw_json']))",
                "target_action_path": "scripts/84_train_student_ae28_official.py build_batch target_source='teacher' lines 1519-1554: raw_teacher_pred -> teacher_model.action_space.traj_to_action",
                "used_for_metrics": "raw teacher pred_xyz, sliced to 64 xyz waypoints",
            },
            "teacher_discrete": {
                "source": "scripts/25_decode_checkpoint_overlays.py load_traj_future_token_ids(sample['hard_target'])",
                "decode_path": "src/inference/checkpoint_eval.py TrajectoryTokenDecoder.decode(history_xyz, history_rot, token_ids)",
                "used_for_metrics": "decoded teacher stored discrete trajectory tokens",
            },
            "gt": {
                "source": "src/training/collator.py load_ego_future_xyz(sample, PROJECT_ROOT)",
                "used_for_metrics": "real ego future xyz, sliced to 64 waypoints",
            },
        },
        "split_summary": split_summary,
        "val_sample_id_check": id_check,
        "reference_availability": reference_availability,
        "trajectory_decoder": {
            "base_model": decoder_base_model,
            **decoder_sig,
        },
        "backbone_checkpoints": {name: str(path) for name, path in checkpoint_by_backbone.items()},
        "tokenizer_signatures": {},
        "decoder_config_by_backbone": {},
        "vram_reservations": {},
        "combo_wall_clock": {},
        "combo_summaries": [],
        "metrics": {},
        "rows_jsonl": str(rows_path),
        "summary_json": str(summary_path),
    }

    accumulators: dict[tuple[str, str, str, str], dict[str, list[float]]] = {}
    tokenizer_sig_expected: dict[str, Any] | None = None
    rows_tmp_path = rows_path
    with rows_tmp_path.open("w", encoding="utf-8") as rows_handle:
        for backbone in selected_backbones:
            checkpoint_dir = checkpoint_by_backbone[backbone]
            model_args = make_model_load_args(args, checkpoint_dir, args.output_dir)
            model, tokenizer, processor, device, base_model = runtime.decode25._load_model_and_processors(model_args)
            signature = validate_checkpoint_tokenizer(
                label=backbone,
                tokenizer=tokenizer,
                expected_signature=tokenizer_sig_expected,
            )
            if tokenizer_sig_expected is None:
                tokenizer_sig_expected = signature
            summary["tokenizer_signatures"][backbone] = {
                "base_model": str(base_model),
                **signature,
            }
            summary["decoder_config_by_backbone"][backbone] = validate_checkpoint_decoder_config(
                runtime=runtime,
                backbone=backbone,
                base_model=str(base_model),
                expected_decoder_signature=decoder_sig,
            )
            reserve_event, reserve_warnings = runtime.ae84.reserve_vram_cache(float(args.reserve_vram_gib), device)
            summary["vram_reservations"][backbone] = {
                "event": reserve_event,
                "warnings": reserve_warnings,
            }
            if reserve_event is not None:
                emit(reserve_event)
            for warning in reserve_warnings:
                emit(warning)

            summary["combo_wall_clock"].setdefault(backbone, {})
            for decode_mode in selected_decode_modes:
                combo_summary = run_combo(
                    args=args,
                    runtime=runtime,
                    backbone=backbone,
                    checkpoint_dir=checkpoint_dir,
                    decode_mode=decode_mode,
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    device=device,
                    decoder=decoder,
                    sample_contexts=sample_contexts,
                    rows_handle=rows_handle,
                    accumulators=accumulators,
                )
                summary["combo_wall_clock"][backbone][decode_mode] = combo_summary["elapsed_sec"]
                summary["combo_summaries"].append(combo_summary)
            del model, tokenizer, processor
            gc.collect()
            if str(device).startswith("cuda") and runtime.torch.cuda.is_available():
                runtime.torch.cuda.empty_cache()

    summary["metrics"] = build_nested_metrics(
        accumulators=accumulators,
        selected_backbones=selected_backbones,
        selected_decode_modes=selected_decode_modes,
        total_samples=len(sample_contexts),
    )
    summary["elapsed_sec"] = float(sum(float(item["elapsed_sec"]) for item in summary["combo_summaries"]))
    summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    emit({"event": "done", "status": "ok", "summary_json": str(summary_path), "rows_jsonl": str(rows_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
