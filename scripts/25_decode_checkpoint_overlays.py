#!/usr/bin/env python3
"""Decode a student checkpoint and write trajectory quality/failure reports.

This is the post-SFT selection evaluator. It runs constrained LM trajectory
generation, decodes the 128 discrete trajectory body to XYZ, computes ADE/FDE,
and adds lightweight failure tags so the next KD/hidden/expert step is based on
actual error modes rather than train loss alone.
"""

from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import LogitsProcessorList, StoppingCriteriaList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.checkpoint_eval import (  # noqa: E402
    TrajectoryTokenDecoder,
    _infer_visual_float_dtype,
    _manual_flex_generate,
    load_ego_history_rot,
    resolve_traj_tokenizer_config_path,
)
from src.inference.decoding import (  # noqa: E402
    StopOnTrajEndCriteria,
    StopOnTrajOnlyEndCriteria,
    TrajDecodingContract,
    TrajOnlyDecodingContract,
    TrajOnlyLogitsProcessor,
    TrajSpanLogitsProcessor,
)
from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint  # noqa: E402
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora  # noqa: E402
from src.model.student_wrapper import StudentWrapperConfig, build_student_model  # noqa: E402
from src.model.tokenizer_ext import distill_trainable_token_ids  # noqa: E402
from src.training.collator import (  # noqa: E402
    build_messages,
    build_traj_only_prompt,
    build_user_prompt,
    fuse_history_tokens_in_input_ids,
    load_ego_future_xyz,
    load_ego_history_xyz,
    load_sample_images,
    load_traj_future_token_ids,
    resolve_image_relative_timestamps,
    resolve_camera_indices,
)
from src.training.flex_batch import compress_batch_for_flex  # noqa: E402
from src.utils.runtime_paths import (  # noqa: E402
    DEFAULT_TEACHER_CACHE_ROOT,
    remap_external_path,
    resolve_student_model_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "distill_v3_2_959.jsonl",
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--student-model", default=resolve_student_model_path())
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of split samples to evaluate. Use 0 or negative for the full split.",
    )
    parser.add_argument("--prompt-mode", choices=("joint", "traj_only"), default="joint")
    parser.add_argument("--target-mode", choices=("joint", "traj_only"), default="joint")
    parser.add_argument(
        "--image-prompt-style",
        choices=("compact", "camera_labeled"),
        default="compact",
        help="Use official Alpamayo-style camera/frame text before image placeholders.",
    )
    parser.add_argument(
        "--prompt-text-style",
        choices=("numeric_history_question", "official_alpamayo"),
        default="numeric_history_question",
        help="Instruction/history prompt style. Use official_alpamayo for the Alpamayo 1.5 input contract.",
    )
    parser.add_argument(
        "--fuse-history-tokens",
        action="store_true",
        help="Replace <|traj_history|> placeholders with encoded Alpamayo history delta token ids.",
    )
    parser.add_argument(
        "--geometry-reference",
        choices=("gt", "teacher"),
        default="gt",
        help="Reference trajectory for ADE/FDE; `teacher` decodes cached teacher trajectory tokens.",
    )
    parser.add_argument(
        "--oracle-cot-prefix",
        action="store_true",
        help=(
            "For target-mode=traj_only, prepend the cached teacher CoT through "
            "`<|traj_future_start|>` so only the 128 trajectory body tokens are free-run."
        ),
    )
    parser.add_argument(
        "--image-ablation",
        choices=("normal", "black", "gray", "noise", "camera_shuffle"),
        default="normal",
        help=(
            "Perturb input images while keeping text/ego unchanged. "
            "camera_shuffle reverses the camera groups within each sample."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of samples per free-run generate() call. Use 1 for the old sample-by-sample path.",
    )
    parser.add_argument("--samples-per-row", type=int, default=1, help="Number of generated trajectories per sample.")
    parser.add_argument("--seed", type=int, default=97, help="Random seed used when samples_per_row > 1.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature when multiple trajectories are requested.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling when multiple trajectories are requested.")
    parser.add_argument("--selected-json", type=Path, help="JSON list of selected sample rows or sample ids.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--skip-overlays", action="store_true")
    parser.add_argument(
        "--disable-failure-tags",
        action="store_true",
        help="Skip heuristic free-run geometry taxonomy tags such as long_horizon_divergence.",
    )
    parser.add_argument(
        "--teacher-text-index",
        type=Path,
        default=DEFAULT_TEACHER_CACHE_ROOT / "text" / "index.jsonl",
    )
    parser.add_argument(
        "--teacher-traj-manifest-dir",
        type=Path,
        default=DEFAULT_TEACHER_CACHE_ROOT / "traj15" / "manifest",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _select_rows(rows: list[dict[str, Any]], split: str, num_samples: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("split") == split]
    if num_samples > 0:
        return selected[:num_samples]
    return selected


def _load_selected_rows(
    path: Path | None,
    rows: list[dict[str, Any]],
    *,
    split: str,
) -> list[dict[str, Any]]:
    if path is None:
        return rows
    if not path.exists():
        raise SystemExit(f"selected-json not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected_ids: list[str] = []
    for item in payload:
        if isinstance(item, str):
            selected_ids.append(str(item))
            continue
        if isinstance(item, dict) and "sample_id" in item:
            selected_ids.append(str(item["sample_id"]))
    if not selected_ids:
        return []
    row_map = {str(row.get("sample_id")): row for row in rows if str(row.get("split") or "") == split}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample_id in selected_ids:
        row = row_map.get(sample_id)
        if row is None or sample_id in seen:
            continue
        selected.append(row)
        seen.add(sample_id)
    return selected


def _candidate_score(ade: float | None, fde: float | None, cand: list[int]) -> tuple[float, float, int]:
    if (
        ade is None
        or fde is None
        or not math.isfinite(float(ade))
        or not math.isfinite(float(fde))
        or math.isnan(float(ade))
        or math.isnan(float(fde))
    ):
        return (float("inf"), float("inf"), -len(cand))
    return (float(ade), float(fde), -len(cand))


def _select_best_candidate(
    candidate_records: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    best_index = 0
    best_score = (float("inf"), float("inf"), 1 << 30)
    best_row = {
        "candidate_index": 1,
        "student_free_run_traj_tokens": [],
        "student_vs_teacher_discrete_ade_m": None,
        "student_vs_teacher_discrete_fde_m": None,
        "student_free_run_unique_token_count": 0,
        "student_free_run_token_match_rate": 0.0,
    }
    if not candidate_records:
        return best_index, best_row
    for idx, cand in enumerate(candidate_records):
        score = _candidate_score(
            cand.get("student_vs_teacher_discrete_ade_m"),
            cand.get("student_vs_teacher_discrete_fde_m"),
            cand.get("student_free_run_traj_tokens") or [],
        )
        if score < best_score:
            best_score = score
            best_row = cand
            best_index = idx
    return best_index, best_row


def _resolve_path(raw_path: str | Path | None) -> Path | None:
    remapped = remap_external_path(raw_path)
    if remapped in (None, ""):
        return None
    path = Path(remapped)
    return path if path.exists() else None


def _extract_generated_text(tokenizer, prompt_ids: torch.Tensor, generated_ids: torch.Tensor, row_index: int = 0) -> str:
    prompt_len = int(prompt_ids.shape[1])
    new_ids = generated_ids[row_index, prompt_len:].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=False)


def _batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    width = max(int(batch_size), 1)
    return [items[index : index + width] for index in range(0, len(items), width)]


def _extract_generated_traj_tokens(text: str) -> list[int]:
    return [int(match.group(1)) for match in re.finditer(r"<i(\d+)>", text)]


def _extract_student_cot(text: str) -> str:
    start_marker = "<|cot_start|>"
    end_marker = "<|cot_end|>"
    traj_marker = "<|traj_future_start|>"
    start = text.find(start_marker)
    start = 0 if start < 0 else start + len(start_marker)
    end = text.find(end_marker, start)
    if end < 0:
        end = text.find(traj_marker, start)
    if end < 0:
        end = len(text)
    cot = re.sub(r"<\|[^|]+\|>", "", text[start:end])
    return " ".join(cot.split())


def _apply_image_ablation(images: list[Any], mode: str, *, sample_id: str) -> list[Any]:
    if mode == "normal":
        return images
    if mode in {"black", "gray"}:
        color = (0, 0, 0) if mode == "black" else (127, 127, 127)
        return [Image.new("RGB", image.size, color) for image in images]
    if mode == "noise":
        out: list[Any] = []
        seed = abs(hash(sample_id)) % (2**32)
        rng = np.random.default_rng(seed)
        for image_index, image in enumerate(images):
            local = np.random.default_rng(seed + image_index * 1009)
            arr = local.integers(0, 256, size=(image.size[1], image.size[0], 3), dtype=np.uint8)
            out.append(Image.fromarray(arr, mode="RGB"))
        return out
    if mode == "camera_shuffle":
        # Materialized order is camera-major: cam0 f0..f3, cam1 f0..f3, ...
        # Reverse camera groups while preserving frame order inside each camera.
        frame_count = 4 if len(images) % 4 == 0 else 1
        groups = [images[index : index + frame_count] for index in range(0, len(images), frame_count)]
        return [image for group in reversed(groups) for image in group]
    raise ValueError(f"Unsupported image_ablation={mode!r}")


def _max_same_token_run(token_ids: list[int]) -> int:
    if not token_ids:
        return 0
    best = current = 1
    for left, right in zip(token_ids, token_ids[1:]):
        if left == right:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _load_teacher_text_cache(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        sample_id = rec.get("sample_id")
        if not sample_id:
            continue
        out[sample_id] = {
            "human_coc": str((rec.get("hard_human") or {}).get("human_coc") or ""),
            "teacher_long_cot": str((rec.get("output") or {}).get("teacher_long_cot") or ""),
            "teacher_motion": str((rec.get("output") or {}).get("teacher_motion_class") or ""),
        }
    return out


def _load_teacher_manifest_map(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for item in path.glob("*.manifest.json"):
        try:
            payload = json.loads(item.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        sample_id = payload.get("sample_id")
        if sample_id:
            out[str(sample_id)] = payload
    return out


def _load_model_and_processors(args: argparse.Namespace):
    checkpoint_dir = args.checkpoint_dir
    train_config_path = checkpoint_dir / "train_config.json"
    train_config = json.loads(train_config_path.read_text(encoding="utf-8")) if train_config_path.exists() else {}
    checkpoint_manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    checkpoint_manifest = (
        json.loads(checkpoint_manifest_path.read_text(encoding="utf-8")) if checkpoint_manifest_path.exists() else {}
    )
    base_model = str((train_config.get("args") or {}).get("student_model") or args.student_model)
    use_lora = not bool((train_config.get("args") or {}).get("disable_lora", False))

    from transformers import AutoProcessor, AutoTokenizer

    tokenizer_dir = checkpoint_dir / "tokenizer"
    processor_dir = checkpoint_dir / "processor"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir if tokenizer_dir.exists() else base_model,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        processor_dir if processor_dir.exists() else base_model,
        local_files_only=True,
    )
    processor.tokenizer = tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    data_view = train_config.get("data_view") or {}
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else None
    wrapper_cfg = StudentWrapperConfig(
        student_model_name=base_model,
        max_length=int((train_config.get("trainer_config") or {}).get("max_length", 4096)),
        torch_dtype=dtype,
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
    print(json.dumps({"event": "load_model_start", "checkpoint": str(checkpoint_dir), "base_model": base_model}))
    model = build_student_model(wrapper_cfg, tokenizer)
    checkpoint_format = detect_checkpoint_format(checkpoint_dir)
    if checkpoint_format == "full_state_dict" and use_lora:
        model.backbone = maybe_apply_lora(
            model.backbone,
            LoraConfigSpec(trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer))),
            enabled=True,
        )
    load_info = load_student_checkpoint(checkpoint_dir, model, use_lora=use_lora)
    model = model.to(device).eval()
    print(
        json.dumps(
            {
                "event": "load_model_done",
                "checkpoint_format": checkpoint_format,
                "load_format": load_info.get("format"),
                "device": str(device),
            }
        )
    )
    return model, tokenizer, processor, device, base_model


def _ade_fde(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    n = min(int(pred.shape[0]), int(target.shape[0]))
    if n <= 0:
        return float("nan"), float("nan")
    dist = np.linalg.norm(pred[:n, :2] - target[:n, :2], axis=-1)
    return float(dist.mean()), float(dist[-1])


def _path_len(xyz: np.ndarray) -> float:
    if xyz.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=-1).sum())


def _final_speed(xyz: np.ndarray, dt: float = 0.1) -> float:
    if xyz.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(xyz[-1, :2] - xyz[-2, :2]) / dt)


def _direction_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    va = a[-1, :2] - a[0, :2]
    vb = b[-1, :2] - b[0, :2]
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-6:
        return float("nan")
    return float(np.dot(va, vb) / denom)


def _path_metrics(student_xyz: np.ndarray | None, reference_xyz: np.ndarray) -> dict[str, float]:
    if student_xyz is None or student_xyz.shape[0] == 0 or reference_xyz.shape[0] == 0:
        return {}
    n = min(int(student_xyz.shape[0]), int(reference_xyz.shape[0]))
    pred = student_xyz[:n]
    reference = reference_xyz[:n]
    ade, fde = _ade_fde(pred, reference)
    early_n = min(20, n)
    late_start = min(20, max(n - 1, 0))
    early_ade, early_fde = _ade_fde(pred[:early_n], reference[:early_n])
    late_ade, _ = _ade_fde(pred[late_start:n], reference[late_start:n])
    reference_len = _path_len(reference)
    pred_len = _path_len(pred)
    return {
        "ade_m": ade,
        "fde_m": fde,
        "early_ade_2s_m": early_ade,
        "early_fde_2s_m": early_fde,
        "late_ade_after_2s_m": late_ade,
        "reference_path_length_m": reference_len,
        "gt_path_length_m": reference_len,
        "student_path_length_m": pred_len,
        "path_length_ratio": float(pred_len / max(reference_len, 1e-6)),
        "reference_final_speed_mps": _final_speed(reference),
        "gt_final_speed_mps": _final_speed(reference),
        "student_final_speed_mps": _final_speed(pred),
        "reference_final_x_m": float(reference[-1, 0]),
        "reference_final_y_m": float(reference[-1, 1]),
        "gt_final_x_m": float(reference[-1, 0]),
        "gt_final_y_m": float(reference[-1, 1]),
        "student_final_x_m": float(pred[-1, 0]),
        "student_final_y_m": float(pred[-1, 1]),
        "final_lateral_error_m": float(abs(pred[-1, 1] - reference[-1, 1])),
        "direction_cosine": _direction_cosine(pred, reference),
    }


def _token_repetition_stats(tokens: list[int]) -> dict[str, Any]:
    counter = Counter(tokens)
    total = max(len(tokens), 1)
    top = counter.most_common(10)
    return {
        "unique": len(counter),
        "max_same_run": _max_same_token_run(tokens),
        "top_tokens": [{"token": int(tok), "count": int(count), "mass": float(count / total)} for tok, count in top],
        "top1_mass": float(top[0][1] / total) if top else 0.0,
        "top2_mass": float(sum(count for _, count in top[:2]) / total) if top else 0.0,
    }


def _failure_tags(
    *,
    sample: dict[str, Any],
    generated_tokens: list[int],
    path_metrics: dict[str, float],
    teacher_manifest: dict[str, Any] | None,
) -> list[str]:
    tags: list[str] = []
    invalid_count = sum(1 for token in generated_tokens if token < 0 or token >= 3000)
    rep = _token_repetition_stats(generated_tokens)
    gt_motion = str((sample.get("derived") or {}).get("gt_motion_class") or "").lower()

    teacher_ade = float((teacher_manifest or {}).get("best_candidate_ade_m") or float("nan"))
    teacher_fde = float((teacher_manifest or {}).get("best_candidate_fde_m") or float("nan"))
    if math.isfinite(teacher_ade) and math.isfinite(teacher_fde) and (teacher_ade > 2.0 or teacher_fde > 6.0):
        tags.append("G_teacher_also_far_from_gt")

    if len(generated_tokens) != 128:
        tags.append("invalid_token_count")
    if invalid_count > 0:
        tags.append("invalid_future_token_i3000_plus")
    if rep["max_same_run"] >= 8 or rep["unique"] <= 8 or rep["top2_mass"] >= 0.85:
        tags.append("F_repetition_or_local_band_oscillation")

    if not path_metrics:
        tags.append("no_decoded_geometry")
        return tags

    ade = path_metrics["ade_m"]
    fde = path_metrics["fde_m"]
    early_ade = path_metrics["early_ade_2s_m"]
    late_ade = path_metrics["late_ade_after_2s_m"]
    ratio = path_metrics["path_length_ratio"]
    direction_cosine = path_metrics["direction_cosine"]
    gt_final_speed = path_metrics["gt_final_speed_mps"]
    student_final_speed = path_metrics["student_final_speed_mps"]
    gt_final_y = path_metrics["gt_final_y_m"]
    student_final_y = path_metrics["student_final_y_m"]
    lateral_error = path_metrics["final_lateral_error_m"]

    stop_like_gt = gt_motion in {"stop", "stopping", "decelerate", "slow"} or gt_final_speed < 0.75
    if stop_like_gt and student_final_speed > 1.5 and ratio > 1.25:
        tags.append("A_stop_or_decel_failure")
    if abs(gt_final_y) > 1.0 and abs(student_final_y) > 1.0 and gt_final_y * student_final_y < 0.0:
        tags.append("B_curvature_or_turn_direction_failure")
    elif lateral_error > 2.5 and ade > 2.0:
        tags.append("B_curvature_or_lateral_failure")
    if math.isfinite(direction_cosine) and direction_cosine > 0.75 and (ratio > 1.35 or ratio < 0.65):
        tags.append("C_speed_scale_failure")
    if early_ade > 2.0 or path_metrics["early_fde_2s_m"] > 3.0:
        tags.append("D_initial_prefix_failure")
    if early_ade <= 2.0 and (late_ade > max(2.5, early_ade * 2.0) or fde > 6.0):
        tags.append("E_long_horizon_divergence")
    if not tags and (ade > 2.0 or fde > 6.0):
        tags.append("unclassified_geometry_error")
    if not tags:
        tags.append("ok_or_low_error")
    return tags


def _polyline(points: np.ndarray, *, xmin: float, ymin: float, scale: float, plot_h: float, margin: float) -> str:
    out: list[str] = []
    for x, y in points[:, :2]:
        px = margin + (float(x) - xmin) * scale
        py = margin + plot_h - (float(y) - ymin) * scale
        out.append(f"{px:.1f},{py:.1f}")
    return " ".join(out)


def _write_overlay_svg(
    path: Path,
    *,
    title: str,
    history: np.ndarray,
    gt: np.ndarray,
    student: np.ndarray | None,
    student_cot: str,
    human_coc: str,
    teacher_cot: str,
    tags: list[str],
) -> None:
    arrays = [arr[:, :2] for arr in (history, gt, student if student is not None else np.zeros((0, 3))) if arr.size]
    if not arrays:
        return
    all_xy = np.concatenate(arrays, axis=0)
    xmin, ymin = all_xy.min(axis=0) - 5.0
    xmax, ymax = all_xy.max(axis=0) + 5.0
    width, height = 980.0, 640.0
    margin = 30.0
    plot_w, plot_h = 540.0, 540.0
    scale = min(plot_w / max(float(xmax - xmin), 1e-3), plot_h / max(float(ymax - ymin), 1e-3))

    def line(arr: np.ndarray, color: str, label: str) -> str:
        if arr.size == 0:
            return ""
        pts = _polyline(arr, xmin=float(xmin), ymin=float(ymin), scale=scale, plot_h=plot_h, margin=margin)
        return f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{pts}'><title>{label}</title></polyline>"

    text_x = 610
    text_y = 42
    text_lines: list[str] = []
    for heading, body in (
        ("tags", ", ".join(tags)),
        ("student", student_cot),
        ("human", human_coc),
        ("teacher", teacher_cot),
    ):
        text_lines.append(f"<text x='{text_x}' y='{text_y}' font-size='13' font-family='monospace' fill='#111'>{html.escape(heading)}:</text>")
        text_y += 18
        words = (body or "").split()
        cur = ""
        for word in words[:120]:
            candidate = f"{cur} {word}".strip()
            if len(candidate) > 46:
                text_lines.append(
                    f"<text x='{text_x}' y='{text_y}' font-size='11' font-family='monospace' fill='#333'>{html.escape(cur)}</text>"
                )
                text_y += 15
                cur = word
            else:
                cur = candidate
        if cur:
            text_lines.append(
                f"<text x='{text_x}' y='{text_y}' font-size='11' font-family='monospace' fill='#333'>{html.escape(cur)}</text>"
            )
            text_y += 15
        text_y += 10

    layers = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{int(width)}' height='{int(height)}'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='white'/>",
        f"<text x='16' y='22' font-size='14' font-family='monospace'>{html.escape(title)}</text>",
        line(history, "#111111", "history"),
        line(gt, "#1b9e77", "GT future"),
        line(student if student is not None else np.zeros((0, 3)), "#e31a1c", "student future"),
        "<line x1='30' y1='590' x2='55' y2='590' stroke='#111111' stroke-width='3'/>",
        "<text x='65' y='594' font-size='12' font-family='monospace'>history</text>",
        "<line x1='145' y1='590' x2='170' y2='590' stroke='#1b9e77' stroke-width='3'/>",
        "<text x='180' y='594' font-size='12' font-family='monospace'>GT</text>",
        "<line x1='230' y1='590' x2='255' y2='590' stroke='#e31a1c' stroke-width='3'/>",
        "<text x='265' y='594' font-size='12' font-family='monospace'>student</text>",
        *text_lines,
        "</svg>",
    ]
    path.write_text("\n".join(layer for layer in layers if layer), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer, processor, device, base_model = _load_model_and_processors(args)
    rows = _load_jsonl(args.corpus_jsonl)
    selected = _load_selected_rows(
        args.selected_json,
        _select_rows(rows, args.split, args.num_samples),
        split=args.split,
    )
    if not selected:
        raise SystemExit(f"No samples selected for split={args.split!r} from {args.corpus_jsonl}")

    decoder_path = resolve_traj_tokenizer_config_path(base_model)
    if decoder_path is None:
        raise SystemExit("Could not find Alpamayo traj tokenizer config.")
    decoder = TrajectoryTokenDecoder(config_path=decoder_path)
    teacher_text = _load_teacher_text_cache(args.teacher_text_index)
    teacher_manifest = _load_teacher_manifest_map(args.teacher_traj_manifest_dir)

    per_sample: list[dict[str, Any]] = []
    all_tokens: Counter[int] = Counter()
    tag_counter: Counter[str] = Counter()
    ade_values: list[float] = []
    fde_values: list[float] = []
    unique_values: list[int] = []
    max_run_values: list[int] = []
    token_match_values: list[float] = []
    invalid_counts: list[int] = []

    for sample_batch in _batched(selected, args.batch_size):
        prepared: list[dict[str, Any]] = []
        texts: list[str] = []
        image_batches: list[list[Any]] = []
        target_token_count: int | None = None

        for sample in sample_batch:
            idx = len(per_sample) + len(prepared) + 1
            sample_id = str(sample.get("sample_id") or f"sample_{idx:04d}")
            history_xyz = load_ego_history_xyz(sample, PROJECT_ROOT)
            history_rot = load_ego_history_rot(sample, PROJECT_ROOT)
            try:
                gt_future = load_ego_future_xyz(sample, PROJECT_ROOT)
            except FileNotFoundError:
                gt_future = np.zeros((0, 3), dtype=np.float32)
            target_tokens = load_traj_future_token_ids(sample.get("hard_target") or {}, PROJECT_ROOT)
            if target_token_count is None:
                target_token_count = len(target_tokens)
            elif len(target_tokens) != target_token_count:
                raise ValueError(
                    f"Mixed trajectory target lengths in one free-run batch: "
                    f"{target_token_count} and {len(target_tokens)}"
                )

            prompt_text = (
                build_traj_only_prompt(sample, PROJECT_ROOT, ego_history_xyz=history_xyz)
                if args.prompt_mode == "traj_only"
                else build_user_prompt(
                    sample,
                    PROJECT_ROOT,
                    ego_history_xyz=history_xyz,
                    prompt_text_style=args.prompt_text_style,
                )
            )
            if args.target_mode == "traj_only":
                if args.oracle_cot_prefix:
                    cot_text = str(
                        (sample.get("teacher_target") or {}).get("cot_text")
                        or (sample.get("hard_target") or {}).get("cot_text")
                        or ""
                    ).strip()
                    assistant_prefix = f"<|cot_start|>{cot_text}<|cot_end|><|traj_future_start|>"
                else:
                    assistant_prefix = "<|traj_future_start|>"
            else:
                assistant_prefix = "<|cot_start|>"
            images = _apply_image_ablation(
                load_sample_images(sample, PROJECT_ROOT),
                args.image_ablation,
                sample_id=sample_id,
            )
            camera_indices = resolve_camera_indices(sample, PROJECT_ROOT, image_count=len(images))
            frames_per_camera = max(len(images) // max(len(camera_indices), 1), 1)
            messages = build_messages(
                prompt_text,
                len(images),
                assistant_prefix=assistant_prefix,
                image_prompt_style=args.image_prompt_style,
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
                    "idx": idx,
                    "sample": sample,
                    "sample_id": sample_id,
                    "history_xyz": history_xyz,
                    "history_rot": history_rot,
                    "gt_future": gt_future,
                    "target_tokens": target_tokens,
                    "camera_indices": camera_indices,
                    "frames_per_camera": frames_per_camera,
                }
            )
            texts.append(text)
            image_batches.append(images)

        batch = processor(text=texts, images=image_batches, return_tensors="pt", padding=True, truncation=True)
        if args.fuse_history_tokens:
            batch["input_ids"] = fuse_history_tokens_in_input_ids(
                batch["input_ids"],
                tokenizer,
                [item["history_xyz"] for item in prepared],
            )
        flex_enabled = bool(hasattr(model, "flex_enabled") and model.flex_enabled())
        if flex_enabled:
            max_cameras = max(len(item["camera_indices"]) for item in prepared)
            max_frames = max(int(item["frames_per_camera"]) for item in prepared)
            camera_indices_tensor = torch.zeros((len(prepared), max_cameras), dtype=torch.long)
            relative_timestamps_tensor = torch.zeros((len(prepared), max_cameras, max_frames), dtype=torch.float32)
            camera_counts = torch.zeros((len(prepared),), dtype=torch.long)
            frames_per_camera_tensor = torch.zeros((len(prepared),), dtype=torch.long)
            for row_index, item in enumerate(prepared):
                sample = item["sample"]
                row_camera_indices = [int(value) for value in item["camera_indices"]]
                row_frames = int(item["frames_per_camera"])
                row_relative_times = resolve_image_relative_timestamps(
                    sample,
                    PROJECT_ROOT,
                    camera_count=len(row_camera_indices),
                    frames_per_camera=row_frames,
                )
                camera_count = len(row_camera_indices)
                camera_indices_tensor[row_index, :camera_count] = torch.tensor(row_camera_indices, dtype=torch.long)
                camera_counts[row_index] = camera_count
                frames_per_camera_tensor[row_index] = row_frames
                for camera_offset, row_times in enumerate(row_relative_times[:camera_count]):
                    count = min(len(row_times), max_frames)
                    if count > 0:
                        relative_timestamps_tensor[row_index, camera_offset, :count] = torch.tensor(
                            row_times[:count],
                            dtype=torch.float32,
                        )
            batch["camera_indices"] = camera_indices_tensor
            batch["relative_timestamps"] = relative_timestamps_tensor
            batch["camera_counts"] = camera_counts
            batch["frames_per_camera"] = frames_per_camera_tensor
            flex_cfg = getattr(model, "flex_scene_config")
            batch = compress_batch_for_flex(
                batch,
                image_token_id=int(getattr(model, "image_token_id")),
                tokens_per_image=int(getattr(flex_cfg, "tokens_per_image")),
                pad_token_id=int(getattr(tokenizer, "pad_token_id", 0) or 0),
            )
        model_dtype = _infer_visual_float_dtype(model)
        batch = {
            key: (
                value.to(device=device, dtype=model_dtype)
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
                else value.to(device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in batch.items()
        }
        # `generate` appends new tokens after the padded input length.  The
        # tokenizer for this model right-pads, so using per-row attention-mask
        # lengths would make pad tokens look like generated tokens.
        prompt_lengths = [int(batch["input_ids"].shape[1])] * len(prepared)
        if args.target_mode == "traj_only":
            contract = TrajOnlyDecodingContract.from_tokenizer(
                tokenizer,
                prompt_lengths=prompt_lengths,
                traj_token_count=int(target_token_count or 0),
            )
            logits_processor = LogitsProcessorList([TrajOnlyLogitsProcessor(contract)])
            stopping_criteria = StoppingCriteriaList([StopOnTrajOnlyEndCriteria(contract)])
        else:
            contract = TrajDecodingContract.from_tokenizer(
                tokenizer,
                prompt_lengths=prompt_lengths,
                traj_token_count=int(target_token_count or 0),
            )
            logits_processor = LogitsProcessorList([TrajSpanLogitsProcessor(contract)])
            stopping_criteria = StoppingCriteriaList([StopOnTrajEndCriteria(contract)])

        samples_per_row = max(int(args.samples_per_row), 1)
        if args.seed is not None:
            seed_value = int(args.seed) + len(per_sample)
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)

        if flex_enabled and samples_per_row > 1:
            print(
                json.dumps(
                    {
                        "event": "warning",
                        "batch_size": len(prepared),
                        "message": "flex path does not support multi-sample generation; falling back to greedy single-sample decode",
                    },
                    flush=True,
                )
            )
            samples_per_row = 1

        with torch.inference_mode():
            if flex_enabled:
                generated = _manual_flex_generate(
                    model,
                    batch,
                    max_new_tokens=args.max_new_tokens,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                )
            else:
                generated = model.backbone.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=samples_per_row > 1,
                    num_return_sequences=samples_per_row,
                    use_cache=True,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                )

        for row_index, item in enumerate(prepared):
            idx = int(item["idx"])
            sample = item["sample"]
            sample_id = str(item["sample_id"])
            history_xyz = item["history_xyz"]
            history_rot = item["history_rot"]
            gt_future = item["gt_future"]
            target_tokens = item["target_tokens"]
            teacher_xyz = (
                decoder.decode(history_xyz, history_rot, target_tokens)
                if len(target_tokens) == decoder.n_waypoints * 2
                else np.zeros((0, 3), dtype=np.float32)
            )
            reference_xyz = teacher_xyz if args.geometry_reference == "teacher" else gt_future
            if args.geometry_reference != "teacher" and reference_xyz.size == 0:
                reference_xyz = teacher_xyz

            row_candidates: list[dict[str, Any]] = []
            row_start = row_index * samples_per_row
            row_end = min(row_start + samples_per_row, generated.shape[0])
            for candidate_index, sample_row in enumerate(range(row_start, row_end), start=1):
                generated_text = _extract_generated_text(
                    tokenizer,
                    batch["input_ids"],
                    generated,
                    row_index=sample_row,
                )
                generated_tokens = _extract_generated_traj_tokens(generated_text)
                all_tokens.update(generated_tokens)
                rep = _token_repetition_stats(generated_tokens)
                invalid_count = sum(1 for token in generated_tokens if token < 0 or token >= 3000)
                token_match = float(
                    sum(1 for left, right in zip(generated_tokens, target_tokens) if int(left) == int(right))
                    / max(len(target_tokens), 1)
                )
                student_xyz = (
                    decoder.decode(history_xyz, history_rot, generated_tokens)
                    if len(generated_tokens) == decoder.n_waypoints * 2
                    else None
                )
                candidate_geom = _path_metrics(student_xyz, reference_xyz)
                row_candidates.append(
                    {
                        "candidate_index": candidate_index,
                        "student_free_run_traj_tokens": generated_tokens,
                        "student_vs_teacher_discrete_ade_m": candidate_geom.get("ade_m"),
                        "student_vs_teacher_discrete_fde_m": candidate_geom.get("fde_m"),
                        "student_free_run_unique_token_count": rep["unique"],
                        "student_free_run_token_match_rate": token_match,
                        "student_free_run_invalid_future_token_count_i3000_plus": invalid_count,
                    }
                )

            if not row_candidates:
                row_candidates = [
                    {
                        "candidate_index": 1,
                        "student_free_run_traj_tokens": [],
                        "student_vs_teacher_discrete_ade_m": None,
                        "student_vs_teacher_discrete_fde_m": None,
                        "student_free_run_unique_token_count": 0,
                        "student_free_run_token_match_rate": 0.0,
                        "student_free_run_invalid_future_token_count_i3000_plus": 0,
                    }
                ]

            best_index, best_row = _select_best_candidate(row_candidates)
            generated_tokens = list(best_row.get("student_free_run_traj_tokens") or [])
            generated_metrics = _path_metrics(
                decoder.decode(history_xyz, history_rot, generated_tokens)
                if len(generated_tokens) == decoder.n_waypoints * 2
                else None,
                reference_xyz,
            )
            best_generated_text = ""
            for candidate_index, sample_row in enumerate(range(row_start, row_end), start=1):
                if candidate_index - 1 != best_index:
                    continue
                best_generated_text = _extract_generated_text(
                    tokenizer,
                    batch["input_ids"],
                    generated,
                    row_index=sample_row,
                )
                break

            rep = _token_repetition_stats(generated_tokens)
            token_match = float(
                sum(1 for left, right in zip(generated_tokens, target_tokens) if int(left) == int(right))
                / max(len(target_tokens), 1)
            )
            invalid_count = int(sum(1 for token in generated_tokens if token < 0 or token >= 3000))
            best_student_xyz = decoder.decode(history_xyz, history_rot, generated_tokens)
            if best_student_xyz is None:
                best_student_xyz = np.zeros((0, 3), dtype=np.float32)

            unique_values.append(int(rep["unique"]))
            max_run_values.append(int(rep["max_same_run"]))
            invalid_counts.append(invalid_count)
            token_match_values.append(token_match)

            geom = generated_metrics if generated_metrics else _path_metrics(None, reference_xyz)
            if geom:
                ade_values.append(geom["ade_m"])
                fde_values.append(geom["fde_m"])

            manifest = teacher_manifest.get(sample_id)
            tags = (
                []
                if args.disable_failure_tags
                else _failure_tags(
                    sample=sample,
                    generated_tokens=generated_tokens,
                    path_metrics=geom,
                    teacher_manifest=manifest,
                )
            )
            tag_counter.update(tags)
            text_entry = teacher_text.get(sample_id) or {}
            human_coc = text_entry.get("human_coc") or str((sample.get("hard_target") or {}).get("cot_text") or "")
            teacher_cot = text_entry.get("teacher_long_cot") or str((sample.get("teacher_target") or {}).get("cot_text") or "")
            student_cot = _extract_student_cot(best_generated_text or generated_text)

            svg_path = None
            if not args.skip_overlays:
                svg_path = args.output_dir / f"{args.split}_{idx:03d}_{sample_id}.svg"
                title = (
                    f"{args.split} {idx}/{len(selected)} {sample_id[:18]} "
                    f"ADE={geom.get('ade_m', float('nan')):.2f} FDE={geom.get('fde_m', float('nan')):.2f}"
                )
                _write_overlay_svg(
                    svg_path,
                    title=title,
                    history=history_xyz,
                    gt=gt_future,
                    student=best_student_xyz,
                    student_cot=student_cot,
                    human_coc=human_coc,
                    teacher_cot=teacher_cot,
                    tags=tags,
                )

            row = {
                "sample_id": sample_id,
                "generated_token_count": len(generated_tokens),
                "target_token_count": len(target_tokens),
                "generated_unique_token_count": int(rep["unique"]),
                "generated_max_same_token_run": int(rep["max_same_run"]),
                "generated_invalid_future_token_count_i3000_plus": int(invalid_count),
                "generated_top_tokens": rep["top_tokens"],
                "generated_traj_tokens": generated_tokens,
                "target_traj_tokens": target_tokens,
                "student_free_run_candidate_records": row_candidates,
                "student_free_run_candidate_count": len(row_candidates),
                "student_free_run_selected_candidate_index": best_index + 1,
                "student_free_run_best_candidate_ade_m": best_row.get("student_vs_teacher_discrete_ade_m"),
                "student_free_run_best_candidate_fde_m": best_row.get("student_vs_teacher_discrete_fde_m"),
                "token_match_rate": token_match,
                **geom,
                "teacher_best_ade_m": (manifest or {}).get("best_candidate_ade_m"),
                "teacher_best_fde_m": (manifest or {}).get("best_candidate_fde_m"),
                "teacher_quality_bucket": (manifest or {}).get("quality_bucket"),
                "failure_tags": tags,
                "student_cot": student_cot,
                "teacher_cot": teacher_cot,
                "human_coc": human_coc,
                "svg": str(svg_path) if svg_path is not None else None,
            }
            per_sample.append(row)
            print(
                json.dumps(
                    {
                        "event": "sample_done",
                        "index": idx,
                        "num_samples": len(selected),
                        "batch_size": len(prepared),
                        "sample_id": sample_id,
                        "ade_m": row.get("ade_m"),
                        "fde_m": row.get("fde_m"),
                        "student_free_run_candidates": len(row_candidates),
                        "tags": tags,
                    }
                ),
                flush=True,
            )

    def mean(values: list[float | int]) -> float | None:
        clean = [float(value) for value in values if math.isfinite(float(value))]
        return float(np.mean(clean)) if clean else None

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "split": args.split,
        "num_samples": len(per_sample),
        "batch_size": int(args.batch_size),
        "prompt_mode": args.prompt_mode,
        "target_mode": args.target_mode,
        "image_prompt_style": args.image_prompt_style,
        "image_ablation": args.image_ablation,
        "prompt_text_style": args.prompt_text_style,
        "fuse_history_tokens": bool(args.fuse_history_tokens),
        "oracle_cot_prefix": bool(args.oracle_cot_prefix),
        "geometry_reference": args.geometry_reference,
        "avg_ade_m": mean(ade_values),
        "avg_fde_m": mean(fde_values),
        "avg_unique_traj_ids": mean(unique_values),
        "avg_max_same_token_run": mean(max_run_values),
        "avg_token_match_rate": mean(token_match_values),
        "invalid_future_token_rate_i3000_plus": float(sum(1 for count in invalid_counts if count > 0) / max(len(invalid_counts), 1)),
        "avg_invalid_future_tokens_i3000_plus": mean(invalid_counts),
        "top_token_histogram": [
            {"token": int(token), "count": int(count), "mass": float(count / max(sum(all_tokens.values()), 1))}
            for token, count in all_tokens.most_common(30)
        ],
        "failure_tag_counts": dict(tag_counter.most_common()),
        "traj_tokenizer_config": str(decoder_path),
        "samples": per_sample,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
