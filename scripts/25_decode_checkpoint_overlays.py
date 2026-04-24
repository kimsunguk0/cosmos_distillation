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
from transformers import LogitsProcessorList, StoppingCriteriaList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.checkpoint_eval import (  # noqa: E402
    TrajectoryTokenDecoder,
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
    load_ego_future_xyz,
    load_ego_history_xyz,
    load_sample_images,
)
from src.utils.runtime_paths import remap_external_path, resolve_student_model_path  # noqa: E402


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
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--skip-overlays", action="store_true")
    parser.add_argument("--teacher-text-index", type=Path, default=Path("/data/teacher_cache/text/index.jsonl"))
    parser.add_argument(
        "--teacher-traj-manifest-dir",
        type=Path,
        default=Path("/data/teacher_cache/traj15/manifest"),
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


def _resolve_path(raw_path: str | Path | None) -> Path | None:
    remapped = remap_external_path(raw_path)
    if remapped in (None, ""):
        return None
    path = Path(remapped)
    return path if path.exists() else None


def _extract_generated_text(tokenizer, prompt_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
    prompt_len = int(prompt_ids.shape[1])
    new_ids = generated_ids[0, prompt_len:].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=False)


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


def _path_metrics(student_xyz: np.ndarray | None, gt_xyz: np.ndarray) -> dict[str, float]:
    if student_xyz is None or student_xyz.shape[0] == 0 or gt_xyz.shape[0] == 0:
        return {}
    n = min(int(student_xyz.shape[0]), int(gt_xyz.shape[0]))
    pred = student_xyz[:n]
    gt = gt_xyz[:n]
    ade, fde = _ade_fde(pred, gt)
    early_n = min(20, n)
    late_start = min(20, max(n - 1, 0))
    early_ade, early_fde = _ade_fde(pred[:early_n], gt[:early_n])
    late_ade, _ = _ade_fde(pred[late_start:n], gt[late_start:n])
    gt_len = _path_len(gt)
    pred_len = _path_len(pred)
    return {
        "ade_m": ade,
        "fde_m": fde,
        "early_ade_2s_m": early_ade,
        "early_fde_2s_m": early_fde,
        "late_ade_after_2s_m": late_ade,
        "gt_path_length_m": gt_len,
        "student_path_length_m": pred_len,
        "path_length_ratio": float(pred_len / max(gt_len, 1e-6)),
        "gt_final_speed_mps": _final_speed(gt),
        "student_final_speed_mps": _final_speed(pred),
        "gt_final_x_m": float(gt[-1, 0]),
        "gt_final_y_m": float(gt[-1, 1]),
        "student_final_x_m": float(pred[-1, 0]),
        "student_final_y_m": float(pred[-1, 1]),
        "final_lateral_error_m": float(abs(pred[-1, 1] - gt[-1, 1])),
        "direction_cosine": _direction_cosine(pred, gt),
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
    selected = _select_rows(rows, args.split, args.num_samples)
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

    for idx, sample in enumerate(selected, start=1):
        sample_id = str(sample.get("sample_id") or f"sample_{idx:04d}")
        history_xyz = load_ego_history_xyz(sample, PROJECT_ROOT)
        history_rot = load_ego_history_rot(sample, PROJECT_ROOT)
        try:
            gt_future = load_ego_future_xyz(sample, PROJECT_ROOT)
        except FileNotFoundError:
            gt_future = np.zeros((0, 3), dtype=np.float32)
        target_tokens = [int(token) for token in (sample.get("hard_target") or {}).get("traj_future_token_ids") or []]

        prompt_text = (
            build_traj_only_prompt(sample, PROJECT_ROOT, ego_history_xyz=history_xyz)
            if args.prompt_mode == "traj_only"
            else build_user_prompt(sample, PROJECT_ROOT, ego_history_xyz=history_xyz)
        )
        assistant_prefix = "<|traj_future_start|>" if args.target_mode == "traj_only" else "<|cot_start|>"
        images = load_sample_images(sample, PROJECT_ROOT)
        messages = build_messages(prompt_text, len(images), assistant_prefix=assistant_prefix)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        batch = processor(text=[text], images=[images], return_tensors="pt", padding=True, truncation=True)
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        prompt_lengths = batch["attention_mask"].sum(dim=1).tolist()
        if args.target_mode == "traj_only":
            contract = TrajOnlyDecodingContract.from_tokenizer(
                tokenizer,
                prompt_lengths=prompt_lengths,
                traj_token_count=len(target_tokens),
            )
            logits_processor = LogitsProcessorList([TrajOnlyLogitsProcessor(contract)])
            stopping_criteria = StoppingCriteriaList([StopOnTrajOnlyEndCriteria(contract)])
        else:
            contract = TrajDecodingContract.from_tokenizer(
                tokenizer,
                prompt_lengths=prompt_lengths,
                traj_token_count=len(target_tokens),
            )
            logits_processor = LogitsProcessorList([TrajSpanLogitsProcessor(contract)])
            stopping_criteria = StoppingCriteriaList([StopOnTrajEndCriteria(contract)])

        with torch.inference_mode():
            generated = model.backbone.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
            )

        generated_text = _extract_generated_text(tokenizer, batch["input_ids"], generated)
        generated_tokens = _extract_generated_traj_tokens(generated_text)
        all_tokens.update(generated_tokens)
        rep = _token_repetition_stats(generated_tokens)
        unique_values.append(int(rep["unique"]))
        max_run_values.append(int(rep["max_same_run"]))
        invalid_count = sum(1 for token in generated_tokens if token < 0 or token >= 3000)
        invalid_counts.append(invalid_count)
        token_match = float(
            sum(1 for left, right in zip(generated_tokens, target_tokens) if int(left) == int(right))
            / max(len(target_tokens), 1)
        )
        token_match_values.append(token_match)

        student_xyz = (
            decoder.decode(history_xyz, history_rot, generated_tokens)
            if len(generated_tokens) == decoder.n_waypoints * 2
            else None
        )
        geom = _path_metrics(student_xyz, gt_future)
        if geom:
            ade_values.append(geom["ade_m"])
            fde_values.append(geom["fde_m"])

        manifest = teacher_manifest.get(sample_id)
        tags = _failure_tags(
            sample=sample,
            generated_tokens=generated_tokens,
            path_metrics=geom,
            teacher_manifest=manifest,
        )
        tag_counter.update(tags)
        text_entry = teacher_text.get(sample_id) or {}
        human_coc = text_entry.get("human_coc") or str((sample.get("hard_target") or {}).get("cot_text") or "")
        teacher_cot = text_entry.get("teacher_long_cot") or str((sample.get("teacher_target") or {}).get("cot_text") or "")
        student_cot = _extract_student_cot(generated_text)

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
                student=student_xyz,
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
                    "sample_id": sample_id,
                    "ade_m": row.get("ade_m"),
                    "fde_m": row.get("fde_m"),
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
        "prompt_mode": args.prompt_mode,
        "target_mode": args.target_mode,
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
