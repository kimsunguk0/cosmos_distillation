#!/usr/bin/env python3
"""Render Q2-only Step A contact sheets for visual audit."""

from __future__ import annotations

import argparse
import os
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vqa.q2_stepa import CAMERA_DISPLAY_NAMES, load_row_frame_tensors


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "vqa_q2_stepa_pilot50k"
    / "teacher_q2_t0p60"
    / "q2_hard_gate_accept.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stepa_q2_vision_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--cell-width", type=int, default=640)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--jpeg-optimize", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            yield line_index, json.loads(stripped)


def tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    if frame.ndim != 3:
        raise ValueError(f"expected CHW frame tensor, got {tuple(frame.shape)}")
    frame = frame.detach().cpu().clamp(0, 255).to(torch.uint8)
    array = frame.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 6
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(0, 0, 0),
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def render_contact_sheet(row: dict[str, Any], *, cell_width: int) -> Image.Image:
    frames, camera_indices = load_row_frame_tensors(row)
    if tuple(frames.shape[:2]) != (4, 1):
        raise ValueError(f"expected 4cam x 1frame tensor, got {tuple(frames.shape)} for {row.get('sample_id')}")
    if cell_width <= 0:
        raise ValueError("--cell-width must be positive")

    source_h = int(frames.shape[-2])
    source_w = int(frames.shape[-1])
    cell_height = max(1, round(cell_width * source_h / source_w))
    label_h = 34
    gap = 8
    canvas_w = cell_width * 2 + gap
    canvas_h = (cell_height + label_h) * 2 + gap
    sheet = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    plans = row.get("frame_plan") or []
    for cell_index in range(4):
        row_i = cell_index // 2
        col_i = cell_index % 2
        x = col_i * (cell_width + gap)
        y = row_i * (cell_height + label_h + gap)
        camera_id = int(camera_indices[cell_index].item())
        plan = plans[cell_index] if cell_index < len(plans) else {}
        display_name = str(plan.get("display_name") or CAMERA_DISPLAY_NAMES.get(camera_id, f"Camera {camera_id}"))
        feature = str(plan.get("feature") or "")
        frame_indices = plan.get("frame_indices") or []
        frame_label = f"frame={frame_indices[0]}" if frame_indices else "frame=?"
        label = f"{display_name} | cam={camera_id} | {frame_label}"
        if feature:
            label = f"{label} | {feature}"

        image = tensor_to_pil(frames[cell_index, 0])
        image = ImageOps.contain(image, (cell_width, cell_height), method=Image.Resampling.BILINEAR)
        paste_x = x + (cell_width - image.width) // 2
        paste_y = y + label_h + (cell_height - image.height) // 2
        sheet.paste(image, (paste_x, paste_y))
        draw_label(draw, (x + 8, y + 9), label, font)

    return sheet


def teacher_flags(row: dict[str, Any]) -> dict[str, Any]:
    teacher = row.get("teacher") if isinstance(row.get("teacher"), dict) else {}
    text_flags = row.get("text_flags") if isinstance(row.get("text_flags"), dict) else {}
    return {
        "hard_reject": bool(teacher.get("hard_reject", row.get("hard_reject", False))),
        "has_coordinate": bool(teacher.get("has_coordinate", text_flags.get("has_coordinate", False))),
        "has_future_language": bool(teacher.get("has_future_language", text_flags.get("has_future_language", False))),
        "has_action_language": bool(teacher.get("has_action_language", text_flags.get("has_action_language", False))),
        "quality_flags": teacher.get("quality_flags", row.get("quality_flags", [])) or [],
    }


def build_manifest_row(
    row: dict[str, Any],
    *,
    audit_index: int,
    image_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    flags = teacher_flags(row)
    candidate = {
        "qid": row.get("qid"),
        "stage": row.get("stage"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        **flags,
    }
    try:
        image_ref = str(image_path.relative_to(output_dir))
    except ValueError:
        image_ref = str(image_path)
    return {
        "audit_index": int(audit_index),
        "sample_id": row.get("sample_id"),
        "base_sample_id": row.get("base_sample_id"),
        "image_path": image_ref,
        "qid": row.get("qid"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "teacher_answer_short": row.get("teacher_answer_short"),
        "split": row.get("split"),
        "clip_id": row.get("clip_id"),
        "slot": row.get("slot"),
        "image_profile": row.get("image_profile"),
        "candidates": [candidate],
    }


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def existing_audit_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    seen: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
                seen.add(int(row["audit_index"]))
            except Exception:
                continue
    return seen


def _configure_worker_threads() -> None:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def render_one(payload: tuple[int, dict[str, Any], str, int, str, int, bool]) -> dict[str, Any]:
    _configure_worker_threads()
    input_index, row, output_dir_text, cell_width, image_format, jpeg_quality, jpeg_optimize = payload
    output_dir = Path(output_dir_text)
    image_dir = output_dir / "images"
    audit_index = input_index
    sample_id = str(row.get("sample_id") or f"row_{audit_index:08d}")
    suffix = "jpg" if image_format == "jpg" else "png"
    image_path = image_dir / f"{audit_index:08d}_{sample_id}.{suffix}"
    if not image_path.exists():
        sheet = render_contact_sheet(row, cell_width=int(cell_width))
        if image_format == "jpg":
            sheet.save(
                image_path,
                format="JPEG",
                quality=int(jpeg_quality),
                optimize=bool(jpeg_optimize),
            )
        else:
            sheet.save(image_path)
    return build_manifest_row(
        row,
        audit_index=audit_index,
        image_path=image_path,
        output_dir=output_dir,
    )


def main() -> None:
    args = parse_args()
    _configure_worker_threads()
    output_dir = args.output_dir
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    if args.resume:
        seen_indices = existing_audit_indices(manifest_path)
    else:
        manifest_path.write_text("", encoding="utf-8")
        seen_indices = set()

    queued = 0
    scanned = 0
    skipped_existing = 0
    payloads: list[tuple[int, dict[str, Any], str, int, str, int, bool]] = []
    for input_index, row in iter_jsonl(args.input_jsonl):
        if input_index < int(args.start_index):
            continue
        if args.limit is not None and queued >= int(args.limit):
            break
        scanned += 1
        if row.get("qid") != "Q2_official" and row.get("family") != "Q2":
            continue
        if input_index in seen_indices:
            skipped_existing += 1
            continue
        if "frame_plan" not in row or "dataset_root" not in row:
            raise ValueError(
                f"input row {input_index} lacks frame_plan/dataset_root; use full judged teacher rows, "
                "not compact text-judge rows"
            )
        payloads.append(
            (
                input_index,
                row,
                str(output_dir),
                int(args.cell_width),
                str(args.image_format),
                int(args.jpeg_quality),
                bool(args.jpeg_optimize),
            )
        )
        queued += 1

    rendered = 0
    workers = max(1, int(args.workers))
    progress_every = max(1, int(args.progress_every))
    with manifest_path.open("a", encoding="utf-8") as handle:
        if workers == 1:
            iterator = map(render_one, payloads)
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            futures = [executor.submit(render_one, payload) for payload in payloads]
            iterator = (future.result() for future in as_completed(futures))
        try:
            for manifest_row in iterator:
                handle.write(json.dumps(manifest_row, ensure_ascii=True, sort_keys=True) + "\n")
                rendered += 1
                if rendered % progress_every == 0:
                    handle.flush()
                    print(
                        json.dumps(
                            {
                                "rendered_new": rendered,
                                "queued": queued,
                                "skipped_existing": skipped_existing,
                                "workers": workers,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        finally:
            if workers != 1:
                executor.shutdown(wait=True, cancel_futures=False)

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "rendered": rendered + skipped_existing,
        "rendered_new": rendered,
        "skipped_existing": skipped_existing,
        "queued": queued,
        "scanned_after_start": scanned,
        "start_index": int(args.start_index),
        "limit": args.limit,
        "cell_width": int(args.cell_width),
        "image_format": str(args.image_format),
        "jpeg_quality": int(args.jpeg_quality),
        "jpeg_optimize": bool(args.jpeg_optimize),
        "workers": workers,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
