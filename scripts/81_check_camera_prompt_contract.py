#!/usr/bin/env python3
"""Check that the student collator preserves Alpamayo's 4V prompt contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_ALPAMAYO_SRC = WORKSPACE_ROOT / "alpamayo_repo" / "alpamayo1.5" / "src"
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl"
DEFAULT_PROCESSOR = WORKSPACE_ROOT / "base_weights" / "cosmos-reason-2b"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.collator import (  # noqa: E402
    build_messages,
    build_user_prompt,
    load_sample_images,
    resolve_camera_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--alpamayo-src", type=Path, default=DEFAULT_ALPAMAYO_SRC)
    parser.add_argument("--processor-model", type=Path, default=DEFAULT_PROCESSOR)
    parser.add_argument("--prompt-text-style", default="official_alpamayo")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "no_nav_distill" / "camera_prompt_contract_sample.json",
    )
    parser.add_argument(
        "--strict-current-prompt",
        action="store_true",
        help="Also fail when the current training prompt text differs from helper.create_message().",
    )
    return parser.parse_args()


def load_record(path: Path, sample_index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == sample_index:
                return json.loads(line)
    raise IndexError(f"sample_index={sample_index} is outside {path}")


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = []
        for item in message.get("content", []):
            if item.get("type") == "image":
                content.append({"type": "image"})
            else:
                content.append({"type": "text", "text": str(item.get("text", ""))})
        normalized.append({"role": message["role"], "content": content})
    return normalized


def user_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(messages[1]["content"])


def image_count(messages: list[dict[str, Any]]) -> int:
    return sum(1 for item in user_content(messages) if item.get("type") == "image")


def camera_labels(messages: list[dict[str, Any]]) -> list[str]:
    return [
        str(item.get("text", ""))
        for item in user_content(messages)
        if item.get("type") == "text" and str(item.get("text", "")).endswith("camera: ")
    ]


def frame_labels(messages: list[dict[str, Any]]) -> list[str]:
    return [
        str(item.get("text", ""))
        for item in user_content(messages)
        if item.get("type") == "text" and str(item.get("text", "")).startswith("frame ")
    ]


def assistant_prefix(messages: list[dict[str, Any]]) -> str:
    return str(messages[-1]["content"][0].get("text", ""))


def render_and_tokenize(processor: Any, messages: list[dict[str, Any]]) -> tuple[str, list[int]]:
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    token_ids = processor.tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return str(rendered), [int(token_id) for token_id in token_ids]


def fail_reasons(checks: dict[str, bool]) -> list[str]:
    return [name for name, ok in checks.items() if not ok]


def main() -> int:
    args = parse_args()
    if str(args.alpamayo_src) not in sys.path:
        sys.path.insert(0, str(args.alpamayo_src))
    from alpamayo1_5 import helper  # noqa: WPS433

    sample = load_record(args.corpus_jsonl, args.sample_index)
    images = load_sample_images(sample, PROJECT_ROOT)
    camera_indices = resolve_camera_indices(sample, PROJECT_ROOT, image_count=len(images))
    num_frames_per_camera = max(len(images) // max(len(camera_indices), 1), 1)
    frames = torch.zeros((len(images), 3, 320, 576), dtype=torch.uint8)

    official_teacher = normalize_messages(
        helper.create_message(
            frames=frames,
            camera_indices=torch.tensor(camera_indices, dtype=torch.int64),
            num_frames_per_camera=num_frames_per_camera,
        )
    )
    official_prompt_text = official_teacher[1]["content"][-1]["text"]

    student_with_official_prompt = build_messages(
        official_prompt_text,
        len(images),
        assistant_prefix="<|cot_start|>",
        image_prompt_style="camera_labeled",
        camera_indices=camera_indices,
        num_frames_per_camera=num_frames_per_camera,
    )
    student_with_official_prompt = normalize_messages(student_with_official_prompt)

    current_prompt_text = build_user_prompt(sample, PROJECT_ROOT, prompt_text_style=args.prompt_text_style)
    current_student = normalize_messages(
        build_messages(
            current_prompt_text,
            len(images),
            assistant_prefix="<|cot_start|>",
            image_prompt_style="camera_labeled",
            camera_indices=camera_indices,
            num_frames_per_camera=num_frames_per_camera,
        )
    )

    processor = AutoProcessor.from_pretrained(
        str(args.processor_model),
        trust_remote_code=True,
        local_files_only=Path(args.processor_model).exists(),
    )
    teacher_rendered, teacher_token_ids = render_and_tokenize(processor, official_teacher)
    student_rendered, student_token_ids = render_and_tokenize(processor, student_with_official_prompt)
    current_rendered, current_token_ids = render_and_tokenize(processor, current_student)

    expected_camera_labels = [
        "Front left camera: ",
        "Front camera: ",
        "Front right camera: ",
        "Front telephoto camera: ",
    ]
    expected_frame_labels = ["frame 0 ", "frame 1 ", "frame 2 ", "frame 3 "] * 4
    image_names = list((sample.get("input") or {}).get("image_names") or [])

    helper_vs_builder_checks = {
        "image_placeholder_count_is_16": image_count(official_teacher) == 16
        and image_count(student_with_official_prompt) == 16,
        "camera_label_order": camera_labels(official_teacher) == expected_camera_labels
        and camera_labels(student_with_official_prompt) == expected_camera_labels,
        "frame_index_order": frame_labels(official_teacher) == expected_frame_labels
        and frame_labels(student_with_official_prompt) == expected_frame_labels,
        "materialized_cam3_maps_to_original_camera_6": len(camera_indices) >= 4 and int(camera_indices[3]) == 6,
        "materialized_cam3_images_are_fourth_group": image_names[12:16]
        in (
            ["cam3_f0.png", "cam3_f1.png", "cam3_f2.png", "cam3_f3.png"],
            [],
        ),
        "prompt_text_after_image_content": user_content(official_teacher)[-1].get("text") == official_prompt_text
        and user_content(student_with_official_prompt)[-1].get("text") == official_prompt_text,
        "assistant_prefix_is_cot_start": assistant_prefix(official_teacher) == "<|cot_start|>"
        and assistant_prefix(student_with_official_prompt) == "<|cot_start|>",
        "apply_chat_template_rendered_text_identical": teacher_rendered == student_rendered,
        "apply_chat_template_token_ids_identical": teacher_token_ids == student_token_ids,
    }
    current_prompt_checks = {
        "current_training_prompt_text_matches_official_helper": current_prompt_text == official_prompt_text,
        "current_training_rendered_text_matches_official_helper": current_rendered == teacher_rendered,
        "current_training_token_ids_match_official_helper": current_token_ids == teacher_token_ids,
    }

    summary = {
        "schema_version": "alpamayo_camera_prompt_contract_v1",
        "sample_id": sample.get("sample_id"),
        "corpus_jsonl": str(args.corpus_jsonl),
        "sample_index": args.sample_index,
        "processor_model": str(args.processor_model),
        "camera_indices": camera_indices,
        "num_images": len(images),
        "num_frames_per_camera": num_frames_per_camera,
        "image_names_12_15": image_names[12:16],
        "teacher_camera_labels": camera_labels(official_teacher),
        "student_camera_labels": camera_labels(student_with_official_prompt),
        "teacher_frame_labels": frame_labels(official_teacher),
        "student_frame_labels": frame_labels(student_with_official_prompt),
        "official_prompt_text_prefix": official_prompt_text[:240],
        "current_training_prompt_text_prefix": current_prompt_text[:240],
        "teacher_rendered_token_count": len(teacher_token_ids),
        "student_builder_rendered_token_count": len(student_token_ids),
        "current_training_rendered_token_count": len(current_token_ids),
        "helper_vs_builder_checks": helper_vs_builder_checks,
        "current_training_prompt_checks": current_prompt_checks,
        "helper_vs_builder_failed_checks": fail_reasons(helper_vs_builder_checks),
        "current_training_prompt_failed_checks": fail_reasons(current_prompt_checks),
        "strict_current_prompt": bool(args.strict_current_prompt),
        "prompt_text_style": args.prompt_text_style,
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    failed = fail_reasons(helper_vs_builder_checks)
    if args.strict_current_prompt:
        failed += [f"current::{name}" for name in fail_reasons(current_prompt_checks)]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
