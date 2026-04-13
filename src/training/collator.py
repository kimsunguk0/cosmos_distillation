"""Corpus collation for v1 text distillation with multimodal inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.data.consistency import ACTION_CLASSES


IGNORE_INDEX = -100
SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."
FRAME_OFFSETS_DEFAULT = (-0.3, -0.2, -0.1, 0.0)


def build_label_mask() -> str:
    """Return the label-mask policy used by the collator."""
    return "assistant_output_only"


def action_class_to_id(value: str | None) -> int:
    """Convert an action class into a compact integer label."""
    if value is None:
        return ACTION_CLASSES.index("unknown")
    value = str(value).strip().lower()
    return ACTION_CLASSES.index(value) if value in ACTION_CLASSES else ACTION_CLASSES.index("unknown")


def resolve_sample_path(sample: dict[str, Any], project_root: Path) -> Path:
    """Resolve a canonical sample path relative to the project root."""
    raw_path = Path(str(sample["input"]["canonical_sample_path"]))
    return raw_path if raw_path.is_absolute() else project_root / raw_path


def frame_offsets_from_sample(sample_dir: Path) -> list[float]:
    """Load the canonical frame offsets for one sample."""
    sample_meta_path = sample_dir / "sample_meta.json"
    if not sample_meta_path.exists():
        return list(FRAME_OFFSETS_DEFAULT)
    sample_meta = json.loads(sample_meta_path.read_text(encoding="utf-8"))
    return [float(value) for value in sample_meta.get("frame_offsets_sec", FRAME_OFFSETS_DEFAULT)]


def load_sample_images(sample: dict[str, Any], project_root: Path) -> list[Image.Image]:
    """Load the canonical multi-camera image set in a stable order."""
    sample_dir = resolve_sample_path(sample, project_root)
    frame_offsets = frame_offsets_from_sample(sample_dir)
    images: list[Image.Image] = []
    for camera_name in sample["input"]["camera_names"]:
        for offset in frame_offsets:
            image_path = sample_dir / "frames" / f"{camera_name}_t{offset:+.1f}.jpg"
            images.append(Image.open(image_path).convert("RGB"))
    return images


def format_history_text(sample_dir: Path) -> str:
    """Serialize ego history into a compact token block for the student input."""
    history_xyz = np.load(sample_dir / "ego_history_xyz.npy")
    steps = []
    for index, point in enumerate(history_xyz):
        steps.append(f"{index}:{point[0]:+.2f},{point[1]:+.2f},{point[2]:+.2f}")
    joined = " | ".join(steps)
    return f"<|traj_history_start|>{joined}<|traj_history_end|>"


def build_user_prompt(sample: dict[str, Any], project_root: Path) -> str:
    """Create the textual instruction paired with the image stack."""
    sample_dir = resolve_sample_path(sample, project_root)
    question = sample["input"]["question"].strip()
    history_text = format_history_text(sample_dir)
    return f"{history_text}\n<|question_start|>{question}<|question_end|>"


def build_messages(prompt_text: str, image_count: int, target_text: str | None = None) -> list[dict[str, Any]]:
    """Construct the multimodal chat message structure for one sample."""
    user_content: list[dict[str, Any]] = [{"type": "image"} for _ in range(image_count)]
    user_content.append({"type": "text", "text": prompt_text})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]
    assistant_text = "<|cot_start|>"
    if target_text is not None:
        assistant_text += f"{target_text.strip()}\n<|cot_end|>"
    messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
    return messages


def format_teacher_target(sample: dict[str, Any]) -> str | None:
    """Return the preferred teacher text target when present."""
    soft_target = sample.get("soft_target", {})
    teacher_text = (
        soft_target.get("teacher_short_reason")
        or soft_target.get("teacher_answer")
        or soft_target.get("teacher_long_cot")
    )
    if not teacher_text:
        return None
    return str(teacher_text).strip()


def _encode_messages(processor, messages_batch: list[list[dict[str, Any]]], image_batch: list[list[Image.Image]], max_length: int):
    texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        for messages in messages_batch
    ]
    return processor(
        text=texts,
        images=image_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def _labels_from_prompt_and_full(prompt_batch, full_batch) -> torch.Tensor:
    labels = full_batch["input_ids"].clone()
    labels[full_batch["attention_mask"] == 0] = IGNORE_INDEX
    prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
    for row_index, prompt_length in enumerate(prompt_lengths.tolist()):
        labels[row_index, : int(prompt_length)] = IGNORE_INDEX
    return labels


@dataclass(slots=True)
class DistillationCollator:
    tokenizer: Any
    processor: Any
    project_root: Path
    max_length: int = 4096

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_messages = []
        full_messages = []
        teacher_messages = []
        image_batch = []
        teacher_image_batch = []
        sample_ids: list[str] = []
        source_sample_ids: list[str] = []
        splits: list[str] = []
        action_labels = []
        teacher_action_labels = []
        teacher_action_present = []
        consistency_scores = []
        teacher_selection_scores = []
        hard_weights = []
        seq_weights = []
        logit_weights = []
        feat_weights = []
        rank_weights = []

        for sample in features:
            prompt_text = build_user_prompt(sample, self.project_root)
            target_text = sample["target"]["text"].strip()
            images = load_sample_images(sample, self.project_root)
            image_count = len(images)
            prompt_messages.append(build_messages(prompt_text, image_count))
            full_messages.append(build_messages(prompt_text, image_count, target_text))

            teacher_target = format_teacher_target(sample)
            if teacher_target is None:
                teacher_messages.append(None)
            else:
                teacher_messages.append(build_messages(prompt_text, image_count, teacher_target))

            image_batch.append(images)
            if teacher_target is not None:
                teacher_image_batch.append(images)

            derived = sample.get("derived", {})
            meta_action = (derived.get("meta_action_from_human") or {}).get("value")
            teacher_action = sample.get("soft_target", {}).get("teacher_action_class")

            action_labels.append(action_class_to_id(meta_action))
            teacher_action_labels.append(action_class_to_id(teacher_action))
            teacher_action_present.append(bool(teacher_action))
            consistency_scores.append(float(sample.get("consistency_score", 0.0)))
            teacher_selection_scores.append(float(sample.get("soft_target", {}).get("teacher_selection_score", 0.0)))

            weights = sample.get("weights", {})
            hard_weights.append(float(weights.get("hard_ce", 1.0)))
            seq_weights.append(float(weights.get("seq_kd", 0.0)))
            logit_weights.append(float(weights.get("logit_kd", 0.0)))
            feat_weights.append(float(weights.get("feat", 0.0)))
            rank_weights.append(float(weights.get("rank", 0.0)))
            sample_ids.append(str(sample["sample_id"]))
            source_sample_ids.append(str(sample.get("source_sample_id", sample["sample_id"])))
            splits.append(str(sample.get("split", "train")))

        prompt_batch = _encode_messages(self.processor, prompt_messages, image_batch, self.max_length)
        full_batch = _encode_messages(self.processor, full_messages, image_batch, self.max_length)
        labels = _labels_from_prompt_and_full(prompt_batch, full_batch)

        teacher_present = [item is not None for item in teacher_messages]
        teacher_batch = None
        if any(teacher_present):
            teacher_prompt_messages = [prompt_messages[idx] for idx, present in enumerate(teacher_present) if present]
            teacher_full_messages = [item for item in teacher_messages if item is not None]
            teacher_prompt_batch = _encode_messages(
                self.processor,
                teacher_prompt_messages,
                teacher_image_batch,
                self.max_length,
            )
            teacher_full_batch = _encode_messages(
                self.processor,
                teacher_full_messages,
                teacher_image_batch,
                self.max_length,
            )
            teacher_batch = {
                "input_ids": teacher_full_batch["input_ids"],
                "attention_mask": teacher_full_batch["attention_mask"],
                "labels": _labels_from_prompt_and_full(teacher_prompt_batch, teacher_full_batch),
                "pixel_values": teacher_full_batch["pixel_values"],
                "image_grid_thw": teacher_full_batch["image_grid_thw"],
                "present_mask": torch.tensor(teacher_present, dtype=torch.bool),
            }

        return {
            "input_ids": full_batch["input_ids"],
            "attention_mask": full_batch["attention_mask"],
            "pixel_values": full_batch["pixel_values"],
            "image_grid_thw": full_batch["image_grid_thw"],
            "labels": labels,
            "teacher_batch": teacher_batch,
            "sample_ids": sample_ids,
            "source_sample_ids": source_sample_ids,
            "splits": splits,
            "action_class_labels": torch.tensor(action_labels, dtype=torch.long),
            "teacher_action_class_labels": torch.tensor(teacher_action_labels, dtype=torch.long),
            "teacher_action_present_mask": torch.tensor(teacher_action_present, dtype=torch.bool),
            "consistency_scores": torch.tensor(consistency_scores, dtype=torch.float32),
            "teacher_selection_scores": torch.tensor(teacher_selection_scores, dtype=torch.float32),
            "hard_ce_weights": torch.tensor(hard_weights, dtype=torch.float32),
            "seq_kd_weights": torch.tensor(seq_weights, dtype=torch.float32),
            "logit_kd_weights": torch.tensor(logit_weights, dtype=torch.float32),
            "feat_weights": torch.tensor(feat_weights, dtype=torch.float32),
            "rank_weights": torch.tensor(rank_weights, dtype=torch.float32),
        }
