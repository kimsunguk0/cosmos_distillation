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
TEXT_TARGET_SPECS = (
    ("teacher_long_cot", 1.0),
    ("teacher_answer", 0.2),
    ("teacher_short_reason", 0.1),
    ("teacher_structured_json", 0.0),
)


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


def teacher_text_targets(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Return teacher text targets without collapsing them into a single string."""
    soft_target = sample.get("soft_target", {})
    signal_targets = soft_target.get("teacher_signal_targets") or {}
    target_weights = soft_target.get("teacher_target_weights") or {}
    targets: list[dict[str, Any]] = []
    for field_name, base_weight in TEXT_TARGET_SPECS:
        text = soft_target.get(field_name)
        if not text:
            continue
        source = soft_target.get(f"{field_name}_source", "missing")
        targets.append(
            {
                "field_name": field_name,
                "text": str(text).strip(),
                "base_weight": float(target_weights.get(field_name, base_weight)),
                "source": source,
                "signal_target": signal_targets.get(field_name) or {},
            }
        )
    return targets


def load_sparse_signal(signal_target: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """Load sparse top-k logits and pooled hidden for one teacher target field."""
    logits_path = signal_target.get("logits_path")
    hidden_path = signal_target.get("hidden_path")
    if not logits_path or not hidden_path:
        return None
    logits_npz = np.load(logits_path)
    pooled_hidden = np.load(hidden_path)
    return {
        "topk_indices": logits_npz["topk_indices"],
        "topk_logits": logits_npz["topk_logits"],
        "target_token_ids": logits_npz["target_token_ids"],
        "target_token_count": np.asarray(logits_npz["target_token_count"]).reshape(-1)[0],
        "pooled_hidden": pooled_hidden,
    }


def pad_sparse_signal_batch(signal_items: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    """Pad sparse signal arrays for a teacher target batch."""
    max_tokens = max(int(item["target_token_count"]) for item in signal_items)
    topk = int(signal_items[0]["topk_indices"].shape[-1])
    hidden_dim = int(signal_items[0]["pooled_hidden"].shape[-1])
    batch_size = len(signal_items)
    topk_indices = torch.zeros((batch_size, max_tokens, topk), dtype=torch.long)
    topk_logits = torch.zeros((batch_size, max_tokens, topk), dtype=torch.float32)
    token_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    pooled_hidden = torch.zeros((batch_size, hidden_dim), dtype=torch.float32)
    for row_index, item in enumerate(signal_items):
        token_count = int(item["target_token_count"])
        topk_indices[row_index, :token_count] = torch.from_numpy(item["topk_indices"]).long()
        topk_logits[row_index, :token_count] = torch.from_numpy(item["topk_logits"]).float()
        token_mask[row_index, :token_count] = True
        pooled_hidden[row_index] = torch.from_numpy(item["pooled_hidden"]).float()
    return {
        "teacher_topk_indices": topk_indices,
        "teacher_topk_logits": topk_logits,
        "teacher_topk_mask": token_mask,
        "teacher_pooled_hidden": pooled_hidden,
    }


@dataclass(slots=True)
class DistillationCollator:
    tokenizer: Any
    processor: Any
    project_root: Path
    max_length: int = 4096

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_messages = []
        full_messages = []
        image_batch = []
        sample_ids: list[str] = []
        source_sample_ids: list[str] = []
        splits: list[str] = []
        action_labels = []
        teacher_action_labels = []
        teacher_action_present = []
        consistency_scores = []
        teacher_selection_scores = []
        hard_weights = []
        rank_weights = []
        teacher_targets_by_field: dict[str, list[dict[str, Any] | None]] = {
            field_name: [] for field_name, _ in TEXT_TARGET_SPECS
        }

        for sample in features:
            prompt_text = build_user_prompt(sample, self.project_root)
            target_text = sample["target"]["text"].strip()
            images = load_sample_images(sample, self.project_root)
            image_count = len(images)
            prompt_messages.append(build_messages(prompt_text, image_count))
            full_messages.append(build_messages(prompt_text, image_count, target_text))
            image_batch.append(images)

            available_targets = {item["field_name"]: item for item in teacher_text_targets(sample)}
            for field_name in teacher_targets_by_field:
                teacher_targets_by_field[field_name].append(available_targets.get(field_name))

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
            rank_weights.append(float(weights.get("rank", 0.0)))
            sample_ids.append(str(sample["sample_id"]))
            source_sample_ids.append(str(sample.get("source_sample_id", sample["sample_id"])))
            splits.append(str(sample.get("split", "train")))

        prompt_batch = _encode_messages(self.processor, prompt_messages, image_batch, self.max_length)
        full_batch = _encode_messages(self.processor, full_messages, image_batch, self.max_length)
        labels = _labels_from_prompt_and_full(prompt_batch, full_batch)

        teacher_target_batches = []
        for field_name, _ in TEXT_TARGET_SPECS:
            targets_for_field = teacher_targets_by_field[field_name]
            present_mask = torch.tensor([item is not None for item in targets_for_field], dtype=torch.bool)
            if not present_mask.any():
                continue
            field_prompt_messages = [prompt_messages[idx] for idx, item in enumerate(targets_for_field) if item is not None]
            field_full_messages = [
                build_messages(
                    build_user_prompt(features[idx], self.project_root),
                    len(image_batch[idx]),
                    item["text"],
                )
                for idx, item in enumerate(targets_for_field)
                if item is not None
            ]
            field_images = [image_batch[idx] for idx, item in enumerate(targets_for_field) if item is not None]
            prompt_encoded = _encode_messages(self.processor, field_prompt_messages, field_images, self.max_length)
            full_encoded = _encode_messages(self.processor, field_full_messages, field_images, self.max_length)
            field_batch = {
                "field_name": field_name,
                "input_ids": full_encoded["input_ids"],
                "attention_mask": full_encoded["attention_mask"],
                "labels": _labels_from_prompt_and_full(prompt_encoded, full_encoded),
                "pixel_values": full_encoded["pixel_values"],
                "image_grid_thw": full_encoded["image_grid_thw"],
                "present_mask": present_mask,
                "weights": torch.tensor(
                    [float(item["base_weight"]) for item in targets_for_field if item is not None],
                    dtype=torch.float32,
                ),
                "sources": [str(item["source"]) for item in targets_for_field if item is not None],
            }
            signal_items = []
            signal_present = True
            for item in targets_for_field:
                if item is None:
                    continue
                signal = load_sparse_signal(item["signal_target"])
                if signal is None:
                    signal_present = False
                    break
                signal_items.append(signal)
            if signal_present and signal_items:
                field_batch.update(pad_sparse_signal_batch(signal_items))
            teacher_target_batches.append(field_batch)

        return {
            "input_ids": full_batch["input_ids"],
            "attention_mask": full_batch["attention_mask"],
            "pixel_values": full_batch["pixel_values"],
            "image_grid_thw": full_batch["image_grid_thw"],
            "labels": labels,
            "teacher_target_batches": teacher_target_batches,
            "sample_ids": sample_ids,
            "source_sample_ids": source_sample_ids,
            "splits": splits,
            "action_class_labels": torch.tensor(action_labels, dtype=torch.long),
            "teacher_action_class_labels": torch.tensor(teacher_action_labels, dtype=torch.long),
            "teacher_action_present_mask": torch.tensor(teacher_action_present, dtype=torch.bool),
            "consistency_scores": torch.tensor(consistency_scores, dtype=torch.float32),
            "teacher_selection_scores": torch.tensor(teacher_selection_scores, dtype=torch.float32),
            "hard_ce_weights": torch.tensor(hard_weights, dtype=torch.float32),
            "rank_weights": torch.tensor(rank_weights, dtype=torch.float32),
        }
