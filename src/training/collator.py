"""Corpus collation for v1 text distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.data.consistency import ACTION_CLASSES


IGNORE_INDEX = -100


def build_label_mask() -> str:
    """Return the label-mask policy used by the collator."""
    return "assistant_output_only"


def action_class_to_id(value: str | None) -> int:
    """Convert an action class into a compact integer label."""
    if value is None:
        return ACTION_CLASSES.index("unknown")
    value = str(value).strip().lower()
    return ACTION_CLASSES.index(value) if value in ACTION_CLASSES else ACTION_CLASSES.index("unknown")


def format_corpus_prompt(sample: dict[str, Any]) -> tuple[str, str]:
    """Build the prompt and target text for a corpus sample."""
    question = sample["input"]["question"].strip()
    prompt = f"<|question_start|>{question}<|question_end|>\n<|cot_start|>"
    target = sample["target"]["text"].strip() + "\n<|cot_end|>"
    return prompt, target


def format_teacher_target(sample: dict[str, Any]) -> str | None:
    """Return the preferred teacher text target when present."""
    soft_target = sample.get("soft_target", {})
    teacher_text = soft_target.get("teacher_short_reason") or soft_target.get("teacher_answer")
    if not teacher_text:
        return None
    return str(teacher_text).strip() + "\n<|cot_end|>"


def _tokenize_prompt_target(tokenizer, prompt: str, target: str, max_length: int) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + target_ids
    input_ids = input_ids[:max_length]
    attention_mask = [1] * len(input_ids)
    labels = [IGNORE_INDEX] * min(len(prompt_ids), len(input_ids))
    if len(input_ids) > len(prompt_ids):
        labels.extend(input_ids[len(prompt_ids) :])
    labels = labels[:max_length]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


@dataclass(slots=True)
class DistillationCollator:
    tokenizer: Any
    max_length: int = 4096

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch_inputs = []
        teacher_inputs = []
        sample_ids: list[str] = []
        source_sample_ids: list[str] = []
        splits: list[str] = []
        action_labels = []
        consistency_scores = []
        hard_weights = []
        seq_weights = []
        logit_weights = []
        feat_weights = []

        for sample in features:
            prompt, target = format_corpus_prompt(sample)
            batch_inputs.append(_tokenize_prompt_target(self.tokenizer, prompt, target, self.max_length))
            teacher_target = format_teacher_target(sample)
            if teacher_target is None:
                teacher_inputs.append(None)
            else:
                teacher_inputs.append(_tokenize_prompt_target(self.tokenizer, prompt, teacher_target, self.max_length))

            derived = sample.get("derived", {})
            meta_action = (derived.get("meta_action_from_human") or {}).get("value")
            action_labels.append(action_class_to_id(meta_action))
            consistency_scores.append(float(sample.get("consistency_score", 0.0)))

            weights = sample.get("weights", {})
            hard_weights.append(float(weights.get("hard_ce", 1.0)))
            seq_weights.append(float(weights.get("seq_kd", 0.0)))
            logit_weights.append(float(weights.get("logit_kd", 0.0)))
            feat_weights.append(float(weights.get("feat", 0.0)))
            sample_ids.append(str(sample["sample_id"]))
            source_sample_ids.append(str(sample.get("source_sample_id", sample["sample_id"])))
            splits.append(str(sample.get("split", "train")))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch_inputs],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch_inputs],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch_inputs],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        teacher_present = [item is not None for item in teacher_inputs]
        teacher_batch = None
        if any(teacher_present):
            teacher_real = [item for item in teacher_inputs if item is not None]
            teacher_batch = {
                "input_ids": torch.nn.utils.rnn.pad_sequence(
                    [item["input_ids"] for item in teacher_real],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                ),
                "attention_mask": torch.nn.utils.rnn.pad_sequence(
                    [item["attention_mask"] for item in teacher_real],
                    batch_first=True,
                    padding_value=0,
                ),
                "labels": torch.nn.utils.rnn.pad_sequence(
                    [item["labels"] for item in teacher_real],
                    batch_first=True,
                    padding_value=IGNORE_INDEX,
                ),
                "present_mask": torch.tensor(teacher_present, dtype=torch.bool),
            }

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "teacher_batch": teacher_batch,
            "sample_ids": sample_ids,
            "source_sample_ids": source_sample_ids,
            "splits": splits,
            "action_class_labels": torch.tensor(action_labels, dtype=torch.long),
            "consistency_scores": torch.tensor(consistency_scores, dtype=torch.float32),
            "hard_ce_weights": torch.tensor(hard_weights, dtype=torch.float32),
            "seq_kd_weights": torch.tensor(seq_weights, dtype=torch.float32),
            "logit_kd_weights": torch.tensor(logit_weights, dtype=torch.float32),
            "feat_weights": torch.tensor(feat_weights, dtype=torch.float32),
        }
