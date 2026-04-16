import pytest
import torch

from src.training.collator import IGNORE_INDEX, _build_traj_only_target_layout, build_messages, _labels_from_prompt_and_full


def _batch(input_ids: list[list[int]], attention_mask: list[list[int]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def test_labels_from_prompt_and_full_keeps_completion_tokens_trainable() -> None:
    prompt_batch = _batch([[10, 11, 12]], [[1, 1, 1]])
    full_batch = _batch([[10, 11, 12, 13, 14]], [[1, 1, 1, 1, 1]])

    labels = _labels_from_prompt_and_full(prompt_batch, full_batch)

    assert labels.tolist() == [[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 13, 14]]


def test_labels_from_prompt_and_full_rejects_prefix_mismatch() -> None:
    prompt_batch = _batch([[10, 11, 99]], [[1, 1, 1]])
    full_batch = _batch([[10, 11, 12, 13]], [[1, 1, 1, 1]])

    with pytest.raises(ValueError, match="Prompt/full chat template prefixes diverged"):
        _labels_from_prompt_and_full(prompt_batch, full_batch)


class _StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        mapping = {
            "<|traj_future_end|>": [9],
        }
        if text in mapping:
            return mapping[text]
        return [1] * len(text)


def test_build_messages_supports_traj_only_assistant_prefix() -> None:
    messages = build_messages("prompt", 1, assistant_prefix="<|traj_future_start|>")
    assert messages[-1]["content"][0]["text"] == "<|traj_future_start|>"


def test_build_traj_only_target_layout_tracks_traj_content_without_prefix() -> None:
    layout = _build_traj_only_target_layout(_StubTokenizer(), [1, 2, 3])
    assert layout.cot_span_len == 0
    assert layout.traj_prefix_len == 0
    assert layout.traj_content_len == 3
    assert layout.traj_suffix_len == 1
