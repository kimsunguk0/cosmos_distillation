"""Constrained decoding helpers for structured CoT and trajectory spans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import LogitsProcessor, StoppingCriteria


def _single_token_id(tokenizer, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Expected a single-token encoding for {token!r}, got {token_ids}")
    return int(token_ids[0])


def _traj_token_ids(tokenizer) -> tuple[int, ...]:
    start = getattr(tokenizer, "traj_token_start_idx", None)
    end = getattr(tokenizer, "traj_token_end_idx", None)
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        try:
            start = tokenizer.convert_tokens_to_ids("<i0>")
            end = tokenizer.convert_tokens_to_ids("<i3999>")
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Tokenizer is missing traj token metadata and ids could not be inferred.") from exc
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        raise ValueError("Tokenizer is missing traj_token_start_idx / traj_token_end_idx metadata.")
    return tuple(range(int(start), int(end) + 1))


@dataclass(slots=True)
class TrajDecodingContract:
    prompt_lengths: tuple[int, ...]
    traj_token_count: int
    cot_end_id: int
    traj_start_id: int
    traj_end_id: int
    traj_token_ids: tuple[int, ...]

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        *,
        prompt_lengths: Sequence[int],
        traj_token_count: int,
    ) -> "TrajDecodingContract":
        return cls(
            prompt_lengths=tuple(int(length) for length in prompt_lengths),
            traj_token_count=int(traj_token_count),
            cot_end_id=_single_token_id(tokenizer, "<|cot_end|>"),
            traj_start_id=_single_token_id(tokenizer, "<|traj_future_start|>"),
            traj_end_id=_single_token_id(tokenizer, "<|traj_future_end|>"),
            traj_token_ids=_traj_token_ids(tokenizer),
        )


@dataclass(slots=True)
class TrajOnlyDecodingContract:
    prompt_lengths: tuple[int, ...]
    traj_token_count: int
    traj_end_id: int
    traj_token_ids: tuple[int, ...]

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer,
        *,
        prompt_lengths: Sequence[int],
        traj_token_count: int,
    ) -> "TrajOnlyDecodingContract":
        return cls(
            prompt_lengths=tuple(int(length) for length in prompt_lengths),
            traj_token_count=int(traj_token_count),
            traj_end_id=_single_token_id(tokenizer, "<|traj_future_end|>"),
            traj_token_ids=_traj_token_ids(tokenizer),
        )


def _row_state(row_tokens: list[int], contract: TrajDecodingContract) -> tuple[bool, bool, int]:
    cot_end_seen = False
    traj_started = False
    traj_tokens_emitted = 0
    for token_id in row_tokens:
        if not cot_end_seen:
            if token_id == contract.cot_end_id:
                cot_end_seen = True
            continue
        if not traj_started:
            if token_id == contract.traj_start_id:
                traj_started = True
            continue
        if token_id == contract.traj_end_id:
            return cot_end_seen, traj_started, traj_tokens_emitted
        if token_id in contract.traj_token_ids:
            traj_tokens_emitted += 1
    return cot_end_seen, traj_started, traj_tokens_emitted


class TrajSpanLogitsProcessor(LogitsProcessor):
    """Force `<traj_future_start> <i...>... <traj_future_end>` once CoT is closed."""

    def __init__(self, contract: TrajDecodingContract) -> None:
        self.contract = contract
        self._traj_token_id_set = set(contract.traj_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        constrained = scores.clone()
        for row_index in range(input_ids.shape[0]):
            prompt_length = self.contract.prompt_lengths[min(row_index, len(self.contract.prompt_lengths) - 1)]
            generated = input_ids[row_index, prompt_length:].tolist()
            cot_end_seen, traj_started, traj_tokens_emitted = _row_state(generated, self.contract)

            allowed: tuple[int, ...] | None = None
            if cot_end_seen and not traj_started:
                allowed = (self.contract.traj_start_id,)
            elif traj_started:
                if traj_tokens_emitted < self.contract.traj_token_count:
                    allowed = self.contract.traj_token_ids
                else:
                    allowed = (self.contract.traj_end_id,)

            if allowed is None:
                continue

            constrained[row_index].fill_(torch.finfo(constrained.dtype).min)
            allowed_indices = torch.tensor(allowed, device=constrained.device, dtype=torch.long)
            constrained[row_index, allowed_indices] = scores[row_index, allowed_indices]
        return constrained


class StopOnTrajEndCriteria(StoppingCriteria):
    """Stop generation after the first complete trajectory span is emitted."""

    def __init__(self, contract: TrajDecodingContract) -> None:
        self.contract = contract

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for row_index in range(input_ids.shape[0]):
            prompt_length = self.contract.prompt_lengths[min(row_index, len(self.contract.prompt_lengths) - 1)]
            generated = input_ids[row_index, prompt_length:].tolist()
            cot_end_seen = False
            traj_started = False
            for token_id in generated:
                if not cot_end_seen:
                    cot_end_seen = token_id == self.contract.cot_end_id
                    continue
                if not traj_started:
                    traj_started = token_id == self.contract.traj_start_id
                    continue
                if token_id == self.contract.traj_end_id:
                    break
            else:
                return False
        return True


class TrajOnlyLogitsProcessor(LogitsProcessor):
    """Force `<i...>` body tokens followed by `<|traj_future_end|>` for traj-only decoding."""

    def __init__(self, contract: TrajOnlyDecodingContract) -> None:
        self.contract = contract

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        constrained = scores.clone()
        for row_index in range(input_ids.shape[0]):
            prompt_length = self.contract.prompt_lengths[min(row_index, len(self.contract.prompt_lengths) - 1)]
            generated = input_ids[row_index, prompt_length:].tolist()
            if len(generated) < self.contract.traj_token_count:
                allowed = self.contract.traj_token_ids
            else:
                allowed = (self.contract.traj_end_id,)
            constrained[row_index].fill_(torch.finfo(constrained.dtype).min)
            allowed_indices = torch.tensor(allowed, device=constrained.device, dtype=torch.long)
            constrained[row_index, allowed_indices] = scores[row_index, allowed_indices]
        return constrained


class StopOnTrajOnlyEndCriteria(StoppingCriteria):
    """Stop traj-only generation after the required body tokens and `<|traj_future_end|>`."""

    def __init__(self, contract: TrajOnlyDecodingContract) -> None:
        self.contract = contract

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for row_index in range(input_ids.shape[0]):
            prompt_length = self.contract.prompt_lengths[min(row_index, len(self.contract.prompt_lengths) - 1)]
            generated = input_ids[row_index, prompt_length:].tolist()
            if len(generated) < self.contract.traj_token_count + 1:
                return False
            if generated[self.contract.traj_token_count] != self.contract.traj_end_id:
                return False
        return True
