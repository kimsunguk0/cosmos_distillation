import torch

from src.inference.decoding import (
    StopOnTrajEndCriteria,
    StopOnTrajOnlyEndCriteria,
    TrajDecodingContract,
    TrajOnlyDecodingContract,
    TrajOnlyLogitsProcessor,
    TrajSpanLogitsProcessor,
)


def _contract(*, traj_token_count: int = 2) -> TrajDecodingContract:
    return TrajDecodingContract(
        prompt_lengths=(3,),
        traj_token_count=traj_token_count,
        cot_end_id=10,
        traj_start_id=11,
        traj_end_id=12,
        traj_token_ids=(20, 21, 22),
    )


def _traj_only_contract(*, traj_token_count: int = 2) -> TrajOnlyDecodingContract:
    return TrajOnlyDecodingContract(
        prompt_lengths=(3,),
        traj_token_count=traj_token_count,
        traj_end_id=12,
        traj_token_ids=(20, 21, 22),
    )


def test_traj_processor_forces_traj_start_after_cot_end() -> None:
    contract = _contract()
    processor = TrajSpanLogitsProcessor(contract)
    input_ids = torch.tensor([[1, 2, 3, 10]], dtype=torch.long)
    scores = torch.arange(30, dtype=torch.float32).unsqueeze(0)
    constrained = processor(input_ids, scores)
    best_token = int(constrained[0].argmax().item())
    assert best_token == contract.traj_start_id


def test_traj_processor_forces_traj_body_then_end() -> None:
    contract = _contract(traj_token_count=2)
    processor = TrajSpanLogitsProcessor(contract)

    body_ids = torch.tensor([[1, 2, 3, 10, 11, 20]], dtype=torch.long)
    body_scores = torch.arange(30, dtype=torch.float32).unsqueeze(0)
    constrained_body = processor(body_ids, body_scores)
    assert int(constrained_body[0].argmax().item()) in contract.traj_token_ids

    end_ids = torch.tensor([[1, 2, 3, 10, 11, 20, 21]], dtype=torch.long)
    end_scores = torch.arange(30, dtype=torch.float32).unsqueeze(0)
    constrained_end = processor(end_ids, end_scores)
    assert int(constrained_end[0].argmax().item()) == contract.traj_end_id


def test_stop_criteria_triggers_after_traj_end() -> None:
    contract = _contract()
    criteria = StopOnTrajEndCriteria(contract)
    unfinished = torch.tensor([[1, 2, 3, 10, 11, 20]], dtype=torch.long)
    finished = torch.tensor([[1, 2, 3, 10, 11, 20, 21, 12]], dtype=torch.long)
    dummy_scores = torch.zeros((1, 30), dtype=torch.float32)
    assert criteria(unfinished, dummy_scores) is False
    assert criteria(finished, dummy_scores) is True


def test_traj_only_processor_forces_body_then_end() -> None:
    contract = _traj_only_contract(traj_token_count=2)
    processor = TrajOnlyLogitsProcessor(contract)
    body_ids = torch.tensor([[1, 2, 3, 20]], dtype=torch.long)
    scores = torch.arange(30, dtype=torch.float32).unsqueeze(0)
    constrained_body = processor(body_ids, scores)
    assert int(constrained_body[0].argmax().item()) in contract.traj_token_ids

    end_ids = torch.tensor([[1, 2, 3, 20, 21]], dtype=torch.long)
    constrained_end = processor(end_ids, scores)
    assert int(constrained_end[0].argmax().item()) == contract.traj_end_id


def test_traj_only_stop_criteria_waits_for_end_token() -> None:
    contract = _traj_only_contract(traj_token_count=2)
    criteria = StopOnTrajOnlyEndCriteria(contract)
    unfinished = torch.tensor([[1, 2, 3, 20, 21]], dtype=torch.long)
    finished = torch.tensor([[1, 2, 3, 20, 21, 12]], dtype=torch.long)
    dummy_scores = torch.zeros((1, 30), dtype=torch.float32)
    assert criteria(unfinished, dummy_scores) is False
    assert criteria(finished, dummy_scores) is True
