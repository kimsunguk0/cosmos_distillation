#!/usr/bin/env python3
"""Static readiness checks for the Step-B weekend queue."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not parse to a mapping")
    return data


def expect(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def count_jsonl_rows(path: Path, *, limit: int | None = None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in handle:
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def main() -> int:
    errors: list[str] = []
    required_paths = [
        "scripts/09_train_distill.py",
        "scripts/launch_stepb_weekend_queue.sh",
        "scripts/stepb_eval_ci.py",
        "scripts/stepb_prepare_eval_slices.py",
        "data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl",
        "data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl",
        "data/corpus/benchmark_semantic_test_clipdisjoint_cap50_seed43.jsonl",
        "/home/pm97/workspace/sukim/base_weights/Cosmos-Reason2-8B",
        "configs/train/stepb_ladder_r0_c_bp3_lora_lr1e4.yaml",
        "configs/train/stepb_ladder_r1_tailkl.yaml",
        "configs/train/stepb_lprime_lora_lr1e4_200k_e1.yaml",
    ]
    for rel in required_paths:
        expect((ROOT / rel).exists(), f"missing required path: {rel}", errors)

    train_script = (ROOT / "scripts/09_train_distill.py").read_text(encoding="utf-8")
    expect("--max-keep-checkpoints" in train_script, "09_train_distill.py lacks --max-keep-checkpoints", errors)
    expect("prune_step_checkpoints" in train_script, "09_train_distill.py lacks checkpoint pruning helper", errors)

    r1 = load_yaml(ROOT / "configs/train/stepb_ladder_r1_tailkl.yaml")
    expect(float(r1.get("learning_rate")) == 3.0e-5, "R1 FullFT LR is not 3e-5", errors)
    expect(bool(r1.get("lora", {}).get("train_lm_head_token_rows")) is False, "R1 FullFT still enables LoRA token rows", errors)
    expect(float(r1.get("loss_weights", {}).get("gt_cot_loss")) == 0.1, "R1 CoT CE weight is not 0.1", errors)
    expect(float(r1.get("loss_weights", {}).get("traj_loss")) == 1.0, "R1 traj CE weight is not 1.0", errors)
    expect(float(r1.get("loss_weights", {}).get("teacher_traj_topk_kd_loss")) == 0.5, "R1 traj KD weight is not 0.5", errors)
    expect(float(r1.get("teacher_traj_topk_kd", {}).get("temperature")) == 1.0, "R1 KD temperature is not 1.0", errors)
    expect(bool(r1.get("teacher_traj_topk_kd", {}).get("tail_bucket")) is True, "R1 tail bucket is not enabled", errors)
    opt = dict(r1.get("optimization") or {})
    expect(bool(opt.get("freeze_visual_tower")) is True, "R1 visual tower is not frozen", errors)
    expect(int(opt.get("language_layers_from", -1)) == 0, "R1 language stack is not fully unfrozen", errors)
    expect(bool(opt.get("unfreeze_multimodal_projector")) is True, "R1 multimodal projector is not trainable", errors)

    soup = load_yaml(ROOT / "configs/train/stepb_ladder_r0_c_bp3_lora_lr1e4.yaml")
    expect(float(soup.get("learning_rate")) == 1.0e-4, "soup@1e-4 LR is not 1e-4", errors)
    expect(bool(soup.get("scheduled_sampling", {}).get("enabled")) is False, "soup@1e-4 scheduled sampling is enabled", errors)

    scale = load_yaml(ROOT / "configs/train/stepb_lprime_lora_lr1e4_200k_e1.yaml")
    expect(float(scale.get("epochs")) == 1.0, "L'@200K epochs is not 1.0", errors)
    expect(float(scale.get("learning_rate")) == 1.0e-4, "L'@200K LR is not 1e-4", errors)

    for cfg_path in sorted((ROOT / "configs/train").glob("stepb_*.yaml")):
        cfg = load_yaml(cfg_path)
        decode = dict(cfg.get("decode_eval") or {})
        if decode.get("enabled", False):
            expect(decode.get("do_sample") is False, f"{cfg_path.name} decode_eval.do_sample is not false", errors)
            expect(
                decode.get("reference_id") == "teacher_greedy_ref_v1",
                f"{cfg_path.name} decode_eval.reference_id is not teacher_greedy_ref_v1",
                errors,
            )
            reference_jsonl = decode.get("reference_jsonl")
            expect(bool(reference_jsonl), f"{cfg_path.name} decode_eval.reference_jsonl is missing", errors)
            if reference_jsonl:
                reference_path = ROOT / str(reference_jsonl)
                expect(reference_path.exists(), f"{cfg_path.name} reference_jsonl is missing: {reference_jsonl}", errors)
                expected_rows = int(decode.get("num_samples", 0) or 0)
                if reference_path.exists() and expected_rows > 0:
                    actual_rows = count_jsonl_rows(reference_path, limit=expected_rows)
                    expect(
                        actual_rows >= expected_rows,
                        f"{cfg_path.name} reference_jsonl has {actual_rows}/{expected_rows} rows: {reference_jsonl}",
                        errors,
                    )
            expect(
                int(decode.get("max_new_tokens", 0) or 0) >= 320,
                f"{cfg_path.name} decode_eval.max_new_tokens is below 320",
                errors,
            )

    if errors:
        print("[ready] FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("[ready] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
