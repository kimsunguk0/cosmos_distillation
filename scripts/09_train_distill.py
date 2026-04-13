#!/usr/bin/env python3
"""WP9-WP13 entrypoint: v1 distillation training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_STUDENT_MODEL = "/home/pm97/workspace/sukim/weights/cosmos-reason-2b"

import torch
import yaml
from torch.utils.data import DataLoader

from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora
from src.model.student_wrapper import (
    StudentWrapperConfig,
    build_student_model,
    load_student_processor,
    load_student_tokenizer,
)
from src.training.collator import DistillationCollator
from src.training.losses import DistillationLossWeights, get_stage_weights
from src.training.trainer import TrainerConfig, move_batch_to_device, run_train_step
from src.utils.seeds import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "strict_human_long_cot.jsonl",
    )
    parser.add_argument(
        "--stage-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train" / "stage_b.yaml",
    )
    parser.add_argument(
        "--student-model",
        default=DEFAULT_STUDENT_MODEL,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "stage_b_v1",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "train_summary.json",
    )
    parser.add_argument(
        "--data-only-dry-run",
        action="store_true",
        help="Inspect and batch the corpus without loading the student model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL corpus file."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def has_materialized_canonical_sample(record: dict) -> bool:
    """Return True when the record's canonical sample artifacts exist locally."""
    sample_path = PROJECT_ROOT / str(record["input"]["canonical_sample_path"])
    required = [
        sample_path / "ego_history_xyz.npy",
        sample_path / "ego_history_rot.npy",
        sample_path / "frames",
    ]
    return all(path.exists() for path in required)


def stage_weights_from_yaml(path: Path) -> tuple[TrainerConfig, DistillationLossWeights]:
    """Load stage config YAML."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    weights = config.get("loss_weights") or {}
    trainer_config = TrainerConfig(
        stage_name=str(config["stage_name"]),
        max_length=int(config.get("max_length", 4096)),
        bf16=bool(config.get("bf16", True)),
        batch_size=int(config.get("batch_size", 1)),
        learning_rate=float(config.get("learning_rate", 2e-5)),
    )
    loss_weights = DistillationLossWeights(
        hard_ce=float(weights.get("hard_ce", get_stage_weights(trainer_config.stage_name).hard_ce)),
        seq_kd=float(weights.get("seq_kd", get_stage_weights(trainer_config.stage_name).seq_kd)),
        logit_kd=float(weights.get("logit_kd", get_stage_weights(trainer_config.stage_name).logit_kd)),
        feat=float(weights.get("feat", get_stage_weights(trainer_config.stage_name).feat)),
        aux=float(weights.get("aux", get_stage_weights(trainer_config.stage_name).aux)),
        self_cons=float(weights.get("self_cons", get_stage_weights(trainer_config.stage_name).self_cons)),
        rank=float(weights.get("rank", get_stage_weights(trainer_config.stage_name).rank)),
    )
    return trainer_config, loss_weights


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_train_records = [record for record in load_jsonl(args.corpus_jsonl) if record.get("split") == "train"]
    records = [record for record in all_train_records if has_materialized_canonical_sample(record)]
    if args.max_train_samples is not None:
        records = records[: args.max_train_samples]

    trainer_cfg, loss_weights = stage_weights_from_yaml(args.stage_config)
    trainer_cfg.batch_size = args.batch_size
    trainer_cfg.learning_rate = args.learning_rate or trainer_cfg.learning_rate
    wrapper_cfg = StudentWrapperConfig(
        student_model_name=args.student_model,
        max_length=trainer_cfg.max_length,
        local_files_only=Path(args.student_model).expanduser().exists(),
    )
    tokenizer = load_student_tokenizer(wrapper_cfg)
    processor = load_student_processor(wrapper_cfg, tokenizer=tokenizer)
    collator = DistillationCollator(
        tokenizer=tokenizer,
        processor=processor,
        project_root=PROJECT_ROOT,
        max_length=trainer_cfg.max_length,
    )
    dataloader = DataLoader(records, batch_size=args.batch_size, shuffle=not args.data_only_dry_run, collate_fn=collator)

    teacher_ready = sum(1 for record in records if record.get("soft_target", {}).get("teacher_short_reason"))
    if args.data_only_dry_run:
        first_batch = next(iter(dataloader), None)
        summary = {
            "mode": "data_only_dry_run",
            "corpus_jsonl": str(args.corpus_jsonl),
            "train_records": len(records),
            "train_records_with_materialized_samples": len(records),
            "all_train_records": len(all_train_records),
            "teacher_ready_records": teacher_ready,
            "stage_name": trainer_cfg.stage_name,
            "batch_size": args.batch_size,
            "max_length": trainer_cfg.max_length,
            "first_batch_shapes": {
                "input_ids": list(first_batch["input_ids"].shape) if first_batch is not None else None,
                "labels": list(first_batch["labels"].shape) if first_batch is not None else None,
                "pixel_values": list(first_batch["pixel_values"].shape) if first_batch is not None else None,
                "image_grid_thw": list(first_batch["image_grid_thw"].shape) if first_batch is not None else None,
            },
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_student_model(wrapper_cfg, tokenizer)
    if hasattr(model.backbone, "gradient_checkpointing_enable"):
        try:
            model.backbone.gradient_checkpointing_enable()
        except Exception:  # noqa: BLE001
            pass
    model.backbone = maybe_apply_lora(
        model.backbone,
        LoraConfigSpec(),
        enabled=not args.disable_lora,
    )
    if hasattr(model.backbone, "enable_input_require_grads"):
        try:
            model.backbone.enable_input_require_grads()
        except Exception:  # noqa: BLE001
            pass
    model = model.to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=trainer_cfg.learning_rate)

    metrics_path = args.output_dir / "metrics.jsonl"
    global_step = 0
    started_at = time.time()
    with metrics_path.open("w", encoding="utf-8") as metrics_handle:
        for batch in dataloader:
            if args.max_steps is not None and global_step >= args.max_steps:
                break
            batch = move_batch_to_device(batch, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if trainer_cfg.bf16 and device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                loss, logs = run_train_step(model, batch, loss_weights)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
            optimizer.step()
            global_step += 1

            row = {
                "timestamp": time.time(),
                "phase": "train",
                "global_step": global_step,
                "logs": {**logs, "grad_norm": float(grad_norm.detach().cpu())},
            }
            metrics_handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            metrics_handle.flush()

    summary = {
        "mode": "train",
        "student_model": args.student_model,
        "stage_name": trainer_cfg.stage_name,
        "train_records": len(records),
        "all_train_records": len(all_train_records),
        "teacher_ready_records": teacher_ready,
        "global_steps": global_step,
        "learning_rate": trainer_cfg.learning_rate,
        "use_lora": not args.disable_lora,
        "elapsed_sec": round(time.time() - started_at, 3),
        "metrics_path": str(metrics_path),
        "output_dir": str(args.output_dir),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
