#!/usr/bin/env python3
"""Run the Stage 0 token-row trainability gate on the local H200 setup."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora
from src.model.student_wrapper import (
    StudentWrapperConfig,
    build_student_model,
    load_student_processor,
    load_student_tokenizer,
)
from src.model.tokenizer_ext import REQUIRED_SPECIAL_TOKENS, distill_trainable_token_ids
from src.training.collator import DistillationCollator
from src.training.losses import (
    DistillationLossWeights,
    export_metric_logs,
    get_stage_weights,
    resolve_loss_weight_value,
    resolve_optional_loss_weight_value,
)
from src.training.trainer import move_batch_to_device, run_train_step
from src.utils.runtime_paths import remap_external_path, resolve_student_model_path
from src.utils.seeds import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "human_coc_teacher_full.jsonl",
    )
    parser.add_argument(
        "--stage-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train" / "stage_h200_clean_human900_stage0.yaml",
    )
    parser.add_argument("--student-model", default=resolve_student_model_path())
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "h200_stage0_token_row_gate.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _path_exists(raw_path: str | Path | None) -> bool:
    if raw_path in (None, ""):
        return False
    remapped = remap_external_path(raw_path)
    if remapped in (None, ""):
        return False
    return Path(remapped).exists()


def has_required_materialized_assets(record: dict[str, Any]) -> bool:
    sample_input = record.get("input") or {}
    hard_target = record.get("hard_target") or {}
    required_paths = [
        sample_input.get("materialized_sample_path"),
        sample_input.get("metadata_path"),
        sample_input.get("ego_history_path"),
        hard_target.get("traj_future_token_ids_path"),
    ]
    if not all(_path_exists(path) for path in required_paths):
        return False
    image_paths = list(sample_input.get("image_paths") or [])
    return bool(image_paths) and all(_path_exists(path) for path in image_paths)


def load_stage(path: Path) -> tuple[dict[str, Any], DistillationLossWeights]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = get_stage_weights(str(config["stage_name"]))
    weights = config.get("loss_weights") or {}
    loss_weights = DistillationLossWeights(
        hard_cot_ce=resolve_loss_weight_value(weights, "hard_cot_ce", defaults.hard_cot_ce),
        teacher_seq_ce=resolve_loss_weight_value(weights, "teacher_seq_ce", defaults.teacher_seq_ce),
        teacher_logit_kd=resolve_loss_weight_value(weights, "teacher_logit_kd", defaults.teacher_logit_kd),
        traj_ce=resolve_loss_weight_value(weights, "traj_ce", defaults.traj_ce),
        format_ce=resolve_loss_weight_value(weights, "format_ce", defaults.format_ce),
        action_aux=resolve_loss_weight_value(weights, "action_aux", defaults.action_aux),
        feat_align=resolve_loss_weight_value(weights, "feat_align", defaults.feat_align),
        traj_aux_reg=resolve_loss_weight_value(weights, "traj_aux_reg", defaults.traj_aux_reg),
        teacher_traj_ce=resolve_optional_loss_weight_value(weights, "teacher_traj_ce"),
        teacher_traj_topk_kd=resolve_loss_weight_value(
            weights,
            "teacher_traj_topk_kd",
            defaults.teacher_traj_topk_kd,
        ),
        teacher_traj_hidden_align=resolve_loss_weight_value(
            weights,
            "teacher_traj_hidden_align",
            defaults.teacher_traj_hidden_align,
        ),
        traj_xyz_reg=resolve_loss_weight_value(weights, "traj_xyz_reg", defaults.traj_xyz_reg),
        traj_delta_reg=resolve_loss_weight_value(weights, "traj_delta_reg", defaults.traj_delta_reg),
        traj_final_reg=resolve_loss_weight_value(weights, "traj_final_reg", defaults.traj_final_reg),
        traj_control_reg=resolve_loss_weight_value(weights, "traj_control_reg", defaults.traj_control_reg),
        traj_control_delta_reg=resolve_loss_weight_value(
            weights,
            "traj_control_delta_reg",
            defaults.traj_control_delta_reg,
        ),
        traj_aux_xyz_reg=resolve_loss_weight_value(weights, "traj_aux_xyz_reg", defaults.traj_aux_xyz_reg),
        traj_aux_final_reg=resolve_loss_weight_value(weights, "traj_aux_final_reg", defaults.traj_aux_final_reg),
        traj_aux_guided_kd=resolve_loss_weight_value(weights, "traj_aux_guided_kd", defaults.traj_aux_guided_kd),
        traj_aux_pseudo_ce=resolve_loss_weight_value(weights, "traj_aux_pseudo_ce", defaults.traj_aux_pseudo_ce),
    )
    return config, loss_weights


def _output_head(model):
    getter = getattr(model.backbone, "get_output_embeddings", None)
    if callable(getter):
        head = getter()
        if head is not None:
            return head
    base_model = getattr(model.backbone, "base_model", None)
    base_inner = getattr(base_model, "model", None)
    head = getattr(base_inner, "lm_head", None)
    if head is not None:
        return head
    raise RuntimeError("Could not locate the output head used by the token adapter.")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    stage_config, loss_weights = load_stage(args.stage_config)
    data_view = dict(stage_config.get("data_view") or {})
    lora_cfg = dict(stage_config.get("lora") or {})
    trainer_batch_size = int(args.batch_size or stage_config.get("batch_size", 1) or 1)
    learning_rate = float(args.learning_rate or stage_config.get("learning_rate", 1.0e-4))
    use_bf16 = bool(stage_config.get("bf16", True))
    gradient_checkpointing = bool(stage_config.get("gradient_checkpointing", True))

    all_rows = load_jsonl(args.corpus_jsonl)
    train_rows = [row for row in all_rows if row.get("split") == "train" and has_required_materialized_assets(row)]
    if args.max_train_samples is not None:
        train_rows = train_rows[: int(args.max_train_samples)]
    if not train_rows:
        raise RuntimeError("No train rows with required local assets were found for the Stage 0 gate.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = resolve_student_model_path(args.student_model)
    wrapper_cfg = StudentWrapperConfig(
        student_model_name=student_model,
        max_length=int(stage_config.get("max_length", 4096)),
        torch_dtype=(torch.bfloat16 if use_bf16 and device.type == "cuda" else None),
        local_files_only=Path(student_model).expanduser().exists(),
    )
    tokenizer = load_student_tokenizer(wrapper_cfg)
    processor = load_student_processor(wrapper_cfg, tokenizer=tokenizer)
    model = build_student_model(wrapper_cfg, tokenizer)
    lora_spec = LoraConfigSpec(
        r=int(lora_cfg.get("rank", 32) or 32),
        alpha=int(lora_cfg.get("alpha", 64) or 64),
        dropout=float(lora_cfg.get("dropout", 0.05) or 0.05),
        trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer)),
    )
    model.backbone = maybe_apply_lora(model.backbone, lora_spec, enabled=True)

    if gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        try:
            model.backbone.gradient_checkpointing_enable()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(model.backbone, "enable_input_require_grads"):
        try:
            model.backbone.enable_input_require_grads()
        except Exception:  # noqa: BLE001
            pass

    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    model = model.to(device)
    model.train()

    collator = DistillationCollator(
        tokenizer=tokenizer,
        processor=processor,
        project_root=PROJECT_ROOT,
        max_length=int(stage_config.get("max_length", 4096)),
        prompt_mode=str(data_view.get("prompt_mode", "joint")),
        target_mode=str(data_view.get("target_mode", "joint")),
        teacher_pair_target=bool(data_view.get("teacher_pair_target", False)),
        enable_teacher_view=bool(data_view.get("enable_teacher_view", False)),
        enable_action_aux=bool(data_view.get("enable_action_aux", False)),
        teacher_traj_cache_dir=None,
        teacher_traj_hidden_source="hidden_upper6",
        teacher_traj_latent_suffix="lat32",
        hard_view_uses_teacher_cot=False,
        teacher_view_force_enable=False,
        teacher_view_uses_teacher_traj=False,
        teacher_view_default_traj_weight=0.0,
        teacher_traj_topk_on_teacher_view=False,
    )
    dataloader = DataLoader(train_rows, batch_size=trainer_batch_size, shuffle=False, collate_fn=collator)
    batch = next(iter(dataloader))
    batch = move_batch_to_device(batch, device)

    output_head = _output_head(model)
    token_adapter = getattr(output_head, "token_adapter", None)
    if token_adapter is None:
        raise RuntimeError("LoRA output head is missing token_adapter, so the token-row gate cannot run.")
    adapter_name = next(iter(token_adapter.token_indices.keys()))
    token_indices = [int(value) for value in token_adapter.token_indices[adapter_name]]
    local_index_by_token_id = {token_id: index for index, token_id in enumerate(token_indices)}

    delta_param_name = None
    delta_param = None
    for name, parameter in model.named_parameters():
        if "token_adapter.trainable_tokens_delta" in name:
            delta_param_name = name
            delta_param = parameter
            break
    if delta_param_name is None or delta_param is None:
        raise RuntimeError("Could not find the token-adapter delta parameter.")

    traj_start_id = int(tokenizer.convert_tokens_to_ids("<i0>"))
    traj_end_id = int(tokenizer.convert_tokens_to_ids("<i3999>"))
    special_tokens = ["<|cot_end|>", "<|traj_future_start|>", "<|traj_future_end|>"]
    special_token_ids = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in special_tokens}

    labels = batch["labels"]
    traj_mask = batch["traj_span_mask"].to(dtype=torch.bool, device=labels.device)
    active_traj_token_ids = sorted(
        {
            int(token_id)
            for token_id in labels[traj_mask & (labels != -100)].detach().cpu().tolist()
            if traj_start_id <= int(token_id) <= traj_end_id
        }
    )
    if not active_traj_token_ids:
        raise RuntimeError("The first Stage 0 batch did not touch any trajectory body token ids.")

    tracked_token_ids = sorted(set(active_traj_token_ids + list(special_token_ids.values())))
    before_rows = delta_param.detach().float().cpu()[[local_index_by_token_id[token_id] for token_id in tracked_token_ids]].clone()

    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    autocast_context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 and device.type == "cuda" else nullcontext()
    with autocast_context:
        loss, logs = run_train_step(
            model,
            batch,
            loss_weights,
            traj_decode_config=None,
            traj_aux_interface_config=None,
            traj_body_prefix_tokens=None,
            traj_hidden_bridge_config=None,
        )
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(args.grad_clip_norm),
    )

    grad_tensor = delta_param.grad.detach().float().cpu()
    special_grad_checks = {}
    for token, token_id in special_token_ids.items():
        local_index = local_index_by_token_id.get(token_id)
        special_grad_checks[token] = {
            "token_id": token_id,
            "present_in_token_adapter": local_index is not None,
            "row_grad_norm": float(grad_tensor[local_index].norm().item()) if local_index is not None else 0.0,
        }
    touched_traj_grad_norms = [
        float(grad_tensor[local_index_by_token_id[token_id]].norm().item())
        for token_id in active_traj_token_ids
        if token_id in local_index_by_token_id
    ]

    optimizer.step()
    after_rows = delta_param.detach().float().cpu()[[local_index_by_token_id[token_id] for token_id in tracked_token_ids]].clone()
    row_delta_norms = (after_rows - before_rows).norm(dim=1)

    special_delta_checks = {}
    for token, token_id in special_token_ids.items():
        tracked_index = tracked_token_ids.index(token_id)
        special_delta_checks[token] = float(row_delta_norms[tracked_index].item())
    touched_traj_delta_norms = [
        float(row_delta_norms[tracked_token_ids.index(token_id)].item())
        for token_id in active_traj_token_ids
    ]

    metrics = export_metric_logs(logs)
    trainable_traj_ids = [token_id for token_id in range(traj_start_id, traj_end_id + 1) if token_id in local_index_by_token_id]
    special_present = {
        token: (token_id in local_index_by_token_id)
        for token, token_id in {
            **special_token_ids,
            **{token: int(tokenizer.convert_tokens_to_ids(token)) for token in REQUIRED_SPECIAL_TOKENS},
        }.items()
    }

    requires_grad_pass = bool(delta_param.requires_grad) and len(trainable_traj_ids) == 4000 and all(special_present.values())
    special_grad_pass = all(item["row_grad_norm"] > 0.0 for item in special_grad_checks.values())
    touched_traj_grad_pass = bool(touched_traj_grad_norms) and all(value > 0.0 for value in touched_traj_grad_norms)
    special_delta_pass = all(value > 0.0 for value in special_delta_checks.values())
    touched_traj_delta_pass = bool(touched_traj_delta_norms) and all(value > 0.0 for value in touched_traj_delta_norms)
    passed = all(
        (
            requires_grad_pass,
            special_grad_pass,
            touched_traj_grad_pass,
            special_delta_pass,
            touched_traj_delta_pass,
        )
    )

    peak_cuda_memory_mb = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    summary = {
        "stage_config": str(args.stage_config),
        "corpus_jsonl": str(args.corpus_jsonl),
        "student_model": student_model,
        "device": str(device),
        "bf16": use_bf16,
        "batch_size": trainer_batch_size,
        "train_rows_used": len(train_rows),
        "sample_ids": [str(row.get("sample_id")) for row in train_rows],
        "token_adapter_param_name": delta_param_name,
        "token_adapter_shape": list(delta_param.shape),
        "lora": {
            "rank": int(lora_spec.r),
            "alpha": int(lora_spec.alpha),
            "dropout": float(lora_spec.dropout),
        },
        "trainable_traj_row_count": len(trainable_traj_ids),
        "traj_token_id_range": [traj_start_id, traj_end_id],
        "active_batch_traj_token_ids": active_traj_token_ids,
        "active_batch_traj_token_count": len(active_traj_token_ids),
        "all_required_special_rows_present": all(special_present.values()),
        "special_row_presence": special_present,
        "special_grad_checks": special_grad_checks,
        "special_delta_norms": special_delta_checks,
        "touched_traj_grad_norm_min": float(min(touched_traj_grad_norms)),
        "touched_traj_grad_norm_max": float(max(touched_traj_grad_norms)),
        "touched_traj_delta_norm_min": float(min(touched_traj_delta_norms)),
        "touched_traj_delta_norm_max": float(max(touched_traj_delta_norms)),
        "metrics": metrics,
        "total_loss": float(loss.detach().cpu().item()),
        "global_grad_norm": float(grad_norm.detach().cpu().item()),
        "learning_rate": learning_rate,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "requires_grad_pass": requires_grad_pass,
        "special_grad_pass": special_grad_pass,
        "touched_traj_grad_pass": touched_traj_grad_pass,
        "special_delta_pass": special_delta_pass,
        "touched_traj_delta_pass": touched_traj_delta_pass,
        "pass": passed,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
