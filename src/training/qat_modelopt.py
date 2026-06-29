"""ModelOpt QAT helpers shared by training and evaluation scripts."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable
from typing import Any

import torch

from src.training.trainer import prepare_flex_batch_for_model


BatchFactory = Callable[[], Iterable[dict[str, Any]]]


def _enabled_quantizer(quantizer: Any) -> bool:
    if quantizer is None:
        return False
    attr = getattr(quantizer, "is_enabled", False)
    return bool(attr() if callable(attr) else attr)


def _iter_calib_batches(factory: BatchFactory | Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    return factory() if callable(factory) else factory


def _move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def apply_modelopt_qat(
    model: Any,
    *,
    quantization: str,
    calib_batches: BatchFactory | Iterable[dict[str, Any]],
    device: torch.device,
    bf16: bool = True,
    calib_samples: int = 512,
    is_rank_zero: bool = True,
) -> dict[str, Any]:
    """Apply ModelOpt fake quantization to a loaded student model.

    The *_vit variants quantize both Qwen language and visual towers. Other
    modes quantize the language tower only. LoRA adapters, lm_head, embeddings,
    first/last language layers, and visual boundary/projection paths stay BF16.
    """

    qat_quantization_requested = str(quantization or "").strip().lower()
    if not qat_quantization_requested:
        return {"enabled": False}

    import modelopt.torch.quantization as mtq

    qat_quantization = qat_quantization_requested
    quantize_visual = qat_quantization in {"fp8_vit", "fp8_pcpt_vit"}
    if qat_quantization == "fp8_vit":
        qat_quantization = "fp8"
    elif qat_quantization == "fp8_pcpt_vit":
        qat_quantization = "fp8_pcpt"

    int4_ffn_only_cfg = copy.deepcopy(mtq.INT4_AWQ_CFG)
    int4_ffn_only_cfg["quant_cfg"]["*q_proj*"] = {"enable": False}
    int4_ffn_only_cfg["quant_cfg"]["*k_proj*"] = {"enable": False}
    int4_ffn_only_cfg["quant_cfg"]["*v_proj*"] = {"enable": False}
    int4_ffn_only_cfg["quant_cfg"]["*o_proj*"] = {"enable": False}
    int4_ffn_only_cfg["quant_cfg"]["*q_norm*"] = {"enable": False}
    int4_ffn_only_cfg["quant_cfg"]["*k_norm*"] = {"enable": False}
    qat_configs = {
        "int4_awq": mtq.INT4_AWQ_CFG,
        "int4_blockwise": getattr(mtq, "INT4_BLOCKWISE_WEIGHT_ONLY_CFG", mtq.INT4_AWQ_CFG),
        "int4_ffn_only": int4_ffn_only_cfg,
        "fp8": copy.deepcopy(mtq.FP8_DEFAULT_CFG),
        "fp8_pcpt": copy.deepcopy(mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG),
    }
    qat_cfg = copy.deepcopy(qat_configs.get(qat_quantization))
    if qat_cfg is None:
        raise ValueError(f"Unknown QAT quantization: {qat_quantization_requested!r}")

    if is_rank_zero:
        print(
            json.dumps(
                {
                    "event": "qat_quantize_start",
                    "quantization": qat_quantization_requested,
                    "base_quantization": qat_quantization,
                    "calib_samples": int(calib_samples),
                    "quantize_visual": quantize_visual,
                }
            ),
            flush=True,
        )

    raw_model = model.module if hasattr(model, "module") else model
    backbone = raw_model.backbone

    unwrapped = backbone
    if hasattr(unwrapped, "base_model"):
        unwrapped = unwrapped.base_model
    if hasattr(unwrapped, "model"):
        unwrapped = unwrapped.model
    qwen_model = getattr(unwrapped, "model", unwrapped)
    language_model = getattr(qwen_model, "language_model", None)
    if language_model is None:
        raise RuntimeError(
            "Could not find language_model submodule for QAT. "
            f"Available: {[name for name, _ in qwen_model.named_children()]}"
        )
    visual_model = getattr(qwen_model, "visual", None)
    if quantize_visual and visual_model is None:
        raise RuntimeError("Requested visual FP8 quantization, but visual module was not found.")

    language_qat_cfg = copy.deepcopy(qat_cfg)
    layer_modules = getattr(language_model, "layers", None)
    excluded_language_layers: list[int] = []
    if layer_modules is not None and hasattr(layer_modules, "__len__"):
        last_layer_idx = max(int(len(layer_modules)) - 1, 0)
        excluded_language_layers = [0, last_layer_idx]
        for excluded_idx in excluded_language_layers:
            language_qat_cfg.setdefault("quant_cfg", {})[f"*layers.{excluded_idx}.*"] = {"enable": False}
    language_qat_cfg.setdefault("quant_cfg", {})["*embed_tokens*"] = {"enable": False}
    language_qat_cfg.setdefault("quant_cfg", {})["*lm_head*"] = {"enable": False}

    visual_qat_cfg = copy.deepcopy(qat_cfg)
    if quantize_visual:
        visual_qat_cfg.setdefault("quant_cfg", {}).update(
            {
                "*patch_embed*": {"enable": False},
                "*embed*": {"enable": False},
                "*pos_embed*": {"enable": False},
                "*rotary*": {"enable": False},
                "*merger*": {"enable": False},
                "*norm*": {"enable": False},
                "*ln*": {"enable": False},
            }
        )

    if is_rank_zero:
        lang_params = sum(p.numel() for p in language_model.parameters())
        vis_params = sum(p.numel() for p in (visual_model or torch.nn.Module()).parameters())
        print(
            json.dumps(
                {
                    "event": "qat_scope",
                    "quantize_target": "language_visual" if quantize_visual else "language_model",
                    "quantization": qat_quantization_requested,
                    "base_quantization": qat_quantization,
                    "language_params_M": round(lang_params / 1e6, 1),
                    "visual_params_M": round(vis_params / 1e6, 1),
                    "visual_excluded": not quantize_visual,
                    "excluded_language_layers": excluded_language_layers,
                    "visual_exclude_patterns": sorted(visual_qat_cfg.get("quant_cfg", {}).keys())
                    if quantize_visual
                    else [],
                }
            ),
            flush=True,
        )

    def make_forward_loop(calib_target: str):
        def calib_forward_loop(_target_model: Any) -> None:
            _target_model.eval()
            raw_model.eval()
            success = 0
            failed = 0
            first_error = None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
                for batch in _iter_calib_batches(calib_batches):
                    if success >= int(calib_samples):
                        break
                    batch = prepare_flex_batch_for_model(batch, raw_model)
                    batch = _move_tensor_batch(batch, device)
                    try:
                        raw_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask"),
                            pixel_values=batch.get("pixel_values"),
                            image_grid_thw=batch.get("image_grid_thw"),
                            return_hidden_states=False,
                            compute_meta_action=False,
                            compute_traj_aux=False,
                        )
                        success += int(batch["input_ids"].shape[0])
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        if first_error is None:
                            first_error = repr(exc)
            if is_rank_zero:
                print(
                    json.dumps(
                        {
                            "event": "qat_calib_done",
                            "target": calib_target,
                            "success": success,
                            "failed": failed,
                            "first_error": first_error,
                        }
                    ),
                    flush=True,
                )
            if success == 0:
                raise RuntimeError(
                    f"QAT calibration target={calib_target!r} produced 0 successful forwards "
                    f"out of {failed} attempts. First error: {first_error}"
                )

        return calib_forward_loop

    vision_attn_originals: list[tuple[Any, dict[str, Any]]] = []
    if visual_model is not None:
        for _, module in visual_model.named_modules():
            attention_functions = getattr(module, "ALL_ATTENTION_FUNCTIONS", None)
            if isinstance(attention_functions, dict) and attention_functions:
                vision_attn_originals.append((module, dict(attention_functions)))

    qwen_model.language_model = mtq.quantize(
        language_model,
        language_qat_cfg,
        forward_loop=make_forward_loop("language_model"),
    )
    if quantize_visual and visual_model is not None:
        qwen_model.visual = mtq.quantize(
            visual_model,
            visual_qat_cfg,
            forward_loop=make_forward_loop("visual"),
        )

    for module, original in vision_attn_originals:
        if hasattr(module, "ALL_ATTENTION_FUNCTIONS"):
            module.ALL_ATTENTION_FUNCTIONS.update(original)

    lora_q_disabled = 0
    for name, module in backbone.named_modules():
        if "lora_" not in name:
            continue
        for attr in ("weight_quantizer", "input_quantizer", "output_quantizer"):
            quantizer = getattr(module, attr, None)
            if quantizer is not None and hasattr(quantizer, "disable"):
                quantizer.disable()
                lora_q_disabled += 1

    counts = {"language": 0, "visual": 0, "lm_head": 0, "lora": 0, "flex": 0, "other": 0}
    for name, module in backbone.named_modules():
        if not hasattr(module, "weight_quantizer"):
            continue
        if not _enabled_quantizer(getattr(module, "weight_quantizer", None)):
            continue
        if "lora_" in name:
            counts["lora"] += 1
        elif "visual" in name:
            counts["visual"] += 1
        elif "lm_head" in name:
            counts["lm_head"] += 1
        elif "language_model" in name or "layers." in name:
            counts["language"] += 1
        else:
            counts["other"] += 1

    flex_model = getattr(raw_model, "flex_scene_encoder", None) or torch.nn.Module()
    flex_clean = not any(
        hasattr(module, "weight_quantizer") and _enabled_quantizer(getattr(module, "weight_quantizer", None))
        for module in flex_model.modules()
    )
    summary = {
        "enabled": True,
        "quantization": qat_quantization_requested,
        "base_quantization": qat_quantization,
        "quantizers_by_family": counts,
        "lora_quantizers_disabled": lora_q_disabled,
        "flex_clean": flex_clean,
        "visual_quantized": quantize_visual,
        "visual_clean": (not quantize_visual and counts["visual"] == 0),
    }
    if is_rank_zero:
        print(json.dumps({"event": "qat_quantize_done", **summary}), flush=True)
    if not quantize_visual and counts["visual"] > 0:
        raise RuntimeError(f"QAT quantized {counts['visual']} visual modules; this should not happen.")
    if quantize_visual and counts["visual"] == 0:
        raise RuntimeError("QAT visual quantization requested, but no enabled visual quantizers were found.")
    if counts["lora"] > 0:
        raise RuntimeError(f"QAT left {counts['lora']} LoRA quantizers enabled.")
    return summary
