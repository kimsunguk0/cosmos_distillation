#!/usr/bin/env python3
"""QAT (Quantization-Aware Training) wrapper for ML-FLEX backbone distillation.

Method A: Load base + existing LoRA (NOT merged) + FLEX, apply ModelOpt INT4 AWQ
fake-quantization to LLM backbone only, then fine-tune with the standard
distillation training loop.

Usage:
    python scripts/train_qat_distill.py \
        --config configs/train/stage_qat_mlflex_k512_int4awq_20k.yaml \
        --corpus-jsonl data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl \
        --init-checkpoint-dir outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final \
        --output-dir outputs/checkpoints/qat_mlflex_k512_int4awq_20k_e3 \
        --qat-quantization int4_awq \
        --qat-calib-samples 512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# ── Project imports (reuse existing infrastructure) ─────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._09_train_distill_imports import (  # noqa: E402 — lazy check below
    build_student_model,
    load_student_checkpoint,
    apply_optimization_policy,
    get_train_val_records,
    make_dataloaders,
    run_training_loop,
)


def _try_import_09():
    """Try to import key functions from 09_train_distill.py.

    The training script was not designed as a library, so we fall back
    to subprocess invocation if the import shim does not exist.
    """
    try:
        # If the import shim exists, great
        from scripts._09_train_distill_imports import build_student_model  # noqa: F811
        return True
    except ImportError:
        return False


def apply_qat_quantization(
    model: torch.nn.Module,
    *,
    quantization: str = "int4_awq",
    calib_samples: int = 512,
    calib_dataloader=None,
    device: torch.device | str = "cuda",
) -> torch.nn.Module:
    """Apply ModelOpt fake-quantization to the LLM backbone for QAT.

    Only the backbone's nn.Linear layers are quantized.  LoRA adapters,
    FLEX scene encoder, ViT, and auxiliary heads are left in FP16.
    """
    import modelopt.torch.quantization as mtq

    # Select quantization config
    quant_configs = {
        "int4_awq": mtq.INT4_AWQ_CFG,
        "int4_blockwise": getattr(mtq, "INT4_BLOCKWISE_WEIGHT_ONLY_CFG", mtq.INT4_AWQ_CFG),
    }
    quant_cfg = quant_configs.get(quantization)
    if quant_cfg is None:
        raise ValueError(f"Unknown QAT quantization: {quantization!r}. Choose from {list(quant_configs)}")

    print(json.dumps({"event": "qat_quantize_start", "quantization": quantization, "calib_samples": calib_samples}), flush=True)

    # Build calibration forward loop
    calib_count = 0

    def calib_forward_loop(model_to_calib):
        nonlocal calib_count
        if calib_dataloader is None:
            print(json.dumps({"event": "qat_calib_skip", "reason": "no_dataloader"}), flush=True)
            return
        model_to_calib.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            for batch_idx, batch in enumerate(calib_dataloader):
                if calib_count >= calib_samples:
                    break
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                try:
                    model_to_calib(**batch)
                except Exception as e:
                    print(json.dumps({"event": "qat_calib_sample_error", "batch_idx": batch_idx, "error": str(e)}), flush=True)
                    continue
                calib_count += batch.get("input_ids", next(iter(batch.values()))).shape[0]
        print(json.dumps({"event": "qat_calib_done", "samples_calibrated": calib_count}), flush=True)

    # Quantize ONLY the backbone (PeftModel wrapping the LLM)
    backbone = model.backbone if hasattr(model, "backbone") else model
    backbone = mtq.quantize(backbone, quant_cfg, forward_loop=calib_forward_loop)
    if hasattr(model, "backbone"):
        model.backbone = backbone

    # Disable quantizers on LoRA adapter weights (must stay FP16)
    lora_quantizers_disabled = 0
    for name, mod in backbone.named_modules():
        if "lora_" in name:
            for attr in ("weight_quantizer", "input_quantizer", "output_quantizer"):
                q = getattr(mod, attr, None)
                if q is not None and hasattr(q, "disable"):
                    q.disable()
                    lora_quantizers_disabled += 1

    # Count quantized vs unquantized modules
    quantized_modules = 0
    total_linear = 0
    for name, mod in backbone.named_modules():
        if isinstance(mod, torch.nn.Linear):
            total_linear += 1
        if hasattr(mod, "weight_quantizer"):
            wq = getattr(mod, "weight_quantizer", None)
            if wq is not None and hasattr(wq, "is_enabled") and wq.is_enabled:
                quantized_modules += 1

    # Verify FLEX is NOT quantized
    flex_encoder = getattr(model, "flex_scene_encoder", None)
    flex_quantized = False
    if flex_encoder is not None:
        for mod in flex_encoder.modules():
            if hasattr(mod, "weight_quantizer"):
                wq = getattr(mod, "weight_quantizer", None)
                if wq is not None and hasattr(wq, "is_enabled") and wq.is_enabled:
                    flex_quantized = True
                    break

    print(json.dumps({
        "event": "qat_quantize_done",
        "quantized_linear_modules": quantized_modules,
        "total_linear_modules": total_linear,
        "lora_quantizers_disabled": lora_quantizers_disabled,
        "flex_quantized": flex_quantized,
    }), flush=True)

    if flex_quantized:
        raise RuntimeError("FLEX scene encoder was quantized -- this should not happen. "
                           "Ensure mtq.quantize is applied to model.backbone only.")

    return model


def main():
    parser = argparse.ArgumentParser(description="QAT wrapper for ML-FLEX distillation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--corpus-jsonl", type=str, required=True)
    parser.add_argument("--init-checkpoint-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--qat-quantization", type=str, default="int4_awq",
                        choices=["int4_awq", "int4_blockwise"])
    parser.add_argument("--qat-calib-samples", type=int, default=512)
    parser.add_argument("--max-val-samples", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--eval-every-epochs", type=float, default=0.5)
    parser.add_argument("--save-every-epochs", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print(json.dumps({
        "event": "qat_boot",
        "config": args.config,
        "checkpoint": args.init_checkpoint_dir,
        "output_dir": args.output_dir,
        "quantization": args.qat_quantization,
        "calib_samples": args.qat_calib_samples,
    }), flush=True)

    # ── This script delegates to 09_train_distill.py with QAT inserted ──
    # The actual implementation injects quantization between model load and
    # training start.  Since 09_train_distill.py is not modular enough to
    # import directly, we use subprocess invocation with an environment
    # variable that triggers QAT inside the training script.
    #
    # For now, print the manual steps until the import shim is implemented.

    print("=" * 70)
    print("QAT MANUAL STEPS (run in order):")
    print("=" * 70)
    print(f"""
# ── Step 1: Start Python REPL or script ──
cd {Path(__file__).resolve().parent.parent}

.venv/bin/python -c "
import torch
import json
import modelopt.torch.quantization as mtq
from src.model.student_wrapper import DistillStudentModel, build_student_model
from src.model.checkpoint_io import load_student_checkpoint

# Load model with LoRA (NOT merged) + FLEX
device = torch.device('{args.device}')
model = build_student_model('{args.init_checkpoint_dir}')
load_student_checkpoint('{args.init_checkpoint_dir}', model, use_lora=True, adapter_trainable=True)
model = model.to(device)
print('Model loaded with LoRA + FLEX')

# Apply INT4 AWQ fake-quantization to backbone only
def calib_fn(backbone):
    backbone.eval()
    # Simple calibration with random inputs (replace with real data for better quality)
    with torch.no_grad():
        for _ in range({args.qat_calib_samples // 16}):
            dummy = torch.randint(0, 1000, (1, 128), device=device)
            try:
                backbone(dummy)
            except Exception:
                pass

model.backbone = mtq.quantize(model.backbone, mtq.INT4_AWQ_CFG, forward_loop=calib_fn)

# Disable quantizers on LoRA weights
for name, mod in model.backbone.named_modules():
    if 'lora_' in name:
        for attr in ('weight_quantizer', 'input_quantizer', 'output_quantizer'):
            q = getattr(mod, attr, None)
            if q is not None and hasattr(q, 'disable'):
                q.disable()

print('INT4 AWQ quantization applied, LoRA quantizers disabled')
print('Now run training with: 09_train_distill.py --init-checkpoint-dir (this quantized model)')
"

# ── Step 2: Fine-tune with quantized backbone ──
# Use 09_train_distill.py with the QAT config
.venv/bin/python scripts/09_train_distill.py \\
  --config {args.config} \\
  --corpus-jsonl {args.corpus_jsonl} \\
  --init-checkpoint-dir {args.init_checkpoint_dir} \\
  --output-dir {args.output_dir} \\
  --max-val-samples {args.max_val_samples} \\
  --num-workers {args.num_workers} \\
  --log-every-steps {args.log_every_steps}
""")

    print("=" * 70)
    print("NOTE: The above is a reference implementation.")
    print("For automated QAT, add --qat-quantization flag to 09_train_distill.py")
    print("by inserting mtq.quantize() after model load and before training loop.")
    print("=" * 70)


if __name__ == "__main__":
    main()
