#!/usr/bin/env python3
"""Detailed E2E latency: ViT / FLEX / Prefill / Decode / AE separately.

Strategy: Hook into model components to measure each stage within build_batch.
"""
from __future__ import annotations
import argparse, copy, importlib, json, os, sys, time
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

SUKIM_ROOT = PROJECT_ROOT.parents[1]
for p in (PROJECT_ROOT, SUKIM_ROOT, SUKIM_ROOT / "alpamayo_repo/alpamayo1.5/src", SUKIM_ROOT / "visualization"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _import_ae():
    spec = importlib.util.spec_from_file_location(
        "ae_train", str(PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint-dir", type=str, required=True)
    parser.add_argument("--ae28-checkpoint", type=str, required=True)
    parser.add_argument("--corpus-jsonl", type=str, required=True)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-bench", type=int, default=20)
    parser.add_argument("--ae-steps", type=int, nargs="+", default=[1, 2, 4, 10])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--student-dtype", type=str, default="bfloat16")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    args = parser.parse_args()

    device = torch.device(args.device)
    ae = _import_ae()

    # Build ae_args with all required fields
    class AEArgs:
        pass
    ae_args = AEArgs()
    for k, v in {
        "student_checkpoint_dir": Path(args.student_checkpoint_dir),
        "corpus_jsonl": Path(args.corpus_jsonl),
        "teacher_checkpoint_path": Path(PROJECT_ROOT.parent.parent / "base_weights" / "Alpamayo-1.5-10B"),
        "student_dtype": args.student_dtype, "device": args.device,
        "student_model": "", "ae_init_mode": "student_backbone_init",
        "attn_implementation": args.attn_implementation,
        "disable_student_deepstack": False, "qat_quantization": "", "qat_calib_samples": 256,
        "num_samples": 100, "val_samples": 20, "val_fraction": 0.1,
        "split_seed": None, "split_cache_json": None, "split": "train",
        "split_scan_all": True, "compressed_layers": 28, "mapping": "linspace_round",
        "ae_dtype": "bfloat16", "prefix_mode": "teacher_forced",
        "preserve_flex_positions": True, "flex_selection_strategy": "uniform",
        "flex_scene_deepstack": True, "target_source": "teacher",
        "max_new_tokens": 160, "max_length": 4096,
        "stage2_attention_mode": "official_none", "seed": 42,
        "teacher_load_device": "cpu",
    }.items():
        setattr(ae_args, k, v)

    # Load models
    student, student_tokenizer, student_processor, base_model = ae.load_student(ae_args)

    def _torch_dtype(name):
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(name, torch.bfloat16)
    _load_fn = getattr(ae, "load_model_and_processor", None)
    if _load_fn is None:
        from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import load_model_and_processor as _load_fn
    teacher_model, _, _, _, _ = _load_fn(
        checkpoint_path=ae_args.teacher_checkpoint_path, dtype=_torch_dtype(args.student_dtype),
        device="cpu", config_json=None, runtime_support=None,
        attn_implementation=args.attn_implementation, min_pixels=163840, max_pixels=196608)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    teacher_model.to(device)

    bundle, _ = ae.build_bundle(teacher_model, ae_args, student=student)
    ae.load_bundle_checkpoint(Path(args.ae28_checkpoint), bundle=bundle)
    bundle.eval()
    for p in bundle.parameters():
        p.requires_grad_(False)
    bundle.to(device)

    train_items, _, _ = ae.select_train_val_items(ae_args)
    sample_items = train_items[:1]

    # ===== Install timing hooks on model components =====
    dtype = _torch_dtype(args.student_dtype)
    timing_log = {}

    # Hook: ViT (visual model)
    qwen_model = student.backbone
    if hasattr(qwen_model, "model"):
        qwen_model = qwen_model.model
    visual = getattr(qwen_model, "visual", None)

    def _make_hook(name):
        def hook(module, input, output):
            torch.cuda.synchronize()
            timing_log[f"{name}_end"] = time.perf_counter()
        def pre_hook(module, input):
            torch.cuda.synchronize()
            timing_log[f"{name}_start"] = time.perf_counter()
        return pre_hook, hook

    handles = []
    if visual is not None:
        pre, post = _make_hook("vit")
        handles.append(visual.register_forward_pre_hook(pre))
        handles.append(visual.register_forward_hook(post))

    # Hook: FLEX encoder
    flex_enc = getattr(student, "flex_encoder", None) or getattr(student, "flex_scene_encoder", None)
    if flex_enc is not None:
        pre, post = _make_hook("flex")
        handles.append(flex_enc.register_forward_pre_hook(pre))
        handles.append(flex_enc.register_forward_hook(post))

    # Hook: LLM (language_model)
    lm = None
    for attr_path in ["backbone.model.language_model", "backbone.model.model.language_model"]:
        obj = student
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            lm = obj
            break
    if lm is not None:
        pre, post = _make_hook("llm")
        handles.append(lm.register_forward_pre_hook(pre))
        handles.append(lm.register_forward_hook(post))

    print(json.dumps({
        "event": "ready",
        "hooks": {
            "vit": visual is not None,
            "flex": flex_enc is not None,
            "llm": lm is not None,
        }
    }), flush=True)

    # ===== Benchmark =====
    all_timings = []

    for trial in range(args.num_warmup + args.num_bench):
        is_bench = trial >= args.num_warmup
        timing_log.clear()
        t = {}

        # --- Processor (CPU): image loading + tokenization ---
        cpu_start = time.perf_counter()
        # (this happens inside build_batch, but we time the whole build_batch)

        # --- Build batch: ViT + FLEX + Prefill ---
        torch.cuda.synchronize()
        build_start = time.perf_counter()
        batch = ae.build_batch(
            args=ae_args, student=student, student_processor=student_processor,
            student_tokenizer=student_tokenizer, teacher_model=teacher_model,
            batch_items=sample_items)
        torch.cuda.synchronize()
        build_end = time.perf_counter()

        t["build_total"] = (build_end - build_start) * 1000

        # Extract component times from hooks
        if "vit_start" in timing_log and "vit_end" in timing_log:
            t["vit"] = (timing_log["vit_end"] - timing_log["vit_start"]) * 1000
        else:
            t["vit"] = 0.0

        if "flex_start" in timing_log and "flex_end" in timing_log:
            t["flex"] = (timing_log["flex_end"] - timing_log["flex_start"]) * 1000
        else:
            t["flex"] = 0.0

        if "llm_start" in timing_log and "llm_end" in timing_log:
            t["prefill"] = (timing_log["llm_end"] - timing_log["llm_start"]) * 1000
        else:
            t["prefill"] = 0.0

        # Processor + other = build_total - vit - flex - prefill
        t["other"] = max(t["build_total"] - t["vit"] - t["flex"] - t["prefill"], 0)

        # teacher_forced mode: decode = 0
        t["decode"] = 0.0

        # --- AE Flow Matching ---
        prompt_cache = batch["cache"]
        context = batch["context"]
        prefill_seq_len = int(context["kv_cache_seq_len"])
        n_diff = int(context["n_diffusion_tokens"])
        action_dims = teacher_model.action_space.get_action_space_dims()
        ae_kwargs = {}
        if bool(getattr(teacher_model.config, "expert_non_causal_attention", False)):
            ae_kwargs["is_causal"] = False

        for n_steps in args.ae_steps:
            def step_fn(*, x, t):
                fut = bundle.action_in_proj(x.to(dtype=dtype), t.to(dtype=dtype))
                if fut.dim() == 2:
                    fut = fut.view(x.shape[0], n_diff, -1)
                attn_m = context.get("attention_mask")
                if attn_m is not None:
                    attn_m = attn_m.to(dtype=fut.dtype)
                out = bundle.expert(
                    inputs_embeds=fut, position_ids=context["position_ids"],
                    past_key_values=prompt_cache, attention_mask=attn_m,
                    use_cache=True, **ae_kwargs)
                prompt_cache.crop(prefill_seq_len)
                hidden = out.last_hidden_state[:, -n_diff:]
                return bundle.action_out_proj(hidden).view(-1, *action_dims)

            torch.manual_seed(42); torch.cuda.manual_seed_all(42)
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s_ev.record()
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                teacher_model.diffusion.sample(
                    batch_size=1, step_fn=step_fn, device=device,
                    inference_step=n_steps, temperature=1.0)
            e_ev.record()
            torch.cuda.synchronize()
            t[f"ae_{n_steps}"] = s_ev.elapsed_time(e_ev)

        if is_bench:
            all_timings.append(t)
        del batch

    # Cleanup hooks
    for h in handles:
        h.remove()

    # Print results
    stages = ["vit", "flex", "prefill", "decode", "other"]
    print(f"\n{'='*65}")
    print(f"  E2E Latency Breakdown (H200, bf16, batch=1, {args.num_bench} samples avg)")
    print(f"{'='*65}")

    for stage in stages:
        vals = [t[stage] for t in all_timings]
        print(json.dumps({
            "stage": stage,
            "mean_ms": round(np.mean(vals), 1),
            "std_ms": round(np.std(vals), 1),
        }), flush=True)

    backbone_vals = [sum(t[s] for s in stages) for t in all_timings]
    print(json.dumps({
        "stage": "backbone_total",
        "mean_ms": round(np.mean(backbone_vals), 1),
    }), flush=True)

    for n_steps in args.ae_steps:
        vals = [t[f"ae_{n_steps}"] for t in all_timings]
        e2e = np.mean(backbone_vals) + np.mean(vals)
        print(json.dumps({
            "stage": f"ae_{n_steps}step",
            "mean_ms": round(np.mean(vals), 1),
            "per_step_ms": round(np.mean(vals) / n_steps, 1),
            "e2e_total_ms": round(e2e, 1),
        }), flush=True)


if __name__ == "__main__":
    main()
