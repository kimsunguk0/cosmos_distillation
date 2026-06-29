#!/usr/bin/env python3
"""Export FLEX encoder and AE28 to ONNX for TensorRT deployment.

Exports two ONNX models:
  1. FLEX encoder:
     ds_level0/ds_level1/ds_level2/final_visual [B, N_vis, 2048]
       + camera/time token metadata
       -> scene_tokens [B, 512, 2048]
       -> deepstack_scene_0/1/2 [B, 512, 2048]
  2. AE single step: noisy_action [B, 64, 2] + timestep [B, 1, 1] -> velocity [B, 64, 2]
     (with KV cache as fixed input)
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

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


# ---- FLEX ONNX Wrapper ----
class FlexEncoderONNX(nn.Module):
    """ML-FLEX encoder for ONNX export.

    This wrapper mirrors the runtime ``MultiLevelFlexEncoder`` path for the
    deployment contract used here: 16 images/sample with uniform visual token
    counts.  It keeps the four DeepStack/final levels explicit and preserves the
    trained camera/time embeddings instead of exporting a final-visual-only
    approximation.
    """

    def __init__(self, flex_encoder):
        super().__init__()
        self.scene_tokens = flex_encoder.scene_tokens          # [512, 1024]
        self.camera_embed = flex_encoder.camera_embed
        self.time_mlp = flex_encoder.time_mlp
        self.local_slot_embed = flex_encoder.local_slot_embed
        self.input_norms = flex_encoder.input_norms             # 4 × LayerNorm(2048)
        self.input_projs = flex_encoder.input_projs             # 4 × Linear(2048→1024)
        self.level_encoders = flex_encoder.level_encoders       # 4 × FlexLevelEncoder
        self.output_norms = flex_encoder.output_norms           # 4 × LayerNorm(1024)
        self.output_projs = flex_encoder.output_projs           # 4 × Linear(1024→2048)
        self.tokens_per_image = flex_encoder.config.tokens_per_image
        self.images_per_sample = flex_encoder.config.expected_images_per_sample
        self.max_camera_types = flex_encoder.config.max_camera_types
        self.use_camera_time_embeddings = bool(flex_encoder.config.use_camera_time_embeddings)

    def _base_queries(
        self,
        *,
        final_vis: torch.Tensor,
        camera_ids: torch.Tensor,
        relative_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = final_vis.shape[0]
        dtype = final_vis.dtype
        device = final_vis.device

        queries = self.scene_tokens.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        if self.local_slot_embed is not None:
            tpi = self.tokens_per_image
            slots = torch.arange(tpi, dtype=torch.long, device=device)
            slots = slots.unsqueeze(0).expand(self.images_per_sample, -1).reshape(-1)
            queries = queries + self.local_slot_embed(slots).to(dtype=dtype).unsqueeze(0)

        if self.use_camera_time_embeddings:
            if self.camera_embed is None or self.time_mlp is None:
                raise RuntimeError("FLEX camera/time embeddings are enabled but modules are missing.")
            image_camera_ids = camera_ids.reshape(B, self.images_per_sample, -1)[:, :, 0]
            image_times = relative_times.reshape(B, self.images_per_sample, -1, 1)[:, :, 0, :]
            query_camera_ids = (
                image_camera_ids.unsqueeze(-1)
                .expand(B, self.images_per_sample, self.tokens_per_image)
                .reshape(B, self.images_per_sample * self.tokens_per_image)
            )
            query_times = (
                image_times.unsqueeze(2)
                .expand(B, self.images_per_sample, self.tokens_per_image, 1)
                .reshape(B, self.images_per_sample * self.tokens_per_image, 1)
            )
            query_camera_ids = query_camera_ids.clamp(min=0, max=max(int(self.max_camera_types) - 1, 0))
            queries = queries + self.camera_embed(query_camera_ids).to(dtype=dtype)
            queries = queries + self.time_mlp(query_times.to(device=device, dtype=dtype)).to(dtype=dtype)
        return queries

    def _project_level(
        self,
        level_idx: int,
        visual_tokens: torch.Tensor,
        *,
        camera_ids: torch.Tensor,
        relative_times: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.input_projs[level_idx](self.input_norms[level_idx](visual_tokens))
        if self.use_camera_time_embeddings:
            clamped_camera_ids = camera_ids.to(device=visual_tokens.device, dtype=torch.long).clamp(
                min=0,
                max=max(int(self.max_camera_types) - 1, 0),
            )
            projected = projected + self.camera_embed(clamped_camera_ids).to(dtype=projected.dtype)
            projected = projected + self.time_mlp(
                relative_times.to(device=visual_tokens.device, dtype=projected.dtype)
            ).to(dtype=projected.dtype)
        return projected

    def forward(
        self,
        ds0: torch.Tensor,    # [B, N_vis, 2048]
        ds1: torch.Tensor,    # [B, N_vis, 2048]
        ds2: torch.Tensor,    # [B, N_vis, 2048]
        final_vis: torch.Tensor,  # [B, N_vis, 2048]
        camera_ids: torch.Tensor,  # [B, N_vis]
        relative_times: torch.Tensor,  # [B, N_vis, 1]
    ) -> torch.Tensor:
        if relative_times.ndim == 2:
            relative_times = relative_times.unsqueeze(-1)
        queries = self._base_queries(
            final_vis=final_vis,
            camera_ids=camera_ids,
            relative_times=relative_times,
        )

        # Level 0
        p0 = self._project_level(0, ds0, camera_ids=camera_ids, relative_times=relative_times)
        c0 = self.level_encoders[0](queries, p0)
        d0 = self.output_projs[0](self.output_norms[0](c0))

        # Level 1
        p1 = self._project_level(1, ds1, camera_ids=camera_ids, relative_times=relative_times)
        c1 = self.level_encoders[1](queries, p1)
        d1 = self.output_projs[1](self.output_norms[1](c1))

        # Level 2
        p2 = self._project_level(2, ds2, camera_ids=camera_ids, relative_times=relative_times)
        c2 = self.level_encoders[2](queries, p2)
        d2 = self.output_projs[2](self.output_norms[2](c2))

        # Level 3 (final)
        p3 = self._project_level(3, final_vis, camera_ids=camera_ids, relative_times=relative_times)
        c3 = self.level_encoders[3](queries, p3)
        out = self.output_projs[3](self.output_norms[3](c3))  # [B, 512, 2048]

        return out, d0, d1, d2


def _onnx_value_shape(value) -> list[str | int]:
    dims: list[str | int] = []
    for dim in value.type.tensor_type.shape.dim:
        dims.append(dim.dim_param or int(dim.dim_value))
    return dims


def _verify_onnx_signature(
    path: Path,
    *,
    expected_inputs: list[str],
    expected_outputs: list[str] | None = None,
) -> None:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    inputs = [value.name for value in model.graph.input]
    outputs = [value.name for value in model.graph.output]
    missing = [name for name in expected_inputs if name not in inputs]
    if missing:
        raise RuntimeError(
            f"ONNX signature check failed for {path}: missing inputs={missing}, actual={inputs}"
        )
    if expected_outputs is not None:
        missing_outputs = [name for name in expected_outputs if name not in outputs]
        if missing_outputs:
            raise RuntimeError(
                f"ONNX signature check failed for {path}: missing outputs={missing_outputs}, actual={outputs}"
            )
    print(json.dumps({
        "event": "onnx_signature_ok",
        "path": str(path),
        "inputs": [(value.name, _onnx_value_shape(value)) for value in model.graph.input],
        "outputs": [(value.name, _onnx_value_shape(value)) for value in model.graph.output],
    }), flush=True)


class AESingleStepONNX(nn.Module):
    """Wraps an AE bundle for single-step ONNX export (no KV cache mutation)."""

    def __init__(self, bundle, is_causal: bool = False):
        super().__init__()
        self.action_in_proj = bundle.action_in_proj
        self.expert = bundle.expert
        self.action_out_proj = bundle.action_out_proj
        self.is_causal = is_causal

    def forward(
        self,
        noisy_action: torch.Tensor,    # [B, 64, 2]
        timestep: torch.Tensor,         # [B, 1, 1]
        position_ids: torch.Tensor,     # [3, B, 64]
        # KV cache as flat tensors per layer (simplified for ONNX)
        past_keys: torch.Tensor,        # [n_layers, B, n_heads, seq_len, head_dim]
        past_values: torch.Tensor,      # [n_layers, B, n_heads, seq_len, head_dim]
    ) -> torch.Tensor:
        """Returns predicted velocity [B, 64, 2]."""
        future_token_embeds = self.action_in_proj(noisy_action, timestep)
        if future_token_embeds.dim() == 2:
            B = noisy_action.shape[0]
            future_token_embeds = future_token_embeds.view(B, 64, -1)

        # Build DynamicCache from flat tensors
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        n_layers = past_keys.shape[0]
        for i in range(n_layers):
            cache.update(past_keys[i], past_values[i], layer_idx=i)

        out = self.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=cache,
            attention_mask=None,
            use_cache=False,  # Don't update cache for ONNX
        )
        last_hidden = out.last_hidden_state[:, -64:]
        return self.action_out_proj(last_hidden).view(-1, 64, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-checkpoint-dir", type=str, required=True)
    parser.add_argument("--ae-checkpoint", type=str, default=None)
    parser.add_argument("--ae28-checkpoint", type=str, default=None)
    parser.add_argument("--ae-output-name", type=str, default=None)
    parser.add_argument("--compressed-layers", type=int, default=28)
    parser.add_argument("--mapping", type=str, default="linspace_round")
    parser.add_argument(
        "--ae-init-mode",
        type=str,
        default="student_backbone_init",
        choices=(
            "teacher_compressed",
            "scratch",
            "student_backbone_init",
            "student_backbone_init_teacher_q",
            "ae_checkpoint_compressed",
        ),
    )
    parser.add_argument("--init-ae-source-checkpoint", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-flex", action="store_true")
    parser.add_argument("--skip-ae", action="store_true")
    parser.add_argument("--flex-n-vis", type=int, default=2880)
    parser.add_argument("--verify-onnx", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float16

    ae = _import_ae()

    # Build minimal args for ae functions
    class AEArgs:
        pass
    ae_args = AEArgs()
    for k, v in {
        "student_checkpoint_dir": Path(args.student_checkpoint_dir),
        "corpus_jsonl": Path("data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl"),
        "teacher_checkpoint_path": Path(SUKIM_ROOT / "base_weights/Alpamayo-1.5-10B"),
        "student_dtype": "bfloat16", "device": args.device, "student_model": "",
        "ae_init_mode": str(args.ae_init_mode), "attn_implementation": "sdpa",
        "disable_student_deepstack": False, "qat_quantization": "", "qat_calib_samples": 256,
        "num_samples": 10, "val_samples": 5, "val_fraction": 0.1,
        "split_seed": None, "split_cache_json": None, "split": "train",
        "split_scan_all": True, "compressed_layers": int(args.compressed_layers), "mapping": str(args.mapping),
        "ae_dtype": "float16", "prefix_mode": "teacher_forced",
        "preserve_flex_positions": True, "flex_selection_strategy": "uniform",
        "flex_scene_deepstack": True, "target_source": "teacher",
        "max_new_tokens": 160, "max_length": 4096,
        "stage2_attention_mode": "official_none", "seed": 42,
        "teacher_load_device": "cpu",
        "init_ae_source_checkpoint": str(args.init_ae_source_checkpoint),
    }.items():
        setattr(ae_args, k, v)

    # Load student (for FLEX encoder)
    student, _, _, _ = ae.load_student(ae_args)
    flex_encoder = getattr(student, "flex_encoder", None) or getattr(student, "flex_scene_encoder", None)

    # ===== 1. Export FLEX Encoder =====
    if not args.skip_flex and flex_encoder is not None:
        print(json.dumps({"event": "flex_export_start"}), flush=True)
        flex_wrapper = FlexEncoderONNX(flex_encoder).to(device=device, dtype=dtype).eval()

        n_vis = int(args.flex_n_vis)  # default: 16 images × 180 tokens each
        if n_vis % int(flex_wrapper.images_per_sample) != 0:
            raise ValueError(
                f"--flex-n-vis must be divisible by images_per_sample={int(flex_wrapper.images_per_sample)}, "
                f"got {n_vis}."
            )
        dummy_ds0 = torch.randn(1, n_vis, 2048, device=device, dtype=dtype)
        dummy_ds1 = torch.randn(1, n_vis, 2048, device=device, dtype=dtype)
        dummy_ds2 = torch.randn(1, n_vis, 2048, device=device, dtype=dtype)
        dummy_final = torch.randn(1, n_vis, 2048, device=device, dtype=dtype)
        tokens_per_source_image = n_vis // int(flex_wrapper.images_per_sample)
        camera_pattern = torch.arange(
            int(flex_wrapper.images_per_sample),
            device=device,
            dtype=torch.long,
        ).view(1, -1, 1)
        camera_pattern = camera_pattern.expand(1, -1, tokens_per_source_image).reshape(1, n_vis)
        dummy_camera_ids = camera_pattern.clamp(max=max(int(flex_wrapper.max_camera_types) - 1, 0))
        time_pattern = torch.arange(
            tokens_per_source_image,
            device=device,
            dtype=dtype,
        ).view(1, 1, -1, 1)
        time_pattern = time_pattern.expand(1, int(flex_wrapper.images_per_sample), -1, -1)
        dummy_relative_times = (time_pattern / max(tokens_per_source_image - 1, 1)).reshape(1, n_vis, 1)

        flex_onnx_path = str(output_dir / "flex_encoder.onnx")
        try:
            with torch.no_grad():
                torch.onnx.export(
                    flex_wrapper,
                    (
                        dummy_ds0,
                        dummy_ds1,
                        dummy_ds2,
                        dummy_final,
                        dummy_camera_ids,
                        dummy_relative_times,
                    ),
                    flex_onnx_path,
                    input_names=[
                        "ds_level0",
                        "ds_level1",
                        "ds_level2",
                        "final_visual",
                        "camera_ids",
                        "relative_times",
                    ],
                    output_names=[
                        "scene_embeds",
                        "deepstack_scene_0",
                        "deepstack_scene_1",
                        "deepstack_scene_2",
                    ],
                    dynamic_axes={
                        "ds_level0": {0: "batch", 1: "n_vis_tokens"},
                        "ds_level1": {0: "batch", 1: "n_vis_tokens"},
                        "ds_level2": {0: "batch", 1: "n_vis_tokens"},
                        "final_visual": {0: "batch", 1: "n_vis_tokens"},
                        "camera_ids": {0: "batch", 1: "n_vis_tokens"},
                        "relative_times": {0: "batch", 1: "n_vis_tokens"},
                        "scene_embeds": {0: "batch"},
                        "deepstack_scene_0": {0: "batch"},
                        "deepstack_scene_1": {0: "batch"},
                        "deepstack_scene_2": {0: "batch"},
                    },
                    opset_version=args.opset,
                    do_constant_folding=True,
                )
            size_mb = os.path.getsize(flex_onnx_path) / 1e6
            print(json.dumps({"event": "flex_export_done", "path": flex_onnx_path, "size_mb": round(size_mb, 1)}), flush=True)
            if bool(args.verify_onnx):
                _verify_onnx_signature(
                    Path(flex_onnx_path),
                    expected_inputs=[
                        "ds_level0",
                        "ds_level1",
                        "ds_level2",
                        "final_visual",
                        "camera_ids",
                        "relative_times",
                    ],
                    expected_outputs=[
                        "scene_embeds",
                        "deepstack_scene_0",
                        "deepstack_scene_1",
                        "deepstack_scene_2",
                    ],
                )
        except Exception as e:
            print(json.dumps({"event": "flex_export_error", "error": str(e)}), flush=True)
            raise
    else:
        print(json.dumps({
            "event": "flex_skip" if args.skip_flex else "flex_not_found",
            "skip_flex": bool(args.skip_flex),
        }), flush=True)

    ae_checkpoint_arg = args.ae_checkpoint or args.ae28_checkpoint
    if bool(args.skip_ae):
        print(json.dumps({"event": "ae_skip", "output_dir": str(output_dir)}), flush=True)
        print(json.dumps({"event": "all_done", "output_dir": str(output_dir)}), flush=True)
        return
    if not ae_checkpoint_arg:
        raise ValueError("--ae-checkpoint or --ae28-checkpoint is required unless --skip-ae is set.")

    # Load teacher (for AE28 bundle)
    _load_fn = getattr(ae, "load_model_and_processor", None)
    if not _load_fn:
        from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import load_model_and_processor as _load_fn
    teacher_model, _, _, _, _ = _load_fn(
        checkpoint_path=ae_args.teacher_checkpoint_path, dtype=torch.bfloat16,
        device="cpu", config_json=None, runtime_support=None,
        attn_implementation="sdpa", min_pixels=163840, max_pixels=196608)
    teacher_model.eval()

    # Build and load AE28
    bundle, _ = ae.build_bundle(teacher_model, ae_args, student=student)
    ae.load_bundle_checkpoint(Path(ae_checkpoint_arg), bundle=bundle)
    bundle.eval()

    # ===== 2. Export AE Single Step =====
    is_causal = not bool(getattr(teacher_model.config, "expert_non_causal_attention", False))
    ae_wrapper = AESingleStepONNX(bundle, is_causal=is_causal).to(device=device, dtype=dtype).eval()

    n_layers = int(bundle.expert.config.num_hidden_layers)
    print(json.dumps({
        "event": "ae_export_start",
        "checkpoint": str(ae_checkpoint_arg),
        "n_layers": n_layers,
    }), flush=True)
    n_kv_heads = int(getattr(bundle.expert.config, "num_key_value_heads", 16))
    head_dim = 128
    kv_seq_len = 893  # typical FLEX-compressed sequence length

    dummy_action = torch.randn(1, 64, 2, device=device, dtype=dtype)
    dummy_t = torch.full((1, 1, 1), 0.5, device=device, dtype=dtype)
    dummy_pos = torch.arange(64, device=device).view(1, 1, 64).expand(3, 1, 64).long()
    dummy_k = torch.randn(n_layers, 1, n_kv_heads, kv_seq_len, head_dim, device=device, dtype=dtype)
    dummy_v = torch.randn(n_layers, 1, n_kv_heads, kv_seq_len, head_dim, device=device, dtype=dtype)

    ae_output_name = args.ae_output_name or f"ae{n_layers}_single_step.onnx"
    ae_onnx_path = str(output_dir / ae_output_name)
    try:
        with torch.no_grad():
            torch.onnx.export(
                ae_wrapper,
                (dummy_action, dummy_t, dummy_pos, dummy_k, dummy_v),
                ae_onnx_path,
                input_names=["noisy_action", "timestep", "position_ids", "past_keys", "past_values"],
                output_names=["velocity"],
                dynamic_axes={
                    "noisy_action": {0: "batch"},
                    "timestep": {0: "batch"},
                    "position_ids": {1: "batch"},
                    "past_keys": {1: "batch", 3: "kv_seq_len"},
                    "past_values": {1: "batch", 3: "kv_seq_len"},
                    "velocity": {0: "batch"},
                },
                opset_version=args.opset,
                do_constant_folding=True,
            )
        size_mb = os.path.getsize(ae_onnx_path) / 1e6
        print(json.dumps({"event": "ae_export_done", "path": ae_onnx_path, "size_mb": round(size_mb, 1)}), flush=True)
        if bool(args.verify_onnx):
            _verify_onnx_signature(
                Path(ae_onnx_path),
                expected_inputs=["noisy_action", "timestep", "position_ids", "past_keys", "past_values"],
                expected_outputs=["velocity"],
            )
    except Exception as e:
        print(json.dumps({"event": "ae_export_error", "error": str(e)}), flush=True)
        raise

    print(json.dumps({"event": "all_done", "output_dir": str(output_dir)}), flush=True)


if __name__ == "__main__":
    main()
