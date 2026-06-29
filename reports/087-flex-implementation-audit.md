# 087 FLEX Implementation Audit

Generated: 2026-06-05

## Verdict

Current `cosmos_distillation` FLEX module is structurally capable of matching the Alpamayo-R1 paper-scale FLEX parameter count when configured for the 10B VLM hidden size.

The current checked-in FLEX training configs are not paper-scale because they are configured for the smaller 2B/student hidden size and a smaller FLEX internal width:

- Alpamayo-R1 paper table: FLEX added parameters = `61.6M`.
- Our current F0 config: `29.917M`.
- Our current F1/F7 configs: `31.378M`.

For the 10B VLM basis, `input_hidden_size=4096`. With the same local architecture, setting `hidden_size=1408`, `num_layers=2`, `num_heads=16` gives `61.35M-62.30M` depending on tokens/image and camera/time embeddings. That matches the paper-reported `61.6M` scale.

Therefore:

- Current configs: not paper-scale.
- Current code structure: can instantiate a paper-scale 10B FLEX module.

## Paper Reference

Alpamayo-R1 reports efficient vision encoder ablation with:

- Baseline: `0` added params, `160` tokens/image.
- Triplane: `6.3M` added params, `104` or `45` tokens/image.
- Flex: `61.6M` added params, `50`, `32`, `16`, or `8` tokens/image.

The paper text says Flex can compress up to `20x` while adding `61.6M` parameters and matching baseline driving quality.

## Local Implementation

Code:

- `src/model/flex_scene_encoder.py`
- `src/model/student_wrapper.py`
- `src/training/flex_batch.py`

Core local architecture:

- LayerNorm on visual tokens.
- Linear projection from input hidden size to FLEX hidden size.
- Optional camera/time embeddings.
- Learned scene tokens.
- `nn.TransformerEncoder` over `[scene_tokens, projected_visual_tokens]`.
- Return only encoded scene tokens, projected back to LLM hidden size.

This matches the general FLEX idea: fixed learned queries attend over multi-view visual tokens and output a compact visual prefix.

## Parameter Counts

Measured by instantiating `FlexSceneEncoder` directly from each config.

| Config | Scene tokens | Hidden | Layers | Camera/time | Params |
|---|---:|---:|---:|---|---:|
| `stage_flex_f0_4cam4frame_k512.yaml` | 512 | 1024 | 2 | no | 29.917M |
| `stage_flex_f1_4cam4frame_k896_camtime_lora_top2_lowlr_norm.yaml` | 896 | 1024 | 2 | yes | 31.378M |
| `stage_flex_f7_4cam4frame_k896_camtime_lora_top8_large_ce.yaml` | 896 | 1024 | 2 | yes | 31.378M |

F1/F7 parameter breakdown:

| Module | Params |
|---|---:|
| Transformer encoder | 25.192M |
| input projection | 2.097M |
| output projection | 2.097M |
| scene tokens | 0.918M |
| time MLP | 1.052M |
| camera embedding | 0.016M |
| norms | 0.006M |

## Why It Misses 61.6M

With this architecture, parameter count is dominated by the internal Transformer hidden size, not token count.

Current configs use:

- `hidden_size: 1024`
- `num_layers: 2`

That naturally lands near `30M`.

## 10B Paper-Scale Check

The local 10B VLM config under `base_weights/alpamayo15_vlm_weights/config.json` has:

- text hidden size: `4096`
- text layers: `36`
- text attention heads: `32`
- vision hidden size: `1152`
- vision output hidden size: `4096`

Qwen3-VL `get_image_features()` returns image embeddings from `self.visual(...)`, split by image grid. Since the vision config has `out_hidden_size=4096`, the FLEX input/output hidden size should be `4096` for the 10B model.

Measured 10B-basis local FLEX counts:

| Camera/time | Tokens/image | Scene tokens | Input hidden | FLEX hidden | Layers | Heads | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| no | 50 | 800 | 4096 | 1408 | 2 | 16 | 60.287M |
| no | 32 | 512 | 4096 | 1408 | 2 | 16 | 59.882M |
| no | 16 | 256 | 4096 | 1408 | 2 | 16 | 59.522M |
| no | 8 | 128 | 4096 | 1408 | 2 | 16 | 59.341M |
| yes | 50 | 800 | 4096 | 1408 | 2 | 16 | 62.297M |
| yes | 32 | 512 | 4096 | 1408 | 2 | 16 | 61.891M |
| yes | 16 | 256 | 4096 | 1408 | 2 | 16 | 61.531M |
| yes | 8 | 128 | 4096 | 1408 | 2 | 16 | 61.351M |

The `tokens/image=16` + camera/time setting lands at `61.531M`, almost exactly the paper's `61.6M`. The `tokens/image=32` setting lands at `61.891M`, also within rounding distance.

A direct random forward check also passes:

- config: `input_hidden_size=4096`, `hidden_size=1408`, `num_layers=2`, `num_heads=16`, `tokens_per_image=32`, `expected_images_per_sample=16`
- input: `[1, 2560, 4096]`
- output: `[1, 512, 4096]`
- params: `61.891M`

## 2B/Student-Scale Variants

Approximate 2B/student-scale variants with the same local architecture:

| Variant | Tokens/image | Scene tokens | Hidden | Layers | Params |
|---|---:|---:|---:|---:|---:|
| close exact | 50 | 800 | 1504 | 2 | 61.698M |
| rounder size | 50 | 800 | 1536 | 2 | 64.190M |
| rounder size | 32 | 512 | 1536 | 2 | 63.748M |
| rounder size | 16 | 256 | 1536 | 2 | 63.355M |

These earlier counts used `input_hidden_size=2048`, which is appropriate for the current 2B/student baseline but not for the 10B paper basis.

## Source Scope

Do not use `/home/pm97/workspace/sukim/alpamayo_repo/*` as an official reference for FLEX. That tree is workspace-local/user-created and is excluded from this audit.

This audit relies only on:

- Alpamayo-R1 paper-reported FLEX parameter/token numbers.
- Current `cosmos_distillation` implementation and direct parameter counts.

## Additional Implementation Caveat

`src/training/flex_batch.py` compresses sequence placeholders by keeping the first `tokens_per_image` image tokens inside each original image block and dropping the rest.

`src/model/student_wrapper.py` then computes global scene tokens from all visual features and writes them back into the remaining image placeholder positions.

This is operationally valid for a smoke experiment, but it may not be exactly paper-faithful. A stricter implementation would define explicitly where the compact scene-token block lives in the prompt/KV sequence and ensure positional semantics match that design.

## Public Source Check: FLEX Insertion Semantics

The FLEX paper/project page publicly specifies the high-level insertion semantics:

- initialize learnable scene tokens;
- prepend scene tokens to all image tokens across cameras and timesteps;
- run a lightweight Transformer encoder with full self-attention over the combined sequence;
- discard image tokens after encoding;
- keep only updated scene tokens as the scene representation passed to the LLM/policy model.

The paper also says image tokens receive camera and timestep embeddings, and the figure caption notes special tokens for modality start/end, camera type, and timestamp information.

For interleaved prediction, the paper further states that the holistic scene tokens are evenly partitioned into sequential non-overlapping chunks, and at prediction step `t` the policy conditions on the first `t` chunks.

What is not publicly available from the checked official sources:

- a full released FLEX runtime implementation in NVlabs Alpamayo or Alpamayo Recipes;
- exact low-level prompt placeholder surgery code;
- exact KV-cache construction code for an Alpamayo/Qwen runtime.

Therefore our implementation should treat the paper as the semantic contract, not as byte-level code to copy.

## Recommended Fix

Do not start serious 10B-paper-scale FLEX training from the current `hidden_size=1024`, `input_hidden_size=2048` configs if the goal is to reproduce the Alpamayo-R1 FLEX ablation scale.

Recommended sequence:

1. Keep existing F0/F1/F7 `hidden_size=1024` configs only as cheap 2B/student wiring smoke tests.
2. Add a 10B paper-scale config:
   - `input_hidden_size: 4096`
   - `tokens_per_image: 32`, `16`, or paper table row under test
   - `hidden_size: 1408`
   - `num_layers: 2`
   - `num_heads: 16`
   - `use_camera_time_embeddings: true`
3. Recount parameters and profile VRAM before training.
4. Only call the result "paper-scale FLEX" after the parameter count is near `61.6M` under the 10B hidden-size basis.

One-line conclusion: the code path is structurally capable; the checked-in configs are not paper-scale, but a 10B config with `input_hidden_size=4096`, `hidden_size=1408`, `layers=2`, `heads=16` matches the paper's `61.6M` parameter scale.
