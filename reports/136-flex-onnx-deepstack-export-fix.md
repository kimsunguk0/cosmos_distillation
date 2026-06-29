# 136. FLEX ONNX DeepStack Export Fix

**Date**: 2026-06-12
**Scope**: FLEX K512 TRT export correction

---

## Summary

The previous FLEX ONNX artifact was not a real DeepStack-aware export. It exposed only:

```text
final_visual -> scene_embeds
```

That was insufficient for Qwen3-VL DeepStack because the runtime FLEX path returns both:

```text
scene_embeds
compressed DeepStack features for layers 0/1/2
```

The FLEX export now exposes four visual feature inputs, camera/time metadata inputs, and four outputs.

## Updated Artifact

```text
outputs/trt_export/flex_k512_fp16/flex/flex_encoder.onnx
```

Backup of the previous one-input artifact:

```text
outputs/trt_export/flex_k512_fp16/flex/flex_encoder.oneinput.bak.onnx
```

Current size:

```text
138.8 MB
```

## New ONNX Contract

Inputs:

```text
ds_level0       [batch, n_vis_tokens, 2048]
ds_level1       [batch, n_vis_tokens, 2048]
ds_level2       [batch, n_vis_tokens, 2048]
final_visual    [batch, n_vis_tokens, 2048]
camera_ids      [batch, n_vis_tokens]
relative_times  [batch, n_vis_tokens, 1]
```

Outputs:

```text
scene_embeds       [batch, 512, 2048]
deepstack_scene_0  [batch, 512, 2048]
deepstack_scene_1  [batch, 512, 2048]
deepstack_scene_2  [batch, 512, 2048]
```

## Important Fixes

1. DeepStack feature inputs are no longer pruned by ONNX export.
   - The first fixed export attempt still lost `ds_level0/1/2` because those branches were calculated but not connected to graph outputs.
   - The wrapper now returns `deepstack_scene_0/1/2`, so all four FLEX levels stay in the ONNX graph.

2. Camera/time embeddings are included.
   - The earlier wrapper only used local slot embeddings.
   - The new wrapper applies `camera_embed` and `time_mlp` to both scene queries and projected visual tokens, matching the training/runtime FLEX path.

3. FLEX wrapper parity was verified against the original PyTorch `MultiLevelFlexEncoder`.

```text
output shapes:
  scene_embeds       [1, 512, 2048]
  deepstack_scene_0  [1, 512, 2048]
  deepstack_scene_1  [1, 512, 2048]
  deepstack_scene_2  [1, 512, 2048]

max_abs_diff_by_output:
  [0.0, 0.0, 0.0, 0.0]
```

4. ONNX checker passed.

## Adjacent Export State

`visual/model.onnx` already exports:

```text
output
deepstack_features_0
deepstack_features_1
deepstack_features_2
```

`llm/model.onnx` already has DeepStack inputs:

```text
deepstack_embeds_0  [batch_size, seq_len, 2048]
deepstack_embeds_1  [batch_size, seq_len, 2048]
deepstack_embeds_2  [batch_size, seq_len, 2048]
```

## Remaining Runtime Wiring

The next deployment-side step is not FLEX export anymore. It is tensor wiring:

```text
visual.deepstack_features_0/1/2 + visual.output
  -> flex.ds_level0/1/2/final_visual

flex.scene_embeds
  -> LLM image placeholder positions in inputs_embeds

flex.deepstack_scene_0/1/2
  -> scatter into full-sequence deepstack_embeds_0/1/2
     at the same 512 FLEX image placeholder positions
```

The LLM inputs expect full sequence tensors, not raw `[B, 512, 2048]` FLEX outputs. The runtime must create `[B, seq_len, 2048]` DeepStack tensors and place FLEX DeepStack tokens at the compressed image positions.

## Files Updated

```text
scripts/export_flex_ae_onnx.py
outputs/trt_export/flex_k512_fp16/flex/flex_encoder.onnx
outputs/trt_export/flex_k512_fp16/DEPLOYMENT_NOTES.md
reports/135-checkpoint-registry.md
```
