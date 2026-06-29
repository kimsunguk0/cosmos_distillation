# FLEX Position And Structure Diagnostic

Date: 2026-06-06

## Context

F5 sequence-level CE proved that stronger generation loss can recover only a small part of B0 camera sensitivity:

- F4b shuffle gap: `+0.053 / +0.128`
- F5 shuffle gap: `+0.324 / +1.021`
- B0 shuffle gap: `+1.064 / +2.740`
- F5 normal regressed to `3.700 / 11.886`

So the remaining question was whether this is mainly token position shift or FLEX content/camera structure loss.

## Code Added

Free-run decode now supports two diagnostic modes:

- `--preserve-flex-positions`
  - compressed sequence
  - preserves original token position ids for kept tokens
  - `_manual_flex_generate` now forwards `position_ids`
  - decode `cache_position` continues from the preserved position range

- `--flex-dummy-image-slots`
  - no sequence compression
  - keeps all original image placeholder slots
  - inserts FLEX scene tokens into the first `K` image positions of each image block
  - leaves surplus image placeholders as default image-token embeddings
  - diagnostic only; no latency benefit

Files touched:

- `src/inference/checkpoint_eval.py`
- `scripts/25_decode_checkpoint_overlays.py`
- `src/model/student_wrapper.py`

Validation:

- `py_compile` passed for modified files.

## Smoke Results

Checkpoint:

- `outputs/checkpoints/flex_f4b_paircontrast_cache_vis68_from_f3_s3000_lr5e6_gc5_20260606/final`

Baseline compressed-mode for first 2 samples from the existing F4b normal summary:

| Sample | ADE | FDE | Unique |
|---|---:|---:|---:|
| `d419...__sg_06` | `2.689` | `9.689` | `51` |
| `9d888...__sg_01` | `8.158` | `26.488` | `2` |

### Preserve Position IDs

Command output:

- summary: `outputs/reports/flex_f6_position_preserve_smoke2_normal_v2_summary.json`
- `preserve_flex_positions=true`
- ADE/FDE: `null / null`
- avg unique traj ids: `2.5`
- avg max same-token run: `186.5`
- avg token match rate: `0.0078`

This collapsed into repeated trajectory tokens.

### Dummy Image Slots

Command output:

- summary: `outputs/reports/flex_f6_dummy_slots_smoke2_normal_summary.json`
- `flex_dummy_image_slots=true`
- ADE/FDE: `null / null`
- avg unique traj ids: `17.0`
- avg max same-token run: `3.5`
- avg token match rate: `0.0`

This did not repeat-collapse like simple position preservation, but it still produced invalid/unusable trajectory tokens.

## Structural Finding

Current `FlexSceneEncoder` is a global scene compressor:

- it projects all visual tokens,
- concatenates learned `scene_tokens` with all projected visual tokens,
- runs one Transformer encoder over the whole sequence,
- returns the first `scene_tokens`.

Important implication:

- `tokens_per_image * 16` controls output count only.
- It does not structurally bind output token ranges to camera/frame blocks.
- The first 56 output tokens are not guaranteed to represent front frame 0, the next 56 front frame 1, etc.
- Camera/time embeddings exist, but the learned global queries can still mix all camera/frame evidence.

This matches the observed failure:

- black-image sensitivity exists, so FLEX carries some visual information.
- camera_shuffle sensitivity is weak, so camera-indexed geometry is not preserved enough.
- stronger sequence CE adds some shuffle gap but damages normal behavior.
- position/dummy-slot inference changes break generation because the model was trained under compressed positional/layout assumptions.

## Conclusion

The main FLEX blocker is not simply insufficient loss or a trivial position-id bug.

Most likely blocker:

- current FLEX architecture is not camera/frame factorized enough for Alpamayo-style camera-order-sensitive driving.

Recommended next design:

1. Replace global scene-token compression with factorized per-image or per-camera compression.
2. Preserve output token block semantics:
   - camera 0 frame 0 tokens
   - camera 0 frame 1 tokens
   - ...
   - camera 6 frame 3 tokens
3. Add optional shallow cross-camera mixing only after per-camera summaries are formed.
4. Re-run the same gate:
   - 16-sample overfit
   - vis68 normal/shuffle/black
   - B0 parity and camera_shuffle gap

One-line status: F6 position/dummy diagnostics FAIL; root is likely FLEX global-compression structure, not just training loss.
