# FLEX Audit - 2026-06-05

## Scope

Checked FLEX dataset contract, model structure, backbone integration, checkpoint loading, training hyperparameters, and existing smoke artifacts.

## Dataset Contract

- Corpus checked:
  - `outputs/tmp/flex_smoke_256.jsonl`: 256/256 rows use 16 images.
  - `data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl`: first 5000 rows checked.
- Image layout:
  - `image_count=16`
  - `camera_count=4`
  - `num_frames_per_camera=4`
  - `image_layout=materialized_4x4_png`
  - `image_names=cam0_f0..cam3_f3`
- Metadata check on first 200 rows:
  - `metadata_path` exists for all checked rows.
  - `camera_indices=(0,1,2,6)` for all checked rows.
  - `absolute_timestamps_us` exists with shape 4x4 for all checked rows.
  - no missing image files found.

Verdict: dataset/image/camera-time contract is OK for 4-camera x 4-frame FLEX.

## Token Compression Contract

Existing smoke logs show:

- Original visual placeholders: 2880 image tokens/sample.
- K512 FLEX: compressed visual placeholders: 512.
- Compression ratio: 5.625x.
- Full training sequence length after compression: about 859-884 tokens in smoke.

Code path:

- `src/training/flex_batch.py` drops surplus image placeholders while preserving aligned tensors.
- `src/model/student_wrapper.py` recomputes full Qwen visual features, feeds all visual tokens into `FlexSceneEncoder`, then replaces the compressed image placeholder positions with scene tokens.
- The wrapper validates that compressed placeholder count equals `scene_tokens`.

Verdict: token count and replacement checks are structurally sound.

## FLEX Model Structure

Current smoke config:

- `tokens_per_image=32`
- `expected_images_per_sample=16`
- `scene_tokens=512`
- `input_hidden_size=2048`
- `hidden_size=1024`
- `num_layers=2`
- `num_heads=8`
- camera/time embeddings enabled.

Parameter count:

- 2B K512 cam/time: 30.985M params.
- 2B K896 cam/time: 31.378M params.
- If using input 2048 and hidden 1536: 66.137M params.
- If using input 2048 and hidden 2048: 114.395M params.

Local model configs:

- Cosmos Reason 2B text hidden size: 2048.
- Alpamayo-1.5-10B expert hidden size: 2048 in local config.

Verdict: current 2B FLEX structure is internally consistent. It is not a 61.6M-parameter replica; matching that paper number would require a larger FLEX hidden size/spec, not the current 1024-hidden 2-layer config.

## Training Hyperparameters

Current smoke launcher:

- config: `configs/train/stage_flex_2b_k512_camtime_smoke.yaml`
- corpus default: `data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl`
- init checkpoint: `outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250`
- `batch_size=1`
- `learning_rate=1e-4`
- `gradient_checkpointing=true`
- `freeze_all_parameters=true`
- `unfreeze_flex_scene_encoder=true`
- loss key aliases are supported, so legacy keys like `gt_cot_loss`, `traj_loss`, `output_format_loss` are not ignored.

Existing smoke result:

- 50 train steps completed.
- final checkpoint saved.
- `flex_scene_encoder.pt` size: 119M.
- optimizer groups included `flex_scene_encoder`.

Verdict: smoke setup is valid for code-path verification, but not sufficient as a real baseline because it has no held-out eval and only 50 steps.

## Integration Gap

Token distillation/eval paths are FLEX-aware:

- Training path compresses batches through `compress_batch_for_flex`.
- Checkpoint eval uses manual FLEX prefill/generation.
- Checkpoint save/load preserves `flex_scene_encoder` and `flex_scene_config`.

Action expert path is not currently FLEX-ready:

- `scripts/84_train_student_ae28_official.py` loads `StudentWrapper`, but generation/prefill uses `student.backbone.generate()` and `student.backbone(...)` directly.
- That bypasses `DistillStudentModel._forward_flex`.
- `84` also does not compress the AE batch with `compress_batch_for_flex`.

Verdict: do not assume a FLEX checkpoint will affect AE training/eval yet. Before using FLEX as the VLM backbone for action expert distillation, patch `84` to use the FLEX wrapper prefill/generation path and compressed batches.

## Recommendation

- For FLEX-only token experiments: OK to proceed.
- For production FLEX run: use a K896 cam/time config with LoRA top layers, not the K512 smoke config, unless VRAM requires K512.
- For AE + FLEX integration: patch and smoke-test `84` first. Current AE path would silently bypass FLEX behavior.

