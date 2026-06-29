# 086 FLEX Two-Track Plan

Generated: 2026-06-05

## Current GPU / AE Track State

- Active run: `outputs/action_expert/stage2_200k_more2ep_b8_nt16_minade6_20260605_final_eval_more2ep`
- PID: `19567`
- Resume checkpoint: `outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603/final.pt`
- Step range: `25000 -> 75000`
- Current logged step at inspection: `25400`
- Next save/eval point: `27500`
- Current batch: `batch_size=8`, `num_time_samples=16`, effective FM batch `128`
- Current eval: temp `1.0`, `num_paths=6`, selection `single`
- Current VRAM: about `130.5 GB` used, about `12.6 GB` free on H200.

Conclusion: FLEX training cannot run meaningfully in parallel while AE stays at batch 8. Batch change requires stopping/restarting the AE process.

## AE Track Recommendation

Preferred path if we can wait:

1. Keep current AE run alive until step `27500`.
2. Let it write the first save/eval artifact.
3. If the 27500 eval is useful, continue AE priority or restart with smaller batch only after checkpoint exists.

Preferred path if FLEX must start now:

1. Stop PID `19567`.
2. Restart AE from the previous stable checkpoint at step `25000`.
3. Use `batch_size=4`, `eval_batch_size=4`, keep `num_time_samples=16`.
4. Do not reduce `num_time_samples` first; that was the setting that fixed random-FM collapse.

Tradeoff:

- `batch8`: best AE throughput and same current recipe, but leaves only about 12 GB free.
- `batch4`: likely frees enough VRAM for FLEX smoke/batch1, but changes effective FM batch from 128 to 64 because script 84 currently does not expose gradient accumulation.
- `batch2`: safer for concurrent FLEX but meaningfully weaker AE optimization signal; use only if FLEX is the priority.

## FLEX Baseline

Use the current best distill/VLM baseline:

- `outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250`

Known baseline metrics:

- Greedy decode vs teacher: ADE/FDE `2.557 / 8.365 m`
- Greedy decode vs GT: ADE/FDE `3.011 / 9.822 m`
- Oracle minADE@4 vs teacher: `1.376 m`
- Oracle minADE@4 vs GT: `1.702 m`

This checkpoint has no FLEX module yet. FLEX must be attached as a new visual-prefix bottleneck, then the action expert must be retrained or finetuned against the new hidden/KV distribution.

## FLEX Track Design

### F0 Smoke: FLEX Encoder Only

Base config:

- `configs/train/stage_flex_f0_4cam4frame_k512.yaml`

Settings:

- K = `32 tokens/image * 16 images = 512 scene tokens`
- Trainable: only `flex_scene_encoder`
- Frozen: existing student backbone and LoRA
- LR: `1e-4`
- Batch: `1`

Purpose:

- Verify FLEX wiring, checkpoint save/load, compressed prefix stats, and memory.
- This is not expected to beat baseline yet. It is a correctness and throughput gate.

Pass gate:

- Train loss decreases.
- No format/token collapse.
- FLEX checkpoint loads in decode/profile scripts.
- Prefill profile confirms image-token compression is active.

### F1 Main: K896 + Camera/Time + Top2 LoRA

Base config:

- `configs/train/stage_flex_f1_4cam4frame_k896_camtime_lora_top2_lowlr_norm.yaml`

Settings:

- K = `56 tokens/image * 16 images = 896 scene tokens`
- Camera/time embeddings enabled.
- Trainable: FLEX scene encoder, top-2 language LoRA, final language norm.
- LR: `1e-5`, FLEX lr scale `5.0`.
- Batch: `1`

Purpose:

- Let the upper LLM layers adapt to the compressed visual interface.
- This is the first serious candidate for beating or matching the distill baseline.

Pass gate:

- Teacher-forced CE/traj token loss improves over F0.
- Free-run decode does not regress badly versus `step_006250`.
- Greedy ADE/FDE and oracle minADE@4 are measured against the same baseline eval set.

### F2 Wider Adaptation: K896 + Top8 LoRA

Base config:

- `configs/train/stage_flex_f7_4cam4frame_k896_camtime_lora_top8_large_ce.yaml`

Settings:

- K = 896.
- Camera/time embeddings enabled.
- Trainable: FLEX scene encoder + LoRA from language layer 20 upward.
- LR: `3e-6`, FLEX lr scale `5.0`.

Purpose:

- Use only if F1 improves teacher-forced metrics but free-run decode still lags.
- This opens enough upper-layer capacity to adapt to the new visual-prefix distribution without dense backbone updates.

## Required FLEX Evaluation

For each FLEX candidate, compare against `step_006250` on the same eval split:

- Greedy ADE/FDE vs teacher discrete.
- Greedy ADE/FDE vs GT.
- Oracle minADE@4 vs teacher and GT.
- Format validity and `traj_future_start` hit rate.
- FLEX prefill profile: original seq len, compressed seq len, image-token compression ratio, prefill latency.

Decision rule:

- If FLEX VLM decode is worse than baseline, do not spend AE training on it yet.
- If FLEX VLM matches or beats baseline, train AE on the FLEX checkpoint using the proven AE recipe:
  - `expert_lr=1e-4`
  - `proj_lr=1e-4`
  - `num_time_samples=16`
  - held-out 200k/10k split
  - paper-style eval: temp `1.0`, `num_paths=6`, `single`, report ADE and minADE@6.

## Immediate Next Actions

1. Keep AE batch8 until step `27500` if waiting is acceptable.
2. In parallel, prepare FLEX F0 smoke output directory and command only; do not launch while free VRAM is 12 GB.
3. After AE checkpoint/eval at `27500`, choose:
   - continue AE batch8 if metrics are clearly improving;
   - restart AE batch4 if two-track GPU work is required;
   - stop AE if FLEX becomes the priority.

One-line plan: keep 200k AE alive to the next checkpoint, start FLEX from `step_006250` with F0 encoder-only smoke, then move to F1 K896/top2-LoRA only if F0 is clean.
