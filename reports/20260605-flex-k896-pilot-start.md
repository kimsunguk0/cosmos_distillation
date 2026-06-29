# FLEX K896 Pilot Start - 2026-06-05

## Baseline

- Backbone init checkpoint: `outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250`
- Base model: `/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b`
- Corpus: `data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl`

## FLEX Target

FLEX replaces the long raw vision placeholder prefix with learned scene tokens.
For the current 16-image input, K896 means `tokens_per_image=56`, so 16 images
produce 896 learned scene tokens.

Primary goal:

- reduce visual token burden from the raw vision-token interface,
- keep camera/time ordering explicit through camera/time embeddings,
- preserve or improve downstream trajectory behavior relative to the best
  backbone baseline after a small LoRA adaptation,
- prepare a compact visual interface that can later be connected to the
  action-expert path once AE training/eval is made FLEX-aware.

Expected result versus baseline:

- short-term: token/traj CE can initially regress because the visual interface
  changed from raw vision tokens to learned scene tokens;
- successful pilot: loss and trajectory token accuracy recover quickly, and
  decode ADE should be close to or better than the baseline after enough data;
- memory/speed: language-side sequence length and KV pressure should drop, but
  the ViT still processes all images, so speedup is mostly from LM/prefill/KV,
  not from image encoding.

## Smoke Test

- Session: `flex_k896_top4_smoke_skipcheck`
- Output dir: `outputs/checkpoints/flex_k896_camtime_top4_smoke_bestbase_20260605_skipcheck`
- Summary: `outputs/reports/flex_k896_camtime_top4_smoke_bestbase_20260605_skipcheck_summary.json`
- Log: `outputs/logs/flex_k896_camtime_top4_smoke_bestbase_20260605_skipcheck.log`
- Data: 512 train records, 9007 val records available
- Steps: 50/50 completed
- Batch: 1
- Result: PASS
- Trainable groups: `language_lora`, `flex_scene_encoder`
- FLEX config: `tokens_per_image=56`, `expected_images_per_sample=16`
- Saved FLEX encoder: `final/flex_scene_encoder.pt` (~120 MB)

## Active 20K Pilot

- Session: `flex_k896_top4_20k_pilot`
- PID at launch check: `16032`
- Output dir: `outputs/checkpoints/flex_k896_camtime_top4_bestbase20k_s3000_b2_20260605`
- Summary target: `outputs/reports/flex_k896_camtime_top4_bestbase20k_s3000_b2_20260605_summary.json`
- Log: `outputs/logs/flex_k896_camtime_top4_bestbase20k_s3000_b2_20260605.log`
- Metrics: `outputs/checkpoints/flex_k896_camtime_top4_bestbase20k_s3000_b2_20260605/metrics.jsonl`
- Data: 20000 train records, 9007 val records available
- Steps: 3000
- Batch: 2
- Effective epoch fraction: 3000 / 10000 steps = 0.3 epoch
- Save cadence: every 1000 steps
- Eval during train: disabled; decode eval should be run on step checkpoints
  after pilot progress is visible.

Observed startup:

- Step 1 total loss: 5.7486
- Step 50 total loss: 1.9454
- Step 100 total loss: 1.7559
- FLEX compressed image tokens per batch sample: 1792 for batch 2, equivalent
  to 896 scene tokens per sample.
- Co-running VRAM with AE Stage2: about 105 GB total, about 38 GB free.

## Caveat

The current AE script path still uses the student backbone directly and does not
yet inject FLEX-compressed scene tokens into AE prefill. This pilot validates and
trains the token/backbone FLEX path. To use FLEX for action expert evaluation or
distillation, the AE script must be patched to build prefill through the FLEX
batch compression / student wrapper path.
