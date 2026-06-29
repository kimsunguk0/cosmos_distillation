# 080 Stage 1 Held-Out Training Results

Date: 2026-06-02

## Run

- Original run: `outputs/action_expert/stage1_heldout20k_val2k_s10000_seed42_full444k_20260531`
- Fast resume: `outputs/action_expert/stage1_fast_resume_s5000_b8_fa2_20260601_081126`
- Resume checkpoint: original `step_005000.pt`
- Final status: `ok`, completed to step 10000
- Best checkpoint by held-out val mean ADE: fast resume `best.pt` at step 9000
- Final checkpoint: fast resume `final.pt` at step 10000

## Config

- Train samples: 20000
- Held-out val samples: 2000
- Train/val overlap: 0 sample ids, 0 split groups
- Prefix mode: `student_free`
- AE init: `student_backbone_init`
- Target source: teacher
- Expert LR: `1e-4`
- Projection LR: `1e-4`
- `num_time_samples`: 16
- Fast resume batch size: 8
- Effective FM batch: 128
- Eval: temperature 0.85, N=16 paths, `mean_traj`
- Attention: `flash_attention_2`

Note: steps 0-5000 evaluated 2000 val samples. Steps 6000-10000 used the fast setting with 512 val samples and 256 train-eval samples.

## Eval Curve

| step | val ADE | val p50 | val h1.6 | val h3.2 | val oracle best-N | train ADE | train h1.6 | train h3.2 | train oracle best-N | gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.678 | 8.612 | 0.829 | 3.060 | 6.491 | 9.094 | 0.739 | 2.662 | 5.562 | 1.584 |
| 1000 | 8.804 | 6.014 | 0.565 | 2.238 | 6.331 | 7.254 | 0.467 | 1.832 | 5.132 | 1.550 |
| 2000 | 3.755 | 3.054 | 0.239 | 0.932 | 2.639 | 3.426 | 0.217 | 0.843 | 2.413 | 0.329 |
| 3000 | 3.139 | 2.440 | 0.196 | 0.781 | 2.135 | 2.800 | 0.175 | 0.688 | 1.860 | 0.339 |
| 4000 | 3.441 | 2.953 | 0.219 | 0.858 | 2.368 | 3.085 | 0.202 | 0.772 | 2.099 | 0.356 |
| 5000 | 2.712 | 1.998 | 0.168 | 0.674 | 1.802 | 2.400 | 0.153 | 0.589 | 1.565 | 0.312 |
| 6000 | 2.713 | 1.996 | 0.164 | 0.668 | 1.667 | 2.400 | 0.141 | 0.567 | 1.382 | 0.313 |
| 7000 | 2.512 | 1.758 | 0.146 | 0.612 | 1.539 | 2.431 | 0.132 | 0.557 | 1.425 | 0.081 |
| 8000 | 2.810 | 2.222 | 0.157 | 0.654 | 1.704 | 2.613 | 0.143 | 0.600 | 1.429 | 0.197 |
| 9000 | 2.503 | 1.824 | 0.162 | 0.645 | 1.311 | 2.282 | 0.137 | 0.553 | 1.170 | 0.221 |
| 10000 | 2.517 | 1.733 | 0.144 | 0.606 | 1.288 | 2.293 | 0.126 | 0.520 | 1.119 | 0.224 |

## Loss / Vector Field

- Loss is noisy per batch but trends down strongly late in training.
- Mean train loss by 1000-step bin:
  - 5000: 0.946
  - 6000: 0.832
  - 7000: 0.784
  - 8000: 0.789
  - 9000: 0.610
  - 10000: 0.275
- `pred_v_abs_mean / target_v_abs_mean` improves from under-amplitude to near matched:
  - 5000 bin: 0.581 / 0.986
  - 9000 bin: 0.827 / 1.042
  - 10000 bin: 0.976 / 1.043

Interpretation: the previous random-FM amplitude collapse is fixed. The model is learning the velocity field.

## Conclusions

- Held-out baseline: best val full ADE = 2.503 m at step 9000.
- Final step 10000 is very close: val full ADE = 2.517 m, with better p50 and oracle best-N.
- Train/val gap is small after 7000 steps, around 0.08-0.22 m. This is not an overfitting-only failure.
- Short horizon is good: held-out h1.6 ADE is about 0.14-0.16 m.
- Medium horizon is moderate: held-out h3.2 ADE is about 0.61-0.65 m late in training.
- The remaining failure is long-horizon 6.4s behavior: full ADE stays around 2.5 m.
- Oracle best-N improves to 1.29 m, but is still far above the 0.5 m Stage 0 strict target. So held-out long-horizon quality is not solved by the current deployable `mean_traj` selection.

One-line result: held-out baseline = 2.50 m, Stage 2 갈 자격 = partial/conditional; train-val generalization is okay, but long-horizon held-out ADE remains too high for deployment-quality action expert.
