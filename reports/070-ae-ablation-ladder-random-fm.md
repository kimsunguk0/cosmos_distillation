# 070 AE Ablation Ladder for Random FM

Date: 2026-05-31

## One-line Verdict

**AE가 random FM 못 배우는 원인 = `action_in_proj`가 아니라 28-layer expert의 optimization 병목. 구체적으로 기본 `expert_lr=1e-5`와 single random draw 조건에서 깊은 expert가 under-trained/gradient-shrunk되어 velocity를 low-amplitude 평균으로 수축한다.**

The fix direction is not another FM sign/target change. The first recipe to validate is:

```text
num_time_samples / random draws per step: 16
expert_lr: 1e-4
proj_lr: 1e-4
```

## Script

Added:

```text
scripts/98_h1_ae_ablation_ladder.py
```

It trains the same one-target random FM task as E3, with no KV:

```text
x0 ~ N(0,I)
t ~ 0.999 - Beta(1.5,1.0) * 0.999
x_t = (1 - t) * x0 + t * x1
target_v = x1 - x0
```

Variants:

- `proj_mlp`: actual `PerWaypointActionInProjV2` -> small token-wise MLP head.
- `expert_N`: actual action projections -> first `N` student expert layers -> action out projection.
- `expert_N_noattn`: same as `expert_N`, but each self-attention update returns zero. This is a temporary H2 ablation.

## H1: Ablation Ladder

Condition:

```text
draws_per_step = 16
steps = 1000
expert_lr = 1e-5
proj_lr = 1e-4
head_lr = 1e-3 for proj_mlp only
```

| rung | architecture | loss | pred_v_abs | target_v_abs | cosine | alpha | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| G0 | independent MLP | 0.1250 | 0.7442 | 0.8195 | 0.9391 | 0.8623 | PASS |
| H1-b | action_in_proj -> MLP head | 0.1794 | 0.7461 | 0.8164 | 0.9107 | 0.8273 | PASS |
| H1-c | action_in_proj -> 2-layer expert -> action_out_proj | 0.4234 | 0.6680 | 0.8242 | 0.7765 | 0.6088 | learns, slower |
| H1-d | action_in_proj -> 28-layer expert -> action_out_proj | 0.7447 | 0.4238 | 0.8164 | 0.5414 | 0.2874 | weak / underfit |
| V2 ref | 28-layer expert, single draw, old E3 | 1.2016 | 0.2324 | 0.8203 | -0.0425 | -0.0123 | FAIL |

H1 conclusion:

- `action_in_proj` is not the blocker. It passes when paired with a small MLP head.
- A shallow 2-layer expert can learn the task, though slower than the MLP head.
- Full 28-layer expert is much harder to optimize at the previous LR.
- Therefore the failure enters with expert depth/optimization, not with the FM target, sampler, or action input projection.

Outputs:

- `outputs/action_expert/h1_ablation_ladder_seed42_draw16/proj_mlp/summary.json`
- `outputs/action_expert/h1_ablation_ladder_seed42_draw16/expert_2/summary.json`
- `outputs/action_expert/h1_ablation_ladder_seed42_draw16/expert_28/summary.json`

## H2: Attention Ablation

Condition:

```text
variant = expert_28_noattn
draws_per_step = 16
steps = 500
expert_lr = 1e-5
proj_lr = 1e-4
```

Result:

| variant | loss | pred_v_abs | target_v_abs | cosine | alpha | verdict |
|---|---:|---:|---:|---:|---:|---|
| 28-layer no-attention update | 1.0359 | 0.1523 | 0.8203 | 0.1604 | 0.0274 | worse |
| 28-layer normal attention, same LR, step500 | 0.9993 | 0.2334 | 0.8203 | 0.2300 | 0.0597 | weak |

H2 conclusion:

Self-attention is not the primary culprit. Removing attention does not rescue learning; it makes the 28-layer stack even more low-amplitude. The problem is deeper than "attention mixes diffusion tokens badly"; the full residual transformer stack is simply not being optimized enough under the old recipe.

Output:

- `outputs/action_expert/h2_noattn_seed42_draw16/expert_28_noattn/summary.json`

## H3: LR / Effective Batch Sensitivity

### Expert LR x10

Condition:

```text
variant = expert_28
draws_per_step = 16
steps = 1000
expert_lr = 1e-4
proj_lr = 1e-4
```

Result:

| variant | loss | pred_v_abs | target_v_abs | cosine | alpha | verdict |
|---|---:|---:|---:|---:|---:|---|
| 28-layer, expert LR x10, draw16 | 0.0427 | 0.8438 | 0.8164 | 0.9803 | 0.9982 | PASS |

This is the key recovery. The same full 28-layer AE that failed at `expert_lr=1e-5` learns the random FM field once expert LR is raised to `1e-4` and each step sees 16 random draws.

### Projection LR x10 Only

Condition:

```text
variant = expert_28
draws_per_step = 16
steps = 500
expert_lr = 1e-5
proj_lr = 1e-3
```

Result:

| variant | loss | pred_v_abs | target_v_abs | cosine | alpha | verdict |
|---|---:|---:|---:|---:|---:|---|
| 28-layer, proj LR x10 only, draw16 | 0.8209 | 0.3008 | 0.8203 | 0.4839 | 0.1955 | still weak |

Projection LR alone does not solve it. The bottleneck is expert update scale.

### Expert LR x10 but Single Draw

Condition:

```text
variant = expert_28
draws_per_step = 1
steps = 1000
expert_lr = 1e-4
proj_lr = 1e-4
```

Result:

| variant | loss | pred_v_abs | target_v_abs | cosine | alpha | verdict |
|---|---:|---:|---:|---:|---:|---|
| 28-layer, expert LR x10, draw1 | 0.9878 | 0.2188 | 0.8164 | 0.2496 | 0.0639 | FAIL |

Expert LR x10 is not enough under single-draw E3. The recovery needs both higher expert LR and more random FM draws per step.

Outputs:

- `outputs/action_expert/h3_lr_grid_seed42_draw16/expert28_x10/expert_28/summary.json`
- `outputs/action_expert/h3_lr_grid_seed42_draw16/proj_only_x10/expert_28/summary.json`
- `outputs/action_expert/h3_lr_grid_seed42_draw1/expert28_x10/expert_28/summary.json`

## Interpretation

The previous random FM collapse looked like an objective/sign issue because single-draw V2 ended with cosine near or below zero. H1/H3 refine that:

1. The task is learnable.
2. Actual `action_in_proj` can encode the task.
3. A shallow expert learns.
4. The full expert learns too, but only when optimization is strong enough.
5. Projection LR alone cannot fix it.
6. Removing attention cannot fix it.
7. Single random draw remains too noisy even with higher expert LR.

So the old recipe was effectively asking a 1.4B-param 28-layer transformer to learn a stochastic vector field from one random `(x0,t)` sample per step with `expert_lr=1e-5`. It responds by shrinking output amplitude instead of learning the per-draw inverse relation.

## Next Gate

Before returning to full Stage 0/KD, patch the training script to support multi-draw FM per sample and run:

```text
--num-time-samples 16
--expert-lr 1e-4
--proj-lr 1e-4
--lr-warmup-steps 0 or short warmup
```

Expected cheap gate:

- E3 actual bundle, random: cosine > 0.9
- Stage 0 32-sample: train loss should stop oscillating at ~1 and `train_inb_ade` should drop materially

If this passes, then the fix is an optimization recipe change, not an architecture rewrite.
