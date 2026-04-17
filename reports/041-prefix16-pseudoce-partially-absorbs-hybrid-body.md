# 041. Prefix-16 Pseudo-CE Partially Absorbs Hybrid Body Into LM Decode

## Summary

The first successful training-side absorption result is now in.

Pure LM body decode from the mixed-from-format checkpoint was previously a full single-token plateau:

- train avg unique traj ids: `1.0`
- train avg max same-token run: `128.0`
- val avg unique traj ids: `1.0`
- val avg max same-token run: `128.0`

After continuing from that checkpoint with a prefix-only auxiliary pseudo-label objective, LM-only decode no longer stays in the old full plateau regime.

## What Was Tried

### Attempt A: soft aux-guided KD only

Config:

- [stage_t1_lm_absorb_auxblend.yaml](/workspace/cosmos_distillation/configs/train/stage_t1_lm_absorb_auxblend.yaml)

Checkpoint and summary:

- [subset_t1_lm_absorb_auxblend_train4_step20/final](/workspace/cosmos_distillation/outputs/checkpoints/subset_t1_lm_absorb_auxblend_train4_step20/final)
- [subset_t1_lm_absorb_auxblend_train4_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_lm_absorb_auxblend_train4_step20.json)

Result:

- stable training
- lower val loss
- but LM-only decode stayed fully collapsed

Metrics:

- train avg unique traj ids: `1.0`
- train avg max same-token run: `128.0`

So soft prior matching by itself did not move LM argmax behavior enough.

### Attempt B: prefix-only pseudo CE + weak soft prior

Config:

- [stage_t1_lm_absorb_auxpseudo_prefix16.yaml](/workspace/cosmos_distillation/configs/train/stage_t1_lm_absorb_auxpseudo_prefix16.yaml)

This stage uses:

- teacher-pair LM supervision
- `traj_aux_guided_kd_loss`
- stronger `traj_aux_pseudo_ce_loss`
- `traj_body_prefix_tokens: 16`
- frozen `traj_aux_head`

Checkpoint and summaries:

- [subset_t1_lm_absorb_auxpseudo_prefix16_train4_step20/final](/workspace/cosmos_distillation/outputs/checkpoints/subset_t1_lm_absorb_auxpseudo_prefix16_train4_step20/final)
- [subset_t1_lm_absorb_auxpseudo_prefix16_train4_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_lm_absorb_auxpseudo_prefix16_train4_step20.json)
- train decode: [subset_t1_lm_absorb_auxpseudo_prefix16_train4_step20_infer_train_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_lm_absorb_auxpseudo_prefix16_train4_step20_infer_train_lm.json)
- val decode: [subset_t1_lm_absorb_auxpseudo_prefix16_train4_step20_infer_val10_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_lm_absorb_auxpseudo_prefix16_train4_step20_infer_val10_lm.json)

## Main Result

LM-only decode improved from a full `1-token / 128-run` plateau to a weaker multi-state plateau.

Comparison artifact:

- [subset_t1_lm_absorb_compare.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_lm_absorb_compare.json)

### Train

- baseline mixed-from-format LM decode
  - avg unique traj ids: `1.0`
  - avg max same-token run: `128.0`
- prefix16 pseudo-CE LM decode
  - avg unique traj ids: `3.0`
  - avg max same-token run: `62.75`

### Val

- baseline mixed-from-format LM decode
  - avg unique traj ids: `1.0`
  - avg max same-token run: `128.0`
- prefix16 pseudo-CE LM decode
  - avg unique traj ids: `3.0`
  - avg max same-token run: `61.3`

Edge saturation stayed at `0.0`, so this gain did not come from falling into the old `<i0>/<i2999>` failure mode.

## Interpretation

This is not the final answer, but it is a real step forward.

What it means:

1. The hybrid body path can be partially absorbed into LM-only decoding.
2. Trying to absorb the whole 128-token body at once was too hard.
3. Absorbing only the first 16 body tokens worked better.
4. The main problem is still not solved, because the resulting LM body is still a coarse plateau (`1512 -> 1497 -> 1499` style staging), not a rich trajectory body.

So the current best reading is:

- full-body absorption: too hard right now
- prefix-only absorption: works partially
- next likely move: extend the prefix curriculum gradually instead of jumping straight to the whole body

## Recommended Next Step

Keep the same recipe, but turn it into a staged curriculum:

1. `prefix16`
2. `prefix32`
3. `prefix64`

and only then test full-body absorption again.

The evidence so far suggests the student can absorb the hybrid path, but only incrementally.

## Follow-up: prefix32

I also repeated the same recipe with:

- [stage_t1_lm_absorb_auxpseudo_prefix32.yaml](/workspace/cosmos_distillation/configs/train/stage_t1_lm_absorb_auxpseudo_prefix32.yaml)
- checkpoint: [subset_t1_lm_absorb_auxpseudo_prefix32_train4_step20/final](/workspace/cosmos_distillation/outputs/checkpoints/subset_t1_lm_absorb_auxpseudo_prefix32_train4_step20/final)
- train decode: [subset_t1_lm_absorb_auxpseudo_prefix32_train4_step20_infer_train_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_lm_absorb_auxpseudo_prefix32_train4_step20_infer_train_lm.json)

Result:

- no meaningful improvement over prefix16
- train avg unique traj ids stayed at `3.0`
- train avg max same-token run stayed around `62.5`

So the first real gain came from moving from full-body absorption to prefix-only absorption.
Going from prefix16 to prefix32 did not produce another clear jump yet.
