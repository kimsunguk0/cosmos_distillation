# 042. Follow-up Absorption Tests Regressed Back To Single-Token Plateau

## Summary

Two follow-up training ideas were tested after the first partial absorption result in [041-prefix16-pseudoce-partially-absorbs-hybrid-body.md](/workspace/cosmos_distillation/reports/041-prefix16-pseudoce-partially-absorbs-hybrid-body.md):

1. a softer `prefix32` continuation starting from the successful `prefix16` checkpoint
2. direct LM self-distillation from the actual non-collapsed `hybrid decode` body

Neither improved on `prefix16`.

The headline is simple:

- `prefix16` remains the only training path that partially weakens the LM-only plateau
- both follow-up tests fell back to a full single-token plateau

Comparison artifact:

- [subset_t1_lm_absorb_followup_compare.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_lm_absorb_followup_compare.json)

## A. Softer Prefix32 Continuation

Config:

- [stage_t1_lm_absorb_auxpseudo_prefix32_soft.yaml](/workspace/cosmos_distillation/configs/train/stage_t1_lm_absorb_auxpseudo_prefix32_soft.yaml)

This run started from:

- [subset_t1_lm_absorb_auxpseudo_prefix16_train4_step20/final](/workspace/cosmos_distillation/outputs/checkpoints/subset_t1_lm_absorb_auxpseudo_prefix16_train4_step20/final)

and used:

- `traj_body_prefix_tokens: 32`
- lower `traj_aux_pseudo_ce_loss: 0.4`
- higher `traj_aux_guided_kd_loss: 0.5`

Artifacts:

- checkpoint: [subset_t1_lm_absorb_auxpseudo_prefix32_soft_from_prefix16_train4_step20/final](/workspace/cosmos_distillation/outputs/checkpoints/subset_t1_lm_absorb_auxpseudo_prefix32_soft_from_prefix16_train4_step20/final)
- summary: [subset_t1_lm_absorb_auxpseudo_prefix32_soft_from_prefix16_train4_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_lm_absorb_auxpseudo_prefix32_soft_from_prefix16_train4_step20.json)
- train decode: [subset_t1_lm_absorb_auxpseudo_prefix32_soft_from_prefix16_train4_step20_infer_train_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_lm_absorb_auxpseudo_prefix32_soft_from_prefix16_train4_step20_infer_train_lm.json)
- val decode: [subset_t1_lm_absorb_auxpseudo_prefix32_soft_from_prefix16_train4_step20_infer_val10_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_lm_absorb_auxpseudo_prefix32_soft_from_prefix16_train4_step20_infer_val10_lm.json)

### Training behavior

This run was stable.

- val `total_loss`: `6.9230 -> 6.5545`
- `traj_aux_guided_kd_loss`: `6.41 -> 6.37`
- `traj_aux_pseudo_ce_loss`: `4.43 -> 3.78`
- no explosion or NaNs

So the optimization itself was fine.

### Decode result

But LM-only body decode fully regressed:

- train avg unique traj ids: `1.0`
- train avg max same-token run: `128.0`
- val avg unique traj ids: `1.0`
- val avg max same-token run: `128.0`

The body was almost entirely `<i1512>` repeats.

Interpretation:

- lowering pseudo-CE and raising guided KD was **not enough** to preserve the partial `prefix16` gain once the prefix horizon expanded
- the current aux-center absorption signal still pushes LM argmax back to a single attractor when the supervised prefix gets longer

## B. Direct Hybrid-Decode Pseudo Pair Distillation

The next idea was to stop using aux-derived pseudo labels and instead train the LM on the **actual non-collapsed hybrid decode body**.

To do this, I built:

- pseudo cache: `/workspace/cosmos_distillation/outputs/pseudo_cache/hybridpair_train4`
- corpus overlay: [distill_v3_2_train4_hybridpair.jsonl](/workspace/cosmos_distillation/data/corpus/distill_v3_2_train4_hybridpair.jsonl)

This overlay replaces the four train samples with:

- `teacher_target.cot_text = generated_cot` from the hybrid decode run
- `teacher_traj15.tokens = actual hybrid decoded 128-token body`

Config:

- [stage_t1_hybridpair_pseudolm.yaml](/workspace/cosmos_distillation/configs/train/stage_t1_hybridpair_pseudolm.yaml)

Artifacts:

- checkpoint: [subset_t1_hybridpair_pseudolm_train4_step20/final](/workspace/cosmos_distillation/outputs/checkpoints/subset_t1_hybridpair_pseudolm_train4_step20/final)
- summary: [subset_t1_hybridpair_pseudolm_train4_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_hybridpair_pseudolm_train4_step20.json)
- train decode: [subset_t1_hybridpair_pseudolm_train4_step20_infer_train_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_hybridpair_pseudolm_train4_step20_infer_train_lm.json)

### Training behavior

This run also trained stably.

- `gt_traj_loss`: `14.375 -> 12.8125`
- grad norm stayed roughly in the `65-112` range
- no instability or divergence

### Decode result

LM-only decode still collapsed completely:

- train avg unique traj ids: `1.0`
- train avg max same-token run: `128.0`

The body reverted to pure `<i1499>` repetition.

Interpretation:

- even giving LM the **actual non-collapsed hybrid sequence** as the hard teacher body was not enough
- this strongly suggests the problem is no longer “bad pseudo labels”
- the bottleneck is now the **LM argmax path itself**, not just the supervision source

## What This Changes

The follow-up tests narrow the situation quite a bit.

### Confirmed

- `prefix16 pseudo-CE` remains the only partial training-side absorption success
- extending the same idea to longer prefixes currently collapses
- replacing aux-derived pseudo labels with actual hybrid body tokens still collapses

### Implication

The current best path is still:

- inference: `hybrid decode`
- training: `prefix16 pseudo-CE` as a weak partial absorber

But the next meaningful research step is **not** more of the same prefix extension.

The evidence now points to:

1. LM-only argmax path is the hard bottleneck
2. richer pseudo targets alone do not fix it
3. the next solution likely needs either:
   - direct training on a hybrid-style blended decode contract
   - scheduled prefix forcing / rollout absorption
   - or a trajectory path that is not reduced to a plain LM argmax body

## Bottom Line

Two plausible continuations were tested.

- softer `prefix32` continuation: stable but regressed to full plateau
- direct hybrid-body pseudo pair distill: also regressed to full plateau

So the current state is:

**hybrid decode still works, `prefix16` still gives partial absorption, but simply training harder or with richer pseudo labels does not yet transfer the hybrid body into stable LM-only decoding.**
