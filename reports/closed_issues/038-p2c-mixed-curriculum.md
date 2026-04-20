# 038. P2c Mixed Curriculum

## Goal

Combine pair-LM contract maintenance with the now-stable paired-interface training.

- pair-LM steps: teacher-paired CoT/traj LM losses
- interface steps: frozen-style teacher control aux loss + weak GT prefix@8 anchor
- schedule: `3 pair-LM : 1 interface`

## Config

- stage config: `configs/train/stage_t1_paired_interface_v2c_mixed.yaml`
- init checkpoint: `outputs/checkpoints/subset_t1_teacherpair_train4_step20/final`
- train probe: `train4`, `12` steps

## Result

Training summary:

- `outputs/reports/subset_t1_pairifacev2c_mixed_train4_step12.json`

Key observations:

- mixed curriculum remained stable
- no return of the previous `1e4~1e5` aux-head blow-up
- pair-LM steps had small, bounded grads (`~8-11`)
- interface steps stayed bounded as well (`~0-109`)
- val total improved modestly:
  - step 4: `5.7752`
  - step 8: `5.7179`
  - step 12: `5.6910`

Teacher-relative pair losses on val remained live:

- `teacher_topk_kd_loss`: `~1.50`
- `teacher_traj_topk_kd_loss`: `~1.91 -> 1.92`

Interface remained stable:

- `traj_aux_loss`: `~2.61 -> 2.62`
- `traj_aux_abs_max`: `~106 -> 104`

## Decode check

Train decode smoke:

- `outputs/reports/subset_t1_pairifacev2c_mixed_train4_step12_infer_train.json`

Result:

- `generated_traj_token_count = 0` for all 4 train samples

Interpretation:

- mixed curriculum stabilizes the coexistence of LM and interface learning
- but this alone does **not** recover the `CoT -> <|traj_future_start|> -> traj body` emission contract yet

## Conclusion

P2c is a stability success but not an emission success.

What changed:

- interface training no longer destroys the LM path
- pair-LM supervision can coexist with interface supervision

What did **not** change:

- the model still fails to emit trajectory tokens during free decode, even on train samples

## Recommended next step

The next bottleneck is no longer aux instability. It is the joint transition/emission contract.

Most likely next moves:

1. increase pair-LM pressure relative to interface for a short run
2. add explicit `cot_end -> traj_start` transition supervision / metric
3. start from a more formatting-capable checkpoint instead of the current pair-sanity base if available
