# 039. Mixed From Formatting-Capable Base Restores Trajectory Emission

## Goal

Recover the full `CoT -> <|traj_future_start|> -> 128 traj tokens` contract while keeping the paired-interface curriculum stable.

## Setup

- training config: `configs/train/stage_t1_paired_interface_v2c_mixed.yaml`
- init checkpoint: `outputs/checkpoints/stage_a_v3_2_format_lr/final`
- train probe: `train4`, `12` steps

Artifacts:

- training summary:
  - `outputs/reports/subset_t1_pairifacev2c_mixed_from_format_train4_step12.json`
- train decode:
  - `outputs/reports/subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_train.json`
- val decode:
  - `outputs/reports/subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_val10.json`

## Result

This restores trajectory emission.

Observed behavior:

- train samples: `generated_traj_token_count = 128` for all `4/4`
- val samples: `generated_traj_token_count = 128` for all `10/10`
- `CoT -> <|cot_end|> -> <|traj_future_start|> -> 128 body tokens -> <|traj_future_end|>` is now present again

This contrasts with report 038, where the same mixed curriculum started from the pair-sanity checkpoint remained stable but produced `generated_traj_token_count = 0`.

## Interpretation

The immediate blocker was not just the mixed curriculum itself.

It was the **combination** of:

- an honest teacher-pair mixed objective
- started from a checkpoint that did not already preserve the formatting / transition contract

Starting from the formatting-capable base changes the outcome:

- mixed curriculum stays stable
- trajectory emission is preserved

## Remaining issue

Although emission is restored, the body is still collapsed.

Current emitted trajectories are dominated by repeated `<i1499>` tokens on both train and val.

So the bottleneck has shifted:

- **fixed:** `CoT -> traj_start -> traj_body` transition/emission
- **still broken:** trajectory body diversity / semantic grounding

## Conclusion

This is the first path in the current teacher-pair reset where the model again emits the full structured trajectory span.

The next optimization target is no longer transition recovery. It is body collapse reduction under the restored emission contract.
