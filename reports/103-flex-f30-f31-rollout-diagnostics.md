# FLEX F30/F31 Rollout Diagnostics

## Purpose

Follow up F29, which showed that 16-sample target-context token CE does not transfer cleanly to actual autoregressive free-run.

The question was whether the blocker is:

- exposure bias from teacher-forced target context,
- malformed student-greedy context extraction,
- CoT/prefix generation,
- or trajectory-body distribution drift under FLEX.

## F30: Unsafe Student-Greedy Context

Run: `flex_f30_studentgreedy_anchor_from_f29_overfit16_s2000_lr1e6_20260607`

Status: stopped early at about step 201.

Reason:

- Initial refresh was valid: exact 128-token rate `1.0`, target match `0.491`.
- By the second refresh, one sample produced 157 `<i...>` tokens without `<|traj_future_start|>` / `<|traj_future_end|>`.
- By step 201, other rows produced empty or malformed trajectory spans.
- CE/grad became unstable: step 200 free-run CE `8.58`, grad norm `961.8`.

Finding:

The old student-greedy context path was unsafe because it extracted `<i...>` tokens from the whole generated text. Malformed CoT text could become a fake trajectory context.

Fix:

`scripts/105_train_flex_teacher_parity.py` now supports `--student-greedy-invalid-context {raw,target,skip}` and records `tokens_source`, `fallback_target_rate`, and `skip_rate`. F30b used `target` fallback.

## F30b: Safe Student-Greedy Context

Run: `flex_f30b_studentgreedy_safe_from_f29_overfit16_s2000_lr5e7_20260607`

Setup:

- Init: F29 final
- LR: `5e-7`
- student-greedy refresh: every 250 steps
- invalid context fallback: B0 target
- extra small text/format KL: `0.02`

Train:

- Final free-run token CE: `3.7776`
- Final free-run token acc: `0.5520`
- Final prefix CE/acc: `0.6643 / 0.8822`
- Final action_pre/cot_end cosine: `0.8838 / 0.6617`
- Refresh fallback rate rose to `0.125-0.25` late in training.

Decode vs B0 free-run targets:

- exact 128-token rate: `0.9375`
- B0 target token match: `0.3833`
- B0 target ADE/FDE: `7.157 / 24.374`
- GT/teacher decode ADE/FDE: `8.510 / 28.638`
- avg max same-token run: `11.31`

Judgment:

Rejected. Safe scheduled/student-greedy context did not solve the overfit. It made actual decode worse than F29.

## F31: Oracle-CoT Trajectory-Only Diagnostic

Run: `flex_f31_trajonly_oraclecot_overfit16_20260607`

Setup:

- prompt mode: joint
- target mode: trajectory-only
- oracle CoT prefix enabled
- 16 heldout samples
- compared outputs to B0 joint normal free-run targets

| Model | exact128 | B0-token match | B0-target ADE/FDE | teacher ADE/FDE |
| --- | ---: | ---: | ---: | ---: |
| B0 traj-only/oracle-CoT | 1.000 | 0.692 | 0.587 / 1.961 | 5.797 / 19.348 |
| F29 traj-only/oracle-CoT | 1.000 | 0.589 | 1.918 / 6.926 | 7.020 / 23.952 |
| F30b traj-only/oracle-CoT | 1.000 | 0.384 | 6.036 / 16.721 | 8.532 / 25.545 |

Interpretation:

Oracle CoT does not rescue FLEX. F29 improves only slightly versus its joint compressed decode, and F30b remains much worse. Since B0 itself reaches B0-joint target ADE `0.587` under the same oracle-CoT trajectory-only diagnostic, the gap is not just because trajectory-only differs from joint. FLEX adaptation is changing the trajectory-body conditional distribution itself.

## Current Diagnosis

The active failure is not only CoT/prefix generation and not only position shift. It is trajectory-body distribution preservation under compressed visual prefix plus limited LoRA/projector adaptation.

The current FLEX training surface can satisfy local teacher-forced parity, but it cannot force the autoregressive trajectory-body distribution back to B0 on even 16 samples without destabilizing format/hidden behavior.

## Next Recommendation

Stop trying to fix this with post-hoc token CE on F28/F29.

The next controlled test should be a clean low-level capacity test:

1. Start from untrained per-image FLEX F0, not from drifted F28/F29.
2. Use trajectory-only/oracle-CoT training first.
3. Freeze text/COT behavior by avoiding joint CoT generation.
4. Train FLEX + projector/LoRA on 16 samples until B0 trajectory-body parity is near B0's own traj-only baseline.
5. Only then reintroduce joint CoT generation and camera-shuffle/black sensitivity.

If trajectory-only/oracle-CoT overfit from clean F0 cannot match B0 on 16 samples, FLEX capacity/placement or the LoRA/projector adaptation surface is insufficient. If it can, the remaining blocker is joint CoT/format rollout.
