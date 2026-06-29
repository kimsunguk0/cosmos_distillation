# FLEX F35/F36 Trajectory-State Diagnostics

## Purpose

Test whether the failed trajectory-body parity gate is caused by weak target-context token CE.

F32/F33/F34 showed that token CE can move while autoregressive decode still fails. F35/F36 add a stricter diagnostic: align the hidden states that produce each of the 128 trajectory-token logits.

Implementation detail:

- For each trajectory label position `p`, the next-token logits come from hidden state `p - 1`.
- The new `traj_state_*` losses align B0 and FLEX hidden states at those `p - 1` positions.
- The input trajectory prefix is still B0 trajectory-only free-run tokens, so this is a controlled target-prefix overfit gate.

## Code Change

`scripts/105_train_flex_teacher_parity.py` now supports:

- `--traj-state-cos-weight`
- `--traj-state-norm-weight`
- `--traj-state-mse-weight`

The teacher cache stores trajectory-state vectors when any of these weights is enabled.

## F35: CE + Trajectory-State Loss

Setup:

- Run: `flex_f35_trajstate_clean_f0_trajonly_overfit16_s3000_lr2e6_20260607`
- Init: clean per-image FLEX F0
- Samples: first 16 val samples
- Prompt/target: `joint` / `traj_only`
- Loss:
  - B0 trajectory-token CE: `1.0`
  - end-token CE: `0.05`
  - trajectory-state cosine/norm: `0.5 / 0.05`
- LR: `2e-6`
- Stopped at step 2000 because the state metric degraded and stayed flat.

Training at step 2000:

- token CE: `0.4355`
- token acc: `0.9328`
- end-token acc: `1.0000`
- trajectory-state cosine: `0.6270`
- trajectory-state loss: `0.1924`

Decode at step 2000:

- teacher/GT ADE/FDE: `7.366 / 24.132`
- teacher-target token match: `0.0508`
- unique trajectory ids: `11.438`
- max same-token run: `1.000`

Compared against B0 trajectory-only free-run targets:

- exact 128-token rate: `1.000`
- B0 target token match: `0.321`
- B0-target ADE/FDE: `4.386 / 15.454`

Judgment: rejected.

Adding trajectory-state loss on top of CE made the state metric worse than its initial value and degraded free-run geometry.

## F36: Trajectory-State Only

Setup:

- Run: `flex_f36_stateonly_clean_f0_trajonly_overfit16_s2000_lr1e6_20260607`
- Init: clean per-image FLEX F0
- Samples: first 16 val samples
- Prompt/target: `joint` / `traj_only`
- Loss:
  - B0 trajectory-token CE: `0.0`
  - end-token CE: `0.0`
  - trajectory-state cosine/norm: `2.0 / 0.2`
- LR: `1e-6`

Training:

| Step | state cosine | state loss | norm ratio |
| ---: | ---: | ---: | ---: |
| 1 | 0.7303 | 0.5525 | 0.8396 |
| 100 | 0.7783 | 0.4556 | 0.9328 |
| 500 | 0.8201 | 0.3672 | 1.0376 |
| 1000 | 0.8257 | 0.3546 | 1.0297 |
| 1950 | 0.8381 | 0.3285 | 1.0206 |
| 2000 | 0.8370 | 0.3308 | 1.0140 |

Decode at final:

- teacher/GT ADE/FDE: `7.619 / 21.823`
- teacher-target token match: `0.0366`
- unique trajectory ids: `14.063`
- max same-token run: `24.750`

Compared against B0 trajectory-only free-run targets:

- exact 128-token rate: `1.000`
- B0 target token match: `0.033`
- B0-target ADE/FDE: `9.691 / 28.390`

Judgment: rejected as a solution, but useful diagnostically.

The trajectory-state loss path is trainable: cosine improves from `0.730` to `0.837`. But this does not preserve autoregressive trajectory tokens or geometry.

## Comparison

| Run | Main change | token CE | token acc | state cos | B0 token match | B0-target ADE/FDE | unique | max run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F32 | CE only, last4 | 0.3279 | 0.9436 | - | 0.401 | 2.346 / 8.214 | 9.938 | 8.500 |
| F33 | CE only, last12 | 0.3151 | 0.9495 | - | 0.414 | 4.255 / 13.526 | 18.313 | 1.125 |
| F34 | CE only, dummy slots | 0.3101 | 0.9572 | - | 0.443 | 3.093 / 10.633 | 15.000 | 1.125 |
| F35 | CE + state | 0.4355 | 0.9328 | 0.627 | 0.321 | 4.386 / 15.454 | 11.438 | 1.000 |
| F36 | state only | - | - | 0.837 | 0.033 | 9.691 / 28.390 | 14.063 | 24.750 |

## Diagnosis

Current FLEX failure is not fixed by aligning target-context trajectory hidden states.

What this rules out:

- token-position deletion alone, from F34;
- last4 LoRA capacity alone, from F33;
- target-context trajectory-token CE alone, from F32/F34;
- target-context trajectory-state alignment alone, from F35/F36.

The most likely remaining blocker is autoregressive rollout-state mismatch under visual replacement. The model can match some target-prefix statistics, but those statistics do not constrain the hidden state distribution encountered during actual free-run generation.

## Next

Do not scale this FLEX line.

The next gate should train or diagnose under actual rollout prefixes:

1. generate B0 and FLEX trajectories step-by-step and compare states under the same generated prefix;
2. use scheduled/student-prefix training only after making the context extraction exact and cheap;
3. test a residual/side-channel FLEX placement that keeps original visual embeddings and adds FLEX, because pure replacement still fails 16-sample overfit.

