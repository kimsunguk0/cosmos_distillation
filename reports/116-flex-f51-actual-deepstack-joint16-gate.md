# 116 - FLEX F51 Actual DeepStack Joint 16-Sample Gate

## Early Decodes

While F51 training was still running, checkpoints were decoded early to avoid waiting for the full 4000-step chain.

| Checkpoint | B0 token match | B0 ADE m | B0 FDE m | Unique tokens | Max same-token run |
|---|---:|---:|---:|---:|---:|
| early step_001000 | 0.707 | 0.398 | 1.460 | 21.06 | 1.12 |
| early step_002000 | 0.698 | 0.585 | 2.225 | 19.88 | 1.12 |

Read: partial at step_001000, then degradation by step_002000. Step_001000 beats F44b step1000 (`0.534 m`) but does not beat F42 no-actual-DeepStack best (`0.380 m`). Step_002000 is worse than both.

F51 was stopped after the step_002000 early decode because the current CE/state objective did not improve actual DeepStack beyond the no-actual-DeepStack F42 gate.

Artifacts:

- `outputs/reports/flex_f51_actualdeepstack_joint_from_f42_overfit16_s4000_base2e7_dsp1e5_20260607_early_step001000_b0_trajonly_parity_summary.json`
- `outputs/reports/flex_f51_actualdeepstack_joint_from_f42_overfit16_s4000_base2e7_dsp1e5_20260607_early_step002000_b0_trajonly_parity_summary.json`

One-line status:

`F51 actual DeepStack joint pass: N. Best partial = step_001000, 0.398 m B0 parity ADE; still worse than F42 no-actual-DeepStack 0.380 m.`
