# Supervision Policy

Every sample must keep supervision split by provenance.

## Provenance groups

1. `hard_human`
2. `weak_derived`
3. `soft_teacher_text`
4. `soft_teacher_joint`

## Allowed pairings

- `human_reasoning + teacher_soft_text`
- `human_reasoning + GT_path`
- `teacher_reasoning_rollout_k + teacher_trajectory_rollout_k`

## Forbidden pairings by default

- `teacher_reasoning + GT_path`
- `human_reasoning + teacher_trajectory`
- `teacher discrete future tokens as GT continuous path`

## Mixed-pair policy

- Default: off
- If ever enabled in ablations:
  - require consistency gate `pass` or `soft_pass`
  - keep a provenance flag
  - keep sample weight at or below `0.25`

## v1 training policy

- Teacher text is soft supervision only
- GT path is stored but not part of the default loss
- Trajectory and planning losses remain off
