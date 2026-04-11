# Training Plan

v1 trains only the reasoning backbone behavior of the student model.

## Student base

- `nvidia/Cosmos-Reason2-2B`

## Teacher base

- `nvidia/Alpamayo-1.5-10B`

## Enabled losses

- `L_hard_ce`
- `L_seq_kd`
- `L_logit_kd`
- `L_feat_align`
- `L_aux_text`
- `L_self_cons`
- `L_rank`

## Disabled losses in v1

- GT path regression
- teacher trajectory imitation
- teacher discrete future token CE
- teacher reasoning plus GT path joint consistency

## Stages

### Stage A

- structured warmup
- JSON, short rationale, and meta-action heavy

### Stage B

- main distillation
- human CoC plus teacher text KD plus hidden alignment

### Stage C

- robustness polish
- anti-generic reasoning, ranking, and hallucination suppression
