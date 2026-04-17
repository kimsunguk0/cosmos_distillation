# 043 Pure-LM Upper-Block Unfreeze Probes Still Plateau

## Status

Completed.

## Goal

Test whether the current bottleneck is mainly a **student hidden manifold mismatch** that can be improved by unfreezing the top of the student language backbone while keeping the trajectory path strictly **LM-only**.

The practical question was:

- if we start from the formatting-capable mixed checkpoint,
- disable hidden-align and aux-losses,
- and let only the upper language stack adapt under teacher paired CE/top-k,

does the LM-only trajectory body stop collapsing to `<i1499>`?

## Code Change

Added selective freeze/unfreeze support in:

- [09_train_distill.py](/workspace/cosmos_distillation/scripts/09_train_distill.py)

New optimization mode:

- `freeze_all_parameters: true`
- selective unfreeze of:
  - language blocks from a chosen layer index,
  - final language norm,
  - optional token embeddings / LM head / projectors / task heads

This was used to run pure-LM reset probes from an existing formatting-capable checkpoint.

## Probe A: Upper-2 Blocks

Config:

- [stage_t1_purelm_upper2_from_format.yaml](/workspace/cosmos_distillation/configs/train/stage_t1_purelm_upper2_from_format.yaml)

Setup:

- init checkpoint:
  - [subset_t1_pairifacev2c_mixed_from_format_train4_step12/final](/workspace/cosmos_distillation/outputs/checkpoints/subset_t1_pairifacev2c_mixed_from_format_train4_step12/final)
- train:
  - last `2` language blocks (`26-27`) + language norm
- hidden align:
  - off
- aux losses:
  - off
- traj path:
  - pure LM only

Artifacts:

- train summary:
  - [subset_t1_purelm_upper2_from_format_train4_step12.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_purelm_upper2_from_format_train4_step12.json)
- LM-only infer train:
  - [subset_t1_purelm_upper2_from_format_train4_step12_infer_train_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_purelm_upper2_from_format_train4_step12_infer_train_lm.json)
- LM-only infer val:
  - [subset_t1_purelm_upper2_from_format_train4_step12_infer_val10_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_purelm_upper2_from_format_train4_step12_infer_val10_lm.json)

Result:

- training was stable
- validation loss improved slightly:
  - `5.2940 -> 5.2764`
- CoT / formatting stayed alive
- but LM-only traj body remained a full plateau:
  - every sample still emitted `<i1499>` repeated `128` times

## Probe B: Upper-6 Blocks

Config:

- [stage_t1_purelm_upper6_from_format.yaml](/workspace/cosmos_distillation/configs/train/stage_t1_purelm_upper6_from_format.yaml)

Setup:

- same init checkpoint
- train:
  - last `6` language blocks (`22-27`) + language norm
- hidden align:
  - off
- aux losses:
  - off
- traj path:
  - pure LM only

Artifacts:

- train summary:
  - [subset_t1_purelm_upper6_from_format_train4_step8.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_purelm_upper6_from_format_train4_step8.json)
- LM-only infer train:
  - [subset_t1_purelm_upper6_from_format_train4_step8_infer_train_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_purelm_upper6_from_format_train4_step8_infer_train_lm.json)
- LM-only infer val:
  - [subset_t1_purelm_upper6_from_format_train4_step8_infer_val10_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_purelm_upper6_from_format_train4_step8_infer_val10_lm.json)

Result:

- training was stable
- CoT / formatting stayed alive
- but LM-only traj body again remained a full `<i1499>` plateau on both train and val

## Interpretation

These probes are useful because they isolate one hypothesis:

> maybe the student just needs a little more upper-backbone freedom and then the pure LM traj body will emerge.

Current answer:

- **not enough**

Upper-stack partial unfreeze by itself:

- does **not** destabilize training,
- does **not** destroy the CoT -> traj contract,
- but also does **not** move LM-only body decoding off the single-token plateau.

## Main Takeaway

The hidden-manifold mismatch is likely real, but:

- **upper-layer unfreeze alone is insufficient**
- the missing piece is probably not just “let the last few blocks move”

This reinforces the current interpretation:

1. there is still a **hidden manifold mismatch**
2. there is also a **readout/interface mismatch**
3. fixing only the first one weakly does not repair the second one

## Next Direction

The next hidden-side experiment should not be another raw projector or another simple upper-block-unfreeze repeat.

More promising options:

1. shared-bottleneck hidden alignment
   - student `2048 -> d_shared`
   - teacher `4096 -> d_shared`
   - short-horizon only
   - normalized / cosine-heavy loss
2. stronger readout adaptation
   - beyond token-row-only adaptation
   - but without opening the full tied embedding matrix too aggressively
3. hybrid-policy absorption paired with structured hidden alignment
   - instead of expecting pure CE/top-k on the existing LM path to fix the body alone

## Conclusion

The current student does **not** recover non-collapsed LM-only trajectory body just by unfreezing the top `2` or `6` language blocks plus norm.

That means the next step should treat:

- hidden manifold repair
- and LM readout repair

as **coupled but distinct** problems.
