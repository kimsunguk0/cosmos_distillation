# 025 Subset Collapse Threshold 1-4-16

- status: in_progress
- scope:
  - probe when trajectory-body collapse begins as the number of train samples increases
  - keep the setup fixed and only vary the train subset size
  - compare both:
    - seen-train decoding
    - held-out validation decoding

## Experiment setup

Base configuration:

- `configs/train/stage_a_contract.yaml`

Runtime settings:

- `max_steps=100`
- `batch_size=1`
- `multi_gpu=off`
- constrained trajectory decoding enabled at inference

Subset corpora:

- `data/corpus/subset_probe/distill_v3_2_train1.jsonl`
- `data/corpus/subset_probe/distill_v3_2_train4.jsonl`
- `data/corpus/subset_probe/distill_v3_2_train16.jsonl`

Run artifacts:

- train `1`
  - `outputs/reports/subset_probe_train1_contract100.json`
  - `outputs/reports/subset_probe_train1_contract100_infer_train.json`
  - `outputs/reports/subset_probe_train1_contract100_infer_val.json`

- train `4`
  - `outputs/reports/subset_probe_train4_contract100.json`
  - `outputs/reports/subset_probe_train4_contract100_infer_train.json`
  - `outputs/reports/subset_probe_train4_contract100_infer_val.json`

- train `16`
  - `outputs/reports/subset_probe_train16_contract100.json`
  - `outputs/reports/subset_probe_train16_contract100_infer_train.json`
  - `outputs/reports/subset_probe_train16_contract100_infer_val.json`

Comparison summary:

- `outputs/reports/subset_probe_1_4_16_compare.json`

## Main result

The transition is sharp:

- `1` sample:
  - rich trajectory-body memorization survives

- `4` samples:
  - trajectory collapse already starts

- `16` samples:
  - trajectory body is effectively a single-token mode collapse

## Quantitative summary

### Seen-train decode

- `train1`
  - average unique traj ids per sample: `60.0`
  - average set Jaccard vs target: `1.0000`
  - interpretation:
    - exact memorization works

- `train4`
  - average unique traj ids per sample: `2.25`
  - average set Jaccard vs target: `0.0262`
  - interpretation:
    - collapse already happens even on seen training examples

- `train16`
  - average unique traj ids per sample: `1.0`
  - average set Jaccard vs target: `0.0130`
  - interpretation:
    - almost complete single-token collapse on seen training examples

### Held-out val decode

- `train1`
  - average unique traj ids per sample: `60.0`
  - average set Jaccard vs target: `0.0999`
  - interpretation:
    - the model simply replays the one memorized trajectory on every val sample

- `train4`
  - average unique traj ids per sample: `2.3`
  - average set Jaccard vs target: `0.0212`
  - interpretation:
    - held-out collapse matches the seen-train collapse pattern

- `train16`
  - average unique traj ids per sample: `1.0`
  - average set Jaccard vs target: `0.0143`
  - interpretation:
    - the held-out path becomes a pure single-token mode

## What this means

This is not just “we do not generalize.”

It is stronger:

- once the train set grows beyond one sample, the trajectory objective starts collapsing even on seen training samples

So the current objective is not merely underfitting held-out data.
It is converging toward a cheap high-frequency trajectory token mode.

## Practical interpretation

The current setup behaves like this:

- with `1` sample:
  - memorize the whole structured target

- with `4+` samples:
  - preserve some sample-specific CoT text
  - but replace trajectory-body learning with a tiny frequent-token band

- with `16` samples:
  - collapse further to a near-single-token trajectory body

## Conclusion

The immediate blocker is **not simply lack of sample count**.

The stronger diagnosis is:

- the present trajectory-token objective has an unstable attractor
- with multiple samples, it prefers a low-entropy frequent-token solution
- more samples make that collapse more obvious rather than fixing it

## Recommended next step

Do not move to `Stage B` yet.

The next fix should target the trajectory objective itself, for example:

- frequency-aware reweighting over trajectory tokens
- a decode-quality-based checkpoint selector
- a trajectory loss that penalizes low-entropy repeated-token collapse more directly
