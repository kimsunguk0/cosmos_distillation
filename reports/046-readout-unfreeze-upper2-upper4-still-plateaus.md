# 046. Readout Unfreeze Probes Still Plateau

## Summary

We tested whether the remaining blocker is mainly the `LM readout/interface` by opening real readout parameters on top of the hidden-first bridge checkpoint.

Starting checkpoint:

- [subset_t1_hiddenfirst_topk_bridge_prefix16_from_format_train4_step12/final](/workspace/cosmos_distillation/outputs/checkpoints/subset_t1_hiddenfirst_topk_bridge_prefix16_from_format_train4_step12/final)

What we opened:

- tied `token_embeddings`
- `lm_head`
- upper language blocks
- final language norm
- shared hidden bridge

Result:

- `upper2 + readout`: trains stably, but pure LM traj body stays full `<i1499>` plateau
- `upper4 + readout`: can finish a short no-eval run, but pure LM traj body still stays full `<i1499>` plateau
- `upper6 + readout`: OOM on 16GB before it can finish

## Configs

- upper2:
  - [stage_t1_hiddenfirst_readout_upper2_prefix16_from_hiddenfirst.yaml](/workspace/cosmos_distillation/configs/train/stage_t1_hiddenfirst_readout_upper2_prefix16_from_hiddenfirst.yaml)
- upper4:
  - [stage_t1_hiddenfirst_readout_upper4_prefix16_from_hiddenfirst.yaml](/workspace/cosmos_distillation/configs/train/stage_t1_hiddenfirst_readout_upper4_prefix16_from_hiddenfirst.yaml)
- upper6:
  - [stage_t1_hiddenfirst_readout_upper6_prefix16_from_hiddenfirst.yaml](/workspace/cosmos_distillation/configs/train/stage_t1_hiddenfirst_readout_upper6_prefix16_from_hiddenfirst.yaml)

## Artifacts

- upper2 train summary:
  - [subset_t1_hiddenfirst_readout_upper2_prefix16_from_hiddenfirst_train4_step12.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_hiddenfirst_readout_upper2_prefix16_from_hiddenfirst_train4_step12.json)
- upper2 LM-only decode:
  - train: [subset_t1_hiddenfirst_readout_upper2_prefix16_from_hiddenfirst_train4_step12_infer_train_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_hiddenfirst_readout_upper2_prefix16_from_hiddenfirst_train4_step12_infer_train_lm.json)
  - val: [subset_t1_hiddenfirst_readout_upper2_prefix16_from_hiddenfirst_train4_step12_infer_val10_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_hiddenfirst_readout_upper2_prefix16_from_hiddenfirst_train4_step12_infer_val10_lm.json)
- upper4 no-eval train summary:
  - [subset_t1_hiddenfirst_readout_upper4_prefix16_from_hiddenfirst_train4_step12_noeval.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_hiddenfirst_readout_upper4_prefix16_from_hiddenfirst_train4_step12_noeval.json)
- upper4 LM-only decode:
  - train: [subset_t1_hiddenfirst_readout_upper4_prefix16_from_hiddenfirst_train4_step12_noeval_infer_train_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_hiddenfirst_readout_upper4_prefix16_from_hiddenfirst_train4_step12_noeval_infer_train_lm.json)
  - val: [subset_t1_hiddenfirst_readout_upper4_prefix16_from_hiddenfirst_train4_step12_noeval_infer_val10_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_hiddenfirst_readout_upper4_prefix16_from_hiddenfirst_train4_step12_noeval_infer_val10_lm.json)
- compare:
  - [subset_t1_hiddenfirst_readout_compare.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_hiddenfirst_readout_compare.json)

## LM-only Result

All successful readout probes still produced the same plateau metrics:

- upper2 train: `avg_unique=1.0`, `avg_maxrun=128.0`
- upper2 val: `avg_unique=1.0`, `avg_maxrun=128.0`
- upper4 train: `avg_unique=1.0`, `avg_maxrun=128.0`
- upper4 val: `avg_unique=1.0`, `avg_maxrun=128.0`

In other words, opening:

- the real `lm_head`
- tied embedding rows
- upper `2` or `4` language blocks

still did not move pure LM body generation off the old `<i1499>` attractor.

## Memory Constraint

The `upper6 + readout` run did not fit on the current 16GB GPU budget.

- first attempt OOM during training
- retry with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` got past early training, but still OOMed during validation

So the practical ceiling in the current setup is somewhere below `upper6` unless we reduce eval pressure or memory footprint.

## Interpretation

This is important because it narrows the remaining readout-side hypothesis.

We now have evidence that:

1. hidden-only bridge is not enough ([044](/workspace/cosmos_distillation/reports/044-shared-bottleneck-hidden-bridge-upper2-still-plateaus.md))
2. opening real readout parameters plus upper2/upper4 is still not enough
3. the current `<i1499>` plateau is not fixed by a small amount of extra readout flexibility

So the remaining problem is not simply “`lm_head` was frozen.”

The more likely readout-side blocker is that the current pure LM body path still cannot absorb the hybrid decode contract we know how to use at inference time.

## Takeaway

At this point:

- `hidden manifold mismatch` is real
- `LM readout/interface mismatch` is also real
- but modest readout unfreeze alone does not repair the pure LM body path

The next candidate should target **contract absorption** more directly, not just more of the same CE/KD with slightly more trainable LM layers.
