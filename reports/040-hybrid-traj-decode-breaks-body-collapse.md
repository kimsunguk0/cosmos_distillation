# 040. Hybrid Trajectory Decode Breaks Body Collapse

## Summary

`CoT -> <|traj_future_start|> -> 128 traj tokens -> <|traj_future_end|>` now works end-to-end with a non-collapsed body when we decode from the mixed-from-format checkpoint using a hybrid LM/aux-head trajectory path.

The important result is not another training win. The important result is that the current student hidden state already contains enough trajectory structure to emit a full 128-token body without the old single-token plateau, as long as we do not force the body through the LM path alone.

## Checkpoint

- Base checkpoint: [subset_t1_pairifacev2c_mixed_from_format_train4_step12/final](/workspace/cosmos_distillation/outputs/checkpoints/subset_t1_pairifacev2c_mixed_from_format_train4_step12/final)

This checkpoint preserves the restored transition contract from the formatting-capable base while also carrying the paired-interface training signal.

## Decode Variants Tested

### 1. LM-only body

- Train: [subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_train.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_train.json)
- Val: [subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_val10.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_val10.json)

Observed behavior:

- `generated_traj_token_count = 128`
- body fully collapsed
- average unique traj ids:
  - train: `1.0`
  - val: `1.0`
- average max same-token run:
  - train: `128.0`
  - val: `128.0`

### 2. Aux-head body

- Train: [subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_train_auxhead.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_train_auxhead.json)

Observed behavior:

- collapse broken strongly
- but edge saturation was very high (`<i0>` / `<i2999>`)

### 3. LM + aux blend

- Train: [subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_train_auxblend.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_train_auxblend.json)
- Val: [subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_val10_auxblend.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_val10_auxblend.json)

Observed behavior:

- collapse broken
- average unique traj ids:
  - train: `31.0`
  - val: `30.7`
- average max same-token run:
  - train: `1.75`
  - val: `1.3`
- but edge saturation remained very high:
  - train: `0.6797`
  - val: `0.6836`

### 4. Best current path: LM + aux blend + bounded aux + edge clamp

- Train: [subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_train_auxblend_clip.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_train_auxblend_clip.json)
- Val: [subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_val10_auxblend_clip.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_val10_auxblend_clip.json)

Settings:

- `traj_source=aux_lm_blend`
- `aux_blend_sigma=96`
- `aux_blend_weight=4`
- `aux_bound_scale=0.5`
- `aux_edge_margin=128`

Observed behavior:

- full 128-token body preserved
- single-token plateau gone
- edge saturation eliminated

Metrics:

- train:
  - avg unique traj ids: `21.0`
  - avg max same-token run: `3.5`
  - avg edge fraction: `0.0`
- val:
  - avg unique traj ids: `22.3`
  - avg max same-token run: `2.0`
  - avg edge fraction: `0.0`

## Comparison Artifact

- [subset_t1_pairifacev2c_decode_hybrid_compare.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_pairifacev2c_decode_hybrid_compare.json)

## Interpretation

The main conclusion is:

1. The student can now reliably emit the full structured sequence:
   - `CoT`
   - `<|cot_end|>`
   - `<|traj_future_start|>`
   - `128` body tokens
   - `<|traj_future_end|>`
2. The old body collapse is no longer an unavoidable property of the checkpoint.
3. The remaining problem is no longer "cannot emit a trajectory body."
4. The remaining problem is "the best current body path is decode-assisted and still semantically weak."

This is strong evidence that the current blocker is not only supervision quantity. It is also the inference interface used for trajectory body emission.

## Current Best Command

```bash
cd /workspace/cosmos_distillation
PYTHONPATH=/workspace/cosmos_distillation ./.venv/bin/python scripts/15_infer_student_smoke.py \
  --checkpoint-dir outputs/checkpoints/subset_t1_pairifacev2c_mixed_from_format_train4_step12/final \
  --split val \
  --num-samples 10 \
  --traj-source aux_lm_blend \
  --aux-blend-sigma 96 \
  --aux-blend-weight 4 \
  --aux-bound-scale 0.5 \
  --aux-edge-margin 128 \
  --summary-json outputs/reports/subset_t1_pairifacev2c_mixed_from_format_train4_step12_infer_val10_auxblend_clip.json
```

## Next Step

The next practical target is not transition recovery anymore. That part is done.

The next target is to make training absorb more of what the hybrid decoder is currently rescuing:

- keep the formatting-capable base
- preserve the paired-interface signal
- reduce the gap between LM-only body decode and hybrid body decode
- improve semantic grounding without reintroducing plateau collapse
