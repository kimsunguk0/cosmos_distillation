# 122 - FLEX F57/F58 MRoPE and DeepStack Passthrough

## Purpose

Find why K1792 passthrough still failed after bypassing the learned FLEX encoder.

The key suspects were:

- compressed FLEX path was not preserving Qwen3-VL official MRoPE position ids
- passthrough restored main image embeddings but not layer-specific DeepStack visual features
- first-K vs uniform visual-token selection

## Code Changes

- `compress_batch_for_flex()` now gathers rank-3 `position_ids` when provided.
- `attach_qwen_mrope_position_ids()` computes official Qwen `get_rope_index()` before compression.
- decode/parity/train/checkpoint eval paths now pass official MRoPE ids when `--preserve-flex-positions` is used.
- passthrough + `--flex-scene-deepstack` now selects original Qwen DeepStack features using the same image-token selection strategy.
- added `--flex-selection-strategy {first,uniform}`.

## Results

All numbers are B0 free-run parity on the same 16 held-out val rows.

| Run | Selection | MRoPE preserved | Original DeepStack | ADE m | FDE m | Token match |
|---|---|---:|---:|---:|---:|---:|
| F55 | first | no | no | 3.305 | 11.786 | 0.251 |
| F56 | uniform | no | no | 5.289 | 18.090 | 0.251 |
| F57 | first | no | yes | 5.044 | 16.080 | 0.260 |
| F57 | uniform | no | yes | 4.132 | 13.090 | 0.195 |
| F58 | first | yes | no | 2.015 | 6.396 | 0.313 |
| F58 | first | yes | yes | 1.744 | 5.059 | 0.326 |
| F58 | uniform | yes | no | 3.544 | 11.720 | 0.268 |
| F58 | uniform | yes | yes | 0.854 | 2.594 | 0.382 |

Artifacts:

- F57 log: `outputs/logs/flex_f57_passthrough_deepstack_k1792_f0_step006250_heldout16.log`
- F58 log: `outputs/logs/flex_f58_mrope_passthrough_k1792_f0_step006250_heldout16.log`
- best F58 parity: `outputs/reports/flex_f58_mrope_passthrough_k1792_f0_step006250_heldout16_uniform_dson_b0_trajonly_parity_summary.json`

## Decision

FLEX compression requires three structural fixes together:

1. preserve official Qwen MRoPE positions through compression
2. keep/infer DeepStack visual features, not only main image embeddings
3. use uniform per-image token selection for K1792

With those fixed, selected-token passthrough reaches B0 parity ADE `0.854m`, so K1792 compression is viable enough to train. Next gate: train learned FLEX under the same structure and compare against this passthrough upper bound.
