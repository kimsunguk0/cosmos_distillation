# 121 - FLEX F56 Uniform Passthrough K1792

## Purpose

Test whether F55 failed because first-K token retention was spatially biased.

F56 keeps the same K1792 budget but uniformly samples 112 tokens across each 180-token image block, then bypasses the FLEX scene encoder and copies those retained Qwen visual features into the compressed slots.

## Settings

- checkpoint: `outputs/checkpoints/flex_f54_f0_perimage_k1792_camtime_from_b0_20260607`
- base B0 target: `outputs/reports/b0_step006250_flexheldout16_decode_trajonly_summary.json`
- corpus: `data/corpus/flex_heldout256_stage2val_seed42.jsonl`
- split / samples: `val`, first 16 rows
- original visual tokens: 2880
- retained scene tokens: 1792, `112/image`
- selection: uniform subsampling from each 180-token image block
- FLEX scene encoder: bypassed
- actual DeepStack: off

## Results

| Eval | ADE m | FDE m | Token match | Unique tokens | Max same-token run |
|---|---:|---:|---:|---:|---:|
| F55 first-K passthrough B0 parity | 3.305 | 11.786 | 0.251 | 19.44 | 1.25 |
| F56 uniform passthrough free-run vs teacher geometry | 6.087 | 21.110 | - | 31.44 | 9.00 |
| F56 uniform passthrough B0 parity | 5.289 | 18.090 | 0.251 | 31.44 | 9.00 |

Artifacts:

- decode: `outputs/reports/flex_f56_uniform_passthrough_k1792_f0_step006250_heldout16_decode_trajonly_summary.json`
- B0 parity: `outputs/reports/flex_f56_uniform_passthrough_k1792_f0_step006250_heldout16_b0_trajonly_parity_summary.json`
- log: `outputs/logs/flex_f56_uniform_passthrough_k1792_f0_step006250_heldout16.log`

## Decision

F56 fails and is worse than first-K. The blocker is not just first-K spatial bias.

Next diagnostic: passthrough the original selected Qwen DeepStack visual features as well as the selected main image embeddings. Current F55/F56 passthrough restored only the main visual embedding path, while no-FLEX B0 also injects layer-specific DeepStack visual features.
