# 120 - FLEX F55 Passthrough K1792 Selection Baseline

## Purpose

Separate learned FLEX encoder failure from visual-token selection failure.

F55 bypasses the FLEX scene encoder and directly copies retained Qwen visual features into the compressed image slots. If this passthrough path matched B0, then the learned encoder would be the main blocker. If it still failed, then token dropping / selection / DeepStack structure itself is already damaging the B0 behavior.

## Settings

- checkpoint: `outputs/checkpoints/flex_f54_f0_perimage_k1792_camtime_from_b0_20260607`
- base B0 target: `outputs/reports/b0_step006250_flexheldout16_decode_trajonly_summary.json`
- corpus: `data/corpus/flex_heldout256_stage2val_seed42.jsonl`
- split / samples: `val`, first 16 rows
- original visual tokens: 2880
- retained scene tokens: 1792, `112/image`
- selection: first 112 tokens from each 180-token image block
- FLEX scene encoder: bypassed
- actual DeepStack: off

## Results

| Eval | ADE m | FDE m | Token match | Unique tokens | Max same-token run |
|---|---:|---:|---:|---:|---:|
| free-run vs teacher geometry | 6.272 | 22.042 | - | 19.44 | 1.25 |
| B0 parity | 3.305 | 11.786 | 0.251 | 19.44 | 1.25 |

Artifacts:

- decode: `outputs/reports/flex_f55_passthrough_k1792_f0_step006250_heldout16_decode_trajonly_summary.json`
- B0 parity: `outputs/reports/flex_f55_passthrough_k1792_f0_step006250_heldout16_b0_trajonly_parity_summary.json`

## Decision

F55 fails. The learned FLEX encoder is not the only blocker. First-K token selection itself breaks the no-FLEX B0 behavior.

Next diagnostic: uniform spatial subsampling passthrough with the same K1792. If uniform improves sharply, the blocker is first-K spatial bias. If uniform also fails, K1792 token dropping and/or missing original DeepStack structure is the blocker.
