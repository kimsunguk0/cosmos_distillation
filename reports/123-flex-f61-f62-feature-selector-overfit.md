# 123 - FLEX F61/F62 Feature and Selector Overfit

## Purpose

Separate learned FLEX encoder capacity from compressed-prefix adaptation.

Previous F58 showed that K1792 compression is structurally viable when using:

- uniform per-image token selection
- official Qwen MRoPE preservation
- selected original DeepStack features

The remaining question was whether a learned FLEX encoder can replace selected original vision features, or whether the stable path should be selector-style compression.

## F61: Learned FLEX Image Feature Only

Setup:

- checkpoint init: `outputs/checkpoints/flex_f54_f0_perimage_k1792_camtime_from_b0_20260607`
- trainable: `flex_scene_encoder` only, `32,295,936` params
- samples: 16 held-out val rows
- loss: selected no-FLEX image feature parity only
- K1792, uniform selection, official MRoPE preserved
- no LoRA, no DeepStack projector, no token CE

Result:

| Run | Step | Image feature cosine | Image feature MSE | B0 parity ADE m | B0 parity FDE m | Token match |
|---|---:|---:|---:|---:|---:|---:|
| F61 | 1000 | 0.817 | 0.047 | 5.437 | 16.896 | 0.165 |

F61b continued from F61 final with LR `3e-4`; at step 500 it was still only image feature cosine `0.841`, so it was stopped.

Interpretation:

The learned query/Transformer FLEX encoder is connected and trainable, but K1792 selected-feature parity is not recovered strongly enough. Even cosine `0.82` destroys free-run trajectory parity. This is not a simple gradient bug.

## F62: Selector Passthrough + LoRA Adaptation

Setup:

- checkpoint init: `outputs/checkpoints/flex_f54_f0_perimage_k1792_camtime_from_b0_20260607`
- compressed prefix uses selected original Qwen image features
- DeepStack uses selected original Qwen DeepStack features
- official MRoPE preserved
- trainable: last-4 language LoRA only, `9,961,472` params
- loss: B0 free-run trajectory token CE + trajectory hidden alignment

Result:

| Run | Step | B0 parity ADE m | B0 parity FDE m | Token match | Unique traj ids | Max same-token run |
|---|---:|---:|---:|---:|---:|---:|
| F58 passthrough diagnostic | 0 | 0.854 | 2.594 | 0.382 | n/a | n/a |
| F62 | 500 | 1.328 | 4.471 | 0.411 | 17.750 | 8.938 |
| F62 | 1000/final | 0.818 | 2.850 | 0.520 | 14.125 | 8.938 |

Artifacts:

- F61 train: `outputs/reports/flex_f61_image_feature_only_from_f0_overfit16_s1000_lr1e4_20260607_train_summary.json`
- F61 parity: `outputs/reports/flex_f61_image_feature_only_final_b0_trajonly_parity_summary.json`
- F62 train: `outputs/reports/flex_f62_selector_passthrough_lora4_from_f0_overfit16_s1000_20260608_train_summary.json`
- F62 parity: `outputs/reports/flex_f62_selector_passthrough_lora4_from_f0_overfit16_s1000_20260608_final_b0_trajonly_parity_summary.json`
- F62 checkpoint: `outputs/checkpoints/flex_f62_selector_passthrough_lora4_from_f0_overfit16_s1000_20260608/final`

## Decision

Current learned FLEX encoder is not ready as the primary compression path.

Important correction: F62 is **not** a completed paper-style FLEX solution.  It is
a selector/passthrough diagnostic that still computes original Qwen visual
features and then keeps a uniform subset.  It proves which structural pieces
matter, but it must not be reported as learned FLEX compression.

The best diagnostic path so far is selector-style K1792 compression:

1. select original per-image vision features uniformly
2. preserve official Qwen MRoPE position ids
3. pass selected original DeepStack features
4. optionally adapt the backbone with small LoRA

Next gate before any claim of "FLEX applied":

- build a learned FLEX compressor that can match selected Qwen image features much more closely than F61
- preserve official Qwen MRoPE positions in every train/eval path
- support DeepStack through a learned projector rather than selected original DeepStack passthrough
- only then evaluate full held-out normal/shuffle/black vision sensitivity against B0

One-line result:

`FLEX issue = learned query compressor loses too much vision feature fidelity; selector-style K1792 is diagnostic only, not final FLEX.`
