# Vision Tokens Into LLM Stream Explainer

Date: 2026-05-27

This note explains how camera images become LLM tokens in the Alpamayo/Qwen-VL style path, including dimensions and where vision/text interaction happens.

## 1. High-Level Picture

The model does not keep vision as a separate memory bank and then cross-attend from text to vision.

Instead, the image features are converted into embeddings with the same hidden dimension as the LLM. They are inserted into the normal token sequence at image placeholder positions.

Then the decoder LLM processes one mixed sequence:

```text
text tokens + vision embeddings + ego-history tokens + prompt tokens
```

So the vision/text mixing happens inside decoder self-attention.

## 2. Per-Sample Input Layout

Current no-nav 4-camera, 4-frame layout:

```text
System:
  You are a driving assistant that generates safe and accurate actions.

User:
  Front left camera:
    frame 0 <image>
    frame 1 <image>
    frame 2 <image>
    frame 3 <image>

  Front camera:
    frame 0 <image>
    frame 1 <image>
    frame 2 <image>
    frame 3 <image>

  Front right camera:
    frame 0 <image>
    frame 1 <image>
    frame 2 <image>
    frame 3 <image>

  Front telephoto camera:
    frame 0 <image>
    frame 1 <image>
    frame 2 <image>
    frame 3 <image>

  <|traj_history_start|>
    48 fused ego-history tokens
  <|traj_history_end|>

  output the chain-of-thought reasoning of the driving process,
  then output the future trajectory.

Assistant:
  <|cot_start|>
```

Camera mapping:

| Materialized camera | Original camera | Text label |
| --- | --- | --- |
| cam0 | camera_cross_left_120fov | Front left camera |
| cam1 | camera_front_wide_120fov | Front camera |
| cam2 | camera_cross_right_120fov | Front right camera |
| cam3 | camera_front_tele_30fov | Front telephoto camera |

## 3. Image Token Count

For the current 320 x 576 ViT-sized image:

```text
effective spatial token stride = patch_size * spatial_merge_size
                               = 16 * 2
                               = 32

height tokens = 320 / 32 = 10
width tokens  = 576 / 32 = 18

tokens per image = 10 * 18 = 180
```

There are:

```text
4 cameras * 4 frames = 16 images/sample
16 images * 180 vision tokens/image = 2880 vision tokens/sample
```

The full prefill length is roughly:

```text
system/camera-label/prompt/history/assistant text tokens
+ 2880 vision tokens
≈ 3086 prefix tokens in the checked no-nav path
```

The exact count can change slightly with nav text or prompt variants.

## 4. Teacher vs Student Dimensions

Teacher VLM from Alpamayo 1.5 runtime support:

```text
text hidden size       = 4096
text layers            = 36
attention heads        = 32
KV heads               = 8
head dim               = 128
vision hidden size     = 1152
vision output hidden   = 4096
vision depth           = 27
```

Student VLM, Cosmos-Reason2-2B:

```text
text hidden size       = 2048
text layers            = 28
attention heads        = 16
KV heads               = 8
head dim               = 128
vision hidden size     = 1024
vision output hidden   = 2048
vision depth           = 24
```

Therefore the LLM input embeddings have these shapes:

```text
Teacher X_prefill: [B, T_prefix, 4096]
Student X_prefill: [B, T_prefix, 2048]
```

For our checked no-nav prompt:

```text
T_prefix ≈ 3086
```

## 5. How Text Tokens Become Embeddings

Text token IDs go through the LLM embedding matrix.

Teacher:

```text
input_ids:      [B, T]
embedding table [vocab, 4096]
text_embeds:    [B, T, 4096]
```

Student:

```text
input_ids:      [B, T]
embedding table [vocab, 2048]
text_embeds:    [B, T, 2048]
```

## 6. How Image Tokens Replace Placeholder Embeddings

The prompt contains image placeholder spans such as:

```text
<|vision_start|> <|image_pad|> ... <|image_pad|> <|vision_end|>
```

The processor also provides:

```text
pixel_values
image_grid_thw
```

The vision tower converts those image patches into visual embeddings.

Teacher:

```text
one image pixels       -> ViT -> [180, 4096]
16 images/sample       ->       [2880, 4096]
```

Student:

```text
one image pixels       -> ViT -> [180, 2048]
16 images/sample       ->       [2880, 2048]
```

Then the placeholder token embeddings at `<|image_pad|>` positions are replaced with the corresponding vision embeddings.

Conceptually:

```text
inputs_embeds = token_embedding(input_ids)
inputs_embeds[image_pad_positions] = vision_embeddings
```

After replacement:

```text
Teacher inputs_embeds: [B, T_prefix, 4096]
Student inputs_embeds: [B, T_prefix, 2048]
```

The order of these replacement positions is what we fixed to match the official Alpamayo contract.

## 7. Where Vision and Text Mix

The LLM is a causal decoder. Each layer computes self-attention over the prefix stream.

For one layer:

```text
X_l: [B, T, D]

Q = X_l W_Q
K = X_l W_K
V = X_l W_V

Attention(i) = softmax(Q_i K_<=i^T / sqrt(head_dim) + causal_mask) V_<=i
```

The key point:

```text
K and V include both text-token states and image-token states.
```

So when a later token, such as `<|cot_start|>`, is processed, it can attend to:

- system prompt text
- camera label text
- all previous image tokens
- ego-history tokens
- user prompt text

This is how vision and text interact.

It is not a separate cross-attention module. It is cross-modal interaction inside self-attention.

## 8. First LLM Layer

At layer 0, the model receives:

```text
X_0 = [
  text embeddings,
  vision embeddings inserted at image positions,
  fused ego-history embeddings,
  prompt embeddings
]
```

Teacher:

```text
X_0_teacher: [B, ~3086, 4096]
Q_teacher:   [B, 32 heads, ~3086, 128]
K_teacher:   [B,  8 KV heads, ~3086, 128]
V_teacher:   [B,  8 KV heads, ~3086, 128]
```

Student:

```text
X_0_student: [B, ~3086, 2048]
Q_student:   [B, 16 heads, ~3086, 128]
K_student:   [B,  8 KV heads, ~3086, 128]
V_student:   [B,  8 KV heads, ~3086, 128]
```

The sequence order and attention-mask semantics are matched, but Q/K/V values are not expected to match because the weights and hidden dimensions differ.

## 9. Prefill vs Decode

### Prefill

Prefill runs the whole prefix once:

```text
inputs_embeds: [B, T_prefix, D]
attention_mask: [B, T_prefix]
        ↓
LLM forward with use_cache=True
        ↓
past_key_values for all prefix tokens
next-token logits
```

Teacher KV cache:

```text
36 layers * (K,V)
each K/V roughly [B, 8 KV heads, T_prefix, 128]
```

Student KV cache:

```text
28 layers * (K,V)
each K/V roughly [B, 8 KV heads, T_prefix, 128]
```

### Decode

During generation, new tokens are appended one at a time:

```text
prefix KV cache
+ generated CoT token 1
+ generated CoT token 2
+ ...
+ <|traj_future_start|>
+ future trajectory tokens
```

Each new token attends to:

```text
all prefix tokens + all previous generated tokens
```

So full-generation KV is longer than prefill-only KV, and it includes generated CoT/future-start states.

## 10. What Is Matched and What Is Not

Matched now:

- camera order
- camera labels
- frame order
- image placeholder count
- image placeholder positions
- history-token placement
- prompt text placement
- assistant `<|cot_start|>` prefix
- causal attention-mask semantics

Not automatically matched:

- vision feature values
- first-layer Q/K/V values
- attention maps
- hidden states
- KV cache values
- camera-specific grounding strength
- action-relevant representation quality

## 11. Why This Matters

Fixing token order makes the student see the correct input contract. It removes a major mismatch.

But it does not force the student to use image/camera tokens like the teacher.

The student can still learn a weaker shortcut:

```text
ego history + coarse visual prior + common road geometry
```

instead of strong camera-specific reasoning:

```text
front-left vs front vs front-right vs telephoto evidence
```

That explains the current observations:

- normal images beat black/noise/shuffled images
- so the student uses vision
- but camera shuffle does not destroy it as much as expected
- so visual grounding is weaker than teacher-like behavior

## 12. Compact Diagram

```text
Raw images
  [B, 16, 3, 320, 576]
        │
        ▼
Qwen ViT / visual merger
        │
        ├─ Teacher: [B, 2880, 4096]
        └─ Student: [B, 2880, 2048]

Text input_ids with image_pad placeholders
  [B, ~3086]
        │
        ▼
Token embedding
        │
        ├─ Teacher text embeds: [B, ~3086, 4096]
        └─ Student text embeds: [B, ~3086, 2048]

Replace image_pad positions with vision embeddings
        │
        ▼
Mixed multimodal prefix
        │
        ├─ Teacher: [B, ~3086, 4096]
        └─ Student: [B, ~3086, 2048]

Decoder LLM causal self-attention
        │
        ├─ text attends to previous text/image/history tokens
        ├─ prompt/cot_start attends to all camera evidence
        └─ produces prefix KV cache

KV cache
        │
        ├─ Teacher: 36 layers, K/V [B, 8, ~3086, 128]
        └─ Student: 28 layers, K/V [B, 8, ~3086, 128]

Generation / action expert
```

## 13. One-Sentence Summary

The teacher and student now receive the same multimodal sequence layout, but the student still has to learn teacher-like visual grounding because the actual ViT/projector/LLM hidden states and attention maps are different.
