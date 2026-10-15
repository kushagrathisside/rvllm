# 04 — LLM Basics You Need for This Repo

This note explains the transformer concepts that appear in the code. Each concept links to where it is implemented. You do not need any machine-learning background beyond "matrices get multiplied".

## 4.1 Tokens and the vocabulary

A model does not read characters. It reads **tokens**: pieces of text such as `"The"`, `" capital"`, `"é"`. Each token has an integer **id**. SmolLM2's vocabulary has roughly 49k tokens.

- Text → ids: `SmollLM230MTokenizer::encode` (`tokenizer.rs`)
- Ids → text: `SmollLM230MTokenizer::decode`

Some ids are **special tokens** with a control meaning:

| Token | Meaning |
|-------|---------|
| EOS (end of sequence) | The model is done. Id comes from GGUF metadata `tokenizer.ggml.eos_token_id` |
| `<\|im_start\|>`, `<\|im_end\|>` | Mark the start and end of a chat turn (ChatML) |

## 4.2 The model's shape (`Config`)

All the hyperparameters are read from the GGUF file into `model::Config`:

| Field | Meaning | SmolLM2-360M (typical) |
|-------|---------|------------------------|
| `n_layers` | Number of transformer blocks stacked | 32 |
| `hidden_dim` | Width of each token's vector | 960 |
| `n_heads` | Query attention heads | 15 |
| `n_kv_heads` | Key/value heads (fewer than queries → GQA) | 5 |
| `head_dim()` | `hidden_dim / n_heads` | 64 |
| `ffn_dim` | Width inside the feed-forward network | 2560 |
| `vocab_size` | Number of tokens | 49152 |
| `context_length` | Max tokens the model supports | 8192 |
| `rope_theta` | Base frequency for RoPE | 100000 |
| `rms_eps` | Small number to avoid divide-by-zero in RMSNorm | ~1e-5 |

> The right-hand column is for orientation only. The code never hardcodes these values; it always reads them from the file.

## 4.3 The forward pass, in order

For a batch of tokens, `Model::forward` (`model.rs`) does:

1. **Embed:** look up each token id's row in `token_embd` → a vector of size `hidden_dim` per token. (`embed`)
2. **Repeat `n_layers` times**, each a `Layer::forward`:
   1. `normed = rms_norm(x)`
   2. Project to **Q, K, V** with three matrix multiplies.
   3. Apply **RoPE** to Q and K.
   4. Write K and V into the **KV cache**; read back the whole sequence's K and V.
   5. **Attention:** each query looks at all earlier keys and mixes their values.
   6. Project back to `hidden_dim` and **add** to `x` (residual connection) → `h`.
   7. `normed2 = rms_norm(h)`, run the **SwiGLU feed-forward**, add to `h` → layer output.
3. **Final norm + LM head:** normalize, then multiply by `output` (size `vocab_size × hidden_dim`) → one **logit** per vocabulary token.

```
token ids ─► embed ─► [ Layer 0 ] ─► [ Layer 1 ] ─► … ─► [ Layer N-1 ] ─► rms_norm ─► LM head ─► logits
                         │
                         ▼
          x ─► rms_norm ─► Q,K,V ─► RoPE ─► KV cache ─► attention ─► out-proj ─(+x)─► h
          h ─► rms_norm ─► SwiGLU FFN ─(+h)─► layer output
```

## 4.4 RMSNorm

`rms_norm(x, weight, eps)` rescales each token vector so its "average size" is about 1, then multiplies each channel by a learned weight:

```
rms = sqrt(mean(x²) + eps)
out = (x / rms) * weight
```

Unlike LayerNorm, it does not subtract the mean. This is cheaper and works well for Llama-family models. It is applied **before** attention and before the FFN ("pre-norm"), which keeps deep stacks stable.

## 4.5 Attention and the causal mask

For each token, attention asks: *which earlier tokens matter to me, and how much?*

1. **Scores:** dot product of this token's query with every key → "how similar".
2. **Scale** by `1 / sqrt(head_dim)` so softmax does not saturate.
3. **Mask:** set the score to `-∞` for any *future* token (causal mask). Also set it to `-∞` for batch padding (see note 06). `softmax(-∞) = 0`, so those positions get zero weight.
4. **Softmax** each row → probabilities summing to 1.
5. **Mix values:** weighted sum of the V vectors.

The mask is built in `batch_mask`. Rule: position `j` is hidden from query `i` if `j > offset + i` (future) or `j ≥ read_num_tokens` (padding).

## 4.6 Multi-head and Grouped-Query Attention (GQA)

The hidden vector is split into `n_heads` independent **heads** that each attend separately. In **GQA**, several query heads **share** one key/value head. With 15 Q heads and 5 KV heads, each KV head serves a group of 3 Q heads.

This makes the KV cache 3× smaller. `gqa_attention` avoids copying K/V three times with a reshaping trick: it folds the 3 query heads of a group into extra *rows* of one matrix multiply against the shared KV head.

## 4.7 RoPE — Rotary Position Embedding

Attention by itself does not know token *order*. RoPE encodes position by **rotating** pairs of numbers in Q and K by an angle that depends on the position:

```
angle[pos, j] = pos × rope_theta^(−2j / head_dim)
(a, b) → (a·cos − b·sin,  a·sin + b·cos)
```

Different pairs rotate at different speeds, like the hands of a clock. Because both Q and K are rotated, their dot product ends up depending on the *distance* between tokens.

**Rotate-half vs interleaved.** There are two ways to choose the pairs:

| Convention | Pairs element `j` with | Used by |
|-----------|------------------------|---------|
| Rotate-half | `j + head_dim/2` | HuggingFace checkpoints, **this engine's kernels** |
| Interleaved | `j + 1` (neighbour) | llama.cpp's GGUF layout |

GGUF files store Q/K weights in the interleaved layout. `weights.rs::unpermute_rope_rows` reorders them at load time so rotate-half is correct. Without this fix, quality degrades as the context gets longer.

## 4.8 SwiGLU feed-forward network

```
FFN(x) = down( silu(gate(x)) * up(x) )
silu(z) = z * sigmoid(z)
```

Two parallel projections widen `x` to `ffn_dim`. One of them is passed through `silu` and used as a soft "gate" on the other. The result is projected back down. Implemented in `swiglu_ffn`.

## 4.9 Residual connections

Each half of a layer **adds** its output to its input (`h = x + attention(...)`) instead of replacing it. Information can flow straight through the stack even when one layer contributes little.

## 4.10 Logits → next token

The LM head produces `vocab_size` logits. A **sampler** turns them into one token id: argmax for greedy, or a random draw for temperature/top-p. See [08-sampling-tokenizer-chat.md](08-sampling-tokenizer-chat.md).

## 4.11 Prefill vs decode

| Phase | Input per sequence | Work | Bottleneck |
|-------|--------------------|------|-----------|
| **Prefill** | The whole prompt (many tokens) | Fill the KV cache, sample the first token | Compute |
| **Decode** | 1 token (the last sampled) | Append to the KV cache, sample the next token | Memory bandwidth (reading weights) |

Because decode processes just one token per sequence, **batching many sequences together** is the main way to use the hardware well. That is why continuous batching matters.
