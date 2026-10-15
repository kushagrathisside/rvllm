# 05 — Model Loading and the CPU Forward Pass

Files: `src/weights.rs`, `src/model.rs`

## 5.1 What is a GGUF file?

GGUF is the file format from the llama.cpp project. One file holds:

- **Metadata:** key/value pairs such as `llama.block_count = 32` and `tokenizer.ggml.eos_token_id = 0`.
- **Tensors:** named weight arrays such as `blk.3.attn_q.weight`, often **quantized** (Q8_0 = 8-bit integers with a scale per block of 32 values).

## 5.2 `Model::load` step by step (`weights.rs`)

1. Open the file and parse the GGUF header with Candle (`gguf_file::Content::read`).
2. Define a helper `load(name)` that reads a tensor and **dequantizes it to F32** on the CPU.
3. Load the global tensors:
   - `token_embd.weight` (`[vocab, hidden]`)
   - `output_norm.weight`
   - `output.weight`. If the file has none, the model uses **tied embeddings**: the output matrix *is* the embedding matrix, so the code reuses it with `token_embd.clone()` (a cheap reference copy, no second full copy).
4. Read the `Config` from the `llama.*` metadata keys.
5. For each layer `i`, load the nine tensors `blk.{i}.*`. **Q and K go through `unpermute_rope_rows`** to undo llama.cpp's interleaved RoPE layout (see note 04 §4.7).
6. Return `Model { config, token_embd, layers, output_norm, output }`.

### Tensor names → struct fields

| GGUF name | `Layer` field | Shape `[out, in]` |
|-----------|---------------|-------------------|
| `blk.i.attn_norm.weight` | `attn_norm` | `[hidden]` |
| `blk.i.attn_q.weight` | `attn_q` | `[n_heads·head_dim, hidden]` |
| `blk.i.attn_k.weight` | `attn_k` | `[n_kv_heads·head_dim, hidden]` |
| `blk.i.attn_v.weight` | `attn_v` | `[n_kv_heads·head_dim, hidden]` |
| `blk.i.attn_output.weight` | `attn_output` | `[hidden, n_heads·head_dim]` |
| `blk.i.ffn_norm.weight` | `ffn_norm` | `[hidden]` |
| `blk.i.ffn_gate.weight` | `ffn_gate` | `[ffn, hidden]` |
| `blk.i.ffn_up.weight` | `ffn_up` | `[ffn, hidden]` |
| `blk.i.ffn_down.weight` | `ffn_down` | `[hidden, ffn]` |

All weights follow the **"Linear" convention** `[out_features, in_features]`, so a projection is `y = x @ Wᵀ`.

### How `unpermute_rope_rows` works

Inside each head, llama.cpp stores the rows in the order `[0, 2, 4, …, 1, 3, 5, …]` turned into pairs. The un-permute is a reshape trick:

```
[rows, cols] → reshape (n_heads, half, 2, cols) → transpose(1,2) → reshape back
```

The test `unpermute_inverts_llama_cpp_layout` checks it against a copy of llama.cpp's own `permute` function.

## 5.3 The building-block functions (`model.rs`)

| Function | Input → Output | Notes |
|----------|----------------|-------|
| `embed(ids, W)` | ids → `[n, hidden]` | `index_select` on rows |
| `rms_norm(x, w, eps)` | any `[..., hidden]` | Normalizes over the last dimension, so it works batched |
| `rope_cos_sin(theta, head_dim, positions)` | → `cos`, `sin` each `[n, head_dim/2]` | Computed once per forward call, shared by every layer |
| `rope_apply(x, cos, sin)` | `[..., n, head_dim]` | Rotate-half pairing |
| `linear(x, W)` | `[..., in]` → `[..., out]` | Flattens leading dims into one 2-D matmul |
| `swiglu_ffn(x, gate, up, down)` | `[..., hidden]` | See note 04 §4.8 |
| `batch_mask(offsets, read_lens, q_len, ctx_len)` | → `[batch, 1, q_len, ctx_len]` | `0` or `-∞` |
| `gqa_attention(q, k, v, mask)` | → `[batch, n_heads, q_len, head_dim]` | Folds the GQA group into the query rows |

> **A performance lesson recorded in the code:** `linear` deliberately avoids Candle's `broadcast_matmul`. That function made a full transposed copy of the weight on *every* call, which made decode spend about 96% of its time copying memory. Read the comment above `linear`.

## 5.4 `BatchItem` — what one sequence contributes to a forward call

```rust
pub struct BatchItem<'a> {
    pub tokens: &'a [u32],                       // new tokens this call
    pub write_positions: &'a [(BlockID, usize)], // where each new token's K/V goes
    pub read_blocks: &'a [BlockID],              // the sequence's whole block table
    pub read_num_tokens: usize,                  // real context length after this call
    pub position_offset: usize,                  // position of tokens[0] in the sequence
}
```

| Situation | Items in batch | `tokens.len()` per item |
|-----------|----------------|-------------------------|
| Prefill | 1 | Many (the uncached part of the prompt) |
| Decode step | 1 per running sequence | Exactly 1 |

**Rule:** every item in one call must have the **same** number of new tokens. The query axis is never padded. That is why prefill runs one sequence at a time.

## 5.5 `StepCtx` — per-call precomputation

Some things depend only on the batch, not on the layer. `StepCtx::new` builds them **once** per forward call, and all 32 layers reuse them:

- RoPE `cos`/`sin` for every row's position
- The attention mask
- The `GatherPlan` (which cache blocks to read)
- The flattened list of write positions

## 5.6 `Layer::forward` walkthrough

Shapes are for `x: [batch, q_len, hidden]`.

1. `normed = rms_norm(x, attn_norm)`
2. `q = linear(normed, attn_q)` → reshape `[batch, q_len, n_heads, head_dim]` → transpose → `[batch, n_heads, q_len, head_dim]`. Same for `k` and `v`, with `n_kv_heads`.
3. `q, k = rope_apply(...)`. V is not rotated; it carries content, not position.
4. `kv_storage.write_tokens(layer, writes, k, v)`: store this call's K/V.
5. `(k_full, v_full) = kv_storage.gather(layer, plan)`: read back the full context, padded to the longest in the batch.
6. `attn = gqa_attention(q, k_full, v_full, mask)` → merge heads → `linear(attn, attn_output)`.
7. `h = x + attn`
8. `out = h + swiglu_ffn(rms_norm(h, ffn_norm), …)`

## 5.7 `forward` vs `forward_last`

| Method | Returns | Used for |
|--------|---------|----------|
| `forward` | logits for **every** row `[batch, q_len, vocab]` | Decode (q_len = 1 anyway) |
| `forward_last` | logits for the **last** row only `[batch, 1, vocab]` | Prefill: a 500-token prompt only needs the final position's logits, so this skips 499 large vocab projections |

Because RMSNorm works row by row, taking the last row *before* the final norm and LM head gives exactly the same result. A test (`forward_last_matches_last_row_of_forward`) checks this.

## 5.8 How the model code is tested

The tests in `model.rs` build a **tiny fake model** (1 layer, `hidden=4`, 2 Q heads, 1 KV head, vocab 5) with random weights and check:

1. Each math piece against hand-computed numbers (RMSNorm, RoPE at positions 0 and 1, SwiGLU with identity weights, GQA).
2. `gqa_attention` against a slow, naive per-head reference.
3. **Batched decode equals unbatched decode**, including sequences of different lengths. This is the key correctness property of batching: putting sequences together must not change any of their outputs.
