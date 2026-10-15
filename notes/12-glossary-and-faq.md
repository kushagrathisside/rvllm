# 12 — Glossary and FAQ

## Glossary

| Term | Meaning | Where in code |
|------|---------|---------------|
| **Token** | A piece of text with an integer id | `tokenizer.rs` |
| **Logits** | Raw scores, one per vocabulary token, output by the model | `Model::forward` |
| **Embedding** | The vector that represents a token | `model.rs::embed` |
| **Hidden state** | A token's vector as it flows through the layers | `x` in `Layer::forward` |
| **RMSNorm** | Normalizes a vector by its root-mean-square | `rms_norm`, `rmsnorm.cu` |
| **Q / K / V** | Query, key, value projections used by attention | `Layer::forward` |
| **Head** | One independent slice of attention | `n_heads`, `head_dim` |
| **GQA** | Grouped-Query Attention: several Q heads share one K/V head | `gqa_attention` |
| **RoPE** | Rotary Position Embedding: encodes position by rotating Q/K | `rope_apply`, `rope.cu` |
| **Rotate-half** | RoPE pairing of element `j` with `j + d/2` | `weights.rs` comment |
| **SwiGLU** | Gated feed-forward: `down(silu(gate(x)) * up(x))` | `swiglu_ffn` |
| **Residual** | Adding a sub-layer's output to its input | `h = x + attn` |
| **Causal mask** | Prevents attending to future tokens | `batch_mask` |
| **LM head** | Final projection from hidden state to logits | `Model::lm_head` |
| **Tied embeddings** | Output matrix = input embedding matrix | `weights.rs` |
| **GGUF** | llama.cpp's model file format | `weights.rs` |
| **Q8_0** | 8-bit quantization with one scale per 32 values | dequantized on load |
| **Dequantize** | Convert quantized weights back to floats | `Model::load` |
| **KV cache** | Stored K/V vectors of past tokens | `kv_storage.rs` |
| **Block** | Fixed-size chunk of KV cache (default 16 tokens) | `BlockID` |
| **Slot** | One token position inside a block | `(BlockID, usize)` |
| **Block table** | A sequence's list of physical blocks | `BlockTable` |
| **Block pool** | All blocks, with a free list and refcounts | `BlockPool` |
| **Prefix cache** | Trie that finds blocks shared by common prompt prefixes | `PrefixCache` |
| **Refcount** | How many sequences use a block; freed at 0 | `incref` / `decref` |
| **Prefill** | Processing the prompt and filling the cache | `admit_waiting` |
| **Decode** | Generating one token per step | `Engine::decode` |
| **Continuous batching** | Rebuilding the batch every step; requests join and leave freely | `Engine::step` |
| **Preemption** | Evicting a running sequence to free memory, then recomputing it later | `handle_pool_full` |
| **LRU** | Least Recently Used eviction policy | `LRUEviction` |
| **TTFT** | Time To First Token | `TokenEvent.ttft` |
| **Greedy** | Always pick the most likely token (temperature 0) | `Sampler` |
| **Temperature** | Sharpens (low) or flattens (high) the distribution | `Sampler` |
| **Top-p / nucleus** | Sample only from the smallest set reaching probability p | `Sampler` |
| **ChatML** | Chat format using `<\|im_start\|>` / `<\|im_end\|>` | `chat.rs` |
| **EOS** | End-of-sequence token | `Config.eos_token_id` |
| **SSE** | Server-Sent Events, the HTTP streaming format | `server::sse` |
| **Backpressure** | Rejecting work (HTTP 429) when overloaded | `AppState.slots` |
| **Backend** | Trait wrapping "forward pass + sampling" for one device | `engine.rs` |
| **NVRTC** | NVIDIA's runtime compiler for CUDA source | `context.rs` |
| **cuBLAS** | NVIDIA's GPU linear-algebra library | `cublas_ops.rs` |
| **Kernel** | A function that runs on the GPU | `kernels/*.cu` |
| **Stream** | An ordered queue of GPU work | `CudaRuntime.stream` |
| **F32 / F16** | 32-bit / 16-bit floating point | — |

## FAQ

### Why is it called rvllm?

It is a **R**ust take on **vLLM**, the well-known Python serving engine that introduced PagedAttention. The name is an inference from the project's description ("Small vLLM-style inference engine"); the repo does not spell it out.

### Can I use a different model such as Llama 3 or Qwen?

Possibly, but it is **untested**. The loader reads generic `llama.*` GGUF keys, so other Llama-architecture models might load. However:

1. The tokenizer wrapper and the ChatML template are written for SmolLM2.
2. Other architectures (Qwen, Mistral variants, …) use different metadata keys or layers.
3. Weights are dequantized to F32, so bigger models need a lot of RAM.

### Why does the base model give strange answers in `--chat` mode?

Base models are trained only to continue text. They have never seen the ChatML format. Use the **-Instruct** GGUF for chat.

### Why is the first response slower than the rest?

The first token needs a **prefill** over the whole prompt. Later tokens are cheap single-token decode steps. On CUDA, kernels are also compiled at startup.

### What happens if I send 100 requests to the server at once?

By default:

- 8 run (`--max-running`).
- 64 wait in the queue (`--max-queue`).
- The other 28 get HTTP 429 with `Retry-After: 1`.

### What is the difference between `max_running` and `num_blocks`?

| Setting | Limits |
|---------|--------|
| `max_running` | How many sequences are **decoded together** in one step |
| `num_blocks` | How much **KV memory** exists in total |

Either one can be the bottleneck. If the blocks run out first, admission waits, or running sequences get preempted.

### Is the output deterministic?

- `temperature = 0` → yes, always.
- `temperature > 0` with a fixed `seed` → yes, even when other requests share the batch, because each request has its own RNG.
- The server picks a **random** seed if you do not send one.

GPU and CPU results may differ slightly because the GPU uses F16 weights.

### Where should I start if I want to modify something?

| Goal | Start here |
|------|------------|
| Add a sampling option (e.g. top-k) | `sampler.rs` (+ `server/api.rs` and `main.rs` flags) |
| Support another chat template | `chat.rs` |
| Add an API field or endpoint | `server/api.rs`, `server/mod.rs` |
| Change scheduling policy | `engine.rs::admit_waiting` / `decode` |
| Speed up the CPU path | `model.rs` (`linear`, `gqa_attention`) |
| Speed up the GPU path | `cuda/model.rs`, `kernels/*.cu`; profile with `RVLLM_TIMING=1` |
| Support a new model architecture | `weights.rs`, `model.rs`, then mirror the changes in `cuda/` |

### What should I read next outside this repo?

1. *Efficient Memory Management for Large Language Model Serving with PagedAttention* (the vLLM paper): the origin of paged KV caches.
2. *Orca: A Distributed Serving System for Transformer-Based Generative Models*: iteration-level (continuous) batching.
3. *RoFormer*: the RoPE paper.
4. *GQA: Training Generalized Multi-Query Transformer Models*: grouped-query attention.
5. The Candle and cudarc crate docs, for the Rust tensor and CUDA APIs used here.
