# 01 — Overview: The Big Picture

## What problem does rvllm solve?

A language model is "just" a big function: give it a list of token ids, and it returns a score (a **logit**) for every possible next token. To generate text you call this function over and over, appending one new token each time.

Doing that **naively** is slow and wasteful:

- Every step would recompute attention over the *entire* text so far.
- Serving many users would mean running them one after another.
- Memory would be reserved for the longest possible answer, even for short ones.

rvllm is a small, readable implementation of the tricks production servers (like vLLM) use to fix these problems. It is a learning-sized project (≈8,500 lines of Rust + CUDA), which makes it a good way to see how serving works.

## The two ways to use it

| Mode | Command | What it does | Typical use |
|------|---------|--------------|-------------|
| CLI (interactive) | `rvllm cli --model … --tokenizer …` | A `>` prompt; you type, it streams an answer | Trying the model out |
| CLI (one-shot) | `rvllm cli … --prompt "a" --prompt "b"` | Runs all prompts *concurrently*, prints answers, exits | Quick tests, demonstrating batching |
| Server | `rvllm serve --model … --port 8000` | OpenAI-compatible HTTP API | Using it from apps, `curl`, or the OpenAI SDK |
| Benchmark | `cargo run --bin bench -- …` | Measures throughput with synthetic load | Performance work |

## The key ideas (in plain words)

1. **KV cache.** When the model reads token *n*, it computes a "key" and "value" vector for it in every layer. Those never change, so we store them and reuse them instead of recomputing. This turns each generation step from "re-read everything" into "read one new token".
2. **Paged KV cache.** Instead of one big contiguous buffer per request, the cache is split into fixed-size **blocks** (default 16 tokens each), like pages of memory in an operating system. A request gets blocks only as it needs them. A **block table** maps "token #37 of this request" → "block 12, slot 5".
3. **Prefix caching.** If two requests start with the same text (e.g. the same system prompt), their first full blocks are identical. When the earlier request is still running, rvllm notices this and **shares** those blocks instead of recomputing them. (Once a request finishes, its blocks are freed and can no longer be shared.)
4. **Continuous batching.** Every running request contributes one token to a single batched forward pass per step. New requests can join at the *next step*. Requests do not have to wait for the others to finish.
5. **Preemption.** If memory (KV blocks) runs out mid-generation, the engine evicts a request, frees its blocks, puts it back in the queue, and recomputes it later.
6. **Backends.** The model math lives behind a `Backend` trait. `CpuBackend` uses the [Candle](https://github.com/huggingface/candle) tensor library. `CudaBackend` uses custom CUDA kernels plus cuBLAS.

## Architecture at a glance

```
                ┌──────────────────────────── main.rs ─────────────────────────────┐
                │   rvllm cli                               rvllm serve             │
                │   (interactive / one-shot)                (tokio + axum HTTP)     │
                └──────────┬───────────────────────────────────────┬────────────────┘
                           │                                       │ Job via channel
                           │                              server/worker.rs (engine thread)
                           ▼                                       ▼
                ┌───────────────────────────── engine.rs ───────────────────────────┐
                │ Engine: add_request / step / abort                                 │
                │   • admit_waiting(): prefill new requests (FIFO)                   │
                │   • decode(): one batched forward for every running sequence       │
                │ uses kv-cache-scheduler crate: BlockPool, BlockTable, PrefixCache  │
                └──────────┬─────────────────────────────────────────────────────────┘
                           │ Backend::forward_sample(items, last_only, samplers)
              ┌────────────┴─────────────┐
              ▼                          ▼
   CpuBackend (engine.rs)       CudaBackend (cuda/engine.rs)
   model.rs + kv_storage.rs     cuda/model.rs + cuda/kv_storage.rs
   (Candle tensors, F32)        (CUDA kernels + cuBLAS, F16 weights)
              │                          │
              └──────────┬───────────────┘
                         ▼
                 sampler.rs → next token id
```

## Life of a single request (step by step)

Here is what happens when you type "The capital of France is" into the CLI:

1. **Tokenize.** `tokenizer.rs` turns the text into token ids, e.g. `[504, 3575, 282, 4649, 314]`.
2. **(Chat mode only)** `chat.rs` first wraps the text in the ChatML template with a system prompt.
3. **Queue.** `Engine::add_request` validates the request and puts it in the scheduler's **waiting** queue.
4. **Admit + prefill.** On the next `Engine::step`, `admit_waiting` checks that there are enough free KV blocks. It asks the prefix cache for reusable blocks, allocates the rest, and runs a **prefill** forward pass over the prompt tokens. Only the last position's logits are computed (`forward_last`).
5. **Sample the first token.** `Sampler` picks a token from the logits: greedy argmax, or temperature + top-p.
6. **Decode loop.** Each later `step` runs **one batched forward pass** where every running request feeds in its last token. Each produces one new token.
7. **Stream out.** Each token becomes a `TokenEvent`. `StreamDecoder` turns token ids into text pieces safely, so multi-byte characters are never cut in half.
8. **Finish.** When the request hits a stop token (EOS or `<|im_end|>`) or `max_tokens`, the engine frees its KV blocks and removes it.

## What is (and isn't) supported

| Supported ✅ | Not supported ❌ |
|-------------|-----------------|
| SmolLM2-360M (base and Instruct), Q8_0 GGUF | Other model families (untested; ChatML template is hardcoded) |
| CPU (Candle) and NVIDIA CUDA | Apple Metal, AMD ROCm |
| Greedy, temperature, top-p, seed | top-k, repetition penalty, logprobs |
| `n = 1`, single prompt per completion | `n > 1`, batched prompt arrays |
| Streaming (SSE), stop strings, usage stats | Tool calling, JSON mode |

> **Note:** Weights are **dequantized** from Q8_0 to F32 when loaded (and narrowed to F16 on the GPU). There is no quantized matrix multiply yet, so the model takes more memory in RAM than on disk.
