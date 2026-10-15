# rvllm — Beginner Notes

These notes explain the **rvllm** repository from the ground up: what it is, how to run it, and how every part of the code works. They assume you know some programming, but not necessarily Rust internals, transformers, or LLM serving.

## What is rvllm in one paragraph?

rvllm is a small LLM **inference engine** written in Rust, modelled on the ideas behind [vLLM](https://github.com/vllm-project/vllm). It loads a language model (SmolLM2-360M) from a GGUF file and generates text. You can use it in two ways: as a **terminal chat/Q&A tool** (`rvllm cli`) or as an **OpenAI-compatible HTTP server** (`rvllm serve`). Under the hood it uses the same techniques big serving systems use: **paged KV cache**, **prefix caching**, **continuous batching**, and **preemption**. It runs on the CPU by default and can also run on an NVIDIA GPU through hand-written CUDA kernels.

> **Want the whole flow in one document?** [13-code-walkthrough.md](13-code-walkthrough.md) follows real requests through the code from `main()` to the printed answer, with code snippets at every step. It is long, but it is self-contained.

## Reading order

Read the files in order if you are new. Each builds on the previous one.

| # | File | What you will learn | Difficulty |
|---|------|---------------------|------------|
| 1 | [01-overview.md](01-overview.md) | The big picture, the main ideas, and how a request flows through the system | Easy |
| 2 | [02-getting-started.md](02-getting-started.md) | How to build, run, and configure every command (all flags explained) | Easy |
| 3 | [03-project-structure.md](03-project-structure.md) | A map of every file and what it is responsible for | Easy |
| 4 | [04-llm-basics.md](04-llm-basics.md) | The transformer concepts you need (tokens, embeddings, attention, RoPE, …) | Medium |
| 5 | [05-model-and-weights.md](05-model-and-weights.md) | How the GGUF file is loaded and how the forward pass is coded | Medium |
| 6 | [06-kv-cache-and-paging.md](06-kv-cache-and-paging.md) | KV cache, blocks, block tables, prefix caching | Medium–Hard |
| 7 | [07-engine-and-scheduling.md](07-engine-and-scheduling.md) | Continuous batching, prefill vs decode, preemption, aborts | Hard |
| 8 | [08-sampling-tokenizer-chat.md](08-sampling-tokenizer-chat.md) | Greedy / temperature / top-p sampling, streaming detokenization, ChatML | Medium |
| 9 | [09-http-server.md](09-http-server.md) | The OpenAI-compatible API, queueing, backpressure, SSE streaming | Medium |
| 10 | [10-cuda-backend.md](10-cuda-backend.md) | How the GPU path mirrors the CPU path with custom kernels and cuBLAS | Hard |
| 11 | [11-benchmarking-and-testing.md](11-benchmarking-and-testing.md) | The `bench` tool, the test suite, and CI | Easy |
| 12 | [12-glossary-and-faq.md](12-glossary-and-faq.md) | Quick definitions and answers to common questions | Reference |
| 13 | [13-code-walkthrough.md](13-code-walkthrough.md) | **The full flow in one document**: a worked example traced through every function, with code snippets, block-by-block KV cache traces, and exercises | Medium–Hard (long) |

## How to use these notes

1. **First pass:** read files 1–3 to get oriented, then actually run the CLI (file 2).
2. **Second pass:** read 4 → 8 with the source code open side by side. Every section names the file and function it describes.
3. **Third pass:** read 9 and 10 when you want to understand serving and GPU acceleration.
4. Keep [12-glossary-and-faq.md](12-glossary-and-faq.md) open as a dictionary.
5. **Alternative path:** read 01, then [13-code-walkthrough.md](13-code-walkthrough.md) straight through. Use notes 04–10 as background whenever a step needs more explanation.

> **Tip:** The source code in this repo is unusually well commented. When a note says "see `engine.rs` → `admit_waiting`", go and read the comments there too. They often explain *why* something was done.
