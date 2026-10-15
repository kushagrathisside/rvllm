# 03 — Project Structure: A Map of Every File

## Top level

| Path | Purpose |
|------|---------|
| `Cargo.toml` | Package manifest: two binaries (`rvllm`, `bench`), dependencies, the optional `cuda` feature |
| `Cargo.lock` | Exact dependency versions (CI uses `--locked`) |
| `README.md` | User-facing documentation |
| `tokenizer.json` | SmolLM2 tokenizer (also used by the tests) |
| `.github/workflows/ci.yml` | CI: runs `cargo test --locked` on Ubuntu on every push / PR |
| `src/` | All the code |

## `src/` — the CPU engine and shared logic

| File | Lines | Responsibility | Key items |
|------|------:|----------------|-----------|
| `lib.rs` | 11 | Declares the library modules (`cuda` only when the feature is on) | — |
| `main.rs` | 438 | The `rvllm` binary: CLI argument parsing, `cli` and `serve` commands | `CliArgs`, `ServeArgs`, `one_shot`, `interactive`, `run_serve` |
| `bin/bench.rs` | 279 | The `bench` binary: throughput sweeps and trace replay | `synthetic_requests`, `parse_trace_file` |
| `model.rs` | 898 | Model structs and the **CPU forward pass** (Candle) | `Config`, `Layer`, `Model`, `rms_norm`, `rope_*`, `gqa_attention`, `swiglu_ffn`, `BatchItem`, `StepCtx` |
| `weights.rs` | 127 | Loads a GGUF file into a `Model` | `Model::load`, `unpermute_rope_rows` |
| `kv_storage.rs` | 239 | CPU KV-cache tensors: write tokens into blocks, gather them back | `KvStorage`, `GatherPlan` |
| `engine.rs` | 832 | **Continuous-batching engine** and the `Backend` trait | `Backend`, `CpuBackend`, `Engine`, `Request`, `TokenEvent`, `run` |
| `runtime.rs` | 63 | Chooses CPU or CUDA and builds the matching backend | `Runtime::new`, `Runtime::backend` |
| `sampler.rs` | 273 | Picks the next token from logits | `Sampler`, `SamplingParams` |
| `tokenizer.rs` | 110 | Wraps HuggingFace `tokenizers`; streaming-safe decoding | `SmollLM230MTokenizer`, `StreamDecoder` |
| `chat.rs` | 63 | ChatML prompt template | `chatml`, `END_OF_TURN` |

## `src/server/` — the OpenAI-compatible HTTP server

| File | Lines | Responsibility |
|------|------:|----------------|
| `mod.rs` | 901 | Routes, request validation, queue slots (semaphore), streaming/non-streaming responses, tests |
| `api.rs` | 136 | JSON request/response structs (serde) matching OpenAI's shapes |
| `worker.rs` | 154 | The single **engine thread**: receives jobs, steps the engine, sends tokens back |
| `stop.rs` | 102 | `StopFilter`: finds stop strings in streamed text, even when they are split across chunks |

## `src/cuda/` — the GPU backend (only with `--features cuda`)

| File | Lines | Responsibility |
|------|------:|----------------|
| `mod.rs` | 7 | Module list |
| `context.rs` | 84 | `CudaRuntime`: device context, stream, cuBLAS handle, compiles all kernels with NVRTC |
| `weights.rs` | 175 | Uploads CPU weights to the GPU once (matmul weights as F16) |
| `model.rs` | 803 | GPU forward pass, a mirror of `model.rs`, with a reusable `Workspace` of buffers |
| `kv_storage.rs` | 236 | GPU KV cache, a mirror of `kv_storage.rs` |
| `engine.rs` | 162 | `CudaBackend`: forward + GPU argmax or host sampling, timing breakdown |
| `ops.rs` | 1471 | Rust wrappers that launch each kernel, plus many kernel tests |
| `cublas_ops.rs` | 352 | Matrix multiplies via cuBLAS (linear layers, attention scores, attention × V) |

### `src/cuda/kernels/*.cu` — the hand-written GPU kernels

| Kernel | What it computes |
|--------|------------------|
| `embed_lookup.cu` | Copy embedding rows for each token id |
| `rmsnorm.cu` | RMS normalization per row (shared-memory reduction) |
| `rope.cu` | Rotary position embedding (rotate-half) |
| `transpose_axes12.cu` | Swap axes 1 and 2 of a 4-D tensor (e.g. `[b, seq, heads, d]` → `[b, heads, seq, d]`) |
| `kv_write.cu` | Scatter new K/V vectors into their (block, slot) in the cache |
| `kv_gather.cu` | Gather each sequence's cached K/V from its block table |
| `mask_add_broadcast.cu` | Add the causal/padding mask to attention scores |
| `softmax.cu` | Row-wise softmax |
| `silu.cu`, `silu_gate_multiply.cu` | SiLU activation; fused `silu(gate) * up` for SwiGLU |
| `add.cu` | Element-wise add (residual connections) |
| `narrow_last_row.cu` | Keep only the last row per sequence (for prefill) |
| `argmax.cu` | Per-row argmax (greedy sampling on the GPU) |
| `f32_to_f16.cu` | Convert activations to half precision before cuBLAS |

## Key dependencies

| Crate | Used for |
|-------|----------|
| `candle-core`, `candle-nn` | Tensors, GGUF reading, dequantization, CPU math |
| `tokenizers` | HuggingFace tokenizer |
| `kv-cache-scheduler` | Block pool, block tables, prefix cache, LRU eviction, scheduler queues (an external crate, see note 06) |
| `rand` | Seeded RNG for sampling |
| `clap` | Command-line parsing |
| `tokio`, `axum`, `tokio-stream`, `futures-util` | Async HTTP server and SSE streaming |
| `serde`, `serde_json` | JSON (de)serialization |
| `cudarc`, `half` *(optional)* | CUDA driver/NVRTC/cuBLAS bindings; `f16` type |

## How the modules depend on each other

```
main.rs ──► runtime.rs ──► engine.rs (CpuBackend) ──► model.rs ──► kv_storage.rs
   │             │                │                       ▲
   │             └──► cuda/engine.rs ──► cuda/model.rs ───┘ (shares Config, BatchItem)
   │                                       └──► cuda/ops.rs, cuda/cublas_ops.rs, kernels/*.cu
   ├──► server/mod.rs ──► server/worker.rs ──► engine.rs
   ├──► tokenizer.rs, chat.rs, sampler.rs
   └──► weights.rs (Model::load)
```

> **A design choice worth noticing:** the CPU and CUDA paths do **not** share a common tensor trait. Each one is written end to end in its own types (Candle `Tensor` vs `CudaSlice`). They only meet at the `Backend` trait boundary in `engine.rs`. The comments in `cuda/kv_storage.rs` and `cuda/weights.rs` explain why: unifying them would leak one side's types into the other.
