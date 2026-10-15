# 11 — Benchmarking, Testing, and CI

## 11.1 The `bench` binary (`src/bin/bench.rs`)

### What it measures

`bench` feeds the engine **synthetic requests**: random token ids. Text content does not affect speed; only sequence lengths and arrival timing do. It therefore needs `--model` (for the config) but **not** a tokenizer.

It reports, for each configuration:

| Metric | Meaning |
|--------|---------|
| Total tokens/sec | Overall throughput across all requests |
| Tokens/sec/request | What each user experiences (the project's "north-star" metric) |
| Prefix cache hit rate | Share of prompt tokens served from the cache |
| Pool utilization (`pool_util`) | Total block allocations ÷ pool size. Blocks that are freed and allocated again count again, so this measures allocation pressure and can exceed 100% |
| TTFT (mean/max), prefill tok/s, decode ms/step | Where the time goes |
| Checksum | FNV-1a hash of every generated token. Same checksum before and after a change ⇒ identical greedy output |

### Mode 1: batch-size sweep

```bash
cargo run --release --bin bench -- \
  --model ../SmolLM2-360M.Q8_0.gguf \
  --batch-sizes 1,4,8,16,32 --prompt-len 128 --gen-len 64
```

All requests in a batch arrive at once with the same lengths. This shows how throughput scales as batches get bigger.

### Mode 2: trace replay (realistic load)

```bash
cargo run --release --bin bench -- --model ../SmolLM2-360M.Q8_0.gguf --trace-file trace.jsonl
```

`trace.jsonl` has one request per line:

```json
{"prompt_len": 128, "gen_len": 64, "arrival_step": 0}
{"prompt_len": 256, "gen_len": 32, "arrival_step": 5}
```

Mixed lengths and staggered arrivals exercise **continuous batching** and **prefix caching** the way real traffic would.

### Bench flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | *(required)* | GGUF path |
| `--batch-sizes` | `1,4,8,16,32` | Sweep values (ignored when `--trace-file` is given) |
| `--prompt-len` | `128` | Synthetic prompt length |
| `--gen-len` | `64` | Synthetic generation length |
| `--block-size` | `16` | KV block size |
| `--seed` | `42` | RNG for the synthetic token ids |
| `--trace-file` | none | JSONL trace file |
| `--device` | `cpu` | `cpu` or `cuda` |

### Methodology notes

1. **Warm-up:** each configuration runs **twice**. The first run is thrown away so first-call allocation costs do not distort the numbers.
2. CPU and CUDA runs print the same report format, so you can compare them directly.
3. For the CUDA path, add `RVLLM_TIMING=1` for the per-step breakdown (see note 10 §10.8).

## 11.2 Running the tests

```bash
cargo test                    # all CPU tests (needs tokenizer.json in the repo root)
cargo test --features cuda    # also the CUDA tests (they skip if no GPU is present)
cargo test engine::           # only tests whose path contains "engine::"
cargo test -- --nocapture     # show println! output
```

### Test inventory by area

| Area | File | Style |
|------|------|-------|
| Math primitives | `model.rs` | Hand-computed expected values |
| Batching correctness | `model.rs` | Batched result == unbatched result |
| GGUF RoPE fix | `weights.rs` | Inverse of llama.cpp's permutation |
| KV storage | `kv_storage.rs` | Write/gather round-trip, isolation, deep clone |
| Scheduling | `engine.rs` | `CountingBackend` (no weights needed) |
| Sampling | `sampler.rs` | Determinism, greedy equivalence, optimized vs reference |
| Tokenizer | `tokenizer.rs` | Round-trip; streaming equals full decode |
| Chat template | `chat.rs` | Exact ChatML output |
| Stop strings | `server/stop.rs` | Split, multi-byte, earliest-wins |
| HTTP server | `server/mod.rs` | End-to-end through the axum router |
| CUDA kernels | `cuda/ops.rs`, `cuda/weights.rs` | GPU output vs CPU reference |

### Testing techniques worth copying

1. **Fake backends.** `CountingBackend` makes the next token predictable (last + 1), so scheduling bugs show up as wrong sequences without loading any weights.
2. **Reference implementations.** Optimized code (partial-sort sampling, folded GQA) is compared with a slow, obviously-correct version.
3. **Invariance tests.** "Batched equals unbatched" and "outputs do not depend on batching" catch whole classes of masking and indexing bugs.
4. **Hardware-optional tests.** GPU tests skip themselves instead of failing, so CI stays green on machines without a GPU.

## 11.3 Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

1. Check out the code.
2. Install stable Rust.
3. Restore the cargo cache (`Swatinem/rust-cache`).
4. Run `cargo test --locked`. `--locked` fails if `Cargo.lock` would need to change.

CI tests the **CPU path only**. The CUDA feature is not built in CI.
