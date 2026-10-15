# 10 — The CUDA Backend

Files: `src/cuda/*.rs`, `src/cuda/kernels/*.cu`. Compiled only with `--features cuda`, used only with `--device cuda`.

## 10.1 The idea in one paragraph

The GPU path is a **hand-written mirror** of the CPU path. It does the same math with the same shapes and the same KV paging. Instead of Candle tensors, it uses raw GPU buffers (`CudaSlice<T>` from the `cudarc` crate), small custom CUDA kernels for element-wise work, and **cuBLAS** for matrix multiplies. It plugs into the same `Engine` through the `Backend` trait, so scheduling, batching, and the server work unchanged.

## 10.2 CPU ↔ GPU correspondence

| CPU (Candle) | GPU equivalent |
|--------------|----------------|
| `model.rs::Model` / `Layer` | `cuda/weights.rs::CudaWeights` / `CudaLayer` |
| `model.rs::StepCtx` | `cuda/model.rs::CudaStepCtx` |
| `kv_storage.rs::KvStorage`, `GatherPlan` | `cuda/kv_storage.rs::CudaKvStorage`, `CudaGatherPlan` |
| `engine.rs::CpuBackend` | `cuda/engine.rs::CudaBackend` |
| `linear`, attention matmuls | `cublas_ops::linear_f16`, `attention_scores`, `attention_apply` |
| `rms_norm`, `rope_apply`, `softmax`, `silu`, … | Kernels in `kernels/*.cu`, launched from `ops.rs` |
| `.transpose(1, 2)` (free, a view) | `transpose_axes12` kernel (a real copy) |

## 10.3 Startup: `CudaRuntime` (`context.rs`)

1. Create a CUDA context for device 0, its default **stream**, and a **cuBLAS** handle.
2. For each `.cu` file, `include_str!` embeds the source **into the binary at compile time**. At startup, `cudarc::nvrtc::compile_ptx` compiles it to PTX, which is loaded as a module. The kernel functions are stored by name in `kernels_mapping`.

Consequences:

- **No `nvcc`** and no build step. You only need the NVIDIA driver (CUDA 13.0 compatible) and the NVRTC library.
- Kernels are compiled every time the program starts (a small startup cost).
- The binary still runs from any directory. An earlier version read the `.cu` files from disk at runtime and only worked from inside `src/cuda/`.

## 10.4 Weight upload and precision (`weights.rs`)

The weights are first loaded on the CPU as usual (`Model::load`), then uploaded **once**:

| Tensor | GPU precision | Why |
|--------|---------------|-----|
| Projection matrices (Q, K, V, O, gate, up, down, LM head) | **F16** | Decode is memory-bandwidth bound; half the bytes ≈ up to 2× faster weight reads |
| Norm weights, token embeddings | F32 | Small, or accessed by row lookup |
| Activations | F32 (converted to F16 just before each GEMM) | cuBLAS has no "F16 weights × F32 activations" GEMM |
| KV cache | F32 | Accuracy; simplicity |

GEMMs **accumulate in F32** and write F32 outputs, so every kernel after them sees F32.

## 10.5 One layer on the GPU (`cuda/model.rs`)

The sequence of launches for one layer:

1. `rmsnorm` → `to_f16` → three `linear_f16` calls (Q, K, V)
2. `transpose_axes12` ×3: `[b, seq, heads, d]` → `[b, heads, seq, d]`
3. `rope` on Q and on K
4. `transpose_axes12` ×2 back to token-major → `kv_write` into the cache
5. `kv_gather` the full context (written directly in the layout attention needs)
6. `attention_scores` (cuBLAS: `Q·Kᵀ / √d`, batched over batch × kv_heads, GQA group folded into rows)
7. `mask_add_broadcast` → `softmax` → `attention_apply` (cuBLAS: `probs·V`)
8. `transpose_axes12` → `linear` (output projection) → `add` (residual)
9. `rmsnorm` → `to_f16` → `linear_f16` (gate, up) → `silu_gate_multiply` (fused) → `linear` (down) → `add`

### The `Workspace`

All about 25 intermediate buffers are allocated **once per forward pass** and reused by every layer, instead of being allocated fresh in each layer. Padding slots in `k_full`/`v_full` may hold stale values from earlier layers; this is safe because the mask hides them.

### RoPE tables on the GPU

`CudaStepCtx` computes `cos`/`sin` on the host and **expands them across heads**, because the rope kernel expects one cos/sin row per input row. It does this separately for Q (`n_heads`) and K (`n_kv_heads`).

## 10.6 Row-major data, column-major cuBLAS

cuBLAS assumes **column-major** matrices, while all buffers here are **row-major**. The trick, explained at the top of `cublas_ops.rs`:

> A row-major `(p × q)` buffer has exactly the same bytes as a column-major `(q × p)` matrix, i.e. its transpose, at no cost.

So every GEMM is set up by writing the desired row-major product, transposing the *whole equation* (`(AB)ᵀ = BᵀAᵀ`), and reading the cuBLAS parameters (m, n, k, leading dimensions, transpose flags) off that. No data is ever moved to change layout.

## 10.7 Anatomy of a kernel: `rope.cu`

```c
extern "C" __global__ void rope(const float *x, const float *cos, const float *sin,
                                float *out, unsigned head_dim, unsigned half, unsigned n_pairs) {
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;   // one thread per (row, pair)
    if (index < n_pairs) {
        unsigned row = index / half, j = index % half;
        float a = x[row*head_dim + j], b = x[row*head_dim + half + j];
        float c = cos[row*half + j],  s = sin[row*half + j];
        out[row*head_dim + j]        = a*c - b*s;
        out[row*head_dim + half + j] = a*s + b*c;
    }
}
```

Common patterns across the kernels:

| Pattern | Example kernels |
|---------|-----------------|
| One thread per output element, bounds check `if (index < n)` | `add`, `silu`, `rope`, `kv_write`, `transpose_axes12` |
| One **block** per row with a shared-memory tree reduction | `rmsnorm`, `softmax`, `argmax` |
| Index arithmetic to flatten multi-dimensional shapes | `kv_write`, `kv_gather`, `mask_add_broadcast` |

## 10.8 `CudaBackend::forward_sample` and timing

1. **Enqueue** the forward pass. Kernel launches are asynchronous, so this returns quickly.
2. If every sampler is greedy, also enqueue the `argmax` kernel.
3. **Synchronize** the stream (wait for the GPU).
4. Copy back either the token ids (greedy) or the full logits (then sample on the CPU).

Set the environment variable **`RVLLM_TIMING=1`** to print a per-step breakdown when the backend is dropped:

```
decode breakdown (ms/step over N steps): enqueue=… gpu_wait=… logits_copy=… sample=…
```

How to read it: if `enqueue` is close to `enqueue + gpu_wait`, the GPU is starved by **kernel-launch overhead** (too many small kernels), not limited by its own compute.

## 10.9 Testing the CUDA code

- `ops.rs` has many unit tests comparing each kernel with a CPU reference (e.g. the KV write/gather round-trip matches the CPU `kv_storage.rs` tests).
- `weights.rs` checks that the uploaded weights match the CPU values (within F16 tolerance).
- Every CUDA test **skips itself** when no GPU is found (`CudaRuntime::new(0).ok()` returns `None`). That is why `cargo test --features cuda` passes on a Mac or on a CI machine without a GPU.

## 10.10 Ideas for future improvement

These are observations from reading the code, not plans stated in the repo:

1. **Fusing kernels** (e.g. the transposes into the GEMMs or into RoPE) would cut launch overhead.
2. A **FlashAttention**-style kernel would read the KV blocks directly instead of gathering a padded copy.
3. **Quantized matmuls** (Q8_0 on the GPU) would reduce memory use and bandwidth further.
4. **CUDA graphs** could replay the decode step with very low launch overhead.
