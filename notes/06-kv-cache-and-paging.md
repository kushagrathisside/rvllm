# 06 — The KV Cache, Paging, and Prefix Caching

Files: `src/kv_storage.rs`, `src/cuda/kv_storage.rs`, and the external crate **`kv-cache-scheduler` 0.2.0** (`block_pool.rs`, `block_table.rs`, `prefix_cache.rs`, `eviction.rs`, `sequence.rs`).

## 6.1 Why cache K and V at all?

At decode step *t*, the new token's query must attend to the keys and values of **all** previous tokens. Those K/V vectors were already computed in earlier steps and never change. Storing them means each decode step computes K/V for just **one** new token. Without a cache it would recompute all *t* tokens every step, which is quadratic work.

**Cache size per token** = `2 (K and V) × n_layers × n_kv_heads × head_dim × 4 bytes (F32)`.
For SmolLM2-360M that is `2 × 32 × 5 × 64 × 4 ≈ 80 KB per token`. A 2048-token sequence needs about 160 MB. Memory is the scarce resource, so it must be managed carefully.

## 6.2 Paging: blocks instead of one big buffer

The KV memory is divided into a fixed number of **blocks** (`--num-blocks`). Each block has **slots** for `--block-size` tokens (default 16).

```
Physical storage, one per layer, for K and for V:
  [num_blocks, block_size, n_kv_heads, head_dim]

   block 0      block 1      block 2      block 3      ...
 ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
 │ 16 slots │ │ 16 slots │ │ 16 slots │ │ 16 slots │
 └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

Each sequence has a **block table**: the list of physical blocks it owns, in logical order.

```
Sequence A (37 tokens): block table = [7, 2, 9]
  token 0..15  → block 7, slots 0..15
  token 16..31 → block 2, slots 0..15
  token 32..36 → block 9, slots 0..4     (11 slots still free)
```

`BlockTable::logical_to_physical(i)` returns `(block_table[i / block_size], i % block_size)`.

### Benefits of paging

| Problem with contiguous buffers | How paging fixes it |
|---------------------------------|---------------------|
| Must reserve `max_len` up front for each request | Allocate one block at a time, as needed |
| Fragmentation when requests finish | Any free block can be reused by anyone |
| Cannot share common prefixes | Several block tables can point to the same block |
| Hard to evict a request | Give its blocks back to the pool in O(blocks) |

## 6.3 The pieces of the `kv-cache-scheduler` crate

| Type | Role |
|------|------|
| `BlockPool` | Free list of `BlockID`s plus a **reference count** per block. `alloc`, `incref`, `decref` (frees at 0), `available()` |
| `BlockTable` | One sequence's list of blocks plus its token count. `append_token` grabs a new block when the last one is full |
| `PrefixCache` | A **trie** keyed by token ids. Every `block_size`-th node remembers the block holding that full prefix |
| `LRUEviction` | Remembers block access order. When memory runs out it picks the least recently used block owned by only one sequence |
| `Scheduler` | Owns all of the above plus the `waiting` queue (FIFO) and the `running` list. Methods: `add_request`, `prefill`, `step`, `finish_sequence`, `handle_pool_full` |

> **Note:** this crate only does **bookkeeping**. It decides *which* block and slot each token goes to. rvllm's own `KvStorage` / `CudaKvStorage` hold the actual numbers.

## 6.4 Writing into the cache: `KvStorage::write_tokens`

Input: K and V for `n` new tokens, plus `(BlockID, slot)` for each token.

The code looks for **runs** of tokens that land in consecutive slots of the same block, and copies each run with a single `slice_set`. A 100-token prefill with block size 16 costs about 7 copies instead of 100.

## 6.5 Reading from the cache: `GatherPlan` + `gather`

Attention needs each sequence's K/V as one contiguous tensor `[batch, n_kv_heads, ctx_len, head_dim]`.

1. `GatherPlan::new` builds a flat index list: each sequence's block table, **padded** to the batch's longest table by repeating its own last block.
2. `gather` runs one `index_select` over the block axis, reshapes to `[batch, max_blocks × block_size, …]`, keeps the first `ctx_len` positions, and transposes.

### Why padding is safe

Sequences in a batch have different lengths, so the shorter ones get padded. The padded positions contain *some other real data*, not zeros. That is fine because `batch_mask` sets their score to `-∞`, and softmax gives them **exactly zero** weight. Using the sequence's own last block also keeps every index in range and every value finite (no NaN).

```
batch = [A (len 5), B (len 3)],  ctx_len = 5
mask row for B's query:   [ 0, 0, 0, -∞, -∞ ]
                                        └──┘ padding: ignored
```

## 6.6 Prefix caching

Many requests share their start (system prompts, few-shot examples). The prefix cache lets a new request **reuse** blocks that already hold those exact tokens, as long as another request is still using them.

How it works:

1. When a sequence is prefilled, `PrefixCache::insert(tokens, blocks)` walks a trie, one node per token. Every `block_size` tokens, the node records which block holds that **full** block of prefix.
2. When a new request arrives, `match_prefix(tokens)` walks the same trie and collects blocks for each complete matching block.
3. `Scheduler::prefill` **increfs** the matched blocks and adds them to the new block table. It only allocates fresh blocks for the rest.
4. `Engine::admit_waiting` computes `skip_tokens = matched_blocks × block_size` and runs only the **uncached tail** through the model.

```
Request 1: [SYS SYS SYS ... (32 tokens)] + "What is 2+2?"
Request 2: [SYS SYS SYS ... (32 tokens)] + "Name a colour."
                └──── 2 full blocks shared ────┘   only the tail is computed
```

Important details:

- **Only full blocks** are shared. A partially filled block is never matched.
- `skip_tokens` is capped at `prompt_len − 1`. Even if the entire prompt is cached, the model must still run **at least one token** to get logits for sampling the first output token.
- When a block's refcount drops to 0, it is also removed from the trie (`prefix_cache.evict`). Cached blocks therefore live only as long as some sequence still uses them. One consequence: in the interactive CLI, a new chat turn does **not** reuse the previous turn's blocks, because that turn's request has already finished.
- The `--debug` flag and `bench` report the **prefix hit rate** = `prefix_hits / prefix_total` tokens.

## 6.7 When the pool runs out: eviction and preemption

During a decode step, `Scheduler::step` appends one slot to every running sequence. If `append_token` fails with `PoolFull`:

1. `handle_pool_full` asks `LRUEviction` for a victim: the least recently used block with refcount 1.
2. The **whole sequence** that owns that block is preempted. All its blocks are decref'd, its block table is cleared, it is moved back to the end of `waiting`, and it is removed from `running`.
3. The allocation is retried.

Later, the engine re-admits the preempted sequence and **re-prefills** it from `prompt + tokens generated so far`. See note 07 §7.5 for how the engine repairs the token list.

## 6.8 CPU vs CUDA KV storage

| Aspect | `KvStorage` (CPU) | `CudaKvStorage` (GPU) |
|--------|-------------------|------------------------|
| Buffer type | Candle `Tensor` per layer | `CudaSlice<f32>` per layer |
| Write | `slice_set` per run of consecutive slots | `kv_write` kernel, one thread per scalar |
| Gather | `index_select` + reshape + transpose | `kv_gather` kernel, writes straight into the transposed layout |
| Precision | F32 | F32 |
| `Clone` | Deep copy (used in tests to snapshot the cache) | — |

> **Gotcha recorded in the code:** `Tensor::clone()` in Candle only bumps a reference count, so a derived `Clone` would make two `KvStorage` values *share* memory. Because `write_tokens` mutates in place, `KvStorage` implements `Clone` by hand as a **deep copy**.
