# 07 — The Engine: Continuous Batching and Scheduling

File: `src/engine.rs` (the heart of the project)

## 7.1 The `Backend` trait — the boundary between scheduling and math

```rust
pub trait Backend {
    fn config(&self) -> &Config;
    fn forward_sample(
        &mut self,
        items: &[BatchItem],
        last_only: bool,              // true = prefill (only the last row's logits)
        samplers: &mut [&mut Sampler],// one per item
    ) -> Result<Vec<u32>, BoxError>;  // one sampled token per item
}
```

The engine knows nothing about tensors or GPUs. It only builds `BatchItem`s and receives token ids back. There are three implementations:

| Backend | Where | Purpose |
|---------|-------|---------|
| `CpuBackend` | `engine.rs` | Real model on the CPU (Candle) |
| `CudaBackend` | `cuda/engine.rs` | Real model on the GPU |
| `CountingBackend` | `engine.rs` tests | Fake model: next token = last token + 1. Lets the scheduling logic be tested without any weights |

`runtime.rs` picks between the first two based on `--device`.

## 7.2 The engine's public API

| Method | What it does |
|--------|--------------|
| `Engine::new(backend, EngineConfig)` | Creates the scheduler with `num_blocks` blocks of `block_size` |
| `add_request(Request) -> SequenceId` | Validates and queues a request (does **not** run it yet) |
| `step() -> Vec<TokenEvent>` | One iteration: admit + prefill, then one batched decode |
| `abort(id)` | Cancels a request and frees its blocks |
| `has_work()`, `num_running()`, `num_waiting()` | Status |
| `max_sequence_tokens()` | `min(num_blocks × block_size, context_length)` |

A `Request` holds `tokens`, `max_tokens`, `stop_tokens`, and its own `Sampler`.
A `TokenEvent` holds `id`, `token`, `finish: Option<Stop | Length>`, and `ttft` (time to first token, set on the first token only).

### Validation in `add_request`

| Check | Error |
|-------|-------|
| Empty prompt | `EmptyPrompt` |
| `max_tokens == 0` | `ZeroMaxTokens` |
| `prompt_len ≥ context_length` | `PromptTooLong` |
| `prompt_len + max_tokens > num_blocks × block_size` | `ExceedsKvCapacity` (it could never fit, even alone) |

`max_tokens` is also quietly **clamped** to `context_length − prompt_len`.

## 7.3 One `step()` = admit, then decode

```
step():
  ┌─ admit_waiting() ─────────────────────────────────────────────┐
  │ while running < max_running and waiting is not empty:          │
  │   seq = waiting.front()                                        │
  │   if pool.available() < ceil(len / block_size): break          │
  │   match prefix cache → skip_tokens                             │
  │   scheduler.prefill(seq)       # allocate blocks, move to running
  │   forward_sample([item], last_only = true)   # prefill forward │
  │   emit(first token)                                            │
  └────────────────────────────────────────────────────────────────┘
  ┌─ decode() ─────────────────────────────────────────────────────┐
  │ remember each running seq's position (= tokens in its cache)   │
  │ scheduler.step()      # reserve 1 slot each; may PREEMPT some  │
  │ repair token lists of preempted sequences                      │
  │ build one BatchItem per still-running seq (1 token each)       │
  │ forward_sample(items, last_only = false)  # ONE batched forward│
  │ emit(token) for each                                           │
  └────────────────────────────────────────────────────────────────┘
```

### Why prefill is separate and sequential

Every item in a forward call must contribute the **same number** of new tokens (the query axis is not padded). Prompts have different lengths, so each prefill is its own forward call with a batch of one. Decode items always have exactly one token, so all of them can share a single call.

### Why this is called "continuous" batching

In *static* batching, a batch starts together and the whole batch waits for its slowest member. Here, the batch is rebuilt **every step**:

- A request that finishes leaves the batch immediately (its blocks are freed in `emit`).
- A request that arrives is admitted at the start of the **next** step.

```
step:     1    2    3    4    5    6    7
req A:   [P]  [d]  [d]  [d]  done
req B:        [P]  [d]  [d]  [d]  [d]  done
req C:                  [P]  [d]  [d]  [d] …
          P = prefill,  d = takes part in that step's batched decode
```

## 7.4 `emit` — recording a token and finishing

For each sampled token:

1. Increase `num_generated` and remember it as `last_token`. This token becomes the input of the next decode step; it is not in the KV cache yet.
2. Decide whether the sequence is done:
   - The token is in `stop_tokens` → `FinishReason::Stop`
   - `num_generated ≥ max_tokens` → `FinishReason::Length`
3. If it is done: `scheduler.finish_sequence` (decref every block) and remove the sequence from all maps, so a long-running server does not leak memory.
4. Push a `TokenEvent`. The final stop token *is* included in the event; callers drop it from the text.

## 7.5 Preemption handling (the tricky part)

When `scheduler.step()` runs out of blocks, it preempts a sequence (see note 06 §6.7). The sequence vanishes from `running` and its blocks are gone. To **re-prefill** it correctly later, its token list must be exactly `prompt + every token generated so far`.

Two subtleties are fixed in `decode()`:

1. The scheduler may or may not have already appended a placeholder `TokenId(0)` before evicting.
2. The most recently sampled token (`last_token`) was never written into the list.

The fix: truncate the list to `prompt_len + num_generated − 1`, then push `last_token`. When the sequence is re-admitted, its prefill forward pass naturally samples the *next* token, and generation continues smoothly. The test `preemption_still_completes_every_request` checks this.

> **Trade-off:** preemption costs a full re-prefill (recompute), but it never deadlocks the system and requires no swap space.

## 7.6 Per-request samplers and determinism

Each request owns its own `Sampler`, which contains its own seeded RNG. The random numbers a request draws therefore do not depend on what else is in the batch. The test `outputs_do_not_depend_on_batching` checks that the result is the same whether requests run alone or together.

## 7.7 `engine::run` — the fixed-workload wrapper

The CLI one-shot mode and `bench` know all their requests up front. `run(backend, requests, sampling, cfg)`:

1. Sorts requests by `arrival_step`.
2. Loops: adds every request whose `arrival_step ≤ current step`, calls `step()`, and collects tokens per request.
3. Stops when nothing is pending or running.
4. Returns `RunResult { generated, stats }`, where `stats` includes steps, elapsed time, output tokens, prefix hits, block allocations, prefill/decode time split, and per-request TTFT.

## 7.8 Statistics glossary

| Stat | Meaning |
|------|---------|
| `steps` | Number of `step()` calls |
| `output_tokens` | Total generated tokens across requests |
| `prefix_hits / prefix_total` | Prompt tokens served from the prefix cache / total prompt tokens |
| `total_block_allocs` | How many times a block was allocated (allocation pressure) |
| `prefill_time`, `prefill_tokens` | Time and tokens spent in prefill forwards |
| `decode_time`, `decode_steps` | Time spent in batched decode, number of decode steps |
| `ttft` | Per request: time from the start of its prefill to its first token |

## 7.9 Engine tests worth reading

All of these use `CountingBackend`:

| Test | Property checked |
|------|------------------|
| `first_sampled_token_is_emitted` | Prefill produces a token event |
| `stop_token_ends_generation` | Stop tokens finish with `Stop` |
| `max_tokens_one_finishes_at_prefill` | The edge case where generation is done right after prefill |
| `max_running_queues_the_rest_fifo` | Concurrency cap and FIFO order |
| `outputs_do_not_depend_on_batching` | Batching does not change results |
| `preemption_still_completes_every_request` | A tiny pool forces evictions; every request still finishes correctly |
| `abort_frees_running_and_waiting` | Cancelling works in both states |
| `rejects_requests_that_can_never_run` | Validation errors |
| `run_wrapper_honours_arrival_steps` | Staggered arrivals in `run` |
