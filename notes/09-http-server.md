# 09 — The OpenAI-Compatible HTTP Server

Files: `src/server/mod.rs`, `src/server/api.rs`, `src/server/worker.rs`, `src/server/stop.rs`, plus `run_serve` in `src/main.rs`.

## 9.1 Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/completions` | Plain text completion (`prompt` → `text`) |
| `POST` | `/v1/chat/completions` | Chat (`messages` → assistant `message`), ChatML applied |
| `GET` | `/v1/models` | Lists the single served model |
| `GET` | `/health` | `{status, running, waiting, max_running, max_queue}` |

### Supported request fields

| Field | Completions | Chat | Notes |
|-------|:-----------:|:----:|-------|
| `prompt` | ✅ | — | A string, or an array of exactly one string |
| `messages` | — | ✅ | `content` can be a string or an array of `{type:"text", text}` parts |
| `max_tokens` | ✅ | ✅ | Chat also accepts `max_completion_tokens` (preferred) |
| `temperature` | ✅ | ✅ | Must be within `[0, 2]` |
| `top_p` | ✅ | ✅ | Must be within `(0, 1]` |
| `seed` | ✅ | ✅ | Random if omitted |
| `stop` | ✅ | ✅ | A string or an array of strings |
| `n` | ✅ | ✅ | Only `1` is allowed |
| `stream` | ✅ | ✅ | Server-Sent Events |
| `stream_options.include_usage` | ✅ | ✅ | Adds a final usage chunk |

Unknown fields are ignored, so OpenAI clients that send extra options still work.

## 9.2 Threading model: many connections, one engine thread

```
 HTTP clients ──► tokio runtime (many async tasks, one per request)
                        │  start_generation(): validate, take a slot, send Job
                        ▼
              unbounded mpsc channel  (Job)
                        │
                        ▼
        ┌────── engine thread (server/worker.rs::run_worker) ──────┐
        │ owns Engine + backend (+ CUDA context). No locks needed. │
        │ loop:                                                    │
        │   idle? → block on channel                               │
        │   drain new jobs → engine.add_request                    │
        │   abort requests whose client hung up                    │
        │   engine.step() → send WorkerEvent::Token per token      │
        │   update WorkerStatus (running / waiting counters)       │
        └──────────────────────────────────────────────────────────┘
                        │ per-request channel (WorkerEvent)
                        ▼
        per-request tokio task: StreamDecoder + StopFilter → Chunk
                        │
                        ▼
              SSE stream, or collect() into one JSON response
```

Why one engine thread?

1. The model and the KV cache are **mutable shared state**. A single owner means no locking.
2. CUDA contexts are tied to a thread. The engine thread builds its own `Runtime` (including the CUDA context), so device state never crosses threads.
3. Batching happens *inside* `engine.step()`. Concurrency comes from batching, not from extra threads.

### Startup sequence (`run_serve`)

1. Load the model and tokenizer, then compute `num_blocks`.
2. Spawn the engine thread in a `std::thread::scope`. It builds the runtime and backend, then reports `max_sequence_tokens` (or an error) through a `ready` channel.
3. The main thread waits for "ready", builds `AppState`, starts a tokio runtime, binds the TCP listener, and runs `axum::serve` with graceful Ctrl-C shutdown.
4. On shutdown, `rt.shutdown_background()` drops the job senders. The worker sees the channel close and exits.

## 9.3 Admission control: queueing and backpressure

| Limit | Enforced by | What happens when it is hit |
|-------|-------------|------------------------------|
| `max_running` (default 8) | The engine (`EngineConfig.max_running`) | Extra requests wait in the engine's FIFO `waiting` queue |
| `max_running + max_queue` (default 72) | A tokio `Semaphore` in `AppState.slots` | New requests get **HTTP 429** `rate_limit_error` / `queue_full` with `Retry-After: 1` |
| KV pool | The engine | Admission waits for free blocks; mid-flight shortages trigger preemption |

The semaphore **permit** travels inside the `Job` and is held for the request's whole life (queued + running). The worker drops it *before* sending the final token, so a client that immediately sends another request does not see the slot still taken.

## 9.4 Validation in `start_generation`

Requests are rejected with **400** before they reach the engine when:

1. `n` is not 1
2. `temperature` is outside `[0, 2]`, or `top_p` is outside `(0, 1]`
3. `max_tokens == 0`
4. The prompt tokenizes to nothing
5. The prompt does not fit the model context, or does not fit `max_sequence_tokens` (the KV pool)

`max_tokens` is then clamped to `max_sequence_tokens − prompt_tokens`.

## 9.5 From tokens to text chunks

A small tokio task per request turns `WorkerEvent`s into `Chunk`s:

1. If the event is the **stop token** (EOS / `<|im_end|>`), flush the decoder. The stop token itself is not shown.
2. Otherwise, `StreamDecoder::push(token)` → a text delta (possibly empty).
3. Pass the text through `StopFilter::push`:
   - If a **stop string** is found, emit only the text before it, send `Done{finish_reason:"stop"}`, and return. Returning drops the event receiver, which makes the worker **abort** the request and free its blocks.
   - Otherwise emit the "safe" text. The filter holds back the longest tail that could still be the *start* of a stop string (e.g. `"ST"` while waiting to see if `"STOP"` follows).
4. On a normal finish, flush the held-back tail and send `Done{finish_reason, usage}`.

### `StopFilter` examples (from its tests)

| Stops | Chunks in | Output | Hit? |
|-------|-----------|--------|:----:|
| `["\n\n"]` | `"hello\n\nworld"` | `hello` | ✅ |
| `["STOP"]` | `"abST"`, `"OPxyz"` | `ab` | ✅ (split across chunks) |
| `["STOP"]` | `"abST"`, `"ART"` | `abSTART` | ❌ (false alarm released) |
| `["cc", "b"]` | `"abcc"` | `a` | ✅ (earliest wins) |

## 9.6 Response formats

**Non-streaming** (`collect`) gathers all chunks and returns OpenAI-shaped JSON:

```json
{"id":"chatcmpl-…","object":"chat.completion","created":…,"model":"…",
 "choices":[{"index":0,"message":{"role":"assistant","content":"Hi!"},"finish_reason":"stop"}],
 "usage":{"prompt_tokens":31,"completion_tokens":3,"total_tokens":34}}
```

**Streaming** (`sse`) sends Server-Sent Events:

1. *(Chat only)* A first chunk with `delta: {role: "assistant", content: ""}`
2. One chunk per text piece: `delta: {content: "…"}` (chat) or `text: "…"` (completions)
3. A final chunk with an empty delta and `finish_reason`
4. *(If `include_usage`)* A chunk with `choices: []` and `usage`
5. `data: [DONE]`

Errors during streaming are sent as an `{"error": …}` event followed by `[DONE]`.

## 9.7 Cancellation

If a client disconnects:

1. Axum drops the response stream, which drops the chunk receiver.
2. The per-request task's next `send` fails, so it returns and drops the `WorkerEvent` receiver.
3. The worker notices `events.is_closed()` (checked every loop) or a failed send, and calls `engine.abort(id)`.
4. The KV blocks and the semaphore slot are freed **immediately**.

## 9.8 Failure handling

If `engine.step()` returns an error, the KV pool may be in an unknown state. The worker therefore **fails every in-flight request** with an error event instead of continuing in a broken state.

## 9.9 Server tests (in `server/mod.rs`)

They use a `SlowBackend` (a `CountingBackend` with a delay) and send real HTTP requests through the axum router with `tower::ServiceExt::oneshot`:

| Test | Checks |
|------|--------|
| `completion_returns_text_and_usage` | Basic completion |
| `chat_completion_non_streaming` | Chat JSON shape |
| `chat_streaming_sends_sse_chunks_then_done` | SSE sequence |
| `stop_string_truncates_and_reports_stop` | Stop strings |
| `rejects_bad_requests` | 400 validation |
| `full_queue_gets_429_and_slot_frees_up_afterwards` | Backpressure |
| `concurrent_requests_queue_fifo_and_all_complete` | Queueing |
| `dropped_client_frees_its_slot` | Cancellation |
| `models_and_health` | Info endpoints |

## 9.10 Using it from Python

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
for chunk in client.chat.completions.create(
        model="any", messages=[{"role": "user", "content": "Hello"}], stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

The `model` field is accepted but ignored; the server always uses the model it loaded.
