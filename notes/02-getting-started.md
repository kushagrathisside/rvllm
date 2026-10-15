# 02 — Getting Started: Build, Run, Configure

## Prerequisites

| Requirement | Why | Notes |
|-------------|-----|-------|
| Rust toolchain (stable, ≥ 1.85) | The crate uses `edition = "2024"` | Install via [rustup](https://rustup.rs) |
| A GGUF model file | The weights | `SmolLM2-360M.Q8_0.gguf` or the `-Instruct` version; the README examples assume it sits one folder *above* the repo (`../`) |
| `tokenizer.json` | Text ↔ token ids | Already in the repo root. For Instruct models, use the Instruct repo's `tokenizer.json` |
| *(optional)* NVIDIA GPU + driver supporting CUDA 13.0 | GPU path | No `nvcc` needed; kernels compile at runtime via NVRTC |

## Building

```bash
cargo build --release                  # CPU only (default)
cargo build --release --features cuda  # also include the CUDA code
```

There are two things to keep apart:

- `--features cuda` is a **compile-time** switch. It puts the CUDA code into the binary.
- `--device cuda` is a **run-time** switch. It tells a command to use the GPU.

You need both to run on the GPU. Without `--device cuda`, even a CUDA build uses the CPU.

> **Why `--release`?** Debug builds of tensor code are extremely slow. The release profile also enables `lto = "fat"` and `codegen-units = 1` for maximum speed (see `Cargo.toml`). The trade-off is slower compiles.

## Running: the four common recipes

### 1. Interactive Q&A in the terminal

```bash
cargo run --release -- cli \
  --model ../SmolLM2-360M.Q8_0.gguf \
  --tokenizer tokenizer.json
```

Type a question and press Enter. The answer streams token by token. Quit with `exit`, `quit`, or Ctrl-D.

Add `--chat` (with an **Instruct** model) to wrap your input in the chat template. Conversation history is then kept between questions.

### 2. One-shot prompts (concurrent)

```bash
cargo run --release -- cli \
  --model ../SmolLM2-360M.Q8_0.gguf --tokenizer tokenizer.json \
  --prompt "The capital of France is" \
  --prompt "The sky is" \
  --max-tokens 20 --debug
```

All prompts run **together** through continuous batching. `--debug` also prints prefix-cache hit rate, block allocations, step count and tokens/sec.

### 3. HTTP server

```bash
cargo run --release -- serve \
  --model ../SmolLM2-360M-Instruct.Q8_0.gguf --tokenizer tokenizer.json --port 8000
```

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "Say hi"}],
  "max_tokens": 32, "stream": true
}'
```

Any OpenAI client works if you set its base URL to `http://localhost:8000/v1`.

### 4. On the GPU

Add `--features cuda` to the cargo command and `--device cuda` to the program arguments:

```bash
cargo run --release --features cuda -- cli \
  --model ../SmolLM2-360M.Q8_0.gguf --tokenizer tokenizer.json \
  --prompt "The capital of France is" --device cuda
```

## All flags, explained

### Shared flags (both `cli` and `serve`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--model <path>` | *(required)* | Path to the `.gguf` file |
| `--tokenizer <path>` | `tokenizer.json` | Path to the HuggingFace tokenizer file |
| `--device <cpu\|cuda>` | `cpu` | Which backend to run on |
| `--block-size <n>` | `16` | Tokens per KV-cache block (the "page size") |
| `--num-blocks <n>` | *(auto)* | Total KV blocks. The auto value depends on the mode (see below) |

### `cli` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--prompt <text>` | none | Repeatable. If given, run these prompts once and exit; otherwise start the interactive loop |
| `--max-tokens <n>` | 20 with `--prompt`, 256 interactive | Max tokens to generate per request |
| `--chat` | off | Wrap input in the ChatML template (use with Instruct models) |
| `--arrival-step <n>` | 0 | Repeatable, one per `--prompt`. The request only becomes eligible at decode step *n* (simulates staggered arrivals) |
| `--temperature <f>` | `0.0` | 0 = greedy (deterministic). Above 0 = random sampling |
| `--top-p <f>` | `1.0` | Nucleus-sampling threshold (ignored when temperature is 0) |
| `--seed <n>` | `42` | RNG seed for reproducible sampling |
| `--debug` | off | Print prompts, labelled completions and stats |

### `serve` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` to accept connections from other machines |
| `--port` | `8000` | Port |
| `--model-name` | file stem | Name reported in `/v1/models` and in responses |
| `--max-running` | `8` | Max sequences decoded together in a batch |
| `--max-queue` | `64` | Extra requests allowed to wait. Beyond `max-running + max-queue` → HTTP 429 |
| `--default-max-tokens` | `256` | Used when a request omits `max_tokens` |
| `--default-temperature` | `1.0` | Used when a request omits `temperature` (**note:** differs from the CLI default of 0.0) |
| `--default-top-p` | `1.0` | Used when a request omits `top_p` |
| `--tokens-per-seq` | `2048` | Sizes the KV pool when `--num-blocks` is not given |

### How `--num-blocks` is chosen automatically

| Mode | Formula | Intuition |
|------|---------|-----------|
| CLI one-shot | `blocks_for(longest prompt+max_tokens, capped at context) × number_of_prompts` | Exactly enough for every prompt to finish at once |
| CLI interactive | `blocks_for(min(context_length, 4096))` | One conversation up to 4K tokens |
| Serve | `max_running × blocks_for(min(context_length, tokens_per_seq))` | Each running slot gets up to 2048 tokens |

`blocks_for(tokens) = ceil(tokens / block_size)`, with a minimum of 1.

> **Why not always size for the full context?** On a GPU, reserving the full context length for every sequence can run out of memory long before the pool is actually used. Sizing by real need avoids that.

## Common pitfalls

1. **Gibberish in `--chat` mode.** You are probably using the *base* model. Only `-Instruct` models are trained to chat.
2. **"built without the `cuda` feature".** You passed `--device cuda` but built without `--features cuda`.
3. **HTTP 429 from the server.** All running slots and the queue are full. Retry after the `Retry-After: 1` second, or raise `--max-running` / `--max-queue`.
4. **"request needs N tokens of KV cache but the pool only holds M".** `prompt + max_tokens` is bigger than the whole pool. Lower `--max-tokens` or raise `--num-blocks`.
5. **Server and CLI give different answers to the same prompt.** The server's default temperature is 1.0 (random). The CLI's is 0.0 (greedy).
