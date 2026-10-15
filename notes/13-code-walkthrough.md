# 13 — Code Walkthrough: Following Requests Through the Code

The other notes each explain one part of rvllm. This document instead follows **real requests through the real code**, from the moment you press Enter until the answer is printed. Every step quotes the code that runs, so you can read the whole flow top to bottom in one place.

It is long on purpose. Read it with the source open beside it, one part at a time.

## How to read this document

- **Code blocks are copied from the repository.** `// …` marks lines left out to keep a snippet short. Nothing else is changed.
- **Circled numbers such as `// ①` are added by these notes.** Each one points to the numbered explanation under its snippet. They are not in the source.
- **Each snippet starts with a link to its location**, for example [src/engine.rs:310](../src/engine.rs#L310). Click it to jump to the code.
- **Some code lives in the external `kv-cache-scheduler` crate** (version 0.2.0). Those snippets are labelled *(crate)*. After `cargo fetch` its source is on your machine at `~/.cargo/registry/src/index.crates.io-*/kv-cache-scheduler-0.2.0/src/`.
- **The numbers in the traces were worked out by hand from the code.** That includes block ids, positions, masks and statistics. The model file is not part of the repo, so generated tokens appear as placeholders (`a1`, `a2`, …) instead of real words. The **prompt token ids are real**: they come from the repo's `tokenizer.json`.

## Contents

| Part | Topic | Main code |
|------|-------|-----------|
| 0 | The map: the whole flow on one page | — |
| 1 | The running example | — |
| 2 | Startup: arguments, weights, tokenizer, device | `main.rs`, `weights.rs`, `runtime.rs` |
| 3 | Turning prompts into requests | `main.rs::one_shot` |
| 4 | The driver loop | `engine.rs::run` |
| 5 | Creating the engine and queueing requests | `Engine::new`, `Engine::add_request` |
| 6 | Prefill: admitting requests | `Engine::admit_waiting` + scheduler crate |
| 7 | The forward pass on the CPU | `model.rs`, `kv_storage.rs` |
| 8 | Sampling the next token | `sampler.rs` |
| 9 | Recording tokens and finishing | `Engine::emit` |
| 10 | Batched decode | `Engine::decode` |
| 11 | Printing the results | `main.rs::one_shot` |
| 12 | When memory runs out: preemption | `Engine::decode` + scheduler crate |
| 13 | Variation: the interactive CLI | `main.rs::interactive`, `tokenizer.rs` |
| 14 | Variation: the HTTP server | `server/` |
| 15 | Variation: the CUDA backend | `cuda/` |
| 16 | Variation: the benchmark | `bin/bench.rs` |
| 17 | Recap, function index and exercises | — |

## Rust you will see in the snippets

Skip this table if you already read Rust comfortably.

| Syntax | Meaning |
|--------|---------|
| `fn f(x: &T) -> R` | A function that borrows a `T` and returns an `R` |
| `Result<T, E>` | Either `Ok(value)` or `Err(error)` |
| `expr?` | If `expr` is an `Err`, return that error from the current function now; otherwise unwrap the `Ok` value |
| `Option<T>` | Either `Some(value)` or `None` |
| `&x` / `&mut x` | Borrow `x` read-only / mutably, without copying it |
| `'a` in `BatchItem<'a>` | A *lifetime*: the struct holds borrowed data that must outlive it |
| `impl Trait for Type` | `Type` implements the interface `Trait` |
| `Box<dyn Trait>` | A heap-allocated value of *some* type that implements `Trait`, chosen at run time |
| `\|x\| x + 1` | A closure (an anonymous function) |
| `.iter().map(…).collect()` | An iterator pipeline: transform each element, then gather the results into a collection |
| `#[cfg(feature = "cuda")]` | Compile the next item only when the `cuda` feature is enabled |
| `#[derive(…)]` | Generate trait implementations automatically (`Clone`, `Debug`, clap's `Parser`, …) |
| `let Some(x) = opt else { break };` | "let-else": bind `x`, or run the `else` block (which must leave the current scope) |
| `'generate: while … { … break 'generate; }` | A labelled loop: `break 'generate` exits that specific loop |
| `unsafe { … }` | Code whose safety the compiler cannot check (raw GPU pointers, kernel launches) |

---

## Part 0 — The map

This is the call tree for the running example (a one-shot CLI run on the CPU). Every function in it is covered below, in this order.

```
main()                                               src/main.rs
└─ run_cli()
   ├─ Model::load()                                  src/weights.rs
   ├─ load_tokenizer()                               src/main.rs → src/tokenizer.rs
   ├─ Runtime::new()                                 src/runtime.rs
   └─ one_shot()                                     src/main.rs
      ├─ tokenizer.encode()  (once per prompt)
      ├─ runtime.backend() → CpuBackend::new() → KvStorage::new()
      ├─ engine::run()                               src/engine.rs
      │  ├─ Engine::new() → Scheduler::new()         (crate)
      │  ├─ Engine::add_request()  (once per prompt)
      │  └─ loop { Engine::step() }
      │     ├─ admit_waiting()                       ── prefill, one request at a time
      │     │  ├─ PrefixCache::match_prefix()        (crate)
      │     │  ├─ Scheduler::prefill()               (crate)
      │     │  ├─ CpuBackend::forward_sample(last_only = true)
      │     │  │  ├─ Model::forward_last()           src/model.rs
      │     │  │  │  ├─ embed(), StepCtx::new()
      │     │  │  │  ├─ Layer::forward()  × 32 layers
      │     │  │  │  │  ├─ rms_norm(), linear(), rope_apply()
      │     │  │  │  │  ├─ KvStorage::write_tokens(), KvStorage::gather()   src/kv_storage.rs
      │     │  │  │  │  ├─ gqa_attention()
      │     │  │  │  │  └─ swiglu_ffn()
      │     │  │  │  └─ lm_head()
      │     │  │  └─ Sampler::sample()               src/sampler.rs
      │     │  └─ emit()
      │     └─ decode()                              ── one batched forward for all running requests
      │        ├─ Scheduler::step()                  (crate)
      │        ├─ CpuBackend::forward_sample(last_only = false) → Model::forward()
      │        └─ emit()  → Scheduler::finish_sequence() when done   (crate)
      └─ tokenizer.decode() + println!
```

| Phase | What happens | Function | File |
|-------|--------------|----------|------|
| Parse | Read the command-line flags | `main`, `run_cli` | `main.rs` |
| Load | GGUF file → F32 tensors | `Model::load` | `weights.rs` |
| Tokenize | Text → token ids | `SmollLM230MTokenizer::encode` | `tokenizer.rs` |
| Device | Pick CPU or CUDA, allocate the KV cache | `Runtime::new`, `Runtime::backend` | `runtime.rs` |
| Queue | Validate a request, put it in the waiting queue | `Engine::add_request` | `engine.rs` |
| Prefill | Run the prompt, sample the first token | `Engine::admit_waiting` | `engine.rs` |
| Forward | The transformer math | `Model::forward(_last)`, `Layer::forward` | `model.rs` |
| Sample | Logits → one token id | `Sampler::sample` | `sampler.rs` |
| Decode | One new token per running request per step | `Engine::decode` | `engine.rs` |
| Finish | Stop or length check, free KV blocks | `Engine::emit` | `engine.rs` |
| Output | Token ids → text, print | `one_shot` | `main.rs` |

---

## Part 1 — The running example

We follow this command from start to finish:

```bash
cargo run --release -- cli \
  --model ../SmolLM2-360M.Q8_0.gguf \
  --tokenizer tokenizer.json \
  --prompt "The capital of France is" \
  --prompt "The sky is" \
  --prompt "The capital of France is" \
  --max-tokens 3 \
  --block-size 4 \
  --debug
```

Each choice is there to make something visible:

1. **Three prompts**, so batching happens. Prompt C repeats prompt A on purpose, to show **prefix caching**.
2. **`--block-size 4`** instead of the default 16, so short prompts already cross block boundaries.
3. **`--max-tokens 3`**, so the whole run takes only two engine steps.
4. **`--debug`**, which prints the statistics we will predict in Part 11.
5. Temperature stays at the CLI default of `0.0`, so sampling is **greedy** and deterministic.

### The three requests

| Name | Prompt | Token ids (real) | Tokens (`Ġ` = a leading space) |
|------|--------|------------------|--------------------------------|
| A | `The capital of France is` | `[504, 3575, 282, 4649, 314]` | `The` `Ġcapital` `Ġof` `ĠFrance` `Ġis` |
| B | `The sky is` | `[504, 6376, 314]` | `The` `Ġsky` `Ġis` |
| C | `The capital of France is` | `[504, 3575, 282, 4649, 314]` | same as A |

Notice that every prompt starts with token `504` (`The`), and that A and C are identical. Both facts matter later.

### The model's dimensions

The code reads these values from the GGUF file. They match SmolLM2-360M's published configuration.

| `Config` field | Value | What it controls |
|----------------|-------|------------------|
| `n_layers` | 32 | `Layer::forward` runs 32 times per forward pass |
| `hidden_dim` | 960 | Width of every token vector |
| `n_heads` | 15 | Query heads |
| `n_kv_heads` | 5 | Key/value heads |
| `head_dim()` | 960 / 15 = **64** | Size of one head |
| group size | 15 / 5 = **3** | Query heads that share one KV head (GQA) |
| `ffn_dim` | 2560 | Width inside the feed-forward network |
| `vocab_size` | 49152 | Logits per position |
| `context_length` | 8192 | Longest allowed sequence |
| `rope_theta` | 100000 | RoPE base frequency |
| `rms_eps` | 1e-5 | RMSNorm epsilon |

Two derived numbers come up again and again:

- **KV cache per token** = 2 (K and V) × 32 layers × 5 KV heads × 64 × 4 bytes = 81,920 bytes = **80 KiB**.
- **Special token ids** in `tokenizer.json`: `<|endoftext|>` = 0, `<|im_start|>` = 1, `<|im_end|>` = 2. SmolLM2's config uses 0 as EOS; the engine uses whatever `tokenizer.ggml.eos_token_id` the GGUF file says.

---

## Part 2 — Startup

### 2.1 `main`: parse the command line

[src/main.rs:117](../src/main.rs#L117)
```rust
fn main() -> Result<(), BoxError> {
    match Cli::parse().command {
        Command::Cli(args) => run_cli(args),
        Command::Serve(args) => run_serve(args),
    }
}
```

`Cli::parse()` is generated by clap's `#[derive(Parser)]`. It reads the program's arguments, and the first word (`cli` or `serve`) selects a variant of the `Command` enum. If a flag is wrong, clap prints an error and exits before this code continues. `BoxError` is an alias (defined in `engine.rs`) for `Box<dyn std::error::Error + Send + Sync>`, meaning "any error type".

The flags are plain struct fields:

[src/main.rs:49](../src/main.rs#L49)
```rust
#[derive(Args)]
struct CliArgs {
    #[command(flatten)]
    model: ModelArgs,
    // One request per occurrence: `--prompt "a" --prompt "b"` runs two
    // concurrent requests through continuous batching, prints the answers and
    // exits. Without any `--prompt`, starts the interactive loop.
    #[arg(long)]
    prompt: Vec<String>,
    // Defaults to 20 with `--prompt`, 256 in the interactive loop.
    #[arg(long)]
    max_tokens: Option<usize>,
    // …
    // 0.0 = greedy argmax (deterministic, no RNG use).
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    // …
}
```

Three clap rules explain every flag in the project:

1. `#[arg(long)]` turns the field name into a `--kebab-case` flag: `max_tokens` becomes `--max-tokens`.
2. A `Vec<…>` field may be repeated, an `Option<…>` field may be left out, and any other field needs a `default_value…` (or it is required).
3. `#[command(flatten)]` copies the shared `ModelArgs` flags (`--model`, `--tokenizer`, `--device`, `--block-size`, `--num-blocks`) into this command.

For our command, `args` ends up holding:

| Field | Value |
|-------|-------|
| `args.model.model` | `"../SmolLM2-360M.Q8_0.gguf"` |
| `args.model.tokenizer` | `"tokenizer.json"` |
| `args.model.device` | `"cpu"` (default) |
| `args.model.block_size` | `4` |
| `args.model.num_blocks` | `None` (will be computed) |
| `args.prompt` | the three prompt strings |
| `args.max_tokens` | `Some(3)` |
| `args.chat` | `false` |
| `args.arrival_step` | `[]` (empty, so every request arrives at step 0) |
| `args.temperature`, `args.top_p`, `args.seed` | `0.0`, `1.0`, `42` |
| `args.debug` | `true` |

### 2.2 `run_cli`: four setup steps

[src/main.rs:142](../src/main.rs#L142)
```rust
fn run_cli(args: CliArgs) -> Result<(), BoxError> {
    let model = Model::load(&args.model.model)?;                                // ①
    let tokenizer = load_tokenizer(&args.model.tokenizer, &model.config)?;     // ②
    let runtime = Runtime::new(&model, &args.model.device)?;                   // ③
    let sampling = SamplingParams {                                            // ④
        temperature: args.temperature,
        top_p: args.top_p,
        seed: args.seed,
    };
    if args.prompt.is_empty() {
        interactive(&args, &runtime, &tokenizer, sampling)
    } else {
        one_shot(&args, &runtime, &tokenizer, sampling)                        // ⑤
    }
}
```

1. Read all the weights from the GGUF file (§2.3).
2. Load the tokenizer. It needs `model.config` because the EOS and BOS ids come from the GGUF metadata (§2.4).
3. Choose the device and prepare it (§2.5).
4. Bundle the sampling settings. Every request will get its own `Sampler` built from these.
5. We passed `--prompt`, so `one_shot` runs (Part 3). Without `--prompt` the interactive loop would run instead (Part 13).

### 2.3 `Model::load`: from a GGUF file to tensors

[src/weights.rs:33](../src/weights.rs#L33)
```rust
    pub fn load(path: &str) -> Result<Model> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path))?;  // ①
        let device = Device::Cpu;
        let mut load = |name: &str| -> Result<Tensor> {                                       // ②
            content
                .tensor(&mut file, name, &device)?
                .dequantize(&device)
        };

        let token_embd = load("token_embd.weight")?;
        let output_norm = load("output_norm.weight")?;
        // Models that tie their input and output embeddings ship no separate
        // output matrix; sharing the tensor avoids a second full copy.
        let output = if content.tensor_infos.contains_key("output.weight") {                // ③
            load("output.weight")?
        } else {
            token_embd.clone()
        };

        let config = Config {                                                                  // ④
            n_layers: metadata(&content, "llama.block_count")?.to_u32()?,
            n_heads: metadata(&content, "llama.attention.head_count")?.to_u32()?,
            n_kv_heads: metadata(&content, "llama.attention.head_count_kv")?.to_u32()?,
            hidden_dim: metadata(&content, "llama.embedding_length")?.to_u32()?,
            ffn_dim: metadata(&content, "llama.feed_forward_length")?.to_u32()?,
            rope_theta: metadata(&content, "llama.rope.freq_base")?.to_f32()?,
            rms_eps: metadata(&content, "llama.attention.layer_norm_rms_epsilon")?.to_f32()?,
            vocab_size: token_embd.dim(0)? as u32,
            context_length: metadata(&content, "llama.context_length")?.to_u32()?,
            eos_token_id: metadata(&content, "tokenizer.ggml.eos_token_id")?.to_u32()?,
            bos_token_id: metadata(&content, "tokenizer.ggml.bos_token_id")?.to_u32()?,
        };

        let layers = (0..config.n_layers as usize)                                           // ⑤
            .map(|i| {
                Ok(Layer {
                    index: i,
                    attn_norm: load(&format!("blk.{i}.attn_norm.weight"))?,
                    attn_q: unpermute_rope_rows(
                        &load(&format!("blk.{i}.attn_q.weight"))?,
                        config.n_heads as usize,
                    )?,
                    attn_k: unpermute_rope_rows(
                        &load(&format!("blk.{i}.attn_k.weight"))?,
                        config.n_kv_heads as usize,
                    )?,
                    attn_v: load(&format!("blk.{i}.attn_v.weight"))?,
                    attn_output: load(&format!("blk.{i}.attn_output.weight"))?,
                    ffn_norm: load(&format!("blk.{i}.ffn_norm.weight"))?,
                    ffn_gate: load(&format!("blk.{i}.ffn_gate.weight"))?,
                    ffn_up: load(&format!("blk.{i}.ffn_up.weight"))?,
                    ffn_down: load(&format!("blk.{i}.ffn_down.weight"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Model {
            config,
            token_embd,
            layers,
            output_norm,
            output,
        })
    }
```

1. `Content::read` parses only the file **header**: the metadata key/value pairs and a table saying where each tensor's bytes are. Tensor data stays on disk until it is asked for.
2. `load` is a closure that reads one named tensor and **dequantizes** it from Q8_0 (8-bit integers plus a scale per 32 values) to F32.
3. SmolLM2 ties its input and output embeddings, so its GGUF file normally has no `output.weight`. `token_embd.clone()` only copies a reference: Candle tensors share their storage, so no second 47-million-value copy is made.
4. The `Config` comes from metadata keys. The `metadata` helper turns a missing key into a clear error message.
5. Each of the 32 layers loads nine tensors. Q and K go through `unpermute_rope_rows`, explained next.

The result uses a lot more memory than the file: about 362 million weights × 4 bytes ≈ **1.45 GB** of F32 in RAM, compared with roughly 0.4 GB for the Q8_0 file.

#### Why Q and K are reordered

[src/weights.rs:20](../src/weights.rs#L20)
```rust
fn unpermute_rope_rows(w: &Tensor, n_heads: usize) -> Result<Tensor> {
    let (rows, cols) = w.dims2()?;
    let half = rows / n_heads / 2;
    w.reshape((n_heads, half, 2, cols))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((rows, cols))
}
```

RoPE rotates *pairs* of numbers inside each head (Part 7). llama.cpp stores Q/K rows so that pairs sit next to each other (row 2j with row 2j+1). This engine's RoPE pairs row j with row j + half, which is how HuggingFace checkpoints are laid out. The reshape–transpose–reshape moves the even rows of each head to the first half and the odd rows to the second half.

A tiny example with one head of `head_dim = 4` (`half = 2`). HuggingFace rows are `[r0, r1, r2, r3]`, with pairs (r0, r2) and (r1, r3):

| Step | Row order |
|------|-----------|
| As stored in GGUF (pairs adjacent) | `r0 r2 r1 r3` |
| `reshape (1, 2, 2, cols)` | `[[r0, r2], [r1, r3]]` |
| `transpose(1, 2)` | `[[r0, r1], [r2, r3]]` |
| `reshape` back | `r0 r1 r2 r3` ✔ |

The test `unpermute_inverts_llama_cpp_layout` in the same file checks exactly this against a copy of llama.cpp's own permutation.

### 2.4 The tokenizer and the stop tokens

[src/main.rs:128](../src/main.rs#L128)
```rust
fn load_tokenizer(path: &str, config: &Config) -> Result<SmollLM230MTokenizer, BoxError> {
    SmollLM230MTokenizer::from_file(path, config.eos_token_id, config.bos_token_id)
}

// Tokens that end a generation: the model's EOS, plus the chat end-of-turn
// marker when the vocabulary has one.
fn stop_tokens(tokenizer: &SmollLM230MTokenizer) -> Vec<u32> {
    let mut stops = vec![tokenizer.eos_token_id];
    stops.extend(tokenizer.token_to_id(chat::END_OF_TURN));
    stops
}
```

`stops.extend(...)` is given an `Option<u32>`. An `Option` behaves like a list of zero or one items, so the line adds `<|im_end|>` (id 2) only if the vocabulary has it. With SmolLM2's EOS of 0, the result is `[0, 2]`.

The tokenizer itself is a thin wrapper:

[src/tokenizer.rs:23](../src/tokenizer.rs#L23)
```rust
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
        let encoding = self.inner.encode(text, false)?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.inner.decode(ids, true)?)
    }
```

- In `encode`, `false` means "do not add special tokens", so no BOS token is put in front. The model sees exactly `[504, 3575, 282, 4649, 314]` for prompt A.
- In `decode`, `true` means "skip special tokens", so markers like `<|im_end|>` never show up in printed text.

### 2.5 `Runtime::new`: choosing the device

[src/runtime.rs:19](../src/runtime.rs#L19)
```rust
    pub fn new(model: &'m Model, device: &str) -> Result<Self, BoxError> {
        match device {
            "cpu" => Ok(Runtime {
                model,
                #[cfg(feature = "cuda")]
                cuda: None,
            }),
            #[cfg(feature = "cuda")]
            "cuda" => {
                let runtime = crate::cuda::context::CudaRuntime::new(0)
                    .map_err(|e| format!("CUDA init failed: {e}"))?;
                let cuda_model = crate::cuda::model::CudaModel::upload(&runtime, model)
                    .map_err(|e| format!("weight upload failed: {e}"))?;
                Ok(Runtime {
                    model,
                    cuda: Some((runtime, cuda_model)),
                })
            }
            #[cfg(not(feature = "cuda"))]
            "cuda" => Err("built without the `cuda` feature; rebuild with --features cuda".into()),
            other => Err(format!("unknown device {other:?} (expected \"cpu\" or \"cuda\")").into()),
        }
    }
```

The `#[cfg(...)]` attributes decide at **compile time** which `match` arms exist. In a normal build, the `"cuda"` arm only returns the error message. With `--features cuda`, it creates a CUDA context and uploads the weights (Part 15). For our run, `device` is `"cpu"`, so the runtime just remembers a reference to the model. Nothing is allocated yet; the KV cache is created in Part 3, once we know how big it must be.

---

## Part 3 — Turning prompts into requests (`one_shot`)

[src/main.rs:158](../src/main.rs#L158)
```rust
fn one_shot(
    args: &CliArgs,
    runtime: &Runtime,
    tokenizer: &SmollLM230MTokenizer,
    sampling: SamplingParams,
) -> Result<(), BoxError> {
    let context_length = runtime.model().config.context_length as usize;
    let max_tokens = args.max_tokens.unwrap_or(20);                            // ①
    let block_size = args.model.block_size;

    let requests: Vec<RequestSpec> = args                                      // ②
        .prompt
        .iter()
        .enumerate()
        .map(|(i, prompt)| {
            let text = if args.chat {
                chat::chatml(&[ChatMessage {
                    role: "user".into(),
                    content: prompt.clone(),
                }])
            } else {
                prompt.clone()
            };
            Ok(RequestSpec {
                tokens: tokenizer.encode(&text)?,
                max_tokens,
                arrival_step: args.arrival_step.get(i).copied().unwrap_or(0),
            })
        })
        .collect::<Result<_, BoxError>>()?;                                    // ③

    // Enough blocks for every request to reach its own prompt+max_tokens at
    // once, capped at the model's context_length - sized off what the
    // requests actually need rather than the full context_length per
    // sequence, which on a GPU can mean OOM well before the pool is full.
    let num_blocks = args.model.num_blocks.unwrap_or_else(|| {                 // ④
        let longest = requests
            .iter()
            .map(|r| (r.tokens.len() + r.max_tokens).min(context_length))
            .max()
            .unwrap_or(0);
        blocks_for(longest, block_size) * requests.len()
    });

    let backend = runtime.backend(block_size, num_blocks)?;                    // ⑤
    let result = engine::run(                                                  // ⑥
        backend,
        &requests,
        sampling,
        EngineConfig {
            block_size,
            num_blocks,
            max_running: usize::MAX,
        },
    )?;
    // … (printing, see Part 11)
```

1. `max_tokens` is `Some(3)`, so it is 3. (Without the flag it would be 20.)
2. Each prompt becomes a `RequestSpec`: its token ids, its `max_tokens`, and its `arrival_step`. `--chat` is off, so the text is used as is. With `--chat`, it would first be wrapped in the ChatML template (Part 13).
3. `collect::<Result<_, BoxError>>()` turns "a list of results" into "a result holding a list". If any prompt fails to tokenize, the whole thing fails with that error, and `?` returns it.
4. `--num-blocks` was not given, so the pool is sized from the requests. Every request needs at most `prompt + max_tokens` token slots:

   | Request | `tokens.len() + max_tokens` |
   |---------|-----------------------------|
   | A | 5 + 3 = 8 |
   | B | 3 + 3 = 6 |
   | C | 5 + 3 = 8 |

   `longest` = 8. `blocks_for(8, 4)` = ⌈8 / 4⌉ = 2 blocks per request, × 3 requests = **6 blocks** (24 token slots in total).
5. `runtime.backend(4, 6)` builds the backend and its KV cache (below).
6. `engine::run` does all the generation (Part 4). `max_running: usize::MAX` means "no limit on how many requests run together; only the block pool limits it".

`blocks_for` is a one-line helper:

[src/main.rs:124](../src/main.rs#L124)
```rust
fn blocks_for(tokens: usize, block_size: usize) -> usize {
    tokens.div_ceil(block_size).max(1)
}
```

### Building the backend and the KV cache

[src/runtime.rs:48](../src/runtime.rs#L48)
```rust
    pub fn backend(
        &self,
        block_size: usize,
        num_blocks: usize,
    ) -> Result<Box<dyn Backend + '_>, BoxError> {
        #[cfg(feature = "cuda")]
        if let Some((runtime, cuda_model)) = &self.cuda {
            return Ok(Box::new(crate::cuda::engine::CudaBackend::new(
                runtime, cuda_model, block_size, num_blocks,
            )?));
        }
        Ok(Box::new(CpuBackend::new(
            self.model, block_size, num_blocks,
        )?))
    }
```

The return type `Box<dyn Backend + '_>` means "some type that implements `Backend`, decided at run time". The caller does not know or care whether it got a CPU or a CUDA backend.

[src/engine.rs:65](../src/engine.rs#L65)
```rust
    pub fn new(model: &'a Model, block_size: usize, num_blocks: usize) -> Result<Self, BoxError> {
        let config = &model.config;
        let kv_storage = KvStorage::new(
            config.n_layers as usize,
            num_blocks,
            block_size,
            config.n_kv_heads as usize,
            config.head_dim(),
            model.device(),
        )?;
        Ok(CpuBackend { model, kv_storage })
    }
```

[src/kv_storage.rs:76](../src/kv_storage.rs#L76)
```rust
    pub fn new(
        n_layers: usize,
        num_blocks: usize,
        block_size: usize,
        n_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<KvStorage> {
        let shape = (num_blocks, block_size, n_kv_heads, head_dim);
        let kv_k = (0..n_layers)
            .map(|_| Tensor::zeros(shape, DType::F32, device))
            .collect::<Result<Vec<_>>>()?;
        let kv_v = (0..n_layers)
            .map(|_| Tensor::zeros(shape, DType::F32, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(KvStorage { kv_k, kv_v })
    }
```

For our run this creates 32 K tensors and 32 V tensors, each of shape `[6 blocks, 4 slots, 5 KV heads, 64]`, all zeros. That is 24 slots × 80 KiB ≈ **1.9 MiB** in total. The tensors are the *storage*; which request owns which block is tracked separately by the scheduler (Part 5).

---

## Part 4 — The driver loop (`engine::run`)

`engine::run` is used when all the requests are known up front: the one-shot CLI and the benchmark. It wraps the engine's step-by-step API in a simple loop.

[src/engine.rs:515](../src/engine.rs#L515)
```rust
pub fn run<B: Backend>(
    backend: B,
    requests: &[RequestSpec],
    sampling: SamplingParams,
    cfg: EngineConfig,
) -> Result<RunResult, BoxError> {
    let mut engine = Engine::new(backend, cfg);                                // ①
    let eos_token_id = engine.config().eos_token_id;

    let mut order: Vec<usize> = (0..requests.len()).collect();                 // ②
    order.sort_by_key(|&i| requests[i].arrival_step);
    let mut next_pending = 0;

    let mut req_of: HashMap<SequenceId, usize> = HashMap::new();
    let mut generated: Vec<Vec<u32>> = vec![Vec::new(); requests.len()];
    let mut ttft = vec![Duration::ZERO; requests.len()];

    let started = Instant::now();
    let mut step_idx = 0usize;
    loop {
        while next_pending < order.len() && requests[order[next_pending]].arrival_step <= step_idx {
            let i = order[next_pending];                                       // ③
            next_pending += 1;
            let id = engine.add_request(Request {
                tokens: requests[i].tokens.clone(),
                max_tokens: requests[i].max_tokens,
                stop_tokens: vec![eos_token_id],
                sampler: Sampler::from_params(sampling),
            })?;
            req_of.insert(id, i);
        }

        if !engine.has_work() && next_pending == order.len() {                 // ④
            break;
        }

        for ev in engine.step()? {                                             // ⑤
            let i = req_of[&ev.id];
            generated[i].push(ev.token);
            if let Some(t) = ev.ttft {
                ttft[i] = t;
            }
        }
        step_idx += 1;
    }

    let scheduler = engine.scheduler();
    let stats = RunStats {                                                     // ⑥
        steps: step_idx,
        // …
    };
    Ok(RunResult { generated, stats })
}
```

1. **Create the engine** (Part 5). `B` is a generic type. Here it is `Box<dyn Backend>`, which works because `engine.rs` also implements `Backend` for any `Box<B>` (line 44), forwarding each call to the boxed value.
2. **Sort by arrival.** `order` lists request indexes by `arrival_step`. Our requests all arrive at step 0, so the order stays `[0, 1, 2]`. (`sort_by_key` is stable, so ties keep their original order.)
3. **Add requests that have "arrived".** At step 0 all three are added. Each gets `stop_tokens = [EOS]` only, and its own `Sampler` built from the same settings. `req_of` remembers which `SequenceId` belongs to which prompt index.
4. **Stop** when nothing is running, nothing is waiting, and no request is still due to arrive.
5. **Run one engine step** and file each produced token under its prompt. The first token of each request carries a `ttft` (time to first token).
6. **Collect statistics** from the scheduler and the engine (Part 11 predicts them).

For our example the loop runs `engine.step()` exactly twice:

| Loop pass | `step_idx` | Requests added | Events returned by `step()` |
|-----------|-----------:|----------------|------------------------------|
| 1 | 0 → 1 | A, B, C | `a1 b1 c1` (prefill) then `a2 b2 c2` (decode) |
| 2 | 1 → 2 | — | `a3 b3 c3` (decode, all finish) |
| 3 | 2 | — | none: no work left, so the loop breaks |

The rest of this document explains how those events come about.

---

## Part 5 — Creating the engine and queueing requests

### 5.1 `Engine::new`

[src/engine.rs:198](../src/engine.rs#L198)
```rust
pub struct Engine<B: Backend> {
    backend: B,
    cfg: EngineConfig,
    scheduler: Scheduler,
    seqs: HashMap<SequenceId, SeqState>,
    pub stats: EngineStats,
}

impl<B: Backend> Engine<B> {
    pub fn new(backend: B, cfg: EngineConfig) -> Self {
        Engine {
            backend,
            cfg,
            scheduler: Scheduler::new(cfg.num_blocks as u32, cfg.block_size),
            seqs: HashMap::new(),
            stats: EngineStats::default(),
        }
    }
```

The engine owns three things: the **backend** (the math), the **scheduler** (bookkeeping, from the `kv-cache-scheduler` crate), and **`seqs`** (the engine's own per-request state). The scheduler is created like this:

*(crate)* `sequence.rs` — `Scheduler::new`
```rust
    pub fn new(num_blocks: u32, block_size: usize) -> Scheduler {
        Scheduler {
            pool: BlockPool::new(num_blocks),
            block_size,
            sequences: HashMap::new(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            next_id: 0,
            eviction: LRUEviction::new(),
            prefix_cache: PrefixCache::new(block_size),
            metrics: Metrics::new(),
        }
    }
```

*(crate)* `block_pool.rs` — `BlockPool`
```rust
    pub fn new(num_blocks: u32) -> Self {
        let ref_count = vec![0; num_blocks as usize];
        let mut free_list = VecDeque::new();
        for i in 0..num_blocks {
            free_list.push_back(BlockID(i));
        }
        BlockPool { free_list, ref_count, total_allocs: 0 }
    }

    // Allocates one block, sets its ref count to 1, and returns its ID.
    // Returns `None` if the pool is exhausted.
    pub fn alloc(&mut self) -> Option<BlockID> {
        let block = self.free_list.pop_back()?;
        self.ref_count[block.0 as usize] = 1;
        self.total_allocs += 1;
        Some(block)
    }

    fn free(&mut self, block_id: BlockID) {
        assert_eq!(self.ref_count[block_id.0 as usize], 1);
        self.ref_count[block_id.0 as usize] = 0;
        self.free_list.push_front(block_id);
    }
    // …
    // Decrements the ref count. Frees the block when it reaches zero.
    pub fn decref(&mut self, block_id: BlockID) {
        match self.ref_count[block_id.0 as usize] {
            0 => panic!("decref on unallocated block {:?}", block_id),
            1 => self.free(block_id),
            _ => self.ref_count[block_id.0 as usize] -= 1,
        }
    }
```

With our 6 blocks the free list starts as `[B0, B1, B2, B3, B4, B5]`. `alloc` takes from the **back**, so the first block handed out is **B5**, then B4, and so on. `free` puts blocks back at the **front**. (`B5` is shorthand for `BlockID(5)` throughout this document.) Each block also has a **reference count**: how many sequences currently use it. A block returns to the free list only when that count reaches zero.

### 5.2 Two places that track a request

Every request is tracked in two maps, both keyed by `SequenceId`:

| Map | Owner | Holds |
|-----|-------|-------|
| `scheduler.sequences` | crate | `token_ids` (as `TokenId(i32)`), the `block_table`, and a state (waiting/decoding/finished) |
| `engine.seqs` | engine | `SeqState`: everything the crate does not know about (below) |

[src/engine.rs:176](../src/engine.rs#L176)
```rust
struct SeqState {
    // Length of the original prompt, before any generated tokens were appended.
    prompt_len: usize,
    max_tokens: usize,
    stop_tokens: Vec<u32>,
    num_generated: usize,
    // Most recently sampled token: the input of the next decode step (it is
    // emitted as soon as it's sampled, but not in the KV cache until then).
    last_token: u32,
    sampler: Sampler,
}
```

### 5.3 `add_request`

[src/engine.rs:248](../src/engine.rs#L248)
```rust
    pub fn add_request(&mut self, req: Request) -> Result<SequenceId, RequestError> {
        if req.tokens.is_empty() {
            return Err(RequestError::EmptyPrompt);
        }
        if req.max_tokens == 0 {
            return Err(RequestError::ZeroMaxTokens);
        }
        let context = self.config().context_length as usize;
        let prompt_len = req.tokens.len();
        if prompt_len >= context {
            return Err(RequestError::PromptTooLong {
                prompt: prompt_len,
                context,
            });
        }
        let max_tokens = req.max_tokens.min(context - prompt_len);
        let capacity = self.cfg.num_blocks * self.cfg.block_size;
        if prompt_len + max_tokens > capacity {
            return Err(RequestError::ExceedsKvCapacity {
                tokens: prompt_len + max_tokens,
                capacity,
            });
        }

        let seq_id = self
            .scheduler
            .add_request(req.tokens.iter().map(|&t| TokenId(t as i32)).collect());
        self.seqs.insert(
            seq_id,
            SeqState {
                prompt_len,
                max_tokens,
                stop_tokens: req.stop_tokens,
                num_generated: 0,
                last_token: 0,
                sampler: req.sampler,
            },
        );
        Ok(seq_id)
    }
```

For request A the checks go like this:

| Check | A's numbers | Result |
|-------|-------------|--------|
| Prompt empty? | 5 tokens | ok |
| `max_tokens == 0`? | 3 | ok |
| `prompt_len >= context`? | 5 vs 8192 | ok |
| Clamp `max_tokens` | `min(3, 8192 − 5)` = 3 | 3 |
| `prompt_len + max_tokens > capacity`? | 5 + 3 = 8 vs 6 × 4 = 24 | ok |

The last check rejects requests that could never fit even if they ran alone. The token ids are then converted to the crate's `TokenId` type and handed to the scheduler:

*(crate)* `sequence.rs` — `Scheduler::add_request`
```rust
    pub fn add_request(&mut self, tokens: Vec<TokenId>) -> SequenceId {
        let seq_id = SequenceId(self.next_id);
        self.sequences.insert(seq_id, Sequence::new(seq_id, tokens, self.block_size));
        self.waiting.push_back(seq_id);
        self.next_id += 1;
        seq_id
    }
```

After the three calls:

| | A | B | C |
|---|---|---|---|
| `SequenceId` | `S0` | `S1` | `S2` |
| Scheduler state | waiting | waiting | waiting |
| Block table | empty | empty | empty |

`waiting = [S0, S1, S2]`, `running = []`, and all 6 blocks are free. **No model computation has happened yet.** Queueing is pure bookkeeping.

---

## Part 6 — Prefill: admitting requests

### 6.1 `step` = admit, then decode

[src/engine.rs:303](../src/engine.rs#L303)
```rust
    pub fn step(&mut self) -> Result<Vec<TokenEvent>, BoxError> {
        let mut events = Vec::new();
        self.admit_waiting(&mut events)?;
        self.decode(&mut events)?;
        Ok(events)
    }
```

Each step first **admits** waiting requests (one prefill forward each), then runs **one batched decode** over everything that is running. This part covers admission; Part 10 covers decode.

### 6.2 `admit_waiting`

[src/engine.rs:310](../src/engine.rs#L310)
```rust
    fn admit_waiting(&mut self, events: &mut Vec<TokenEvent>) -> Result<(), BoxError> {
        let block_size = self.cfg.block_size;
        while self.scheduler.running.len() < self.cfg.max_running {                       // ①
            let Some(&seq_id) = self.scheduler.waiting.front() else {
                break;
            };
            let needed_blocks = self.scheduler.sequences[&seq_id]                         // ②
                .token_ids
                .len()
                .div_ceil(block_size);
            if self.scheduler.pool.available() < needed_blocks {
                break;
            }

            let token_ids = self.scheduler.sequences[&seq_id].token_ids.clone();
            let (_matched_tokens, matched_blocks) =                                       // ③
                self.scheduler.prefix_cache.match_prefix(&token_ids);
            let skip_tokens = (matched_blocks.len() * block_size).min(token_ids.len() - 1);

            let prefill_started = Instant::now();
            self.scheduler.prefill(seq_id);                                               // ④

            let seq = &self.scheduler.sequences[&seq_id];
            let tokens: Vec<u32> = seq.token_ids.iter().map(|t| t.0 as u32).collect();
            let write_positions: Vec<(BlockID, usize)> = (skip_tokens..tokens.len())      // ⑤
                .map(|i| seq.block_table.logical_to_physical(i))
                .collect();

            // Only the tokens past the cached prefix run through the model,
            // but attention reads the sequence's whole block table.
            let item = BatchItem {                                                        // ⑥
                tokens: &tokens[skip_tokens..],
                write_positions: &write_positions,
                read_blocks: seq.block_table.blocks(),
                read_num_tokens: seq.block_table.num_tokens(),
                position_offset: skip_tokens,
            };
            let state = self.seqs.get_mut(&seq_id).unwrap();
            let next = self                                                               // ⑦
                .backend
                .forward_sample(&[item], true, &mut [&mut state.sampler])?[0];
            let prefill_elapsed = prefill_started.elapsed();
            self.stats.prefill_time += prefill_elapsed;
            self.stats.prefill_tokens += tokens.len() - skip_tokens;

            let first = state.num_generated == 0;                                         // ⑧
            self.emit(seq_id, next, first.then_some(prefill_elapsed), events);
        }
        Ok(())
    }
```

1. **Room to run?** Keep admitting while fewer than `max_running` requests are running (unlimited here). The `let … else` takes the request at the front of the FIFO queue, or leaves the loop if the queue is empty.
2. **Room in memory?** The prompt needs ⌈prompt_len / block_size⌉ blocks. If the pool does not have that many free blocks, stop admitting; this request (and everything behind it) waits for a later step. This check counts *all* the prompt's blocks, even ones that could come from the prefix cache. That keeps it simple and makes sure step ④ can never run out of blocks.
3. **Ask the prefix cache** which leading blocks of this prompt are already in memory. `skip_tokens` is how many tokens we can skip computing. It is capped at `prompt_len − 1`: at least one token must run through the model, because we need its logits to sample the first output token.
4. **Let the scheduler allocate blocks** (shown below) and move the request from `waiting` to `running`.
5. **Work out where each new token's K/V goes.** `logical_to_physical(i)` turns "token *i* of this sequence" into "(block, slot)".
6. **Describe the work** as a `BatchItem`: the tokens to run, where to write their K/V, which blocks to read for attention, how long the real context is, and the position of the first new token.
7. **Run the model** on a batch of one item, with `last_only = true` because only the last position's logits are needed. One token id comes back.
8. **Record the token** (Part 9). If this is the request's first token, its prefill time is reported as its TTFT. (It is not the first token when a preempted request is re-admitted; see Part 12.)

### 6.3 The scheduler side of admission *(crate)*

*(crate)* `prefix_cache.rs` — `match_prefix` and `insert`
```rust
    pub fn insert(&mut self, tokens: &[TokenId], blocks: &[BlockID]) {
        let mut current_node = &mut self.root;
        for (i, token) in tokens.iter().enumerate() {
            current_node = current_node.children.entry(*token).or_insert_with(TrieNode::new);
            if (i + 1) % self.block_size == 0 {
                current_node.block_id = Some(blocks[(i + 1) / self.block_size - 1]);
            }
        }
    }

    pub fn match_prefix(&self, tokens: &[TokenId]) -> (usize, Vec<BlockID>) {
        let mut matched_tokens = 0;
        let mut matched_blocks = Vec::new();
        let mut current_node = &self.root;

        for token in tokens {
            if let Some(child) = current_node.children.get(token) {
                current_node = child;
                matched_tokens += 1;
                if matched_tokens % self.block_size == 0 {
                    if let Some(id) = current_node.block_id {
                        matched_blocks.push(id);
                    } else {
                        break;
                    }
                }
            } else {
                break;
            }
        }

        (matched_tokens, matched_blocks)
    }
```

The prefix cache is a **trie** (a tree of tokens). Walking down from the root one token at a time spells out a prompt prefix. Every 4th node (with our block size) records which block holds those 4 tokens' K/V. `match_prefix` walks the trie with a new prompt and collects a block for every *complete* block of matching tokens.

*(crate)* `sequence.rs` — `Scheduler::prefill`
```rust
    // Allocates blocks for the full prompt, consulting the prefix cache first.
    // Moves the sequence from waiting to running.
    pub fn prefill(&mut self, seq_id: SequenceId) {
        let token_ids: Vec<TokenId> = self.sequences[&seq_id].token_ids.clone();
        let token_count = token_ids.len();
        self.metrics.prefix_total += token_count;
        let (_matched_tokens, matched_blocks) = self.prefix_cache.match_prefix(&token_ids);
        let cache_block_count = matched_blocks.len();
        for &block_id in &matched_blocks {
            self.pool.incref(block_id);
            self.sequences.get_mut(&seq_id).unwrap().block_table.push_full_block(block_id);
            self.eviction.on_access(&block_id);
        }
        let prefix_token_count = cache_block_count * self.block_size;
        self.metrics.prefix_hits += prefix_token_count;
        for _ in prefix_token_count..token_count {
            self.sequences.get_mut(&seq_id).unwrap().block_table.append_token(&mut self.pool).unwrap();
            let block_id = self.sequences[&seq_id].block_table.last_block().unwrap();
            self.eviction.on_access(&block_id);
        }
        let all_blocks: Vec<_> = self.sequences[&seq_id].block_table.blocks().to_vec();
        self.prefix_cache.insert(&token_ids, &all_blocks);
        self.sequences.get_mut(&seq_id).unwrap().state = SequenceState::Decoding;
        self.waiting.retain(|id| id != &seq_id);
        self.running.push(seq_id);
    }
```

In order, `prefill`:

1. Counts the prompt tokens toward `prefix_total` (for the hit-rate statistic).
2. **Shares** each matched block: increases its reference count and appends it to this sequence's block table.
3. Counts the shared tokens toward `prefix_hits`.
4. Reserves one slot per remaining prompt token with `append_token`, allocating a new block whenever the last one is full.
5. Records this prompt's full blocks in the trie, so later prompts can share them.
6. Moves the sequence from `waiting` to `running`.

The block table methods it uses:

*(crate)* `block_table.rs`
```rust
    pub fn logical_to_physical(&self, token_index: usize) -> (BlockID, usize) {
        let block_idx: usize = token_index / self.block_size;
        let offset: usize = token_index % self.block_size;
        (self.blocks[block_idx], offset)
    }
    pub fn append_token(&mut self, pool: &mut BlockPool) -> Result<(), PoolFull> {
        let block_boundary: usize = self.num_tokens % self.block_size;
        match block_boundary {
            0 => {
                let new_block = pool.alloc();
                match new_block {
                    Some(block) => {
                        self.blocks.push(block);
                        self.num_tokens += 1;
                        Ok(())
                    }
                    _ => Err(PoolFull),
                }
            }
            _ => {
                self.num_tokens += 1;
                Ok(())
            }
        }
    }
    // …
    // Insert a pre-existing shared block (from prefix cache) without allocating from the pool
    // Caller is responsible for having already incref'd the block.
    pub fn push_full_block(&mut self, block_id: BlockID) {
        self.blocks.push(block_id);
        self.num_tokens += self.block_size;
    }
```

`append_token` does not store any data. It only reserves a slot: if the token count is a multiple of the block size (the last block is full, or there is none yet), it allocates a new block first.

### 6.4 Trace: admitting A, B and C

The admission loop runs three times in step 1.

**Request A (`S0`), prompt `[504, 3575, 282, 4649, 314]`**

| What | Value |
|------|-------|
| `needed_blocks` | ⌈5 / 4⌉ = 2; 6 are free → admit |
| `match_prefix` | trie is empty → no blocks → `skip_tokens` = 0 |
| `Scheduler::prefill` | allocates **B5** (tokens 0–3) and **B4** (token 4); block table `[B5, B4]`, `num_tokens` = 5 |
| `prefix_cache.insert` | the node after `504 → 3575 → 282 → 4649` now points at **B5** |
| `write_positions` | `(B5,0) (B5,1) (B5,2) (B5,3) (B4,0)` |
| `BatchItem` | tokens = all 5; `read_blocks` = `[B5, B4]`; `read_num_tokens` = 5; `position_offset` = 0 |
| Result | the model computes all 5 positions and samples **`a1`** |

**Request B (`S1`), prompt `[504, 6376, 314]`**

| What | Value |
|------|-------|
| `needed_blocks` | ⌈3 / 4⌉ = 1; 4 are free → admit |
| `match_prefix` | `504` matches (A put it in the trie), `6376` does not. Only 1 token matched, which is not a whole block → no blocks → `skip_tokens` = 0 |
| `Scheduler::prefill` | allocates **B3**; block table `[B3]`, `num_tokens` = 3 |
| `write_positions` | `(B3,0) (B3,1) (B3,2)` |
| Result | 3 positions computed, samples **`b1`** |

**Request C (`S2`), prompt `[504, 3575, 282, 4649, 314]` (same as A)**

| What | Value |
|------|-------|
| `needed_blocks` | 2; 3 are free → admit |
| `match_prefix` | walks `504 → 3575 → 282 → 4649`: 4 tokens, one whole block, and the node points at **B5** → `matched_blocks` = `[B5]` |
| `skip_tokens` | `min(1 × 4, 5 − 1)` = **4** |
| `Scheduler::prefill` | B5's refcount goes 1 → **2**; `push_full_block(B5)`; `prefix_hits` += 4; token 4 needs a new block → allocates **B2**; block table `[B5, B2]` |
| `write_positions` | only `(B2,0)`, for token 4 |
| `BatchItem` | tokens = `[314]` (just 1!); `read_blocks` = `[B5, B2]`; `read_num_tokens` = 5; `position_offset` = 4 |
| Result | **1** position computed instead of 5, samples **`c1`** |

The trie after these three prefills:

```
root
└─ 504 "The"
   ├─ 3575 "Ġcapital" ─ 282 "Ġof" ─ 4649 "ĠFrance" [→ B5] ─ 314 "Ġis"
   └─ 6376 "Ġsky" ─ 314 "Ġis"                       (too short to fill a block)
```

And the physical KV cache (`X:n` means "K/V of request X's token at position n"):

```
         slot 0   slot 1   slot 2   slot 3     owner(s)           refcount
 B0    [   -    |   -    |   -    |   -    ]   free
 B1    [   -    |   -    |   -    |   -    ]   free
 B2    [  C:4   |   -    |   -    |   -    ]   C                  1
 B3    [  B:0   |  B:1   |  B:2   |   -    ]   B                  1
 B4    [  A:4   |   -    |   -    |   -    ]   A                  1
 B5    [ A/C:0  | A/C:1  | A/C:2  | A/C:3  ]   A and C (shared)   2
```

> **Why prefill handles one request at a time.** In a single forward call, every item must contribute the same number of new tokens: the code never pads the query dimension. A has 5 new tokens, B has 3, and C has 1, so they cannot share a call. Decode is different: every item contributes exactly one token, so all running requests share one call (Part 10).

---

## Part 7 — The forward pass on the CPU

We now follow request A's prefill call, `forward_sample(&[item], true, …)`, into the model.

### 7.1 `CpuBackend::forward_sample`

[src/engine.rs:84](../src/engine.rs#L84)
```rust
    fn forward_sample(
        &mut self,
        items: &[BatchItem],
        last_only: bool,
        samplers: &mut [&mut Sampler],
    ) -> Result<Vec<u32>, BoxError> {
        let logits = if last_only {
            self.model.forward_last(items, &mut self.kv_storage)?
        } else {
            self.model.forward(items, &mut self.kv_storage)?
        }; // [batch, 1, vocab_size]
        let mut next = Vec::with_capacity(items.len());
        for (i, sampler) in samplers.iter_mut().enumerate() {
            next.push(sampler.sample(&logits.get(i)?.get(0)?)?);
        }
        Ok(next)
    }
```

The backend runs the model, then gives each item's logit row to that item's own sampler. `logits.get(i)?.get(0)?` picks batch item `i`, position 0, leaving a 1-D tensor of 49,152 logits.

### 7.2 `Model`: embed, 32 layers, LM head

[src/model.rs:421](../src/model.rs#L421)
```rust
    fn hidden(&self, items: &[BatchItem], kv_storage: &mut KvStorage) -> Result<Tensor> {
        let q_len = items[0].tokens.len();
        let device = self.device();

        // Turn token ids into vectors, in (batch, q_len) row-major order.
        let flat_tokens: Vec<u32> = items
            .iter()
            .flat_map(|item| item.tokens.iter().copied())
            .collect();
        let mut x = embed(&flat_tokens, &self.token_embd, device)?.reshape((
            items.len(),
            q_len,
            self.config.hidden_dim as usize,
        ))?;

        let ctx = StepCtx::new(&self.config, items, device)?;
        // Run through every layer in sequence, each one refining x a bit more.
        for layer in &self.layers {
            x = layer.forward(&x, &self.config, &ctx, kv_storage)?;
        }
        Ok(x)
    }

    // One final norm after the last layer, then project to vocab-sized logits.
    fn lm_head(&self, x: &Tensor) -> Result<Tensor> {
        let normed = rms_norm(x, &self.output_norm, self.config.rms_eps)?;
        linear(&normed, &self.output)
    }
    // …
    pub fn forward_last(&self, items: &[BatchItem], kv_storage: &mut KvStorage) -> Result<Tensor> {
        let x = self.hidden(items, kv_storage)?;
        let last = x.narrow(1, x.dim(1)? - 1, 1)?;
        self.lm_head(&last)
    }
```

`hidden` does three things:

1. **Embedding lookup.** Each token id selects one row of `token_embd` (`[49152, 960]`):

   [src/model.rs:128](../src/model.rs#L128)
   ```rust
   pub fn embed(token_ids: &[u32], embedding_weight: &Tensor, device: &Device) -> Result<Tensor> {
       // Wrap the ids so we can use them as row indices.
       let tokens_tensor = Tensor::from_slice(token_ids, token_ids.len(), device)?;
       // Pull out row `token_ids[i]` for each i - that row is that token's embedding.
       embedding_weight.index_select(&tokens_tensor, 0)
   }
   ```
   For A, `x` becomes `[batch = 1, tokens = 5, 960]`.
2. **Per-call precomputation** with `StepCtx::new` (§7.3).
3. **32 layers**, each refining `x` (§7.4).

`forward_last` then keeps only the last position (`narrow` on dimension 1) before the LM head. A's prompt has 5 positions, but we only need the logits after `Ġis`, so 4 of the 5 vocabulary-sized projections are skipped. The result has shape `[1, 1, 49152]`.

### 7.3 `StepCtx`: work shared by all 32 layers

[src/model.rs:285](../src/model.rs#L285)
```rust
    pub fn new(config: &Config, items: &[BatchItem], device: &Device) -> Result<Self> {
        let q_len = items[0].tokens.len();
        debug_assert!(
            items.iter().all(|item| item.tokens.len() == q_len),
            "every item must contribute the same number of new tokens"
        );
        let head_dim = config.head_dim();

        // Every item is at its own position, so build per-row positions in
        // (batch, q_len) row-major order, then fold back to broadcast shape.
        let positions: Vec<usize> = items
            .iter()
            .flat_map(|item| item.position_offset..item.position_offset + q_len)
            .collect();
        let (cos, sin) = rope_cos_sin(config.rope_theta, head_dim, &positions, device)?;
        let rope_shape = (items.len(), 1, q_len, head_dim / 2);
        let cos = cos.reshape(rope_shape)?;
        let sin = sin.reshape(rope_shape)?;

        let gather_items: Vec<(&[BlockID], usize)> = items
            .iter()
            .map(|item| (item.read_blocks, item.read_num_tokens))
            .collect();
        let gather = GatherPlan::new(&gather_items, device)?;

        let position_offsets: Vec<usize> = items.iter().map(|i| i.position_offset).collect();
        let read_num_tokens: Vec<usize> = items.iter().map(|i| i.read_num_tokens).collect();
        let mask = batch_mask(
            &position_offsets,
            &read_num_tokens,
            q_len,
            gather.ctx_len(),
            device,
        )?;

        let writes = items
            .iter()
            .flat_map(|item| item.write_positions.iter().copied())
            .collect();
        Ok(StepCtx {
            cos,
            sin,
            mask,
            gather,
            writes,
        })
    }
```

Four things depend only on the batch, not on the layer, so they are computed once and reused 32 times:

| Field | For A's prefill |
|-------|-----------------|
| RoPE `cos`/`sin` | positions `[0, 1, 2, 3, 4]` → two tensors of shape `[1, 1, 5, 32]` |
| `gather` (which blocks to read) | blocks `[B5, B4]`, real context 5 tokens |
| `mask` | 5 × 5 causal mask (below) |
| `writes` | `(B5,0) (B5,1) (B5,2) (B5,3) (B4,0)` |

The gather plan lists each sequence's blocks, padding short lists by repeating their last block:

[src/kv_storage.rs:42](../src/kv_storage.rs#L42)
```rust
    pub fn new(items: &[(&[BlockID], usize)], device: &Device) -> Result<Self> {
        let batch = items.len();
        let ctx_len = items.iter().map(|&(_, n)| n).max().unwrap_or(0);
        let max_blocks = items
            .iter()
            .map(|&(blocks, _)| blocks.len())
            .max()
            .unwrap_or(0);

        let mut idx: Vec<u32> = Vec::with_capacity(batch * max_blocks);
        for &(blocks, _) in items {
            idx.extend(blocks.iter().map(|b| b.0));
            // The padding only ever lands at positions the attention mask
            // rules out, so any valid block will do; the sequence's own last
            // block keeps every index in range and every value finite.
            let pad = blocks.last().map_or(0, |b| b.0);
            idx.resize(idx.len() + (max_blocks - blocks.len()), pad);
        }
        let block_idx = Tensor::from_vec(idx, batch * max_blocks, device)?;
        Ok(GatherPlan {
            block_idx,
            batch,
            max_blocks,
            ctx_len,
        })
    }
```

The mask says which positions each query may look at:

[src/model.rs:180](../src/model.rs#L180)
```rust
pub fn batch_mask(
    position_offset: &[usize],
    read_num_tokens: &[usize],
    q_len: usize,
    ctx_len: usize,
    device: &Device,
) -> Result<Tensor> {
    let batch = position_offset.len();
    let mut mask_data = vec![0f32; batch * q_len * ctx_len];
    for b in 0..batch {
        let offset = position_offset[b];
        let valid_len = read_num_tokens[b];
        for i in 0..q_len {
            for j in 0..ctx_len {
                if j >= valid_len || j > offset + i {
                    mask_data[(b * q_len + i) * ctx_len + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Tensor::from_vec(mask_data, (batch, 1, q_len, ctx_len), device)
}
```

A position `j` is hidden (`-∞`) from query row `i` if it is in the **future** (`j > offset + i`) or is **padding** (`j ≥ valid_len`). For A's prefill (`offset` 0, `valid_len` 5) this is the familiar triangle:

```
                 key position j →   0     1     2     3     4
query row 0 (position 0)          [ 0    -∞    -∞    -∞    -∞ ]
query row 1 (position 1)          [ 0     0    -∞    -∞    -∞ ]
query row 2 (position 2)          [ 0     0     0    -∞    -∞ ]
query row 3 (position 3)          [ 0     0     0     0    -∞ ]
query row 4 (position 4)          [ 0     0     0     0     0 ]
```

For C's prefill there is one query row at position 4 (`offset` 4, `valid_len` 5), so its mask is `[0 0 0 0 0]`: it may look at positions 0–4. Positions 0–3 are read from the shared block B5, which A's prefill filled. Position 4 is read from B2, which C writes in this very call.

### 7.4 `Layer::forward`: one transformer block

[src/model.rs:339](../src/model.rs#L339)
```rust
    pub fn forward(
        &self,
        x: &Tensor,
        config: &Config,
        ctx: &StepCtx,
        kv_storage: &mut KvStorage,
    ) -> Result<Tensor> {
        let (batch, q_len, _) = x.dims3()?;
        let n_heads = config.n_heads as usize;
        let n_kv_heads = config.n_kv_heads as usize;
        let head_dim = config.head_dim();

        // Normalize before attention (pre-norm), not after - this is the
        // convention Llama uses, and it's what keeps deep stacks of these
        // blocks stable to train/run.
        let normed = rms_norm(x, &self.attn_norm, config.rms_eps)?;                       // ①

        // Project into query/key/value space. Q gets n_heads worth of output,
        // K/V get fewer (n_kv_heads) since GQA shares them across several Q
        // heads. Then split the projected dim into per-head chunks and move
        // the head axis in front of the sequence axis:
        // [batch, q_len, heads * head_dim] -> [batch, heads, q_len, head_dim].
        let q = linear(&normed, &self.attn_q)?                                            // ②
            .reshape((batch, q_len, n_heads, head_dim))?
            .transpose(1, 2)?;
        let k = linear(&normed, &self.attn_k)?
            .reshape((batch, q_len, n_kv_heads, head_dim))?
            .transpose(1, 2)?;
        let v = linear(&normed, &self.attn_v)?
            .reshape((batch, q_len, n_kv_heads, head_dim))?
            .transpose(1, 2)?;

        // RoPE rotates every head identically given the position, so one call
        // per tensor covers all heads. Only Q and K get rotated - V carries
        // content, not position.
        let q = rope_apply(&q, &ctx.cos, &ctx.sin)?;                                      // ③
        let k = rope_apply(&k, &ctx.cos, &ctx.sin)?;

        // Write this call's tokens into their block slots. The cache wants
        // token-major rows: [batch * q_len, n_kv_heads, head_dim].
        kv_storage.write_tokens(                                                          // ④
            self.index,
            &ctx.writes,
            &k.transpose(1, 2)?
                .reshape((batch * q_len, n_kv_heads, head_dim))?,
            &v.transpose(1, 2)?
                .reshape((batch * q_len, n_kv_heads, head_dim))?,
        )?;
        // Read back every sequence's full cached K/V (padded to the batch's
        // longest context) and attend against it.
        let (k_full, v_full) = kv_storage.gather(self.index, &ctx.gather)?;               // ⑤
        let attn_out = gqa_attention(&q, &k_full, &v_full, &ctx.mask)?;                   // ⑥

        // Merge the heads back into one dimension - inverse of the earlier
        // split - and project down to hidden_dim so it can be added to the
        // residual stream.
        let attn_out = attn_out
            .transpose(1, 2)?
            .reshape((batch, q_len, n_heads * head_dim))?;
        let attn_out = linear(&attn_out, &self.attn_output)?;

        // Residual connection: the block adds *to* x rather than replacing it,
        // so gradients (and, at inference time, information) can flow straight
        // through even if attention contributes very little.
        let h = x.add(&attn_out)?;                                                        // ⑦

        // Same pre-norm pattern before the FFN half of the block.
        let normed2 = rms_norm(&h, &self.ffn_norm, config.rms_eps)?;
        let ffn_out = swiglu_ffn(&normed2, &self.ffn_gate, &self.ffn_up, &self.ffn_down)?; // ⑧

        // Second residual connection - this is the block's output.
        h.add(&ffn_out)
    }
```

The shapes for A's prefill, with every dimension labelled (A has 5 tokens and the model has 5 KV heads, so the number 5 means two different things):

| Step | Code | Shape |
|------|------|-------|
| Input | `x` | `[batch 1, tokens 5, 960]` |
| ① Pre-norm | `rms_norm(x, …)` | `[1, 5, 960]` |
| ② Q projection | `linear(…, attn_q)` → reshape → transpose | `[1, 5, 960]` → `[1, 5, 15 heads, 64]` → `[1, 15 heads, 5 tokens, 64]` |
| ② K and V projections | `linear(…, attn_k / attn_v)` → … | `[1, 5, 320]` → `[1, 5 tokens, 5 heads, 64]` → `[1, 5 heads, 5 tokens, 64]` |
| ③ RoPE | `rope_apply` on Q and K | shapes unchanged |
| ④ Cache write | `write_tokens` | rows `[5 tokens, 5 heads, 64]` go into B5 slots 0–3 and B4 slot 0 |
| ⑤ Cache read | `gather` | `k_full`, `v_full`: `[1, 5 heads, context 5, 64]` |
| ⑥ Attention | `gqa_attention` | `[1, 15 heads, 5 tokens, 64]` |
| Merge heads | transpose → reshape → `linear(…, attn_output)` | `[1, 5, 960]` |
| ⑦ Residual | `h = x + attn_out` | `[1, 5, 960]` |
| ⑧ Feed-forward | inside `swiglu_ffn`: gate and up | `[1, 5, 2560]` each |
| ⑧ Feed-forward | result | `[1, 5, 960]` |
| Output | `h + ffn_out` | `[1, 5, 960]` |

The order of ④ and ⑤ matters: this call's tokens are **written first**, so the gather that follows already includes them. A token can therefore attend to itself and to the earlier tokens of the same call.

The helper functions, one by one:

**RMSNorm** — scale each row so its root-mean-square is about 1, then apply a learned per-channel weight.

[src/model.rs:60](../src/model.rs#L60)
```rust
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    // …
    let last_dim = x.rank() - 1;
    let rms = x
        .powf(2.0)?
        .mean_keepdim(last_dim)?
        .affine(1.0, eps as f64)?
        .sqrt()?;
    // Divide each row by its own rms, then scale each channel by `weight`.
    let normed = x.broadcast_div(&rms)?;
    normed.broadcast_mul(weight)
}
```

**Linear** — `y = x · Wᵀ`, written to avoid a costly hidden copy.

[src/model.rs:141](../src/model.rs#L141)
```rust
// y = x @ weight^T for x [..., in_features] and weight [out_features,
// in_features] (candle Linear convention). Leading dims are flattened into one
// 2D matmul, which candle runs on the transposed weight view directly. Do not
// use `broadcast_matmul` here: it calls `.contiguous()` on the broadcast
// weight, materializing a full transposed copy of it on every call, which
// made decode ~96% memcpy.
pub fn linear(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let in_features = x.dim(D::Minus1)?;
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let mut out_dims = dims.to_vec();
    *out_dims.last_mut().unwrap() = weight.dim(0)?;
    x.reshape((rows, in_features))?
        .matmul(&weight.t()?)?
        .reshape(out_dims)
}
```

For A's Q projection: `x` `[1, 5, 960]` becomes 5 rows × 960, is multiplied by `attn_qᵀ` (`[960, 960]`), and is reshaped back to `[1, 5, 960]`.

**RoPE** — rotate pairs of numbers by a position-dependent angle.

[src/model.rs:81](../src/model.rs#L81)
```rust
pub fn rope_cos_sin(
    rope_theta: f32,
    head_dim: usize,
    positions: &[usize],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let theta: Vec<f32> = (0..half)
        .map(|j| rope_theta.powf(-2.0 * j as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::from_vec(theta, (1, half), device)?;

    let positions_f: Vec<f32> = positions.iter().map(|&p| p as f32).collect();
    let positions = Tensor::from_vec(positions_f, (positions.len(), 1), device)?;

    // angle[pos, j] = pos * theta_j - how far pair j has rotated by this position.
    let angles = positions.broadcast_mul(&theta)?;
    Ok((angles.cos()?, angles.sin()?))
}

// …
pub fn rope_apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    // Rotate-half pairing: dim i pairs with dim i + half, not its neighbor.
    let first = x.narrow(D::Minus1, 0, half)?;
    let second = x.narrow(D::Minus1, half, half)?;

    // Standard 2D rotation applied to each (first, second) pair.
    let first_rot = first.broadcast_mul(cos)?.sub(&second.broadcast_mul(sin)?)?;
    let second_rot = first.broadcast_mul(sin)?.add(&second.broadcast_mul(cos)?)?;

    Tensor::cat(&[&first_rot, &second_rot], D::Minus1)
}
```

With `head_dim` 64 there are 32 pairs. Pair `j` turns by `θⱼ = 100000^(−2j/64)` radians per position: pair 0 by 1 radian per position, the last pair by only about 0.000014. For A's token at position 4, pair 0 is rotated by 4 radians. The `cos`/`sin` shape `[1, 1, 5, 32]` broadcasts over all heads, which is why one call rotates every head.

**Grouped-query attention** — 15 query heads share 5 KV heads, 3 per group.

[src/model.rs:209](../src/model.rs#L209)
```rust
pub fn gqa_attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (batch, n_heads, q_len, head_dim) = q.dims4()?;
    let (_, n_kv_heads, ctx_len, _) = k.dims4()?;

    // GQA shares each KV head across `group_size` consecutive Q heads. Rather
    // than duplicating K and V group_size times to line up with every Q head
    // (a full copy of the context per layer), fold the group into the query
    // rows: Q heads [j*group_size .. (j+1)*group_size) all become extra query
    // rows against KV head j, so K and V are read as-is.
    let group_size = n_heads / n_kv_heads;
    let q = q
        .contiguous()?
        .reshape((batch, n_kv_heads, group_size * q_len, head_dim))?;

    // …
    let scores = q.matmul(&k.transpose(2, 3)?)?;
    // …
    let scores = scores.affine(1.0 / (head_dim as f64).sqrt(), 0.0)?;

    // Unfold the group to add the mask, which is the same for every head.
    let scores = scores
        .reshape((batch, n_kv_heads, group_size, q_len, ctx_len))?
        .broadcast_add(&mask.unsqueeze(1)?)?;

    // …
    let probs = softmax(&scores, D::Minus1)?;
    // …
    probs
        .reshape((batch, n_kv_heads, group_size * q_len, ctx_len))?
        .matmul(v)?
        .reshape((batch, n_heads, q_len, head_dim))
}
```

Followed with A's prefill numbers:

1. `q` `[1, 15, 5, 64]` is regrouped to `[1, 5 KV heads, 3 × 5 = 15 rows, 64]`. For KV head `j`, the 15 rows are query heads `3j`, `3j+1` and `3j+2` at positions 0–4. No data is copied for K or V.
2. `scores = q · kᵀ`: `[1, 5, 15, 64] × [1, 5, 64, 5]` → `[1, 5, 15, 5]`. Each entry is "how well this query matches this key".
3. Multiply by 1/√64 = 0.125 so that softmax does not saturate.
4. Reshape to `[1, 5, 3, 5, 5]` and add the mask (`[1, 1, 1, 5, 5]` after `unsqueeze`, broadcast over KV heads and groups).
5. Softmax over the last dimension turns each row into weights that sum to 1. Masked entries (`-∞`) get weight exactly 0.
6. Weighted sum of V: `[1, 5, 15, 5] × [1, 5, 5, 64]` → `[1, 5, 15, 64]`, reshaped back to `[1, 15 heads, 5 tokens, 64]`.

**SwiGLU feed-forward**

[src/model.rs:156](../src/model.rs#L156)
```rust
pub fn swiglu_ffn(
    x: &Tensor,
    gate_weight: &Tensor,
    up_weight: &Tensor,
    down_weight: &Tensor,
) -> Result<Tensor> {
    // Two parallel projections of the same input...
    let gate = linear(x, gate_weight)?;
    let up = linear(x, up_weight)?;
    // ...gate gets squashed through silu (a smooth "soft gate" nonlinearity)...
    let silu_gate = silu(&gate)?;
    // ...then used to scale up, elementwise - this is the "gating" in SwiGLU.
    let result = silu_gate.mul(&up)?;
    // Project back down to hidden_dim.
    linear(&result, down_weight)
}
```

960 → 2560 (twice, gate and up) → elementwise `silu(gate) × up` → 2560 → 960.

### 7.5 Writing to and reading from the KV cache

[src/kv_storage.rs:99](../src/kv_storage.rs#L99)
```rust
    pub fn write_tokens(
        &mut self,
        layer: usize,
        positions: &[(BlockID, usize)],
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        let (n_tokens, n_kv_heads, head_dim) = k.dims3()?;
        debug_assert_eq!(n_tokens, positions.len());
        let mut start = 0;
        while start < positions.len() {
            let (block_id, offset) = positions[start];
            let mut len = 1;
            while start + len < positions.len()
                && positions[start + len] == (block_id, offset + len)
            {
                len += 1;
            }
            let block = block_id.0 as usize;
            let run_k = k
                .narrow(0, start, len)?
                .reshape((1, len, n_kv_heads, head_dim))?;
            let run_v = v
                .narrow(0, start, len)?
                .reshape((1, len, n_kv_heads, head_dim))?;
            self.kv_k[layer]
                .narrow(0, block, 1)?
                .slice_set(&run_k, 1, offset)?;
            self.kv_v[layer]
                .narrow(0, block, 1)?
                .slice_set(&run_v, 1, offset)?;
            start += len;
        }
        Ok(())
    }
```

The inner `while` looks for a **run** of tokens going into consecutive slots of the same block, so each run is copied with one `slice_set`. A's write positions `(B5,0) (B5,1) (B5,2) (B5,3) (B4,0)` form two runs:

| Run | Tokens | Destination |
|-----|--------|-------------|
| 1 | positions 0–3 | block B5, slots 0–3 |
| 2 | position 4 | block B4, slot 0 |

That is two copies per tensor per layer, instead of five.

[src/kv_storage.rs:140](../src/kv_storage.rs#L140)
```rust
    pub fn gather(&self, layer: usize, plan: &GatherPlan) -> Result<(Tensor, Tensor)> {
        let read = |cache: &Tensor| -> Result<Tensor> {
            let blocks = cache.index_select(&plan.block_idx, 0)?;
            let (_, block_size, n_kv_heads, head_dim) = blocks.dims4()?;
            blocks
                .reshape((
                    plan.batch,
                    plan.max_blocks * block_size,
                    n_kv_heads,
                    head_dim,
                ))?
                .narrow(1, 0, plan.ctx_len)?
                .transpose(1, 2)?
                .contiguous()
        };
        Ok((read(&self.kv_k[layer])?, read(&self.kv_v[layer])?))
    }
```

For A's prefill (`block_idx` = `[5, 4]`, `ctx_len` = 5):

| Step | Shape |
|------|-------|
| The layer's cache tensor | `[6 blocks, 4 slots, 5 heads, 64]` |
| `index_select` blocks 5 and 4 | `[2, 4, 5, 64]` |
| `reshape` into one sequence | `[1, 8 positions, 5, 64]` |
| `narrow` to the real context | `[1, 5 positions, 5, 64]` |
| `transpose(1, 2)` + `contiguous` | `[1, 5 heads, 5 positions, 64]` |

### 7.6 From hidden state to logits

After the 32nd layer, `forward_last` keeps the last position (`[1, 1, 960]`), and `lm_head` applies the final RMSNorm and multiplies by `output` (`[49152, 960]`; with tied embeddings this *is* the embedding matrix). The result is one row of **49,152 logits**: a score for every possible next token. Back in `CpuBackend::forward_sample`, that row goes to A's sampler.

---

## Part 8 — Sampling the next token

[src/sampler.rs:31](../src/sampler.rs#L31)
```rust
pub struct Sampler {
    temperature: f32,
    top_p: f32,
    rng: StdRng,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: u64) -> Self {
        Sampler {
            temperature,
            top_p,
            rng: StdRng::seed_from_u64(seed),
        }
    }
    // …
    pub fn sample(&mut self, logits: &Tensor) -> candle_core::Result<u32> {
        if self.temperature <= 0.0 {
            return logits.argmax(0)?.to_scalar::<u32>();
        }
        Ok(self.sample_slice(&logits.to_vec1::<f32>()?))
    }
```

Our run uses temperature 0, so `sample` takes the **greedy** branch: `argmax` returns the index of the highest of the 49,152 logits, and that index *is* the token id. No randomness is involved, so the run is reproducible.

With a positive temperature (the HTTP server's default is 1.0), the logits are copied into a plain `Vec<f32>` and `sample_slice` does **temperature + nucleus (top-p) sampling**:

[src/sampler.rs:67](../src/sampler.rs#L67)
```rust
    pub fn sample_slice(&mut self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 {
            // … greedy: a plain scan for the first maximum
        }

        let inv_temp = 1.0 / self.temperature;                                     // ①
        let max_logit = logits
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b * inv_temp));
        let mut probs: Vec<f32> = logits
            .iter()
            .map(|&l| (l * inv_temp - max_logit).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // … (comment explaining the partial sort)
        let descending = |&a: &usize, &b: &usize| probs[b].total_cmp(&probs[a]);   // ②
        let mut order: Vec<usize> = (0..probs.len()).collect();
        let mut k = INITIAL_CANDIDATES.min(order.len());
        let cutoff = loop {
            if k < order.len() {
                order.select_nth_unstable_by(k - 1, descending);
            }
            order[..k].sort_unstable_by(descending);
            let mut cumulative = 0.0f32;
            let reached = order[..k].iter().position(|&idx| {
                cumulative += probs[idx];
                cumulative >= self.top_p
            });
            match reached {
                Some(i) => break i + 1,
                None if k == order.len() => break k,
                None => k = (k * 4).min(order.len()),
            }
        };
        let kept = &order[..cutoff];                                               // ③
        let kept_sum: f32 = kept.iter().map(|&i| probs[i]).sum();

        let r: f32 = self.rng.random::<f32>() * kept_sum;                          // ④
        let mut acc = 0.0f32;
        for &idx in kept {
            acc += probs[idx];
            if acc >= r {
                return idx as u32;
            }
        }
        *kept.last().unwrap() as u32
    }
```

1. **Softmax with temperature.** Divide every logit by the temperature, subtract the largest one (so `exp` cannot overflow), exponentiate, and normalize so the probabilities sum to 1. A low temperature sharpens the distribution; a high one flattens it.
2. **Find the nucleus cheaply.** Sorting all 49,152 probabilities would be slow. `select_nth_unstable_by` moves the 64 largest (`INITIAL_CANDIDATES`) to the front in linear time, and only those 64 are sorted. Walking them in descending order, the code stops as soon as the running total reaches `top_p`. If 64 is not enough (a very flat distribution), it tries 256, then 1,024, and so on.
3. **Keep** just those tokens: the nucleus.
4. **Draw** a random number `r` in `[0, kept_sum)` from this request's own RNG, and walk the kept tokens until the running total passes `r`. More probable tokens cover more of the range, so they are picked more often.

Every request owns a separate `Sampler` with its own RNG, so a seeded request draws the same random numbers no matter which other requests share its batch. (In `engine::run` all requests get the same seed, 42, but each has its own RNG instance.)

---

## Part 9 — Recording tokens and finishing (`emit`)

Every sampled token, whether from prefill or decode, goes through `emit`:

[src/engine.rs:449](../src/engine.rs#L449)
```rust
    fn emit(
        &mut self,
        seq_id: SequenceId,
        token: u32,
        ttft: Option<Duration>,
        events: &mut Vec<TokenEvent>,
    ) {
        let state = self.seqs.get_mut(&seq_id).unwrap();
        state.num_generated += 1;                                                  // ①
        state.last_token = token;
        let finish = if state.stop_tokens.contains(&token) {                       // ②
            Some(FinishReason::Stop)
        } else if state.num_generated >= state.max_tokens {
            Some(FinishReason::Length)
        } else {
            None
        };
        if finish.is_some() {                                                      // ③
            self.scheduler.finish_sequence(seq_id);
            // Finished sequences would otherwise pile up forever in a
            // long-running server.
            self.scheduler.sequences.remove(&seq_id);
            self.seqs.remove(&seq_id);
        }
        events.push(TokenEvent {                                                   // ④
            id: seq_id,
            token,
            finish,
            ttft,
        });
    }
```

1. Count the token and remember it as `last_token`. It is the **input** of the next decode step. Its K/V is not in the cache yet; that happens when it is fed back in.
2. Decide whether the request is done. The stop-token check comes first, so a stop token that is also the last allowed token reports `Stop`, not `Length`.
3. If done, release its KV blocks (`finish_sequence`, shown in Part 10) and forget the request in both maps.
4. Report the token to the caller. The final event carries `finish`. If the request ended on a stop token, that token is still reported; the caller removes it before showing text.

[src/engine.rs:127](../src/engine.rs#L127)
```rust
pub struct TokenEvent {
    pub id: SequenceId,
    pub token: u32,
    pub finish: Option<FinishReason>,
    // Set on a request's first token: how long its prefill forward took.
    pub ttft: Option<Duration>,
}
```

After the three prefills in step 1, the engine state is:

| | A (`S0`) | B (`S1`) | C (`S2`) |
|---|---|---|---|
| `num_generated` | 1 | 1 | 1 |
| `last_token` | `a1` | `b1` | `c1` |
| Event | `a1`, with TTFT | `b1`, with TTFT | `c1`, with TTFT |
| Finished? | no (1 < 3) | no | no |

---

## Part 10 — Batched decode

Still inside step 1, `admit_waiting` has returned and `decode` runs. All three requests are now running.

### 10.1 `decode`

[src/engine.rs:363](../src/engine.rs#L363)
```rust
    fn decode(&mut self, events: &mut Vec<TokenEvent>) -> Result<(), BoxError> {
        if self.scheduler.running.is_empty() {
            return Ok(());
        }
        let decode_started = Instant::now();
        let running = self.scheduler.running.clone();
        let position_offsets: HashMap<SequenceId, usize> = running                     // ①
            .iter()
            .map(|&id| (id, self.scheduler.sequences[&id].block_table.num_tokens()))
            .collect();

        self.scheduler.step();                                                         // ②

        // A sequence that dropped out of `running` was preempted (its blocks
        // evicted). Rebuild its token list as prompt + every token sampled so
        // far, so it re-prefills cleanly - and samples its next token - when
        // readmitted. The scheduler may or may not have appended its
        // placeholder token before the eviction, and the latest sampled
        // token was never written back into the list, so fix up the tail.
        for &seq_id in &running {                                                      // ③
            if !self.scheduler.running.contains(&seq_id) {
                let state = &self.seqs[&seq_id];
                let real_len = state.prompt_len + state.num_generated;
                let token_ids = &mut self.scheduler.sequences.get_mut(&seq_id).unwrap().token_ids;
                token_ids.truncate(real_len - 1);
                token_ids.push(TokenId(state.last_token as i32));
            }
        }

        let mut rows: Vec<DecodeRow> = Vec::with_capacity(running.len());             // ④
        for &seq_id in &running {
            if !self.scheduler.running.contains(&seq_id) {
                continue;
            }
            let last_token = self.seqs[&seq_id].last_token;
            let seq = self.scheduler.sequences.get_mut(&seq_id).unwrap();
            if let Some(t) = seq.token_ids.last_mut() {
                *t = TokenId(last_token as i32);
            }
            let position_offset = position_offsets[&seq_id];
            rows.push(DecodeRow {
                seq_id,
                token: [last_token],
                write: [seq.block_table.logical_to_physical(position_offset)],
                position_offset,
            });
        }
        if rows.is_empty() {
            return Ok(());
        }

        let next_ids = {
            let items: Vec<BatchItem> = rows                                           // ⑤
                .iter()
                .map(|row| {
                    let table = &self.scheduler.sequences[&row.seq_id].block_table;
                    BatchItem {
                        tokens: &row.token,
                        write_positions: &row.write,
                        read_blocks: table.blocks(),
                        read_num_tokens: table.num_tokens(),
                        position_offset: row.position_offset,
                    }
                })
                .collect();
            let mut by_id: HashMap<SequenceId, &mut Sampler> = self                    // ⑥
                .seqs
                .iter_mut()
                .map(|(&id, state)| (id, &mut state.sampler))
                .collect();
            let mut samplers: Vec<&mut Sampler> = rows
                .iter()
                .map(|row| by_id.remove(&row.seq_id).unwrap())
                .collect();
            self.backend.forward_sample(&items, false, &mut samplers)?                 // ⑦
        };

        for (row, &next) in rows.iter().zip(&next_ids) {                               // ⑧
            self.emit(row.seq_id, next, None, events);
        }
        self.stats.decode_time += decode_started.elapsed();
        self.stats.decode_steps += 1;
        Ok(())
    }
```

1. **Where does each request's next token go?** A sequence's `num_tokens` counts the slots it already uses: the prompt plus every generated token except the newest. So it is exactly the **position of the token we are about to feed in**, `last_token`.
2. **Reserve one more slot** per running sequence (crate code below). This is also where a full pool can trigger preemption.
3. **Repair preempted sequences** (Part 12). Nothing is preempted in our example.
4. **Build one row per running sequence.** The scheduler pushed a placeholder `TokenId(0)` into `token_ids`; it is replaced with the real `last_token`. The write slot comes from `logical_to_physical(position)`.
5. **Build the batch.** `BatchItem` only holds references (`&'a [u32]`), so the data it points to has to live somewhere for the duration of the call. `DecodeRow` owns the token and write slot, and the block tables are borrowed from the scheduler without copying.
6. **Collect the samplers in row order.** The backend needs one `&mut Sampler` per row. Rust allows several mutable borrows into a `HashMap` only if they are all taken at once, which is what `iter_mut` does. The code then picks them out in row order.
7. **One forward call** for every running sequence at once, with `last_only = false`. Each item has a single token, so `Model::forward` returns `[batch, 1, vocab]`.
8. **Emit** one token per row.

The crate's half of the work:

*(crate)* `sequence.rs` — `Scheduler::step`
```rust
    // Advances every running sequence by one decode token using standard (non-CoW) allocation.
    pub fn step(&mut self) {
        let running = self.running.clone();
        for seq_id in &running {
            if !self.running.contains(seq_id) { continue; }
            self.sequences.get_mut(seq_id).unwrap().token_ids.push(TokenId(0));
            match self.sequences.get_mut(seq_id).unwrap().block_table.append_token(&mut self.pool) {
                Ok(_) => {
                    let block_id = self.sequences[seq_id].block_table.last_block().unwrap();
                    self.eviction.on_access(&block_id);
                }
                Err(PoolFull) => {
                    self.handle_pool_full();
                    if self.running.contains(seq_id) {
                        self.sequences.get_mut(seq_id).unwrap().block_table.append_token(&mut self.pool).unwrap();
                        let block_id = self.sequences[seq_id].block_table.last_block().unwrap();
                        self.eviction.on_access(&block_id);
                    }
                }
            }
        }
    }
```

> **Why no copy-on-write is needed.** Shared blocks are always *full*, because the prefix cache only shares complete blocks. Decode writes into the slot after `num_tokens`. If the last block is full, `append_token` allocates a fresh private block first. So decode never writes into a shared block, and the plain `step()` is safe. (The crate also offers `step_cow()`, which the engine does not need.)

### 10.2 Trace: step 1's decode

`Scheduler::step` reserves one slot each: A goes from 5 to 6 slots (5 % 4 = 1, so B4 still has room), B from 3 to 4, C from 5 to 6. No new blocks are needed.

| Row | Request | Token fed in | `position_offset` | Write slot | `read_blocks` | `read_num_tokens` |
|-----|---------|--------------|------------------:|------------|---------------|------------------:|
| 0 | A (`S0`) | `a1` | 5 | `(B4, 1)` | `[B5, B4]` | 6 |
| 1 | B (`S1`) | `b1` | 3 | `(B3, 3)` | `[B3]` | 4 |
| 2 | C (`S2`) | `c1` | 5 | `(B2, 1)` | `[B5, B2]` | 6 |

`StepCtx` for this batch:

- **Gather plan:** `max_blocks` = 2, so B's one-block table is padded with its own last block: `block_idx` = `[5, 4,  3, 3,  5, 2]`. `ctx_len` = 6 (the longest real context). Block B5 appears twice: A and C read the same physical memory in the same call.
- **Mask** (one query row per request):

  ```
                         position →   0    1    2    3    4    5
  A  (offset 5, valid 6)            [ 0    0    0    0    0    0 ]
  B  (offset 3, valid 4)            [ 0    0    0    0   -∞   -∞ ]   ← padding
  C  (offset 5, valid 6)            [ 0    0    0    0    0    0 ]
  ```

  For B, gathered positions 4–5 actually hold B3's slots 0–1 again (B's own tokens, read a second time because of the padding). The `-∞` gives them zero weight, so the padded data cannot change the result.

The shapes inside `Layer::forward` for this decode step:

| Tensor | Shape |
|--------|-------|
| `x` | `[batch 3, tokens 1, 960]` |
| `q` | `[3, 15 heads, 1, 64]` |
| `k`, `v` (new) | `[3, 5 heads, 1, 64]` |
| `k_full`, `v_full` | `[3, 5 heads, context 6, 64]` |
| attention scores | `[3, 5 KV heads, 3 group × 1, 6]` |
| logits | `[3, 1, 49152]` |

The forward pass samples `a2`, `b2` and `c2`. `emit` raises each `num_generated` to 2. Step 1 is over and has produced six events.

### 10.3 Trace: step 2, the final step

1. `admit_waiting`: the queue is empty, so there is nothing to do.
2. `decode`: positions are A 6, B 4, C 6. `Scheduler::step` reserves slots. **B has used all 4 slots of B3** (4 % 4 = 0), so it gets a new block. The free list is `[B0, B1]`, and `alloc` takes from the back: **B1**.

| Row | Request | Token fed in | `position_offset` | Write slot | `read_blocks` | `read_num_tokens` |
|-----|---------|--------------|------------------:|------------|---------------|------------------:|
| 0 | A | `a2` | 6 | `(B4, 2)` | `[B5, B4]` | 7 |
| 1 | B | `b2` | 4 | `(B1, 0)` | `[B3, B1]` | 5 |
| 2 | C | `c2` | 6 | `(B2, 2)` | `[B5, B2]` | 7 |

The gather plan is `[5, 4,  3, 1,  5, 2]` with `ctx_len` 7. B's mask hides positions 5 and 6. The model samples `a3`, `b3`, `c3`, and `emit` sees `num_generated` = 3 = `max_tokens` for all three: **`FinishReason::Length`**.

The KV cache at the end of this forward pass (positions 5 and 6 hold the fed-back tokens `a1`, `a2`, and so on):

```
         slot 0   slot 1   slot 2   slot 3
 B0    [   -    |   -    |   -    |   -    ]   free
 B1    [  B:4   |   -    |   -    |   -    ]   B
 B2    [  C:4   |  C:5   |  C:6   |   -    ]   C
 B3    [  B:0   |  B:1   |  B:2   |  B:3   ]   B
 B4    [  A:4   |  A:5   |  A:6   |   -    ]   A
 B5    [ A/C:0  | A/C:1  | A/C:2  | A/C:3  ]   A and C
```

The last tokens (`a3`, `b3`, `c3`) are sampled but never fed back, so their K/V is never computed. That is why A needs only 5 + 2 = 7 slots, not 8.

### 10.4 Releasing the blocks

Each finishing `emit` calls:

*(crate)* `sequence.rs` — `Scheduler::finish_sequence`
```rust
    pub fn finish_sequence(&mut self, seq_id: SequenceId) {
        let blocks: Vec<_> = self.sequences[&seq_id].block_table.blocks().to_vec();
        for block_id in &blocks {
            self.pool.decref(*block_id);
            self.eviction.on_free(block_id);
            if self.pool.is_free(*block_id) {
                self.prefix_cache.evict(*block_id);
            }
        }
        self.sequences.get_mut(&seq_id).unwrap().block_table.clear();
        self.sequences.get_mut(&seq_id).unwrap().state = SequenceState::Finished;
        self.running.retain(|id| id != &seq_id);
    }
```

Every block's reference count goes down by one. A block that reaches zero returns to the free list and is **also removed from the prefix cache**: its data can be overwritten now, so it must not be matched again. Watch the shared block B5:

| Moment | B5 | B4 | B3 | B2 | B1 | Free blocks |
|--------|---:|---:|---:|---:|---:|------------:|
| After all three prefills | 2 | 1 | 1 | 1 | — | 2 |
| Step 2: B allocates B1 | 2 | 1 | 1 | 1 | 1 | 1 |
| A finishes | **1** (still used by C, stays in the trie) | free | 1 | 1 | 1 | 2 |
| B finishes | 1 | free | free | 1 | free | 4 |
| C finishes | **free** (removed from the trie) | free | free | free | free | 6 |

### 10.5 The whole run at a glance

| Engine step | Phase | A | B | C | Free blocks after |
|-------------|-------|---|---|---|------------------:|
| 1 | admit | prefill 5 tokens → `a1` | prefill 3 tokens → `b1` | prefill **1** token (4 from cache) → `c1` | 2 |
| 1 | decode | `a1` → `a2` | `b1` → `b2` | `c1` → `c2` | 2 |
| 2 | decode | `a2` → `a3` ✔ | `b2` → `b3` ✔ (new block B1) | `c2` → `c3` ✔ | 6 |

Two things to notice:

1. In the step where a request is admitted, it gets **two** tokens: one from its prefill and one from that step's decode.
2. Three requests made progress together in each decode step. That is **continuous batching**: had a fourth request arrived at step 2, it would have been prefilled at the start of step 2 and joined the same decode call.

---

## Part 11 — Printing the results

Back in `one_shot`, `engine::run` has returned `result.generated` (one token list per prompt) and `result.stats`.

[src/main.rs:214](../src/main.rs#L214)
```rust
    let stops = stop_tokens(tokenizer);
    for (i, generated) in result.generated.iter().enumerate() {
        // A request that ended on a stop token has it as its last entry.
        let ids: Vec<u32> = match generated.split_last() {
            Some((last, rest)) if stops.contains(last) => rest.to_vec(),
            _ => generated.clone(),
        };
        let text = tokenizer.decode(&ids)?;
        if args.debug {
            println!("[{i}] prompt: {:?}", args.prompt[i]);
            println!("[{i}] completion: {text}");
        } else {
            println!("{text}");
        }
    }

    if args.debug {
        let stats = &result.stats;
        println!(
            "prefix cache: {} / {} tokens hit ({:.1}%), {} total block allocations",
            stats.prefix_hits,
            stats.prefix_total,
            if stats.prefix_total == 0 {
                0.0
            } else {
                stats.prefix_hits as f64 / stats.prefix_total as f64 * 100.0
            },
            stats.total_block_allocs,
        );
        println!(
            "{} steps, {} output tokens in {:.2?} ({:.1} tok/s)",
            stats.steps,
            stats.output_tokens,
            stats.elapsed,
            stats.output_tokens as f64 / stats.elapsed.as_secs_f64(),
        );
    }
    Ok(())
}
```

1. If the last generated token is a stop token, it is dropped (`split_last` splits off the last element).
2. The ids are decoded to text in one go. Special tokens are skipped.
3. With `--debug`, each prompt is printed with `{:?}` (so it appears in quotes), followed by its completion and then the statistics.

Putting Parts 4–10 together, the output has this shape (`‹…›` marks what depends on the model's actual predictions or on timing):

```
[0] prompt: "The capital of France is"
[0] completion: ‹text of a1 a2 a3›
[1] prompt: "The sky is"
[1] completion: ‹text of b1 b2 b3›
[2] prompt: "The capital of France is"
[2] completion: ‹text of c1 c2 c3›
prefix cache: 4 / 13 tokens hit (30.8%), 5 total block allocations
2 steps, 9 output tokens in ‹time› (‹rate› tok/s)
```

Every statistic can be traced back to an earlier part:

| Statistic | Value | Where it comes from |
|-----------|------:|---------------------|
| `prefix_total` | 13 | Prompt tokens counted by `Scheduler::prefill`: 5 + 3 + 5 (§6.3) |
| `prefix_hits` | 4 | C reused one full 4-token block, B5 (§6.4) |
| Hit rate | 30.8% | 4 / 13 |
| `total_block_allocs` | 5 | B5, B4 (A) + B3 (B) + B2 (C) + B1 (B, during step 2) |
| `steps` | 2 | `step_idx` when the loop ended (Part 4) |
| `output_tokens` | 9 | 3 requests × 3 tokens |

Because sampling is greedy and C's context is identical to A's, C's completion will normally be the same text as A's. (Tiny floating-point differences between computing 5 positions and computing 1 position could, in principle, flip a near-tie.)

---

## Part 12 — When memory runs out: preemption

Our example had enough blocks. What happens when the pool runs dry in the middle of decoding? The engine test `preemption_still_completes_every_request` sets up exactly that situation, and we trace it here.

[src/engine.rs:748](../src/engine.rs#L748)
```rust
    #[test]
    fn preemption_still_completes_every_request() {
        // 6 blocks of 4 slots can't hold three 10-token sequences at once, so
        // some get evicted and re-prefilled; all must still finish with the
        // right tokens.
        let mut engine = Engine::new(CountingBackend::new(64), cfg(6, 8));
        let ids: Vec<_> = (0..3)
            .map(|i| engine.add_request(req(vec![10 * (i + 1)], 9)).unwrap())
            .collect();
        let events = drain(&mut engine);
        for (i, id) in ids.iter().enumerate() {
            let start = 10 * (i as u32 + 1) + 1;
            assert_eq!(
                tokens_of(&events, *id),
                (start..start + 9).collect::<Vec<_>>()
            );
        }
        assert_eq!(engine.scheduler().pool.available(), 6);
    }
```

The setup:

- `cfg(6, 8)`: 6 blocks of 4 slots, at most 8 running.
- Three requests with one-token prompts `[10]`, `[20]` and `[30]`, each with `max_tokens` = 9.
- `CountingBackend` is a fake model whose "next token" is always the last input token + 1. So request `[10]` should produce 11, 12, …, 19.

Each request needs 1 + 8 = 9 slots (the last generated token is never stored), which is 3 blocks. Three requests need 9 blocks, but only 6 exist.

### 12.1 Timeline

| Step | S0 (prompt `[10]`) | S1 (`[20]`) | S2 (`[30]`) | Free blocks after |
|------|--------------------|-------------|-------------|------------------:|
| 1 | prefill → 11, decode → 12 (block B5) | 21, 22 (B4) | 31, 32 (B3) | 3 |
| 2–3 | 13, 14 | 23, 24 | 33, 34 | 3 |
| 4 | 2nd block (B2) → 15 | 2nd block (B1) → 25 | 2nd block (B0) → 35 | 0 |
| 5–7 | 16, 17, 18 | 26, 27, 28 | 36, 37, 38 | 0 |
| 8 | needs a 3rd block, pool empty → **preempted** | 3rd block → 29 ✔ | 3rd block → 39 ✔ | 6 |
| 9 | re-admitted: prefill `[10, 11, …, 18]` → **19** ✔ | — | — | 6 |

### 12.2 Step 8 in slow motion

1. `decode` records the positions (8 for everyone) and calls `Scheduler::step`.
2. The scheduler handles S0 first. It pushes a placeholder token, then `append_token` needs a new block (8 % 4 = 0), but the pool is empty: `PoolFull`.
3. `handle_pool_full` asks the LRU policy for a victim:

   *(crate)* `eviction.rs` and `sequence.rs`
   ```rust
   impl EvictionPolicy for LRUEviction {
       fn select_victim(&mut self, pool: &BlockPool) -> Option<BlockID> {
           let block_ids = self.blocks_map.keys();
           for block in block_ids {
               let ref_count = pool.get_ref_count(*block);
               if ref_count == 1 {
                   return Some(*block);
               }
           }
           None
       }
   }

       pub fn handle_pool_full(&mut self) {
           let victim_id = self.eviction.select_victim(&self.pool)
               .ok_or(SchedulerError::NothingEvictable)
               .expect("pool full and no blocks eligible for eviction");
           let victim_seq_id = self.sequences
               .iter()
               .find(|(_, seq)| seq.block_table.blocks().contains(&victim_id))
               .map(|(seq_id, _)| *seq_id)
               .unwrap();
           let blocks: Vec<_> = self.sequences[&victim_seq_id].block_table.blocks().to_vec();
           for block_id in &blocks {
               self.pool.decref(*block_id);
               self.eviction.on_free(block_id);
               self.prefix_cache.evict(*block_id);
           }
           self.metrics.blocks_evicted += 1;
           self.sequences.get_mut(&victim_seq_id).unwrap().state = SequenceState::Waiting;
           self.sequences.get_mut(&victim_seq_id).unwrap().block_table.clear();
           self.waiting.push_back(victim_seq_id);
           self.running.retain(|id| id != &victim_seq_id);
       }
   ```

   `on_access` (called whenever a sequence appends into a block) removes the block from the ordered map and inserts it again, which puts it at the **end**. `keys()` starts from the **oldest** entry, so `select_victim` returns the least recently used block that only one sequence uses. Here that is **B5**, S0's first block: it was last appended to in step 3. Every newer block was touched in steps 4–7.
4. The victim is the **sequence** that owns B5, S0, which is the very sequence asking for a block. All of S0's blocks (B5 and B2) are released. S0 goes to the back of `waiting`, and the retry for S0 is skipped because it is no longer running.
5. S1 and S2 then get their third blocks from the two just freed.
6. Back in `decode`, step ③ repairs S0's token list:

   | Moment | `token_ids` of S0 | Length |
   |--------|-------------------|-------:|
   | Start of step 8 | `[10, 11, …, 17]` | 8 = prompt + 7 generated |
   | `Scheduler::step` pushes a placeholder | `[10, 11, …, 17, 0]` | 9 |
   | Preempted (blocks freed, table cleared) | unchanged | 9 |
   | `truncate(real_len − 1)` with `real_len` = 1 + 8 | `[10, 11, …, 17]` | 8 |
   | `push(last_token)` | `[10, 11, …, 18]` | 9 = prompt + all 8 generated |

   The placeholder may or may not be there, depending on whether the sequence was evicted before or after its own turn in the scheduler's loop. Truncating to `real_len − 1` and pushing `last_token` gives the right list either way.
7. Only S1 and S2 go into the forward pass. They produce 29 and 39 and finish, freeing all 6 blocks.

### 12.3 Step 9: re-admission

`admit_waiting` finds S0 at the front of the queue. It needs ⌈9 / 4⌉ = 3 blocks, and 6 are free. S0 is **re-prefilled from scratch** with its 9 tokens `[10, 11, …, 18]`, and the prefill's sampled token is 19. Because `num_generated` is already 8, this is not the request's first token, so no TTFT is recorded. `emit` counts it as the 9th token, and S0 finishes.

The client sees an unbroken sequence 11 … 19, exactly what the test checks. The price of preemption is **recomputation**: 9 tokens of prefill were done twice. In exchange, the engine never deadlocks and never needs extra memory to swap K/V out.

---

## Part 13 — Variation: the interactive CLI

Run the CLI without `--prompt` and `run_cli` calls `interactive` instead of `one_shot`. The engine is the same; the differences are in how it is driven and how text comes out.

### 13.1 `interactive`

[src/main.rs:254](../src/main.rs#L254)
```rust
fn interactive(
    args: &CliArgs,
    runtime: &Runtime,
    tokenizer: &SmollLM230MTokenizer,
    sampling: SamplingParams,
) -> Result<(), BoxError> {
    let context_length = runtime.model().config.context_length as usize;
    let block_size = args.model.block_size;
    let max_tokens = args.max_tokens.unwrap_or(256);
    let num_blocks = args                                                          // ①
        .model
        .num_blocks
        .unwrap_or_else(|| blocks_for(context_length.min(4096), block_size));
    let mut engine = Engine::new(                                                  // ②
        runtime.backend(block_size, num_blocks)?,
        EngineConfig {
            block_size,
            num_blocks,
            max_running: 1,
        },
    );
    let stops = stop_tokens(tokenizer);

    eprintln!("Type a question and press Enter. Ctrl-D or `exit` to quit.");
    let mut history: Vec<ChatMessage> = Vec::new();
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    loop {
        eprint!("> ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if matches!(line, "exit" | "quit") {
            break;
        }

        let prompt = if args.chat {                                                // ③
            history.push(ChatMessage {
                role: "user".into(),
                content: line.to_string(),
            });
            chat::chatml(&history)
        } else {
            line.to_string()
        };
        let request = Request {                                                    // ④
            tokens: tokenizer.encode(&prompt)?,
            max_tokens,
            stop_tokens: stops.clone(),
            sampler: Sampler::from_params(sampling),
        };
        let id = match engine.add_request(request) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("error: {e}");
                if args.chat {
                    history.pop();
                }
                continue;
            }
        };

        // Stream the answer as it is generated.
        let mut decoder = StreamDecoder::new();                                    // ⑤
        let mut answer = String::new();
        'generate: while engine.has_work() {
            for ev in engine.step()? {
                debug_assert_eq!(ev.id, id);
                let stopped = ev.finish == Some(engine::FinishReason::Stop);
                let text = if stopped {
                    decoder.flush(tokenizer)?
                } else {
                    decoder.push(tokenizer, ev.token)?
                };
                answer.push_str(&text);
                write!(stdout, "{text}")?;
                stdout.flush()?;
                if ev.finish.is_some() {
                    break 'generate;
                }
            }
        }
        writeln!(stdout)?;
        if args.chat {                                                             // ⑥
            history.push(ChatMessage {
                role: "assistant".into(),
                content: answer,
            });
        }
    }
    Ok(())
}
```

1. **One conversation's worth of KV.** By default the pool holds `min(context_length, 4096)` tokens: 4096 / 16 = 256 blocks, about 320 MiB.
2. **The CLI drives the engine itself**, calling `add_request` and `step` directly instead of `engine::run`. `max_running: 1` because there is only ever one question in flight.
3. **Chat mode** appends the user's line to `history` and renders the *whole* conversation with ChatML (§13.3). Without `--chat`, the raw line is the prompt.
4. **Stop tokens** here are EOS *and* `<|im_end|>`. (`engine::run` in one-shot mode uses only EOS.)
5. **Streaming.** Each `step` returns the new token(s). They go through a `StreamDecoder` (§13.2) and are printed immediately, with `stdout.flush()` so they appear right away instead of waiting in a buffer. `break 'generate` leaves the outer loop as soon as the answer finishes. On a stop token, the decoder is flushed instead of fed the stop token, so `<|im_end|>` never appears in the output.
6. **History.** The finished answer is stored as an `assistant` message, so the next question sees the whole conversation.

### 13.2 `StreamDecoder`: printing text one token at a time

[src/tokenizer.rs:44](../src/tokenizer.rs#L44)
```rust
#[derive(Default)]
pub struct StreamDecoder {
    ids: Vec<u32>,
    // ids[prefix_offset..read_offset] is already-emitted context kept so the
    // next delta is decoded with the right spacing.
    prefix_offset: usize,
    read_offset: usize,
}

impl StreamDecoder {
    // …
    pub fn push(
        &mut self,
        tok: &SmollLM230MTokenizer,
        id: u32,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.ids.push(id);
        let prefix = tok.decode(&self.ids[self.prefix_offset..self.read_offset])?;
        let text = tok.decode(&self.ids[self.prefix_offset..])?;
        if text.len() > prefix.len() && !text.ends_with('\u{FFFD}') {
            let delta = text.get(prefix.len()..).unwrap_or_default().to_string();
            self.prefix_offset = self.read_offset;
            self.read_offset = self.ids.len();
            Ok(delta)
        } else {
            Ok(String::new())
        }
    }
    // … flush() returns whatever is still held back
}
```

Some characters are split across tokens. With this tokenizer, `é🙂` is three tokens (real ids):

| Token id | Token | Bytes |
|---------:|-------|-------|
| 2756 | `Ã©` | `C3 A9`, the whole of `é` |
| 10813 | `ðŁ` | `F0 9F`, the first half of `🙂` |
| 38887 | `ĻĤ` | `99 82`, the second half of `🙂` |

Feeding them through `push` one by one:

| Push | `prefix` (already shown) | `text` (with the new token) | Returned | Offsets after |
|------|--------------------------|-----------------------------|----------|---------------|
| 2756 | `""` | `"é"` | `"é"` | prefix 0, read 1 |
| 10813 | `"é"` | `"é�"`: ends in U+FFFD, an incomplete character | `""` (held back) | unchanged |
| 38887 | `"é"` | `"é🙂"` | `"🙂"` | prefix 1, read 3 |

Decoding one token at a time would have printed broken characters. The decoder instead waits until the text no longer ends in the replacement character `�`. It also always decodes a little already-printed context (`prefix_offset..`), so leading spaces come out right.

### 13.3 The chat template

[src/chat.rs:8](../src/chat.rs#L8)
```rust
const DEFAULT_SYSTEM: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

// Terminates each turn; also the token generation should stop on.
pub const END_OF_TURN: &str = "<|im_end|>";

pub fn chatml(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    if messages.first().is_none_or(|m| m.role != "system") {
        out.push_str(&format!(
            "<|im_start|>system\n{DEFAULT_SYSTEM}{END_OF_TURN}\n"
        ));
    }
    for m in messages {
        out.push_str(&format!(
            "<|im_start|>{}\n{}{END_OF_TURN}\n",
            m.role, m.content
        ));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}
```

For a single user message `Say hi`, this produces the text below, which tokenizes to **32 tokens** (real count):

```
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
Say hi<|im_end|>
<|im_start|>assistant
```

| Tokens | Text |
|--------|------|
| 0–20 (21 tokens) | `<\|im_start\|>system\n…Hugging Face<\|im_end\|>\n` |
| 21–23 (3) | `<\|im_start\|>user\n` |
| 24–25 (2) | `Say hi` |
| 26–27 (2) | `<\|im_end\|>\n` |
| 28–31 (4) | `<\|im_start\|>assistant\n` |

The prompt ends right where the assistant's reply should start, so the model's next tokens *are* the reply. It signals the end of the reply with `<|im_end|>` (id 2), which is a stop token.

> **Prefix caching does not carry over between turns.** Turn 2's prompt starts with the whole of turn 1, so it looks like a perfect prefix-cache hit. But turn 1's request finishes before turn 2 is added, and `finish_sequence` frees its blocks *and removes them from the trie* (§10.4). Each turn therefore prefills the entire conversation again. Blocks are shared only between requests that are **alive at the same time**.

---

## Part 14 — Variation: the HTTP server

We follow one streaming chat request:

```bash
cargo run --release -- serve \
  --model ../SmolLM2-360M-Instruct.Q8_0.gguf --tokenizer tokenizer.json --port 8000

curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "messages": [{"role": "user", "content": "Say hi"}],
  "max_tokens": 32, "stream": true
}'
```

The engine work is the same as before. What changes is everything *around* it: threads, channels, queueing, streaming, and cancellation.

```
 curl ──HTTP──► axum handler (tokio task)
                  │ start_generation(): validate → take a slot → send Job
                  ▼
           job channel (tokio mpsc, unbounded)
                  │
                  ▼
        engine thread: run_worker()  ── owns the Engine; calls step() in a loop
                  │ WorkerEvent::Token per token (one channel per request)
                  ▼
        per-request tokio task: StreamDecoder + StopFilter → Chunk
                  │ chunk channel
                  ▼
        sse(): Chunk → "data: {json}" events → curl
```

### 14.1 `run_serve`: two worlds, one process

[src/main.rs:354](../src/main.rs#L354)
```rust
fn run_serve(args: ServeArgs) -> Result<(), BoxError> {
    if args.max_running == 0 {
        return Err("--max-running must be at least 1".into());
    }
    let model = Model::load(&args.model.model)?;
    let tokenizer = load_tokenizer(&args.model.tokenizer, &model.config)?;
    let context_length = model.config.context_length as usize;
    let block_size = args.model.block_size;
    let num_blocks = args.model.num_blocks.unwrap_or_else(|| {                     // ①
        args.max_running * blocks_for(context_length.min(args.tokens_per_seq), block_size)
    });
    let engine_cfg = EngineConfig {
        block_size,
        num_blocks,
        max_running: args.max_running,
    };
    let model_name = args.model_name.clone().unwrap_or_else(|| {
        std::path::Path::new(&args.model.model)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "rvllm".into())
    });

    let (job_tx, job_rx) = tokio::sync::mpsc::unbounded_channel();                 // ②
    let status = Arc::new(WorkerStatus::default());
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<usize, String>>();

    std::thread::scope(|scope| -> Result<(), BoxError> {
        // The engine thread builds its own runtime (CUDA context included) so
        // device state never crosses threads.
        let worker_status = status.clone();
        let device = args.model.device.clone();
        let model = &model;
        scope.spawn(move || {                                                      // ③
            let runtime = match Runtime::new(model, &device) {
                Ok(r) => r,
                Err(e) => return drop(ready_tx.send(Err(e.to_string()))),
            };
            let backend = match runtime.backend(block_size, num_blocks) {
                Ok(b) => b,
                Err(e) => return drop(ready_tx.send(Err(e.to_string()))),
            };
            let engine = Engine::new(backend, engine_cfg);
            let _ = ready_tx.send(Ok(engine.max_sequence_tokens()));
            run_worker(engine, job_rx, worker_status);
        });
        let max_sequence_tokens = ready_rx                                         // ④
            .recv()
            .map_err(|_| "engine thread died during startup")??;

        let state = AppState::new(                                                 // ⑤
            ServerConfig {
                model_name: model_name.clone(),
                max_running: args.max_running,
                max_queue: args.max_queue,
                default_max_tokens: args.default_max_tokens,
                default_temperature: args.default_temperature,
                default_top_p: args.default_top_p,
                context_length,
                max_sequence_tokens,
            },
            tokenizer,
            job_tx,
            status,
        );

        let rt = tokio::runtime::Builder::new_multi_thread()                       // ⑥
            .enable_all()
            .build()?;
        let addr = format!("{}:{}", args.host, args.port);
        let result = rt.block_on(async {
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            eprintln!(
                "serving {model_name:?} on http://{addr} ({} on {}; {} running slots, queue of {}; KV pool {} blocks x {} tokens)",
                args.model.model, args.model.device, args.max_running, args.max_queue, num_blocks, block_size,
            );
            server::serve(listener, state).await
        });
        // Drops any request tasks still holding the job channel, so the
        // engine thread sees it close and exits.
        rt.shutdown_background();                                                  // ⑦
        result?;
        Ok(())
    })
}
```

1. **KV pool size** with the defaults: 8 running slots × ⌈min(8192, 2048) / 16⌉ = 8 × 128 = **1,024 blocks** = 16,384 token slots × 80 KiB ≈ **1.25 GiB** of F32 KV cache.
2. **Channels.** `job_tx`/`job_rx` carries new requests from HTTP handlers to the engine thread. `ready` lets the engine thread report that startup worked.
3. **The engine thread** builds its own `Runtime` and backend (so a CUDA context is created and used on one thread only), then runs `run_worker` until the job channel closes. `std::thread::scope` lets this thread borrow `model` from the enclosing function.
4. **Wait for "ready".** The double `??` unwraps two layers: the channel `recv` could fail (the thread died), and the thread could have sent an error string (for example "CUDA init failed").
5. **Shared state for the handlers** (§14.2). With the default settings, `max_sequence_tokens` = min(1,024 × 16, 8192) = 8192.
6. **The async world.** A multi-threaded tokio runtime accepts connections and runs axum. With our command, the banner reads: `serving "SmolLM2-360M-Instruct.Q8_0" on http://127.0.0.1:8000 (../SmolLM2-360M-Instruct.Q8_0.gguf on cpu; 8 running slots, queue of 64; KV pool 1024 blocks x 16 tokens)`.
7. **Shutdown.** After Ctrl-C, shutting the runtime down drops the last job senders. The worker sees the channel close and returns, and the scope then joins its thread.

### 14.2 `AppState` and the routes

[src/server/mod.rs:67](../src/server/mod.rs#L67)
```rust
    pub fn new(
        cfg: ServerConfig,
        tokenizer: SmollLM230MTokenizer,
        jobs: UnboundedSender<Job>,
        status: Arc<WorkerStatus>,
    ) -> Arc<Self> {
        let slots = Arc::new(Semaphore::new(cfg.max_running + cfg.max_queue));
        let mut stop_tokens = vec![tokenizer.eos_token_id];
        if let Some(id) = tokenizer.token_to_id(END_OF_TURN) {
            stop_tokens.push(id);
        }
        Arc::new(AppState {
            cfg,
            tokenizer,
            jobs,
            slots,
            status,
            stop_tokens,
        })
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .with_state(state)
}
```

- **`slots`** is a semaphore with 8 + 64 = **72 permits**: one per request allowed in the system (running or queued). It is the backpressure mechanism.
- The tokenizer lives here, on the async side. Tokenizing and detokenizing happen in the handlers and tasks, not on the engine thread.
- `Arc` ("atomically reference-counted") lets every handler share one `AppState`.

### 14.3 The handler: `chat_completions`

[src/server/mod.rs:447](../src/server/mod.rs#L447)
```rust
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    if body.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    let messages: Vec<chat::ChatMessage> = body
        .messages
        .into_iter()
        .map(|m| chat::ChatMessage {
            role: m.role,
            content: m.content.map(ChatContent::into_text).unwrap_or_default(),
        })
        .collect();
    let rx = start_generation(
        &state,
        GenRequest {
            prompt: chat::chatml(&messages),
            max_tokens: body.max_completion_tokens.or(body.max_tokens),
            temperature: body.temperature,
            top_p: body.top_p,
            seed: body.seed,
            stop: body.stop.map(StopField::into_vec).unwrap_or_default(),
            n: body.n,
        },
    )?;
    let id = new_id("chatcmpl");
    let created = now();
    let model = state.cfg.model_name.clone();

    if body.stream {
        let include_usage = body.stream_options.is_some_and(|o| o.include_usage);
        let chunk_of = {
            let (id, model) = (id.clone(), model.clone());
            move |delta: Value, finish: Option<&str>, usage: Option<Usage>| {
                let mut v = json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": if usage.is_some() { json!([]) } else {
                        json!([{"index": 0, "delta": delta, "finish_reason": finish}])
                    },
                });
                if let Some(u) = usage {
                    v["usage"] = serde_json::to_value(u).unwrap();
                }
                v
            }
        };
        let first = chunk_of(json!({"role": "assistant", "content": ""}), None, None);
        return Ok(sse(
            rx,
            include_usage,
            move |text, finish, usage| match text {
                Some(t) => chunk_of(json!({"content": t}), None, None),
                None => chunk_of(json!({}), finish, usage),
            },
            Some(first),
        ));
    }

    let (text, finish_reason, usage) = collect(rx).await?;
    // … (build one "chat.completion" JSON object)
}
```

1. axum's `Json(body)` extractor parses the request body into `ChatRequest` (from `server/api.rs`). Bad JSON is rejected before this function runs.
2. The OpenAI message list becomes the engine's `ChatMessage` list and then a ChatML prompt (§13.3): the same 32 tokens as before.
3. `start_generation` (next section) does the real work and returns a channel of text chunks.
4. With `"stream": true`, the response is a stream of Server-Sent Events built by `sse` (§14.6). `chunk_of` is a closure that formats one OpenAI "chat.completion.chunk" object. Otherwise `collect` waits for the whole answer and returns a single JSON object.

### 14.4 `start_generation`: validate, queue, translate

[src/server/mod.rs:208](../src/server/mod.rs#L208)
```rust
fn start_generation(
    state: &Arc<AppState>,
    req: GenRequest,
) -> Result<UnboundedReceiver<Chunk>, ApiError> {
    if req.n.is_some_and(|n| n != 1) {                                             // ①
        return Err(ApiError::bad_request("only n=1 is supported"));
    }
    let temperature = req.temperature.unwrap_or(state.cfg.default_temperature);
    let top_p = req.top_p.unwrap_or(state.cfg.default_top_p);
    if !(0.0..=2.0).contains(&temperature) {
        return Err(ApiError::bad_request("temperature must be between 0 and 2"));
    }
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(ApiError::bad_request("top_p must be in (0, 1]"));
    }
    let max_tokens = req.max_tokens.unwrap_or(state.cfg.default_max_tokens);
    if max_tokens == 0 {
        return Err(ApiError::bad_request("max_tokens must be at least 1"));
    }

    let tokens = state                                                             // ②
        .tokenizer
        .encode(&req.prompt)
        .map_err(|e| ApiError::internal(format!("tokenization failed: {e}")))?;
    if tokens.is_empty() {
        return Err(ApiError::bad_request("prompt is empty"));
    }
    let prompt_tokens = tokens.len();
    if prompt_tokens >= state.cfg.context_length {
        return Err(ApiError::bad_request(format!(
            "prompt is {prompt_tokens} tokens, which does not fit the model's {}-token context",
            state.cfg.context_length
        )));
    }
    if prompt_tokens >= state.cfg.max_sequence_tokens {
        return Err(ApiError::bad_request(format!(
            "prompt is {prompt_tokens} tokens but the server's KV cache holds at most {} per request",
            state.cfg.max_sequence_tokens
        )));
    }
    // Generation is capped by whatever room the context and KV pool leave.
    let max_tokens = max_tokens.min(state.cfg.max_sequence_tokens - prompt_tokens);

    let permit = state                                                             // ③
        .slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::overloaded())?;

    let (event_tx, mut event_rx) = mpsc::unbounded_channel();                     // ④
    state
        .jobs
        .send(Job {
            tokens,
            max_tokens,
            stop_tokens: state.stop_tokens.clone(),
            sampling: SamplingParams {
                temperature,
                top_p,
                seed: req.seed.unwrap_or_else(|| rand::rng().random()),
            },
            events: event_tx,
            permit,
        })
        .map_err(|_| ApiError::internal("inference worker is not running"))?;

    let (chunk_tx, chunk_rx) = mpsc::unbounded_channel();                         // ⑤
    let state = state.clone();
    let mut stop_filter = StopFilter::new(req.stop);
    tokio::spawn(async move {
        let mut decoder = StreamDecoder::new();
        let mut completion_tokens = 0usize;
        let done = |finish_reason, completion_tokens| Chunk::Done {
            finish_reason,
            usage: Usage::new(prompt_tokens, completion_tokens),
        };
        while let Some(ev) = event_rx.recv().await {
            let (token, finish) = match ev {
                WorkerEvent::Token { token, finish } => (token, finish),
                WorkerEvent::Error(message) => {
                    let _ = chunk_tx.send(Chunk::Error(message));
                    return;
                }
            };
            completion_tokens += 1;
            // The stop token ends the turn; it isn't part of the text.
            let hit_stop_token = finish == Some(FinishReason::Stop);
            let text = if hit_stop_token {
                decoder.flush(&state.tokenizer).unwrap_or_default()
            } else {
                match decoder.push(&state.tokenizer, token) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = chunk_tx.send(Chunk::Error(format!("detokenization failed: {e}")));
                        return;
                    }
                }
            };
            let (mut out, hit_stop_string) = stop_filter.push(&text);              // ⑥
            if finish.is_some() && !hit_stop_string {
                out.push_str(&stop_filter.flush());
            }
            if !out.is_empty() && chunk_tx.send(Chunk::Text(out)).is_err() {
                return;
            }
            if hit_stop_string {
                let _ = chunk_tx.send(done("stop", completion_tokens));
                return; // drops event_rx -> worker aborts the request
            }
            if let Some(f) = finish {
                let _ = chunk_tx.send(done(finish_str(f), completion_tokens));
                return;
            }
        }
        let _ = chunk_tx.send(Chunk::Error("engine stopped before finishing".into()));
    });
    Ok(chunk_rx)
}
```

For our request:

1. **Validation and defaults.** `n` is absent: fine. `temperature` is absent, so the server default **1.0** applies (random sampling, unlike the CLI's 0.0). `top_p` = 1.0, `max_tokens` = 32.
2. **Tokenize and check the length.** The prompt is 32 tokens, which fits both the context (8192) and `max_sequence_tokens` (8192). `max_tokens` = min(32, 8192 − 32) = 32. Everything that could make the engine reject the request is checked here, so the request fails fast with HTTP 400 instead of failing later on the engine thread.
3. **Take a slot.** `try_acquire_owned` takes one of the 72 permits *without waiting*. If none is left, the client gets **HTTP 429** with `Retry-After: 1` (`ApiError::overloaded`). The permit travels inside the `Job` and is released when the request ends.
4. **Hand the job to the engine thread**, together with the sending half of a new per-request event channel. No seed was given, so a random one is chosen.
5. **Spawn a translator task** that turns token events into text chunks: it counts tokens for `usage`, runs them through a `StreamDecoder`, and flushes the decoder instead of decoding the stop token.
6. **Stop strings.** `StopFilter` looks for user-supplied stop strings (none in our request) in the decoded text, holding back text that might be the start of one. If one is found, the task sends `Done` with `"stop"` and returns. Returning drops `event_rx`, and that is how the engine learns to abort the request (§14.7).

### 14.5 `run_worker`: the engine thread's loop

[src/server/worker.rs:50](../src/server/worker.rs#L50)
```rust
pub fn run_worker<B: Backend>(
    mut engine: Engine<B>,
    mut jobs: UnboundedReceiver<Job>,
    status: Arc<WorkerStatus>,
) {
    let mut live: HashMap<SequenceId, Live> = HashMap::new();

    let admit = |engine: &mut Engine<B>, live: &mut HashMap<SequenceId, Live>, job: Job| {
        let request = Request {
            tokens: job.tokens,
            max_tokens: job.max_tokens,
            stop_tokens: job.stop_tokens,
            sampler: Sampler::from_params(job.sampling),
        };
        match engine.add_request(request) {
            Ok(id) => {
                live.insert(
                    id,
                    Live {
                        events: job.events,
                        permit: job.permit,
                    },
                );
            }
            // The HTTP layer pre-validates, so this is a safety net.
            Err(e) => {
                let _ = job.events.send(WorkerEvent::Error(e.to_string()));
            }
        }
    };

    loop {
        if !engine.has_work() {                                                    // ①
            match jobs.blocking_recv() {
                Some(job) => admit(&mut engine, &mut live, job),
                None => break,
            }
        }
        while let Ok(job) = jobs.try_recv() {                                      // ②
            admit(&mut engine, &mut live, job);
        }

        // Clients that went away (dropped their receiver) stop costing
        // anything: free their KV blocks and their queue slot now.
        let gone: Vec<SequenceId> = live                                           // ③
            .iter()
            .filter(|(_, l)| l.events.is_closed())
            .map(|(&id, _)| id)
            .collect();
        for id in gone {
            engine.abort(id);
            live.remove(&id);
        }

        match engine.step() {                                                      // ④
            Ok(events) => {
                for ev in events {
                    if ev.finish.is_some() {
                        // Release the queue slot *before* the final event goes
                        // out: once the client has its response, an immediate
                        // follow-up request must not see the slot still taken.
                        let Some(Live { events, permit }) = live.remove(&ev.id) else {
                            continue;
                        };
                        drop(permit);
                        let _ = events.send(WorkerEvent::Token {
                            token: ev.token,
                            finish: ev.finish,
                        });
                        continue;
                    }
                    let Some(l) = live.get(&ev.id) else { continue };
                    let delivered = l
                        .events
                        .send(WorkerEvent::Token {
                            token: ev.token,
                            finish: None,
                        })
                        .is_ok();
                    if !delivered {
                        engine.abort(ev.id);
                        live.remove(&ev.id);
                    }
                }
            }
            Err(e) => {                                                            // ⑤
                // A failed forward pass leaves the KV pool in an unknown state,
                // so fail everything in flight rather than limp on.
                eprintln!("engine step failed: {e}");
                for (id, l) in live.drain() {
                    let _ = l
                        .events
                        .send(WorkerEvent::Error(format!("engine error: {e}")));
                    engine.abort(id);
                }
            }
        }
        status                                                                     // ⑥
            .running
            .store(engine.num_running(), Ordering::Relaxed);
        status
            .waiting
            .store(engine.num_waiting(), Ordering::Relaxed);
    }
}
```

1. **Idle?** Sleep on the channel (`blocking_recv`) until a job arrives, so an idle server uses no CPU. If every sender is gone, the server is shutting down and the loop ends.
2. **Busy?** Take every job that has arrived since the last step, without waiting. This is continuous batching in the server: a request that arrives mid-generation joins the very next step.
3. **Drop requests whose client went away.** `is_closed()` is true once the receiving end of the event channel has been dropped.
4. **One engine step**, exactly as in Parts 6–10. Every token is forwarded to its request's channel. On the final token, the permit is dropped *before* the event is sent, so a client that immediately sends another request finds its slot free.
5. **If a forward pass fails**, the KV cache may be in an unknown state, so every live request gets an error and is aborted.
6. **Publish the queue lengths** for `GET /health` through two atomic counters. No lock is needed.

The engine is the same `Engine` the CLI uses. The only difference is `max_running` = 8, so at most 8 requests decode together and the rest wait in the engine's FIFO queue.

### 14.6 `sse`: chunks become Server-Sent Events

[src/server/mod.rs:345](../src/server/mod.rs#L345)
```rust
fn sse(
    rx: UnboundedReceiver<Chunk>,
    include_usage: bool,
    mut render: impl FnMut(Option<&str>, Option<&str>, Option<Usage>) -> Value + Send + 'static,
    first: Option<Value>,
) -> Response {
    let mut first = first;
    let stream = UnboundedReceiverStream::new(rx).flat_map(move |chunk| {
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();
        if let Some(f) = first.take() {
            events.push(Ok(Event::default().data(f.to_string())));
        }
        match chunk {
            Chunk::Text(t) => events.push(Ok(
                Event::default().data(render(Some(&t), None, None).to_string())
            )),
            Chunk::Done {
                finish_reason,
                usage,
            } => {
                events.push(Ok(
                    Event::default().data(render(None, Some(finish_reason), None).to_string())
                ));
                if include_usage {
                    events.push(Ok(
                        Event::default().data(render(None, None, Some(usage)).to_string())
                    ));
                }
                events.push(Ok(Event::default().data("[DONE]")));
            }
            Chunk::Error(message) => {
                let body = json!({"error": {"message": message, "type": "server_error"}});
                events.push(Ok(Event::default().data(body.to_string())));
                events.push(Ok(Event::default().data("[DONE]")));
            }
        }
        tokio_stream::iter(events)
    });
    Sse::new(stream).into_response()
}
```

Every chunk from the translator task becomes one or more `data: …` lines. The "role" chunk (`first`) goes out together with the first real chunk. For our request, if the model answered "Hello!" (an illustration: the words and the split into chunks depend on the model), curl would receive:

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"SmolLM2-360M-Instruct.Q8_0","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {…,"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {…,"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {…,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

`finish_reason` is `"stop"` because generation ended on `<|im_end|>`. Had it run into the 32-token limit, it would be `"length"`. (The order of the JSON keys in the real output may differ.)

### 14.7 Cancellation: what happens when the client disconnects

Nothing ever calls a "cancel" function directly. Cancellation travels by **dropping channel ends**:

1. curl is closed, so axum drops the response stream, and with it the chunk receiver `chunk_rx`.
2. The translator task's next `chunk_tx.send(...)` fails, so it returns, which drops `event_rx`.
3. On the engine thread, `l.events.is_closed()` becomes true (or the next `send` fails), so `run_worker` calls `engine.abort(id)`.
4. `abort` removes the request from the queue or calls `finish_sequence`, freeing its KV blocks. Dropping the `Live` entry releases its semaphore permit.

[src/engine.rs:290](../src/engine.rs#L290)
```rust
    pub fn abort(&mut self, id: SequenceId) {
        if self.seqs.remove(&id).is_none() {
            return;
        }
        self.scheduler.waiting.retain(|&w| w != id);
        if self.scheduler.running.contains(&id) {
            self.scheduler.finish_sequence(id);
        }
        self.scheduler.sequences.remove(&id);
    }
```

### 14.8 Two clients at once

If a second client sends a different question while the first is still generating:

1. Its handler runs on another tokio task at the same moment. It takes a second permit and sends a second `Job`.
2. `run_worker` picks the job up before its next `step()` (②). The engine prefills it at the start of that step, and it then shares every decode step with the first request.
3. Both prompts begin with the same 24 ChatML tokens (the system message plus `<|im_start|>user\n`). With the default block size of 16, the first **16** tokens fill one block, so the second request reuses it through the prefix cache. This works only because the first request is still running when the second is admitted (Part 13's note).
4. Each request has its own sampler and RNG, so sharing a batch does not change either one's output for a given seed.

---

## Part 15 — Variation: the CUDA backend

Run the running example with `--device cuda` (on a build with `--features cuda`). The engine, the scheduler, the traces in Parts 4–12, and every block id stay **exactly the same**. Only the backend behind `forward_sample` changes.

### 15.1 Startup: context, kernels and weights

`Runtime::new(&model, "cuda")` (§2.5) calls two things.

[src/cuda/context.rs:15](../src/cuda/context.rs#L15)
```rust
    pub fn new(device_id: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ctx = cudarc::driver::CudaContext::new(device_id)?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;
        let mut cuda_runtime = Self {
            stream,
            ctx,
            kernels_mapping: HashMap::new(),
            blas,
        };
        cuda_runtime.load_kernels()?;
        Ok(cuda_runtime)
    }
    pub fn load_kernels(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // …
        self.load_single_kernel(include_str!("kernels/silu.cu"), "silu".to_string())?;
        // … one call per .cu file
        Ok(())
    }

    pub fn load_single_kernel(
        &mut self,
        source: &str,
        kernel_name: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(source)?;
        let module = self.ctx.load_module(ptx)?;
        let kernel = module.load_function(kernel_name.as_str())?;
        self.kernels_mapping.insert(kernel_name, kernel);

        Ok(())
    }
```

1. A CUDA **context** on GPU 0 and its default **stream**: an ordered queue of GPU work. Everything in this backend goes into this one stream, so operations run in the order they were issued.
2. A **cuBLAS** handle bound to that stream, for the matrix multiplications.
3. Each kernel's source is put into the binary by `include_str!` at compile time. At startup, **NVRTC** compiles it to PTX (GPU assembly), loads it, and stores the function under its name. That is why no `nvcc` or separate build step is needed.

Then `CudaModel::upload` copies the weights to the GPU once:

[src/cuda/weights.rs:44](../src/cuda/weights.rs#L44)
```rust
fn upload_f16(
    cuda_runtime: &CudaRuntime,
    t: &Tensor,
) -> Result<CudaSlice<f16>, Box<dyn std::error::Error>> {
    let host: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let host: Vec<f16> = host.into_iter().map(f16::from_f32).collect();
    Ok(cuda_runtime.stream.clone_htod(&host)?)
}
```

[src/cuda/weights.rs:54](../src/cuda/weights.rs#L54)
```rust
                Ok(CudaLayer {
                    attn_norm: upload(cuda_runtime, &l.attn_norm)?,
                    attn_q: upload_f16(cuda_runtime, &l.attn_q)?,
                    attn_k: upload_f16(cuda_runtime, &l.attn_k)?,
                    attn_v: upload_f16(cuda_runtime, &l.attn_v)?,
                    attn_output: upload_f16(cuda_runtime, &l.attn_output)?,
                    ffn_norm: upload(cuda_runtime, &l.ffn_norm)?,
                    ffn_gate: upload_f16(cuda_runtime, &l.ffn_gate)?,
                    ffn_up: upload_f16(cuda_runtime, &l.ffn_up)?,
                    ffn_down: upload_f16(cuda_runtime, &l.ffn_down)?,
                })
```

The big matrices are narrowed to **F16** (half the bytes, and decode speed is limited by how fast weights can be read), while the small norm vectors stay F32. `clone_htod` means "copy host to device". Approximate GPU memory for the weights:

| Part | Precision | Size |
|------|-----------|-----:|
| 32 layers of matrices (≈315M values) | F16 | ≈ 629 MB |
| `output` (LM head, 49152 × 960) | F16 | ≈ 94 MB |
| `token_embd` (49152 × 960) | F32 | ≈ 189 MB |
| **Total** | | **≈ 0.91 GB** |

With tied embeddings, the same matrix is therefore stored **twice** on the GPU: as F32 for the embedding lookup, and as F16 for the LM head.

Finally, `runtime.backend(4, 6)` creates a `CudaBackend` with a `CudaKvStorage`: 64 zeroed F32 device buffers, with the same layout and size (≈ 1.9 MiB) as the CPU version.

### 15.2 `CudaBackend::forward_sample`

[src/cuda/engine.rs:100](../src/cuda/engine.rs#L100)
```rust
    fn forward_sample(
        &mut self,
        items: &[BatchItem],
        last_only: bool,
        samplers: &mut [&mut Sampler],
    ) -> Result<Vec<u32>, BoxError> {
        let rows = items.len();
        let vocab_size = self.model.config.vocab_size as usize;
        let runtime = self.runtime;

        let enqueue_started = Instant::now();
        let logits = if last_only {                                                // ①
            self.model
                .forward_last(runtime, items, &mut self.kv_storage)
        } else {
            self.model.forward(runtime, items, &mut self.kv_storage)
        }
        .map_err(boxed)?;
        // Queue the per-row argmax right behind the forward pass.
        let greedy_ids = if samplers.iter().all(|s| s.is_greedy()) {               // ②
            let mut ids = runtime.stream.alloc_zeros::<u32>(rows).map_err(boxed)?;
            ops::argmax_wrapper(runtime, &logits, vocab_size as u32, &mut ids).map_err(boxed)?;
            Some(ids)
        } else {
            None
        };
        let enqueue = enqueue_started.elapsed();

        // Sync first so the time spent waiting on the GPU is separate from the
        // copy (clone_dtoh would sync anyway).
        let wait_started = Instant::now();
        runtime.stream.synchronize().map_err(boxed)?;                              // ③
        let gpu_wait = wait_started.elapsed();

        let copy_started = Instant::now();
        let (next, copy, sample) = match greedy_ids {                              // ④
            Some(ids) => {
                let next = runtime.stream.clone_dtoh(&ids).map_err(boxed)?;
                (next, copy_started.elapsed(), Duration::ZERO)
            }
            None => {
                let host = runtime.stream.clone_dtoh(&logits).map_err(boxed)?;
                let copy = copy_started.elapsed();
                let sample_started = Instant::now();
                let next = samplers
                    .iter_mut()
                    .enumerate()
                    .map(|(i, s)| s.sample_slice(&host[i * vocab_size..(i + 1) * vocab_size]))
                    .collect();
                (next, copy, sample_started.elapsed())
            }
        };

        if !last_only {                                                            // ⑤
            self.breakdown.steps += 1;
            self.breakdown.enqueue += enqueue;
            self.breakdown.gpu_wait += gpu_wait;
            self.breakdown.copy += copy;
            self.breakdown.sample += sample;
        }
        Ok(next)
    }
```

1. **Enqueue the forward pass.** Kernel launches are *asynchronous*: these calls return as soon as the work is queued, long before the GPU finishes it.
2. **Greedy shortcut.** Our run is greedy, so an `argmax` kernel is queued right behind the forward pass. For step 1's decode, only 3 token ids (12 bytes) will come back instead of 3 × 49,152 logits (≈ 590 KB).
3. **Wait** for the GPU to finish everything queued so far.
4. **Copy back** either the ids, or the logits for CPU sampling (`clone_dtoh` = device to host).
5. **Timing.** For decode steps, the time spent is split into enqueue, GPU wait, copy and sample. Set `RVLLM_TIMING=1` and the totals are printed when the backend is dropped.

### 15.3 The GPU forward pass

[src/cuda/model.rs:523](../src/cuda/model.rs#L523)
```rust
    fn hidden(
        &self,
        cuda_runtime: &CudaRuntime,
        items: &[BatchItem],
        kv_storage: &mut CudaKvStorage,
    ) -> Result<(CudaSlice<f32>, usize, usize), Box<dyn std::error::Error>> {
        let batch = items.len();
        let q_len = items[0].tokens.len();
        let hidden_dim = self.config.hidden_dim as usize;
        let rows = batch * q_len;

        let flat_tokens: Vec<u32> = items
            .iter()
            .flat_map(|item| item.tokens.iter().copied())
            .collect();
        let tokens_dev = cuda_runtime.stream.clone_htod(&flat_tokens)?;
        let mut x = cuda_runtime.stream.alloc_zeros::<f32>(rows * hidden_dim)?;
        ops::embed_lookup_wrapper(
            cuda_runtime,
            &tokens_dev,
            &self.weights.token_embd,
            hidden_dim as u32,
            &mut x,
        )?;

        let step_ctx = CudaStepCtx::new(cuda_runtime, &self.config, items)?;
        let mut ws = Workspace::new(cuda_runtime, &self.config, &step_ctx)?;
        // Ping-pong between two hidden-state buffers: a layer reads `x` and
        // writes `x_next`, then they swap.
        let mut x_next = cuda_runtime.stream.alloc_zeros::<f32>(rows * hidden_dim)?;
        for (i, layer) in self.weights.layers.iter().enumerate() {
            layer_forward(
                cuda_runtime,
                &self.config,
                i,
                layer,
                &step_ctx,
                kv_storage,
                &x,
                &mut x_next,
                &mut ws,
            )?;
            std::mem::swap(&mut x, &mut x_next);
        }
        Ok((x, batch, q_len))
    }
```

This has the same structure as the CPU `hidden` (§7.2), with three GPU-specific ideas:

1. **`CudaStepCtx`** builds the same per-call data as `StepCtx`, on the host, and copies it to the GPU. For step 1's decode in our example, the write slots `(B4,1) (B3,3) (B2,1)` become two small arrays, `block_ids = [4, 3, 2]` and `offsets = [1, 3, 1]`, for the `kv_write` kernel. RoPE `cos`/`sin` are pre-expanded per head: 3 × 15 = 45 rows for Q and 3 × 5 = 15 rows for K, each with 32 values.
2. **`Workspace`** allocates every intermediate buffer **once per forward pass**, sized for this batch, and all 32 layers reuse it.
3. **Ping-pong buffers:** each layer reads `x` and writes `x_next`, and then the two are swapped. No new memory is needed per layer.

`layer_forward` performs the CPU `Layer::forward` math as a sequence of **31 launches per layer**:

| # | Call | Runs on | CPU counterpart |
|--:|------|---------|-----------------|
| 1 | `rmsnorm_wrapper` | kernel `rmsnorm` | `rms_norm` |
| 2 | `to_f16` | kernel `f32_to_f16` | — (CPU stays in F32) |
| 3–5 | `linear_f16` for Q, K, V | cuBLAS GEMM | `linear` |
| 6–8 | `transpose_axes12_wrapper` ×3 | kernel | `.transpose(1, 2)` (free on CPU) |
| 9–10 | `rope_wrapper` for Q and K | kernel `rope` | `rope_apply` |
| 11–12 | `transpose_axes12_wrapper` ×2 (K and V back to token-major) | kernel | `.transpose(1, 2)` |
| 13 | `kv_storage.write_tokens` | kernel `kv_write` (K and V together) | `KvStorage::write_tokens` |
| 14–15 | `kv_storage.gather` | kernel `kv_gather` ×2 (K, V) | `KvStorage::gather` |
| 16 | `attention_scores` | cuBLAS batched GEMM, scaled by 1/√64 | `q.matmul(kᵀ)` + scale |
| 17 | `mask_add_broadcast_wrapper` | kernel | `broadcast_add(mask)` |
| 18 | `softmax_wrapper` | kernel `softmax` | `softmax` |
| 19 | `attention_apply` | cuBLAS batched GEMM | `probs.matmul(v)` |
| 20 | `transpose_axes12_wrapper` | kernel | merge heads |
| 21–22 | `linear` (output projection) | `f32_to_f16` + GEMM | `linear` |
| 23 | `add_wrapper` | kernel `add` | `x.add(&attn_out)` |
| 24–25 | `rmsnorm_wrapper`, `to_f16` | kernels | `rms_norm` |
| 26–27 | `linear_f16` for gate and up | cuBLAS GEMM | inside `swiglu_ffn` |
| 28 | `silu_gate_multiply_wrapper` | kernel (fused) | `silu(gate) * up` |
| 29–30 | `linear` (down projection) | `f32_to_f16` + GEMM | inside `swiglu_ffn` |
| 31 | `add_wrapper` | kernel `add` | `h.add(&ffn_out)` |

Over 32 layers that is about **1,000 launches per forward pass**. When the batch is small, launching that many small operations can cost more time than the GPU spends computing. That is exactly what the `RVLLM_TIMING` breakdown is designed to reveal.

### 15.4 One launch, up close: RoPE

The Rust side checks the sizes and launches the kernel:

[src/cuda/ops.rs:107](../src/cuda/ops.rs#L107)
```rust
pub fn rope_wrapper(
    cuda_runtime: &CudaRuntime,
    x: &CudaSlice<f32>,
    cos: &CudaSlice<f32>,
    sin: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    head_dim: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    // … length checks
    let half = head_dim / 2;
    let n_rows = n / head_dim;
    let n_pairs = n_rows * half;
    // …
    match cuda_runtime.kernels_mapping.get("rope") {
        Some(kernel) => {
            let mut builder = cuda_runtime.stream.launch_builder(kernel);

            builder.arg(x);
            builder.arg(cos);
            builder.arg(sin);
            builder.arg(output);
            builder.arg(&head_dim);
            builder.arg(&half);
            builder.arg(&n_pairs);
            unsafe { builder.launch(LaunchConfig::for_num_elems(n_pairs)) }?;
        }
        None => return Err("Error: kernel 'rope' not found".into()),
    }
    Ok(())
}
```

The GPU side runs one thread per (row, pair):

[src/cuda/kernels/rope.cu](../src/cuda/kernels/rope.cu)
```c
extern "C" __global__ void rope(
    const float *x,
    const float *cos,
    const float *sin,
    float *out,
    unsigned int head_dim,
    unsigned int half,
    unsigned int n_pairs
) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n_pairs) {
        // which row it takes care of
        unsigned int row = index / half;
        // which pair within that row
        unsigned int j = index % half;
        // it loads one pair using offsets
        float first = x[row * head_dim + j];
        float second = x[row * head_dim + half + j];
        // pre comouted values
        float c = cos[row * half + j];
        float s = sin[row * half + j];
        // actual rotation
        float first_rot  = first * c - second * s;
        float second_rot = first * s + second * c;
        // store it back
        out[row * head_dim + j] = first_rot;
        out[row * head_dim + half + j] = second_rot;
    }
}
```

For Q in step 1's decode there are 45 rows × 32 pairs = **1,440** pairs. `LaunchConfig::for_num_elems(1440)` uses blocks of 1,024 threads, so it launches 2 blocks (2,048 threads). The `if (index < n_pairs)` check leaves the last 608 threads idle. Each working thread computes exactly the same rotation as `rope_apply` on the CPU, for one pair.

Kernels that need a whole row at once (`rmsnorm`, `softmax`, `argmax`) are launched differently: **one block of 256 threads per row**. The threads first each handle every 256th element, then combine their partial results in shared memory, halving the number of active threads each round (a tree reduction):

[src/cuda/ops.rs:60](../src/cuda/ops.rs#L60) (from `rmsnorm_wrapper`)
```rust
            let cfg = LaunchConfig {
                grid_dim: (num_rows, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: block_size * std::mem::size_of::<f32>() as u32,
            };
```

### 15.5 Matrix multiplies through cuBLAS

[src/cuda/cublas_ops.rs:44](../src/cuda/cublas_ops.rs#L44) (the call inside `linear_f16`)
```rust
    unsafe {
        result::gemm_ex(
            *cuda_runtime.blas.handle(),
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            out_features as i32,
            rows as i32,
            in_features as i32,
            (&alpha as *const f32).cast(),
            w_ptr as *const c_void,
            cudaDataType::CUDA_R_16F,
            in_features as i32,
            x_ptr as *const c_void,
            cudaDataType::CUDA_R_16F,
            in_features as i32,
            (&beta as *const f32).cast(),
            o_ptr as *mut c_void,
            cudaDataType::CUDA_R_32F,
            out_features as i32,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }?;
```

We want `Y = X · Wᵀ` with row-major buffers, but cuBLAS reads every buffer as **column-major**. The trick (explained at the top of `cublas_ops.rs`) is that a row-major `[p, q]` buffer has exactly the same bytes as a column-major `[q, p]` matrix, its transpose:

| Buffer (row-major) | What cuBLAS sees (column-major) | Operation passed | Matrix used |
|--------------------|---------------------------------|------------------|-------------|
| `W` `[out, in]` | `Wᵀ` `[in, out]` | `OP_T` | `W` `[out, in]` |
| `X` `[rows, in]` | `Xᵀ` `[in, rows]` | `OP_N` | `Xᵀ` `[in, rows]` |
| result | `W · Xᵀ` `[out, rows]` | — | read back row-major: `[rows, out]` = `X · Wᵀ` ✔ |

For step 1's Q projection that is `m` = 960, `n` = 3 rows, `k` = 960. The inputs are F16 (`CUDA_R_16F`), the arithmetic and the output are F32 (`CUBLAS_COMPUTE_32F`, `CUDA_R_32F`), and `TENSOR_OP` lets cuBLAS use the GPU's tensor cores.

The two attention GEMMs (`attention_scores`, `attention_apply`) use `gemm_strided_batched`: one call performs `batch × n_kv_heads` = 3 × 5 = 15 small matrix multiplies, one for each (sequence, KV head) pair. The `1/√64` scale is passed as cuBLAS's `alpha`, so it needs no extra kernel.

### 15.6 The greedy pick on the GPU

[src/cuda/kernels/argmax.cu](../src/cuda/kernels/argmax.cu)
```c
// One block per row: index of the row's maximum. Ties go to the lowest index
// and NaN is ignored, same as Sampler::sample_slice's greedy scan, so the
// GPU and CPU paths pick identical tokens. A row with no value above -inf
// (all -inf / NaN) returns 0, like the CPU scan.
extern "C" __global__ void argmax(
    const float *x,
    unsigned int *out,
    unsigned int row_len
) {
    extern __shared__ unsigned char smem[];
    float *vals = (float *)smem;
    unsigned int *idxs = (unsigned int *)(smem + blockDim.x * sizeof(float));

    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    const float *row_in = x + (size_t)row * row_len;

    // NVRTC has no INFINITY; build -inf from its bit pattern.
    float best = -__int_as_float(0x7f800000);
    unsigned int best_idx = 0xffffffffu;
    for (unsigned int i = tid; i < row_len; i += blockDim.x) {
        float v = row_in[i];
        if (v > best) {
            best = v;
            best_idx = i;
        }
    }
    vals[tid] = best;
    idxs[tid] = best_idx;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float v2 = vals[tid + s];
            unsigned int i2 = idxs[tid + s];
            if (v2 > vals[tid] || (v2 == vals[tid] && i2 < idxs[tid])) {
                vals[tid] = v2;
                idxs[tid] = i2;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = idxs[0] == 0xffffffffu ? 0u : idxs[0];
    }
}
```

1. Each of the 256 threads scans every 256th logit of its row (49,152 / 256 = 192 values) and keeps its best value and index.
2. The tree reduction halves the number of active threads each round (128, 64, …, 1), keeping the larger value, or the **lower index on a tie**.
3. Thread 0 writes the winner. The tie rule matches the CPU's greedy scan, so the CPU and GPU pick the same token for the same logits.

Because of F16 weights and a different order of additions, GPU logits differ slightly from CPU logits, so a near-tie can occasionally resolve differently on the two paths. The test `cuda_forward_matches_cpu_forward` in `cuda/model.rs` checks that the two paths agree within a tolerance and pick the same greedy token on a small model.

---

## Part 16 — Variation: the benchmark

`bench` is a second binary that drives the same `engine::run` with synthetic requests.

```bash
cargo run --release --bin bench -- \
  --model ../SmolLM2-360M.Q8_0.gguf --batch-sizes 1,4 --prompt-len 128 --gen-len 64
```

[src/bin/bench.rs:228](../src/bin/bench.rs#L228) (trimmed)
```rust
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    let model = Model::load(&args.model)?;
    let runtime = Runtime::new(&model, &args.device)?;
    // …
    let mut rng = StdRng::seed_from_u64(args.seed);
    // … (a --trace-file run returns early here)
    for &batch_size in &batch_sizes {
        let requests = synthetic_requests(
            &mut rng,
            model.config.vocab_size,
            batch_size,
            args.prompt_len,
            args.gen_len,
        );
        run_and_report(
            &format!("batch_size={batch_size}"),
            &runtime,
            &requests,
            args.block_size,
            args.seed,
        )?;
    }

    Ok(())
}
```

1. **No tokenizer.** Prompts are random token ids (`synthetic_requests`). Throughput depends on lengths and timing, not on what the text says.
2. **Pool sizing** (`num_blocks_for`) works like the CLI's: (128 + 64) = 192 tokens → 12 blocks per request. For batch size 4 that is 48 blocks.
3. **Warm-up, then measure.** `run_and_report` calls `engine::run` twice, each with a fresh backend, and only reports the second run:

[src/bin/bench.rs:186](../src/bin/bench.rs#L186) (trimmed)
```rust
    let sampling = SamplingParams {
        seed,
        ..SamplingParams::GREEDY
    };

    // Warmup run (discarded) so the measured run isn't paying for first-touch
    // allocation of the KV storage tensors, per the plan's "steady-state
    // throughput once warmup is excluded".
    engine::run(
        runtime.backend(block_size, num_blocks)?,
        requests,
        sampling,
        cfg,
    )?;

    let result = engine::run(
        runtime.backend(block_size, num_blocks)?,
        requests,
        sampling,
        cfg,
    )?;
    report(label, &result, batch_size, num_blocks)
```

`report` turns `RunStats` into numbers:

| Printed as | Formula | For batch 4 in this example |
|------------|---------|-----------------------------|
| `tok/s(total)` | `output_tokens / elapsed` | depends on the machine |
| `tok/s/req` | total ÷ batch size | depends on the machine |
| `prefix_hit` | `prefix_hits / prefix_total` | ≈ 0%: random prompts share no block |
| `pool_util` | `total_block_allocs / num_blocks` | 48 / 48 = 100% (8 blocks per request at prefill + 4 during decode) |
| `ttft_ms(mean/max)` | per-request prefill times | depends on the machine |
| `prefill_tok/s` | `prefill_tokens / prefill_time` | depends on the machine |
| `decode_ms/step` | `decode_time / decode_steps` | depends on the machine |
| `checksum` | FNV-1a hash over every generated token | equal before and after a change ⇒ identical greedy output |

The checksum is a handy tool when optimizing: if a speed-up changes the checksum, it changed the model's output too.

---

## Part 17 — Recap, function index and exercises

### 17.1 The whole journey in 20 steps

1. `main` parses the flags with clap and calls `run_cli`.
2. `Model::load` reads the GGUF file, dequantizes every tensor to F32 and fixes the Q/K row order for RoPE.
3. `load_tokenizer` loads `tokenizer.json`; `Runtime::new` picks the CPU (or creates a CUDA context and uploads the weights).
4. `one_shot` tokenizes each prompt into a `RequestSpec` and sizes the KV pool (6 blocks of 4 slots in our example).
5. `runtime.backend` creates the backend and its zeroed KV tensors.
6. `engine::run` creates the `Engine` (and with it the crate's `Scheduler` and `BlockPool`).
7. `Engine::add_request` validates each request and puts it in the scheduler's FIFO `waiting` queue.
8. `Engine::step` starts with `admit_waiting`.
9. For each waiting request that fits, `match_prefix` finds cached blocks and `Scheduler::prefill` shares or allocates blocks.
10. The prompt's uncached tokens become a single-item `BatchItem`, and `forward_sample(…, last_only = true)` runs the prefill.
11. `Model::forward_last` embeds the tokens, builds a `StepCtx` and runs 32 × `Layer::forward`.
12. Each layer normalizes, projects to Q/K/V, applies RoPE, **writes** the new K/V into its block slots, **gathers** the full context, runs grouped-query attention and the SwiGLU feed-forward, with residual connections around both.
13. `lm_head` turns the last position into 49,152 logits, and `Sampler::sample` picks a token.
14. `emit` records the token, checks for stop or length, and returns a `TokenEvent`.
15. `Engine::decode` reserves one slot per running request (`Scheduler::step`), preempting if the pool is full.
16. All running requests feed in their last token together in **one** batched `forward_sample(…, last_only = false)`.
17. Each new token goes through `emit`. Finished requests release their blocks through `finish_sequence`.
18. `engine::run` repeats `step` until nothing is waiting or running, collecting tokens per prompt.
19. `one_shot` removes a trailing stop token, decodes the ids to text and prints it.
20. With `--debug`, it prints the prefix-cache, allocation and throughput statistics.

### 17.2 Function index

| Function | Location | Role |
|----------|----------|------|
| `main` | [main.rs:117](../src/main.rs#L117) | Entry point; dispatches to `cli` or `serve` |
| `run_cli` | [main.rs:142](../src/main.rs#L142) | Loads everything, picks one-shot or interactive |
| `one_shot` | [main.rs:158](../src/main.rs#L158) | Builds requests, sizes the pool, runs, prints |
| `interactive` | [main.rs:254](../src/main.rs#L254) | REPL loop with streaming output |
| `run_serve` | [main.rs:354](../src/main.rs#L354) | Starts the engine thread and the HTTP server |
| `Model::load` | [weights.rs:33](../src/weights.rs#L33) | GGUF → `Model` |
| `unpermute_rope_rows` | [weights.rs:20](../src/weights.rs#L20) | Fix the Q/K row order for rotate-half RoPE |
| `Runtime::new` / `backend` | [runtime.rs:19](../src/runtime.rs#L19) / [:48](../src/runtime.rs#L48) | Pick the device; build a backend + KV pool |
| `CpuBackend::forward_sample` | [engine.rs:84](../src/engine.rs#L84) | Forward + sample on the CPU |
| `Engine::add_request` | [engine.rs:248](../src/engine.rs#L248) | Validate and queue |
| `Engine::abort` | [engine.rs:290](../src/engine.rs#L290) | Cancel and free |
| `Engine::step` | [engine.rs:303](../src/engine.rs#L303) | Admit, then decode |
| `Engine::admit_waiting` | [engine.rs:310](../src/engine.rs#L310) | Prefill new requests |
| `Engine::decode` | [engine.rs:363](../src/engine.rs#L363) | One batched decode step |
| `Engine::emit` | [engine.rs:449](../src/engine.rs#L449) | Record a token; finish if done |
| `engine::run` | [engine.rs:515](../src/engine.rs#L515) | Drive a fixed set of requests to completion |
| `Model::forward` / `forward_last` | [model.rs:451](../src/model.rs#L451) / [:461](../src/model.rs#L461) | Logits for all rows / the last row |
| `StepCtx::new` | [model.rs:285](../src/model.rs#L285) | Per-call RoPE tables, mask, gather plan, writes |
| `Layer::forward` | [model.rs:339](../src/model.rs#L339) | One transformer block |
| `gqa_attention` | [model.rs:209](../src/model.rs#L209) | Grouped-query attention |
| `batch_mask` | [model.rs:180](../src/model.rs#L180) | Causal + padding mask |
| `KvStorage::write_tokens` / `gather` | [kv_storage.rs:99](../src/kv_storage.rs#L99) / [:140](../src/kv_storage.rs#L140) | Store / read K and V |
| `GatherPlan::new` | [kv_storage.rs:42](../src/kv_storage.rs#L42) | Which blocks each sequence reads |
| `Sampler::sample` / `sample_slice` | [sampler.rs:56](../src/sampler.rs#L56) / [:67](../src/sampler.rs#L67) | Greedy or temperature + top-p |
| `StreamDecoder::push` | [tokenizer.rs:57](../src/tokenizer.rs#L57) | Safe incremental detokenization |
| `chatml` | [chat.rs:13](../src/chat.rs#L13) | Chat template |
| `start_generation` | [server/mod.rs:208](../src/server/mod.rs#L208) | Validate, take a slot, send the job, translate tokens |
| `run_worker` | [server/worker.rs:50](../src/server/worker.rs#L50) | The engine thread's loop |
| `sse` | [server/mod.rs:345](../src/server/mod.rs#L345) | Chunks → Server-Sent Events |
| `StopFilter::push` | [server/stop.rs:21](../src/server/stop.rs#L21) | Stop strings across chunk boundaries |
| `CudaRuntime::new` | [cuda/context.rs:15](../src/cuda/context.rs#L15) | Context, stream, cuBLAS, NVRTC kernels |
| `CudaBackend::forward_sample` | [cuda/engine.rs:100](../src/cuda/engine.rs#L100) | Forward + GPU argmax or host sampling |
| `layer_forward` (CUDA) | [cuda/model.rs:254](../src/cuda/model.rs#L254) | One block as 31 launches |
| `linear_f16` | [cuda/cublas_ops.rs:44](../src/cuda/cublas_ops.rs#L44) | F16 GEMM with F32 output |
| `run_and_report` | [bin/bench.rs:186](../src/bin/bench.rs#L186) | Warm-up + measured benchmark run |

### 17.3 Exercises

Predict the `--debug` statistics before looking at the answers. Each one can be worked out from the code in this document. Unless stated otherwise, use the running example's command with one change.

**1.** Change `--block-size 4` to `--block-size 16`.

<details>
<summary>Answer</summary>

The pool is `blocks_for(8, 16)` × 3 = **3 blocks**, one per request. No prompt reaches 16 tokens, so no block is ever complete and nothing can be shared: `prefix cache: 0 / 13 tokens hit (0.0%), 3 total block allocations`. Every request fits its 7 or 5 used slots in its single block, so no allocation happens during decode. Still `2 steps, 9 output tokens`.
</details>

**2.** Change `--max-tokens 3` to `--max-tokens 1`.

<details>
<summary>Answer</summary>

Every request finishes inside `admit_waiting`, because its first token already reaches `max_tokens`. A finishes, and its blocks are freed and **removed from the trie** before C is admitted, so C gets no hit: `0 / 13 tokens hit (0.0%), 5 total block allocations` (A 2, B 1, C 2). `decode` finds nothing running, so the run takes `1 steps` and produces `3 output tokens`.
</details>

**3.** Keep the original command but add `--arrival-step 0 --arrival-step 0 --arrival-step 5`, so C arrives at step 5.

<details>
<summary>Answer</summary>

A and B finish at step 2 (as before) and free their blocks. Steps 3–5 are idle: `engine::run` keeps calling `step()` because C has not arrived yet. C is admitted at step 6 with no cache hit, because A's blocks are long gone, and it finishes at step 7. Output: `0 / 13 tokens hit (0.0%), 6 total block allocations` (A 2, B 2, C 2) and `7 steps, 9 output tokens`.
</details>

**4.** In the original run, C only allocates *one* new block. Why does `admit_waiting` still require *two* free blocks before admitting it?

<details>
<summary>Answer</summary>

`needed_blocks` is computed from the full prompt length, ⌈5 / 4⌉ = 2, before the prefix cache is consulted. The check is deliberately conservative: it guarantees that the `append_token(…).unwrap()` calls inside `Scheduler::prefill` can never fail.
</details>

**5.** During step 1's decode, which physical memory does request B's attention read for context positions 4 and 5, and why is the result still correct?

<details>
<summary>Answer</summary>

B's block table `[B3]` is padded to two entries by repeating B3, so positions 4–5 are read from B3's slots 0–1 (B's own first two tokens, read a second time). `batch_mask` sets those two positions to `-∞` because `j ≥ read_num_tokens` (4), so softmax gives them exactly zero weight.
</details>

---

**Where to go next:** [05-model-and-weights.md](05-model-and-weights.md), [06-kv-cache-and-paging.md](06-kv-cache-and-paging.md) and [07-engine-and-scheduling.md](07-engine-and-scheduling.md) cover the same code topic by topic, and [12-glossary-and-faq.md](12-glossary-and-faq.md) defines every term used here.
