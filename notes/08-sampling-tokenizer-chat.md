# 08 — Sampling, Streaming Detokenization, and Chat Templates

Files: `src/sampler.rs`, `src/tokenizer.rs`, `src/chat.rs`

## 8.1 Sampling: choosing the next token

The model outputs one **logit** per vocabulary token. Higher means more likely. The `Sampler` turns those into a single token id.

### The parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| `temperature` | `0.0` = greedy; `> 0` = random | Divides logits before softmax. Low values make output sharper and safer; high values make it more random and creative |
| `top_p` | `(0, 1]` | Nucleus sampling: only sample from the smallest set of tokens whose probabilities add up to ≥ `top_p` |
| `seed` | any `u64` | Seeds this request's own RNG, so runs are reproducible |

### Algorithm (`Sampler::sample_slice`)

1. **Greedy shortcut:** if `temperature ≤ 0`, return the index of the largest logit (the first one wins on ties). No RNG is used, so it is fully deterministic.
2. **Softmax with temperature:** `p_i = exp(l_i / T − max) / Σ`. Subtracting the max is the standard trick to avoid overflow.
3. **Nucleus filter, done efficiently:**
   - Instead of sorting all ~49k probabilities, *partition* out the top `k = 64` (`select_nth_unstable_by`, O(n)) and sort only those.
   - Walk them, adding up probability, until the sum reaches `top_p`.
   - If 64 is not enough (a very flat distribution), grow `k ×4` and try again.
4. **Draw:** pick a random `r ∈ [0, kept_sum)` and walk the kept tokens until the running sum passes `r`.

A test (`partial_selection_matches_full_sort_reference`) checks that this optimized version picks exactly the same tokens as a simple full-sort version.

### Worked example

Logits `[2.0, 1.0, 0.1]`, `T = 1` → probabilities ≈ `[0.66, 0.24, 0.10]`.

| `top_p` | Kept set | Behaviour |
|---------|----------|-----------|
| 0.5 | `{0}` | Always token 0 (0.66 ≥ 0.5 already) |
| 0.9 | `{0, 1}` | Token 0 about 73% of the time, token 1 about 27% |
| 1.0 | `{0, 1, 2}` | Full distribution |

### Where sampling happens

| Backend | Greedy | Non-greedy |
|---------|--------|-----------|
| CPU | `Sampler::sample` on each row's tensor | Same |
| CUDA | `argmax` **kernel on the GPU**; copies back one `u32` per row | Copies all logits to the host, then `sample_slice` per row |

Greedy on the GPU avoids copying megabytes of logits to the CPU every step.

## 8.2 The tokenizer wrapper

`SmollLM230MTokenizer` is a thin wrapper around HuggingFace's `tokenizers::Tokenizer`:

- `encode(text)` → ids. Special tokens are **not** added automatically (`add_special_tokens = false`).
- `decode(ids)` → text (skipping special tokens).
- `token_to_id("<|im_end|>")` is used to find the chat stop token.

## 8.3 `StreamDecoder` — safe streaming output

**Problem:** many characters (accents, emoji, CJK) are split across *several* byte-level tokens. If you decode each token by itself, you get broken characters (`�`, U+FFFD) or lost leading spaces.

**Solution:** keep a small sliding window of ids:

```
ids:  [ … already emitted … | prefix_offset … read_offset | new ids ]
```

On each `push(id)`:

1. Decode the window *without* the new ids → `prefix`.
2. Decode the window *with* them → `text`.
3. If `text` is longer **and** does not end in `�`, emit `text[prefix.len()..]` and slide the window forward.
4. Otherwise, emit nothing yet and wait for more tokens to complete the character.

`flush()` releases whatever is still held back when generation ends. The test `stream_decoder_matches_full_decode` streams `"Héllo wörld 🙂 — streaming, ok?"` and checks that the joined pieces equal a one-shot decode.

## 8.4 Chat templates: ChatML (`chat.rs`)

Instruct models are trained on a specific conversation format. SmolLM2-Instruct uses **ChatML**:

```
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
What is Rust?<|im_end|>
<|im_start|>assistant
```

`chatml(messages)`:

1. Adds the **default system prompt** unless the first message already has role `system`.
2. Writes each message as `<|im_start|>{role}\n{content}<|im_end|>\n`.
3. Ends with `<|im_start|>assistant\n` so the model writes the assistant's reply next.

The model signals the end of its reply by producing `<|im_end|>`. `main.rs::stop_tokens` and the server both add that id to the stop list next to EOS.

### Multi-turn chat in the CLI

With `--chat`, the interactive loop keeps a `history: Vec<ChatMessage>`:

1. Push the user message and render the **whole** history with `chatml`.
2. Generate the reply and push it as an `assistant` message.

Each new prompt starts with the whole previous conversation, so you might expect **prefix caching** to reuse the earlier turns' blocks. It does not: the previous turn's request has already finished, and finishing frees its blocks and removes them from the prefix cache. Every turn therefore prefills the full conversation again. (Prefix sharing only happens between requests that are running at the same time, such as concurrent server requests with the same system prompt.)

## 8.5 Stop strings (server only)

Besides stop **tokens**, OpenAI clients can send stop **strings** (`"stop": ["\n\n"]`). They are handled on decoded text by `server/stop.rs::StopFilter`. See note 09 §9.5.
