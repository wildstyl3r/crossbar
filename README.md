# crossbar

crossbar is a small Gemma-3 inference engine in Rust on
[candle](https://github.com/huggingface/candle): chat generation from
safetensors or GGUF weights with streaming decode, sampling controls, and an
interactive REPL with tool calling and schema-constrained sampling.

> **Note:** this repository was produced with guided AI generation and human
> review. It is experimental research code, shared as-is.

Experimental mechanisms layered on top of the engine — softmax-feedback
decoding, a trained gated state-readout enricher, a feedback-impact
benchmark suite, and the enricher's TBPTT training — are isolated in
[`src/experiments/`](src/experiments/) together with their design docs
(`ENRICHER.md`, `FEEDBACK.md`). The experimental specs, CLI groups, training
and bench tooling live entirely in that directory; the decode-time hooks
they plug into are part of the core backends.

## Quickstart

Requirements: Rust 1.85+ (the crate uses edition 2024). CPU-only is fine —
the 270M model needs ~1 GB RAM as safetensors and ~200 MB as Q4_0 GGUF.

Model weights are gated; accept Google's Gemma license on Hugging Face
first, then place them as:

- `models/gemma-3-270m-it` — safetensors backend, from
  [google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)
  (keep `config.json`, `tokenizer.json`, `chat_template.jinja`,
  `generation_config.json`, `model.safetensors`).
- `models/gemma-3-270m-it-qat-GGUF/` — quantized backend, a QAT Q4_0 GGUF
  build of the same model (see the HF model page; the original is
  [google/gemma-3-270m-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-270m-it-qat-q4_0-unquantized)).
  Only the `.gguf` file is needed here; the tokenizer/template/config fall
  back to `--fallback-dir` (default `models/gemma-3-270m-it`).

```sh
cargo run --release -- --cpu -p "Name three colors:" -n 128 --seed 42
cargo run --release -- --cpu --model-dir models/gemma-3-270m-it -p "Name three colors:" -n 128
cargo run --release -- --cpu --repl --tools auto   # interactive chat with tool calling
```

`--model-dir` auto-detects GGUF vs safetensors. Sampling defaults come from
`generation_config.json`; override with `--temperature` (0 = greedy),
`--top-k`, `--top-p`, `--repeat-penalty`, `--seed`.

## Core architecture

| Module | Responsibility |
|---|---|
| `main.rs` | thin binary: parse `cli::Args`, dispatch to train / bench / generate |
| `cli.rs` | clap definitions for the core arguments; the experimental arg groups are flattened in from `experiments::args` |
| `lib.rs` | library crate root; the binary consumes everything through it |
| `engine.rs` | `Engine` enum dispatching every model call to the safetensors or quantized backend |
| `assets.rs` | model-directory discovery (safetensors/GGUF/tokenizer/template) and `generation_config.json` parsing |
| `config.rs` | `Config` / `LayerType`, shared by both backends |
| `model/mod.rs` | safetensors model: architecture wiring + inference paths (plain forward, decode feedback, prefill cascade, enricher cascade) |
| `model/layers.rs` | transformer building blocks: `RmsNorm`, `RotaryEmbedding`, `Mlp`, `Attention`, `DecoderLayer`, KV-cache kinds, attention masks |
| `quantized_model.rs` | GGUF backend; mirrors the inference paths of `model/mod.rs` |
| `generation.rs` | streaming decode pipeline (`TextGeneration`): per-request `SamplingParams`/`GenerateOptions`, stop-strings, cancellation, `LogitsFilter` hook, streaming sink |
| `sampling.rs` | `LogitsProcessor` (temperature / top-k / top-p) + repeat penalty |
| `chat.rs` | minijinja chat-template rendering: one-shot (`render`) and multi-turn (`render_messages`, tool results as framed user turns) |
| `repl.rs` | interactive terminal chat (`--repl`): session history, `/`-commands, Ctrl+C cancellation, tool loop, schema-constrained output |
| `tools.rs` | built-in tool registry (current_time / calculator / echo), system-prompt tool spec, lenient tool-call extraction from free text |
| `grammar.rs` | schema-constrained sampling: vocab token trie + char-driven JSON state machine masking logits to a JSON-Schema subset (objects, arrays, scalars, string enums) |

## Flows

- **Generation**: `main` -> `Engine` -> `generation::TextGeneration`; per
  step the engine returns logits. The engine also exposes the decode-time
  hooks the experiments plug into (expected-embedding decode feedback
  `forward_feedback`, prefill cascade `forward_prompt_blend`, sequential
  enricher cascade `cascade_step`); their specs come from `experiments`.
- **REPL**: `main` (`--repl`) -> `repl::run`, fully synchronous. Turns
  reuse the KV cache: the ids actually forwarded by each generation are
  remembered, and the next prompt is assembled at the token level so only
  the suffix is pre-filled; on any mismatch the cache is cleared and the
  whole prompt re-prefilled — correctness never depends on the cache
  hitting. `/tool required` attaches a `grammar::SchemaMasker` forcing the
  tool-call envelope; `/schema <file>` masks the reply to any supported
  JSON schema.

## Known limitations

- The safetensors and GGUF backends implement the inference paths in
  parallel (the candle-examples convention); changes to the feedback math
  must be made in both.
- Both model backends have no unit tests; behavior-parity smokes
  (byte-identical generation at fixed seeds) are run manually during
  refactors.

## Tests

`cargo build --release` builds the binary; `cargo test` runs unit tests over
the pure logic (sampling, sliding-window pattern, attention masks, chat
rendering, schema grammar).
