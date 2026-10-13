# crossbar

crossbar is a Gemma-3 inference engine in Rust on
[candle](https://github.com/huggingface/candle): chat generation from
safetensors or GGUF weights with streaming decode, sampling controls, and an
interactive REPL with tool calling and schema-constrained sampling.

Experimental mechanisms layered on top of the engine — softmax-feedback
decoding, a trained gated state-readout enricher, a feedback-impact
benchmark suite, and the enricher's TBPTT training — are isolated in
[`src/experiments/`](src/experiments/) together with their design docs
(`ENRICHER.md`, `FEEDBACK.md`). Nothing in that directory is required for
plain generation; the core only imports types from it and exposes the
decode-time hooks the experiments plug into.

## Core architecture

Required to run the model:

| Module | Responsibility |
|---|---|
| `main.rs` | thin binary: parse `cli::Args`, dispatch to train / probe / bench / generate |
| `cli.rs` | clap definitions for the core arguments; the experimental arg groups are flattened in from `experiments::args` |
| `lib.rs` | library crate root; the binary consumes everything through it |
| `engine.rs` | `Engine` enum dispatching every model call to the F32 or quantized backend |
| `assets.rs` | model-directory discovery (safetensors/GGUF/tokenizer/template) and `generation_config.json` parsing |
| `config.rs` | `Config` / `LayerType`, shared by both backends |
| `model/mod.rs` | F32 model: architecture wiring + inference paths (plain forward, decode feedback, prefill cascade, enricher cascade, probe) |
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
- **REPL**: `main` (`--repl`) -> `repl::run` on the main thread (fully
  synchronous). Turns reuse the KV cache: the ids actually forwarded by each
  generation (`GenerationOutcome::context_ids`) are remembered with the
  history length they cover, and the next prompt is assembled at the token
  level (cached ids + the new turns rendered by the template), so only the
  suffix is pre-filled; on any mismatch (`/reset`, `/system`, `/tool`,
  cancelled retries, out-of-line edits) the cache is cleared and the whole
  prompt re-prefilled — correctness never depends on the cache hitting.
  Multi-token delta chunks are only fed while the sliding ring has not
  wrapped; past that the delta is fed token by token (ring-ordered cache
  contents pair badly with positional masks). `/tool required` additionally
  attaches a `grammar::SchemaMasker` forcing the tool-call
  envelope (`tools::required_call_schema`); `/schema <file>` masks the reply
  to any supported JSON schema.

## Tests

`cargo build --release` builds the binary; `cargo test` runs unit tests over
the pure logic (sampling, sliding-window pattern, attention masks, chat
rendering, schema grammar). Behavior-parity smokes (byte-identical
generation across backends/experiment modes at fixed seeds) are run
manually during refactors.
