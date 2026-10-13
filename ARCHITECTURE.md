# Architecture

crossbar is a Gemma-3 inference / training playground on candle: chat
generation from safetensors or GGUF weights, experimental softmax-feedback
decoding, a feedback-impact benchmark suite, and TBPTT training of a gated
state-readout enricher (see `src/experiments/ENRICHER.md`,
`src/experiments/FEEDBACK.md`).

## Module map

Core (required to run the model):

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

Experiments (nothing here is needed for plain generation; the core only
imports types from `experiments` and exposes the decode-time hooks the
experiments plug into):

| Module | Responsibility |
|---|---|
| `experiments/feedback.rs` | softmax-feedback types (`FeedbackConfig`, stats) and the entropy-solver math (`FEEDBACK.md`) |
| `experiments/enrich.rs` | the enricher: variant spec, serialization (`enrich.cfg` tensor), gated readout math, monitors (`ENRICHER.md`) |
| `experiments/cascade.rs` | the training-only cascade forward (TBPTT, detached-head CE trick) — lives outside `model` but touches `Model`/`DecoderLayer` internals via `pub(crate)` fields |
| `experiments/train.rs` | enricher training loop: batching, gradient accumulation, tau-sweep diagnostics, PCA calibration, eval, probe |
| `experiments/tasks.rs` | synthetic task generators (state-track, induce-d) + exact-match scoring, shared by trainer and bench |
| `experiments/bench.rs` | feedback-impact benchmark suite: tasks, modes, McNemar stats, CSV output |
| `experiments/args.rs` | clap argument groups (`FeedbackArgs`, `TrainArgs`, `BenchArgs`) + CLI-to-domain converters (`feedback_config`, `enricher_spec`) |
| `experiments/ENRICHER.md` | enricher design, BPTT analysis, plan |
| `experiments/FEEDBACK.md` | softmax-feedback experiments and findings |

## Flows

- **Generation**: `main` -> `Engine` -> `generation::TextGeneration`; per step the
  engine returns logits and (optionally) feeds back expected embeddings
  (`forward_feedback`, prefill cascade via `forward_prompt_blend`) or, with an
  enricher attached, runs the sequential cascade (`cascade_step`). The
  feedback/enricher specs come from `experiments`.
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
- **Training**: `main` -> `experiments::train::run`; batches come from
  `experiments::tasks::build_dataset`, the graph is built by
  `experiments::cascade::cascade_train_forward` (via
  `Engine::cascade_train_forward`), gradients flow only through the enricher
  (the backbone stays frozen).
- **Bench**: `main` -> `experiments::bench::run_bench`, reusing `Engine`
  factories and `experiments::tasks` scoring.

## Checkpoint formats

Enricher checkpoints carry an `enrich.cfg` tensor (5 floats in v2.2+: rank,
gelu, content-gate, a-mode, residual). M1-era files (`enrich.w_q/w_k/w_v/w_o`,
no cfg) are incompatible with the current residual-stream readout
(src/experiments/ENRICHER.md) — expected to fail loading. Saves are atomic: serialized to
`<path>.tmp` then renamed over the target, so a mid-write kill cannot corrupt
the last good checkpoint (`Enricher::save`).

## Tests

`cargo test` runs unit tests over the pure logic (entropy solver, sampling,
enricher spec round-trip, sliding-window pattern, attention masks, task
scoring, chat rendering). Behavior-parity smokes (byte-identical generation
across backends/enricher modes at fixed seeds) are run manually per
`/tmp/opencode/baseline` SHA256SUMS during refactors.
