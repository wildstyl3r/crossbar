# Experiments

Experimental mechanisms layered on top of the plain Gemma-3 engine:
softmax-feedback decoding, a trained gated state-readout enricher, the
enricher's TBPTT training, and a synthetic-task feedback-impact benchmark
suite. See the repository README for the core architecture and the
decode-time hooks these mechanisms plug into.

## Module map

| Module | Responsibility |
|---|---|
| `feedback.rs` | softmax-feedback types (`FeedbackConfig`, stats) and the entropy-solver math (`FEEDBACK.md`) |
| `enrich.rs` | the enricher: variant spec, serialization (`enrich.cfg` tensor), gated readout math, monitors (`ENRICHER.md`) |
| `cascade.rs` | the training-only cascade forward (TBPTT, detached-head CE trick) — lives outside `model` but touches `Model`/`DecoderLayer` internals via `pub(crate)` fields |
| `train.rs` | enricher training loop: batching, gradient accumulation, tau-sweep diagnostics, PCA calibration, eval, probe |
| `tasks.rs` | synthetic task generators (state-track, induce-d) + exact-match scoring, shared by trainer and bench |
| `bench.rs` | feedback-impact benchmark suite: tasks, modes, McNemar stats, CSV output |
| `args.rs` | clap argument groups (`FeedbackArgs`, `TrainArgs`, `BenchArgs`) + CLI-to-domain converters (`feedback_config`, `enricher_spec`) |
| `ENRICHER.md` | enricher design, BPTT analysis, results |
| `FEEDBACK.md` | softmax-feedback experiments and findings |

## Flows

- **Decode hooks**: the engine's `forward_feedback`, `forward_prompt_blend`
  and `cascade_step` (listed in the repository README) implement what
  `feedback.rs` / `enrich.rs` specify; the CLI groups come from `args.rs`.
- **Training**: `main` -> `experiments::train::run`; batches come from
  `experiments::tasks::build_dataset`, the graph is built by
  `experiments::cascade::cascade_train_forward` (via
  `Engine::cascade_train_forward`), gradients flow only through the
  enricher (the backbone stays frozen).
- **Bench**: `main` -> `experiments::bench::run_bench`, reusing `Engine`
  factories and `experiments::tasks` scoring.

## Checkpoint formats

Enricher checkpoints carry an `enrich.cfg` tensor (5 floats in v2.2+: rank,
gelu, content-gate, a-mode, residual). M1-era files (`enrich.w_q/w_k/w_v/w_o`,
no cfg) are incompatible with the current residual-stream readout
(ENRICHER.md) — expected to fail loading. Saves are atomic: serialized to
`<path>.tmp` then renamed over the target, so a mid-write kill cannot corrupt
the last good checkpoint (`Enricher::save`).

## Tests

Unit tests here cover the entropy solver, the enricher spec round-trip, and
task scoring (`cargo test`).
