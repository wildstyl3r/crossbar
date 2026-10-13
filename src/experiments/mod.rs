//! Experimental mechanisms layered on top of the plain Gemma-3 engine.
//!
//! Nothing in this module is required to run the base model: the core
//! (`model`, `quantized_model`, `engine`, `generation`, ...) only imports
//! types from here and exposes the decode-time hooks these experiments plug
//! into (the softmax-feedback / prefill-cascade / enricher-cascade paths
//! inside the backends). Design notes and experiment logs live alongside
//! the code they describe: `FEEDBACK.md` (softmax-feedback decoding) and
//! `ENRICHER.md` (the gated state-readout enricher and its training).
//!
//! | Module | Responsibility |
//! |---|---|
//! | `feedback` | softmax-feedback types (`FeedbackConfig`, stats) and the entropy-solver math |
//! | `enrich` | the enricher: variant spec, serialization (`enrich.cfg` tensor), gated readout math, monitors |
//! | `cascade` | the training-only cascade forward (TBPTT, detached-head CE trick) — touches `Model`/`DecoderLayer` internals |
//! | `train` | enricher training loop: batching, gradient accumulation, tau-sweep diagnostics, PCA calibration, eval, probe |
//! | `tasks` | synthetic task generators (state-track, induce-d) + exact-match scoring, shared by trainer and bench |
//! | `bench` | feedback-impact benchmark suite: tasks, modes, McNemar stats, CSV output |
//! | `args` | CLI argument groups (`FeedbackArgs`, `TrainArgs`, `BenchArgs`) + CLI-to-domain converters |

pub mod args;
pub mod bench;
pub mod cascade;
pub mod enrich;
pub mod feedback;
pub mod tasks;
pub mod train;
