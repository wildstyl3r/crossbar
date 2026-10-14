//! crossbar: Gemma-3 inference on candle. The experimental softmax-feedback
//! decoding and gated state-readout enricher work (specs, CLI groups,
//! training, bench) lives in the `experiments` submodule; the decode-time
//! hooks it plugs into are part of the core backends.

pub mod assets;
pub mod chat;
pub mod cli;
pub mod config;
pub mod download;
pub mod engine;
pub mod experiments;
pub mod generation;
pub mod grammar;
pub mod model;
pub mod quantized_model;
pub mod repl;
pub mod sampling;
pub mod tools;

pub use engine::Engine;
