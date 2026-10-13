//! crossbar: Gemma-3 inference on candle, with the experimental softmax
//! -feedback decoding and gated state-readout enricher work isolated in
//! the `experiments` submodule.

pub mod assets;
pub mod chat;
pub mod cli;
pub mod config;
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
