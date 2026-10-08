//! crossbar: Gemma-3 inference, softmax-feedback decoding, and gated
//! state-readout enricher training on candle.

pub mod assets;
pub mod bench;
pub mod chat;
pub mod cli;
pub mod config;
pub mod engine;
pub mod enrich;
pub mod feedback;
pub mod generation;
pub mod model;
pub mod quantized_model;
pub mod sampling;
pub mod tasks;
pub mod train;

pub use engine::Engine;
