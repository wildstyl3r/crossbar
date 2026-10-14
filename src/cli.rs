//! Command-line interface: core argument definitions. The experimental
//! argument groups (`FeedbackArgs`, `TrainArgs`, `BenchArgs`) live in
//! `experiments::args` and are flattened in here.

use clap::Parser;

use crate::experiments::args::{BenchArgs, FeedbackArgs, TrainArgs};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run an interactive chat REPL instead of one-shot generation
    /// (--prompt is ignored).
    #[arg(long)]
    pub repl: bool,

    /// Tool calling mode for the REPL: off, auto (parse replies for tool
    /// calls) or required (retry when a reply is not a tool call).
    #[arg(long, value_enum, default_value = "off")]
    pub tools: crate::tools::ToolMode,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,

    /// The prompt to run the generator with (required unless --bench,
    /// --train-enricher or --repl is set).
    #[arg(long, short = 'p')]
    pub prompt: Option<String>,

    /// Optional system prompt, folded into the first user turn.
    #[arg(long)]
    pub system: Option<String>,

    /// Directory containing the model files; auto-detects GGUF (quantized) vs safetensors.
    #[arg(long, default_value = "models/gemma-3-270m-it-qat-GGUF")]
    pub model_dir: String,

    /// Directory to fall back to for tokenizer/chat template/generation config
    /// when they are missing from --model-dir.
    #[arg(long, default_value = "models/gemma-3-270m-it")]
    pub fallback_dir: String,

    /// The temperature used to generate samples, 0 means greedy decoding.
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Top-k sampling, 0 disables it.
    #[arg(long)]
    pub top_k: Option<usize>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    pub top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    pub sample_len: usize,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,

    /// Load a trained gated state-readout enricher checkpoint; prefill
    /// becomes the sequential forced-token cascade and every decode step is
    /// enriched from the previous state (parallel prefill is never used).
    #[arg(long)]
    pub enricher: Option<String>,

    #[command(flatten)]
    pub bench_args: BenchArgs,

    #[command(flatten)]
    pub feedback_args: FeedbackArgs,

    #[command(flatten)]
    pub train_args: TrainArgs,
}
