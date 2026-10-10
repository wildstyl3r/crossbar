//! Command-line interface: argument definitions and CLI-type conversions.

use clap::Parser;

use crate::enrich;
use crate::model;

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
    /// --probe or --repl is set).
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

    /// Print the top-5 logits of the first generated token (debug).
    #[arg(long)]
    pub debug_logits: bool,

    /// Dump embedding row and single-token logits for a token (debug).
    #[arg(long)]
    pub probe: bool,

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

#[derive(clap::Args, Debug)]
pub struct BenchArgs {
    /// Run the feedback-impact benchmark suite instead of generating from a
    /// prompt: greedy decoding over synthetic tasks (digit copy, arithmetic,
    /// counting, induction, trivia) across baseline and feedback modes.
    #[arg(long)]
    pub bench: bool,

    /// Number of examples per task for --bench (trivia caps at 40).
    #[arg(long, default_value_t = 100)]
    pub bench_n: usize,

    /// Output CSV path for --bench with per-example rows.
    #[arg(long, default_value = "bench_results.csv")]
    pub bench_out: String,

    /// Comma-separated task filter for --bench (digits-5, digits-10,
    /// digits-20, induce, induce-sh, induce-early, trivia); default runs all
    /// tasks.
    #[arg(long)]
    pub bench_tasks: Option<String>,

    /// Comma-separated mode filter for --bench (baseline, blend-d0.5,
    /// prefill-p0.5, both-0.5, cascade-k2, cascade-k3, cascade-k2-a25);
    /// 'baseline' is always required. Default runs all modes.
    #[arg(long)]
    pub bench_modes: Option<String>,
}

#[derive(clap::Args, Debug)]
pub struct FeedbackArgs {
    /// Feed the previous step's expected token embedding softmax(logits / T)
    /// @ E back into the first block during decoding, instead of embedding
    /// the sampled token; textual decoding stays normal.
    #[arg(long)]
    pub softmax_feedback: bool,

    /// Softmax temperature for --softmax-feedback; 0 approaches the argmax
    /// token's embedding (i.e. normal generation), large values approach the
    /// mean embedding. Used as the base value by the heuristic mode.
    #[arg(long, default_value_t = 1.0)]
    pub softmax_feedback_temperature: f64,

    /// Weight of the expected embedding in the decode input
    /// (1 - a) * E[sampled] + a * expected; 1.0 is the pure expected input.
    #[arg(long, default_value_t = 1.0)]
    pub softmax_feedback_blend: f64,

    /// Weight of the per-position expected embedding in the prefill input
    /// (prompt feedback, two-pass); 0 disables it.
    #[arg(long, default_value_t = 0.0)]
    pub prefill_feedback_blend: f64,

    /// Number of prefill-feedback refinement iterations (Schedule B
    /// cascade): iteration 0 observes the raw prompt, iteration k re-forwards
    /// (1 - a_p) * E + a_p * expected(logits_{k-1}) over all positions; 1 is
    /// the single refinement pass (the original prefill feedback), >= 2 the
    /// iterated cascade.
    #[arg(long, default_value_t = 1)]
    pub prefill_feedback_iters: usize,

    /// How the feedback temperature is chosen each step: the fixed value,
    /// an entropy-matched servo, or a threshold-based heuristic.
    #[arg(long, value_enum, default_value = "fixed")]
    pub softmax_feedback_mode: model::FeedbackTempMode,

    /// Target entropy (nats) of the feedback distribution for
    /// --softmax-feedback-mode entropy-servo (exact) and heuristic (band
    /// center).
    #[arg(long, default_value_t = 2.5)]
    pub softmax_feedback_target_entropy: f64,

    /// Half-width of the entropy band for --softmax-feedback-mode heuristic.
    #[arg(long, default_value_t = 1.0)]
    pub softmax_feedback_band: f64,

    /// Multiplicative decay/growth factor for --softmax-feedback-mode
    /// heuristic.
    #[arg(long, default_value_t = 0.8)]
    pub softmax_feedback_decay: f64,

    /// Consecutive-feedback cosine above which --softmax-feedback-mode
    /// heuristic decays the temperature.
    #[arg(long, default_value_t = 0.999)]
    pub softmax_feedback_max_cosine: f64,

    /// Print per-step feedback diagnostics as CSV lines to stderr.
    #[arg(long)]
    pub softmax_feedback_diagnostics: bool,
}

#[derive(clap::Args, Debug)]
pub struct TrainArgs {
    /// Run the enricher training loop instead of generating from a prompt.
    #[arg(long)]
    pub train_enricher: bool,

    /// Teacher-forced CE steps before the hard-sampled ST phase.
    #[arg(long, default_value_t = 150)]
    pub train_steps_warmup: usize,

    /// Hard-sampled ST steps after warmup (scheduled forcing 1 -> 0).
    #[arg(long, default_value_t = 300)]
    pub train_steps_st: usize,

    /// Where the trained enricher checkpoint is written / reloaded from.
    #[arg(long, default_value = "enricher.safetensors")]
    pub train_out: String,

    /// Initial gate bias of a fresh enricher (+2.0 => sigmoid ~0.88: the
    /// gate starts open and must learn to close).
    #[arg(long, default_value_t = 2.0)]
    pub train_gate_bias: f64,

    /// Linear lr ramp length (steps) at the start of warmup (0 = constant).
    #[arg(long, default_value_t = 50)]
    pub train_lr_warmup: usize,

    /// Gate floor during warmup. The ST phase anneals 0.6 * this -> 0.
    #[arg(long, default_value_t = 0.5)]
    pub train_gate_floor: f64,

    /// Starting sampling temperature of the ST phase (annealed to
    /// --train-gt-end).
    #[arg(long, default_value_t = 1.0)]
    pub train_gt_start: f64,

    /// Anneal target of the ST sampling temperature (0.1 ~= the greedy
    /// deployment regime).
    #[arg(long, default_value_t = 0.1)]
    pub train_gt_end: f64,

    /// Relative pre-gate injection penalty weight: loss + lambda * mean(
    /// ||W shat||^2 / ||q||^2 ). 0 disables.
    #[arg(long, default_value_t = 0.1)]
    pub train_inj_l2: f64,

    /// BPTT truncation horizon (in-graph cascade steps at the sequence end).
    /// Measured (ENRICHER.md): the state-chain dies by ~4 steps (c^4 <= 6e-3
    /// at healthy injection); the exact-KV path is the dominant cross-step
    /// credit at init. tau=6 is the knee of the gradient/memory curve:
    /// 72% of tau=8's gradient at 78% of its memory (5.3 vs 6.8 GB peak).
    #[arg(long, default_value_t = 6)]
    pub train_tau: usize,

    /// Keep the full KV-path gradient in the training graph (default, exact:
    /// at W=0 the KV path is the ONLY cross-step credit; detaching it
    /// captured just 32% of the gradient in the tau-sweep). Disabling it
    /// detaches keys/values for a smaller graph at a large gradient cost.
    #[arg(long, default_value_t = true)]
    pub train_kv_grad: bool,

    /// Lockstep batch size per micro-step (4 halves the in-graph memory vs
    /// 8; the cascade forward is weight-streaming-bound, so batch 4 costs
    /// nearly the same wall time as 8).
    #[arg(long, default_value_t = 4)]
    pub train_batch: usize,

    /// Micro-steps accumulated per optimizer step: effective batch =
    /// batch x accum. batch 4 x accum 4 (effective 16) is the default
    /// overnight recipe: ~3.5-4 GB peak, ~50-55 s/step, warmup 150 + ST 300
    /// ~ 7 h. accum 8 halves the step count budget; accum 16 only if noise
    /// demonstrably dominates.
    #[arg(long, default_value_t = 4)]
    pub train_accum: usize,

    /// Override the tau-sweep diagnostic grid ("tau:kv,tau:kv", e.g.
    /// "8:1"); empty runs the default comparison grid.
    #[arg(long, default_value = "")]
    pub train_sweep: String,

    /// Global-norm gradient clip over the enricher parameters.
    #[arg(long, default_value_t = 1.0)]
    pub train_grad_clip: f64,

    /// Seed for the training-data shuffle and the ST sampling rng.
    #[arg(long, default_value_t = 31337)]
    pub train_seed: u64,

    /// Skip the terminal evaluation (smoke runs).
    #[arg(long)]
    pub train_skip_eval: bool,

    /// Probe a trained enricher checkpoint (--train-out) for the ST
    /// soft-mixture artifact instead of training.
    #[arg(long)]
    pub train_probe: bool,

    /// Enricher readout rank: 0 = v0 full-rank W (default, unchanged), d > 0
    /// = factored readout inj = B phi(A shat) with A [d x H], B [H x d]
    /// (ENRICHER.md, variants v2.1).
    #[arg(long, default_value_t = 0)]
    pub train_enrich_rank: usize,

    /// Nonlinearity inside the bottleneck of a factored readout
    /// (rank > 0): none = V1 (linear), gelu = V2 (GeluPytorchTanh, matching
    /// the backbone MLP activation).
    #[arg(long, value_enum, default_value = "none")]
    pub train_enrich_act: EnrichAct,

    /// Per-bottleneck-coordinate content gate conditioned on the incoming
    /// token (V3; requires rank > 0): u = sigmoid(q Wu + U z + b), inj =
    /// g B(u * z); the injection penalty stays on the pre-u path.
    #[arg(long, default_value_t = false)]
    pub train_enrich_content_gate: bool,

    /// Treatment of the read-side matrix A (rank > 0): trainable = trained
    /// jointly with B; fixed = fixed random rows ~ N(0, 1/H); pca = set once
    /// from a PCA of collected states via a no-grad calibration pass.
    #[arg(long, value_enum, default_value = "trainable")]
    pub train_enrich_a: EnrichAMode,

    /// Readout-source treatment (ENRICHER.md, Claim R): pick = the readout
    /// sees the pick-orthogonalized state `rmsnorm(shat − alpha·q)` with
    /// alpha = <shat, q>/||q||² (the thought-residual — the state minus its
    /// projection onto the fed token's direction); none = the raw state
    /// (v2.1). Parameter-free; the gate keeps the raw-state view.
    #[arg(long, value_enum, default_value = "none")]
    pub train_enrich_residual: EnrichResidual,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum EnrichAct {
    None,
    Gelu,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum EnrichAMode {
    Trainable,
    Fixed,
    Pca,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum EnrichResidual {
    None,
    Pick,
}

impl Args {
    pub fn feedback_config(&self) -> model::FeedbackConfig {
        let fb = &self.feedback_args;
        model::FeedbackConfig {
            mode: fb.softmax_feedback_mode,
            temperature: fb.softmax_feedback_temperature,
            target_entropy: fb.softmax_feedback_target_entropy,
            band: fb.softmax_feedback_band,
            decay: fb.softmax_feedback_decay,
            max_cosine: fb.softmax_feedback_max_cosine,
            blend: fb.softmax_feedback_blend,
            prefill_blend: fb.prefill_feedback_blend,
            prefill_iters: fb.prefill_feedback_iters,
        }
    }

    pub fn enrich_residual(&self) -> enrich::ResidualMode {
        match self.train_args.train_enrich_residual {
            EnrichResidual::None => enrich::ResidualMode::None,
            EnrichResidual::Pick => enrich::ResidualMode::Pick,
        }
    }

    pub fn enricher_spec(&self) -> enrich::EnricherSpec {
        let t = &self.train_args;
        enrich::EnricherSpec {
            rank: t.train_enrich_rank,
            gelu: matches!(t.train_enrich_act, EnrichAct::Gelu),
            content_gate: t.train_enrich_content_gate,
            a_mode: match t.train_enrich_a {
                EnrichAMode::Trainable => enrich::AMode::Trainable,
                EnrichAMode::Fixed => enrich::AMode::FixedRandom,
                EnrichAMode::Pca => enrich::AMode::Pca,
            },
            residual: self.enrich_residual(),
        }
    }
}
